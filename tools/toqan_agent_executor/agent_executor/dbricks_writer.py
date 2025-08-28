import json
import textwrap
import asyncio
from typing import Dict, List, Optional
import httpx
import uuid
import random
from datetime import datetime

def _escape_json_for_sql(value):
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
        escaped = json.dumps(parsed, ensure_ascii=False)
        escaped = escaped.replace('\\', '\\\\')
        escaped = escaped.replace("'", "''")
        return escaped
    except Exception:
        return str(value).replace("'", "''").replace("\\", "\\\\")


class DatabricksWriter:
    """
    Envia linhas para o Databricks via SQL Statements API (2.0).

    Esquema recomendado:
      CREATE TABLE <schema>.<table> (
        id STRING,                -- chave única da linha (UUID por evento)
        orchestration_id STRING,  -- agrupa uma execução/fluxo (UUID do produto/run)
        input STRING,
        answer STRING,
        created_at TIMESTAMP,
        agent STRING
      )
      USING DELTA;

    Chave de upsert: ON target.id = source.id
    """

    def __init__(
        self,
        poll_interval: float = 2.0,
        max_poll_attempts: int = 60,
    ):
        self.host = 'https://ifood-prod-main.cloud.databricks.com'
        self.token = ''
        self.warehouse_id = ''
        self.table = 'groceries_sandbox.toqan_answers'
        self.poll_interval = poll_interval
        self.max_poll_attempts = max_poll_attempts

        missing = []
        if not self.host: missing.append("DATABRICKS_HOST")
        if not self.token: missing.append("DATABRICKS_TOKEN")
        if not self.warehouse_id: missing.append("DATABRICKS_WAREHOUSE_ID")
        if not self.table: missing.append("DATABRICKS_TABLE")
        self.enabled = len(missing) == 0
        self._missing = missing

        # Evita MERGE concorrente no mesmo processo
        self._merge_lock = asyncio.Lock()

    def is_enabled(self) -> bool:
        return self.enabled

    def missing_vars(self) -> List[str]:
        return self._missing

    async def _post_statement(self, client: httpx.AsyncClient, sql: str) -> Dict:
        api_base = self.host.rstrip("/") + "/api/2.0/sql/statements"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        payload = {"statement": sql, "warehouse_id": self.warehouse_id}
        r = await client.post(api_base, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()

    async def _poll_statement(self, client: httpx.AsyncClient, statement_id: str) -> Dict:
        api_base = self.host.rstrip("/") + "/api/2.0/sql/statements"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        for _ in range(self.max_poll_attempts):
            await asyncio.sleep(self.poll_interval)
            r = await client.get(f"{api_base}/{statement_id}", headers=headers, timeout=120)
            r.raise_for_status()
            data = r.json()
            state = (data.get("status", {}) or {}).get("state")
            if state not in ("PENDING", "RUNNING"):
                return data
        return {"status": {"state": "TIMEOUT"}}

    async def _execute_with_retry(self, client: httpx.AsyncClient, sql: str, *, max_retries: int = 5) -> Dict:
        """
        Executa um statement com retry exponencial em caso de erro de concorrência do Delta.
        """
        backoff = 0.6
        for attempt in range(max_retries):
            try:
                result = await self._post_statement(client, sql)

                # Poll caso necessário
                state = (result.get("status", {}) or {}).get("state")
                if state not in ("SUCCEEDED", "FAILED"):
                    statement_id = result.get("statement_id")
                    if statement_id:
                        result = await self._poll_statement(client, statement_id)

                final_state = (result.get("status", {}) or {}).get("state")
                if final_state == "SUCCEEDED":
                    return result

                # Se falhou, vê se é concorrência (DELTA_CONCURRENT...)
                err_msg = ((result.get("status", {}) or {}).get("error", {}) or {}).get("message", "")
                msg = (err_msg or "").upper()
                is_concurrency = (
                    "DELTA_CONCURRENT" in msg
                    or "CONCURRENTAPPEND" in msg
                    or "CONCURRENT APPEND" in msg
                    or "CONCURRENT UPDATE" in msg
                )
                if is_concurrency and attempt < max_retries - 1:
                    await asyncio.sleep(backoff + random.uniform(0, 0.35))
                    backoff = min(backoff * 2, 5.0)
                    continue

                return result  # outro erro: não retry
            except Exception:
                if attempt < max_retries - 1:
                    await asyncio.sleep(backoff + random.uniform(0, 0.35))
                    backoff = min(backoff * 2, 5.0)
                    continue
                raise

    async def merge_rows(self, rows: List[Dict[str, str]], batch_size: int = 200) -> None:
        """
        Faz upsert por id.
        Espera linhas no formato:
          {
            "id": "<uuid por evento>",                 # se ausente, será gerado
            "orchestration_id": "<uuid do run>",       # agrupa a execução
            "input": "...", "answer": "...",
            "created_at": "YYYY-MM-DD HH:MM:SS[.ffffff]",
            "agent": "..."
          }
        """
        if not self.enabled or not rows:
            return

        timeout = httpx.Timeout(120.0)
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            for i in range(0, len(rows), batch_size):
                sub = rows[i:i + batch_size]
                values_sql_parts = []
                for r in sub:
                    _id = (r.get("id") or str(uuid.uuid4())).replace("'", "''")  # usa o id do caller; gera se faltar
                    _orch = (r.get("orchestration_id") or "").replace("'", "''")
                    _input = (r.get("input") or "").replace("'", "''")
                    _answer = _escape_json_for_sql(r.get("answer", "{}"))
                    _created_at = (r.get("created_at") or "")
                    _agent = _escape_json_for_sql(r.get("agent", ""))

                    if not _input:
                        # sem input não grava
                        continue

                    values_sql_parts.append(
                        f"('{_input}', '{_answer}', '{_created_at}', '{_agent}', '{_orch}', '{_id}')"
                    )

                if not values_sql_parts:
                    continue

                values_sql = ",".join(values_sql_parts)
                sql_statement = f"""
                    MERGE INTO {self.table} AS target
                    USING (
                        SELECT
                            col1 AS input,
                            col2 AS answer,
                            to_timestamp(col3) AS created_at,
                            col4 AS agent,
                            col5 AS orchestration_id,
                            col6 AS id
                        FROM (VALUES {values_sql}) AS temp(col1, col2, col3, col4, col5, col6)
                    ) AS source
                    ON target.id = source.id
                    WHEN MATCHED AND source.created_at > target.created_at THEN
                      UPDATE SET input = source.input,
                                 answer = source.answer,
                                 created_at = source.created_at,
                                 agent = source.agent,
                                 orchestration_id = source.orchestration_id
                    WHEN NOT MATCHED THEN
                      INSERT (input, answer, created_at, agent, orchestration_id, id)
                      VALUES (source.input, source.answer, source.created_at, source.agent, source.orchestration_id, source.id)
                """
                sql_statement = textwrap.dedent(sql_statement)

                # serializa MERGE no processo + retry em concorrência
                async with self._merge_lock:
                    result = await self._execute_with_retry(client, sql_statement)

                final_state = (result.get("status", {}) or {}).get("state")
                if final_state == "FAILED":
                    err_msg = (result.get("status", {}).get("error", {}) or {}).get("message", "Erro desconhecido")
                    print(f"❌ [DBRICKS] Sub-lote falhou: {err_msg}")
                elif final_state == "SUCCEEDED":
                    print(f"✅ [DBRICKS] Sub-lote ({len(sub)}) enviado.")
                else:
                    print(f"⚠️ [DBRICKS] Estado final: {final_state}")

    async def insert_rows(self, rows: List[Dict[str, str]], batch_size: int = 200) -> None:
        """
        Modo append-only (sem upsert). Evita conflitos de MERGE.
        """
        if not self.enabled or not rows:
            return

        timeout = httpx.Timeout(120.0)
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            for i in range(0, len(rows), batch_size):
                sub = rows[i:i + batch_size]
                values_sql_parts = []
                for r in sub:
                    _id = (r.get("id") or str(uuid.uuid4())).replace("'", "''")
                    _orch = (r.get("orchestration_id") or "").replace("'", "''")
                    _input = (r.get("input") or "").replace("'", "''")
                    _answer = _escape_json_for_sql(r.get("answer", "{}"))
                    _created_at = (r.get("created_at") or "")
                    _agent = _escape_json_for_sql(r.get("agent", ""))

                    if not _input:
                        continue

                    values_sql_parts.append(
                        f"('{_id}', '{_orch}', '{_input}', '{_answer}', '{_created_at}', '{_agent}')"
                    )

                if not values_sql_parts:
                    continue

                values_sql = ",".join(values_sql_parts)
                sql_statement = f"""
                    INSERT INTO {self.table} (id, orchestration_id, input, answer, created_at, agent)
                    SELECT col1, col2, col3, col4, to_timestamp(col5), col6
                    FROM (VALUES {values_sql}) AS temp(col1, col2, col3, col4, col5, col6)
                """
                sql_statement = textwrap.dedent(sql_statement)

                # INSERTs concorrentes são seguros, mas ainda usamos retry para robustez
                result = await self._execute_with_retry(client, sql_statement)
                final_state = (result.get("status", {}) or {}).get("state")
                if final_state == "FAILED":
                    err_msg = (result.get("status", {}).get("error", {}) or {}).get("message", "Erro desconhecido")
                    print(f"❌ [DBRICKS] Sub-lote (INSERT) falhou: {err_msg}")
                elif final_state == "SUCCEEDED":
                    print(f"✅ [DBRICKS] Sub-lote (INSERT) ({len(sub)}) enviado.")
                else:
                    print(f"⚠️ [DBRICKS] Estado final (INSERT): {final_state}")