# dbricks_writer.py
import json
import textwrap
import asyncio
from typing import Dict, List, Optional
import httpx
import re
import os
import uuid
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
    Schema esperado (ajuste conforme sua tabela):
      CREATE TABLE <schema>.<table> (
        input STRING,
        answer STRING,
        created_at TIMESTAMP,
        agent STRING
      );
    Chave de merge: ON target.input = source.input
    """

    def __init__(
        self,
        poll_interval: float = 2.0,
        max_poll_attempts: int = 60,
    ):
        self.host = ''
        self.token = ''
        self.warehouse_id = ''
        self.table = 'groceries_sandbox.toqan_answers'
        self.poll_interval = poll_interval
        self.max_poll_attempts = max_poll_attempts

        # validação mínima
        missing = []
        if not self.host: missing.append("DATABRICKS_HOST")
        if not self.token: missing.append("DATABRICKS_TOKEN")
        if not self.warehouse_id: missing.append("DATABRICKS_WAREHOUSE_ID")
        if not self.table: missing.append("DATABRICKS_TABLE")
        self.enabled = len(missing) == 0
        self._missing = missing

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

    async def merge_rows(self, rows: List[Dict[str, str]], batch_size: int = 200) -> None:
        """
        rows: [ {"input": str, "answer": str(json), "created_at": str(UTC 'YYYY-MM-DD HH:MM:SS'), "agent": str}, ...]
        """
        if not self.enabled or not rows:
            return

        timeout = httpx.Timeout(120.0)
        limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            for i in range(0, len(rows), batch_size):
                sub = rows[i:i+batch_size]
                values_sql_parts = []
                for r in sub:
                    _input = (r.get("input") or "").replace("'", "''")
                    _answer = _escape_json_for_sql(r.get("answer", "{}"))
                    _created_at = (r.get("created_at") or "")
                    _agent = _escape_json_for_sql(r.get("agent", ""))
                    _orchestration_id = _escape_json_for_sql(r.get("orchestration_id", ""))

                    if not _input:
                        continue
                    values_sql_parts.append(
                        f"('{_input}', '{_answer}', '{_created_at}', '{_agent}', '{_orchestration_id}', '{str(uuid.uuid4())}')"
                    )
                if not values_sql_parts:
                    continue

                values_sql = ",".join(values_sql_parts)
                sql_statement = f"""
                    MERGE INTO {self.table} AS target
                    USING (
                        SELECT col1 AS input, col2 AS answer, to_timestamp(col3) AS created_at, col4 AS agent, col5 AS orchestration_id, col6 AS id
                        FROM (VALUES {values_sql}) AS temp(col1, col2, col3, col4, col5, col6)
                    ) AS source
                    ON target.id  = source.id 
                    WHEN MATCHED AND source.created_at > target.created_at THEN
                      UPDATE SET answer = source.answer, created_at = source.created_at, agent = source.agent
                    WHEN NOT MATCHED THEN
                      INSERT (input, answer, created_at, agent, orchestration_id, id) VALUES (source.input, source.answer, source.created_at, source.agent, source.orchestration_id, source.id)
                """
                sql_statement = textwrap.dedent(sql_statement)

                try:
                    result = await self._post_statement(client, sql_statement)
                except Exception as e:
                    print(f"[DBRICKS][POST][ERRO] {e}")
                    continue

                state = (result.get("status", {}) or {}).get("state")
                if state in ("SUCCEEDED", "FAILED"):
                    poll_result = result
                else:
                    statement_id = result.get("statement_id")
                    if statement_id:
                        poll_result = await self._poll_statement(client, statement_id)
                    else:
                        poll_result = result

                final_state = (poll_result.get("status", {}) or {}).get("state")
                if final_state == "FAILED":
                    err_msg = (poll_result.get("status", {}).get("error", {}) or {}).get("message", "Erro desconhecido")
                    print(f"❌ [DBRICKS] Sub-lote falhou: {err_msg}")
                elif final_state == "SUCCEEDED":
                    print(f"✅ [DBRICKS] Sub-lote ({len(sub)}) enviado.")
                else:
                    print(f"⚠️ [DBRICKS] Estado final: {final_state}")
