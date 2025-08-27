#!/usr/bin/env python3
import asyncio
import aiohttp
import csv
import os
import sys
import json
import time
import argparse
import subprocess
import re
import hashlib
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any

# ------------------------------
# Databricks debug flag and logger
# ------------------------------
DBRICKS_DEBUG = False
def dbg(msg: str):
    if DBRICKS_DEBUG:
        print(f"[DBRICKS][DEBUG] {msg}")

API_BASE = "https://api.coco.prod.toqan.ai/api"

# ------------------------------
# Utils
# ------------------------------

def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def load_inputs(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        print(f"[ERRO] CSV não encontrado: {csv_path}", file=sys.stderr)
        sys.exit(1)

    inputs: List[str] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            val = (row[0] or "").strip()
            if val:
                inputs.append(val)
    return inputs


def rewrite_pending_csv(csv_path: Path, all_rows: List[Dict[str, str]], pending_inputs: set) -> None:
    """
    Reescreve o CSV mantendo apenas os inputs pendentes, sem cabeçalho e com
    apenas uma coluna (valor bruto), pois o arquivo original não possui header.
    """
    # Extrai a lista (na mesma ordem do arquivo original) dos valores ainda pendentes
    remaining_inputs: List[str] = []
    for r in all_rows:
        val = (r.get("input") or "").strip()
        if val and val in pending_inputs:
            remaining_inputs.append(val)

    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for v in remaining_inputs:
            writer.writerow([v])
    tmp.replace(csv_path)

def escape_json_for_sql(value):
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
        escaped = json.dumps(parsed, ensure_ascii=False)
        escaped = escaped.replace('\\', '\\\\')
        escaped = escaped.replace("'", "''")
        return escaped
    except Exception:
        return str(value).replace("'", "''").replace("\\", "\\\\")


def safe_filename(name: str, max_length: int = 100) -> str:
    # Substitui caracteres inválidos por "_"
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    # Limita tamanho
    safe = safe[:max_length]
    # Acrescenta hash curto para garantir unicidade
    digest = hashlib.sha1(name.encode()).hexdigest()[:8]
    return f"{safe}_{digest}"

# ------------------------------
# Helpers for JSON -> CSV consolidation
# ------------------------------

def flatten(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Achata dicionários aninhados em chaves com notação ponto (a.b.c)."""
    items: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten(v, key, sep=sep))
        else:
            items[key] = v
    return items


def extract_json_from_answer(answer: Any) -> Dict[str, Any]:
    """Retorna um dict a partir de `answer`:
       - se já for dict, retorna;
       - se for string, extrai o trecho entre a 1ª '{' e a última '}' e faz json.loads;
       - caso contrário, lança ValueError."""
    if isinstance(answer, dict):
        return answer

    if isinstance(answer, str):
        start = answer.find("{")
        end = answer.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = answer[start:end+1]
            snippet = re.sub(r"^```(?:json)?\s*|\s*```$", "", snippet.strip(), flags=re.IGNORECASE)
            return json.loads(snippet)

    raise ValueError("answer não contém um JSON válido.")


def consolidate_answers(indir: Path, out_csv: Path, pattern: str = "*.json", sep: str = ".") -> None:
    files = sorted(indir.glob(pattern))
    if not files:
        print(f"[AVISO] Nenhum arquivo encontrado em {indir} com padrão {pattern}.", file=sys.stderr)
        return

    rows: List[Dict[str, Any]] = []
    all_keys: set = set()

    for fp in files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ERRO] Falha ao ler {fp}: {e}", file=sys.stderr)
            continue

        answer = data.get("answer")
        try:
            parsed = extract_json_from_answer(answer)
        except Exception as e:
            print(f"[ERRO] {fp.name}: não foi possível extrair JSON de 'answer' ({e})", file=sys.stderr)
            continue

        flat = flatten(parsed, sep=sep) if isinstance(parsed, dict) else {}
        flat["__source_file"] = str(fp)
        # Drop hash suffix from filename (Databricks rule)
        stem = fp.stem
        m = re.match(r"^(.*)_[0-9a-fA-F]{8}$", stem)
        input_nohash = m.group(1) if m else stem
        flat["__ean_from_filename"] = input_nohash
        rows.append(flat)
        all_keys.update(flat.keys())

    if not rows:
        print("[AVISO] Nenhum registro válido para exportar.", file=sys.stderr)
        return

    fieldnames = sorted(all_keys)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"[OK] CSV gerado: {out_csv} ({len(rows)} linhas)")

# Row shape:
# {
#   "input": str,
#   "answer": dict (JSON),
#   "created_at": str (timestamp),
#   "agent": str
# }

# --- Databricks helpers ---
def _parse_schema_table(ident: str) -> Dict[str, str]:
    ident = (ident or "").strip()
    parts = ident.split(".")
    if len(parts) == 3:
        catalog, schema, table = parts
        return {"catalog": catalog.strip(), "schema": schema.strip(), "table": table.strip(), "full": f"{catalog.strip()}.{schema.strip()}.{table.strip()}"}
    elif len(parts) == 2:
        schema, table = parts
        schema = schema.strip()
        table = table.strip()
        return {"catalog": "main", "schema": schema, "table": table, "full": f"main.{schema}.{table}"}
    else:
        # fallback para nome simples -> main.<ident>
        ident = ident.strip()
        return {"catalog": "main", "schema": "main", "table": ident, "full": f"main.main.{ident}"}

async def dbricks_exec_sql(session: aiohttp.ClientSession, host: str, token: str, warehouse_id: str, sql: str, catalog: str = "", schema: str = "", poll_interval: float = 2.0, max_poll_attempts: int = 60) -> Dict:
    api_base = host.rstrip("/") + "/api/2.0/sql/statements"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {"statement": textwrap.dedent(sql), "warehouse_id": warehouse_id}
    if catalog:
        payload["catalog"] = catalog
    if schema:
        payload["schema"] = schema
    # Debug logs before submit
    dbg(f"Warehouse: {warehouse_id} | Catalog: {catalog or '(default)'} | Schema: {schema or '(default)'}")
    dbg(f"Endpoint: {api_base}")
    try:
        sql_trim = textwrap.dedent(sql).strip()
        dbg(f"SQL ({len(sql_trim)} chars): {sql_trim[:1200]}{' ...[truncado]' if len(sql_trim) > 1200 else ''}")
    except Exception:
        pass
    dbg(f"Payload keys: {list(payload.keys())}")
    # Submit
    async with session.post(api_base, headers=headers, json=payload, timeout=120) as resp:
        status_code = resp.status
        text_body = await resp.text()
        dbg(f"POST status={status_code}, body_len={len(text_body)}")
        try:
            result = json.loads(text_body)
        except Exception:
            result = {"raw_text": text_body, "status_code": status_code}
        if status_code >= 400:
            return {"status": {"state": "FAILED", "error": {"message": f"HTTP {status_code}", "details": text_body}}, "status_code": status_code}
    state = (result.get("status", {}) or {}).get("state")
    if state in ("SUCCEEDED", "FAILED"):
        return result
    statement_id = result.get("statement_id")
    poll_result = result
    for _ in range(max_poll_attempts):
        await asyncio.sleep(poll_interval)
        dbg(f"Polling {api_base}/{statement_id}")
        async with session.get(f"{api_base}/{statement_id}", headers=headers, timeout=120) as poll_resp:
            poll_result = await poll_resp.json()
        dbg(f"Poll state={ (poll_result.get('status', {}) or {}).get('state') }")
        state = (poll_result.get("status", {}) or {}).get("state")
        if state not in ("PENDING", "RUNNING"):
            break
    return poll_result

async def dbricks_ensure_table_exists(session: aiohttp.ClientSession, host: str, token: str, warehouse_id: str, table_ident: str) -> None:
    parts = _parse_schema_table(table_ident)
    full = parts["full"]
    catalog = parts.get("catalog", "")

    # cria schema se não existir
    if catalog:
        create_schema_sql = f"CREATE SCHEMA IF NOT EXISTS {catalog}.{parts['schema']}"
        dbg(f"CREATE SCHEMA SQL: {create_schema_sql}")
        await dbricks_exec_sql(session, host, token, warehouse_id, create_schema_sql, catalog=catalog, schema=parts["schema"])
    else:
        create_schema_sql = f"CREATE SCHEMA IF NOT EXISTS {parts['schema']}"
        dbg(f"CREATE SCHEMA SQL: {create_schema_sql}")
        await dbricks_exec_sql(session, host, token, warehouse_id, create_schema_sql, schema=parts["schema"])

    # cria tabela se não existir (Delta)
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {full} (
        input STRING,
        answer STRING,
        created_at TIMESTAMP,
        agent STRING
    ) USING DELTA
    """
    dbg(f"CREATE TABLE SQL for {full}: {create_table_sql.strip()[:800]}{' ...[truncado]' if len(create_table_sql) > 800 else ''}")
    res = await dbricks_exec_sql(session, host, token, warehouse_id, create_table_sql, catalog=catalog, schema=parts["schema"])
    state = (res.get("status", {}) or {}).get("state")
    if state != "SUCCEEDED":
        msg = (res.get("status", {}).get("error", {}) or {}).get("message", "Erro ao garantir tabela")
        details = (res.get("status", {}).get("error", {}) or {}).get("details", "")
        print(f"❌ [Databricks] Falha ao garantir tabela {full}: {msg}. {details[:300]}", file=sys.stderr)
        return
    print(f"✅ [Databricks] Tabela garantida: {full}")

async def dbricks_merge_rows(session: aiohttp.ClientSession, host: str, token: str, warehouse_id: str, table: str, rows: List[Dict[str, str]], batch_size: int = 1000, poll_interval: float = 2.0, max_poll_attempts: int = 60) -> None:
    if not rows:
        return
    api_base = host.rstrip("/") + "/api/2.0/sql/statements"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    parts = _parse_schema_table(table)
    catalog = parts.get("catalog", "")
    schema = parts["schema"]
    full_table = parts["full"]
    dbg(f"MERGE target: {full_table} | Catalog: {catalog or '(default)'} | Schema: {schema}")
    dbg(f"Total rows to send: {len(rows)} | Batch size: {batch_size}")
    # Envia em sublotes
    for i in range(0, len(rows), batch_size):
        sub = rows[i:i+batch_size]
        dbg(f"Batch rows: {len(sub)} (idx {i}-{i+len(sub)-1})")
        if sub:
            sample = sub[0].copy()
            sample["answer"] = "<json omitted>"
            dbg(f"Sample row: {sample}")
        values_sql_parts = []
        for r in sub:
            input = (r.get("input") or "").strip()
            if not input:
                continue
            answer = escape_json_for_sql(r.get("answer", "{}"))
            created_at = (r.get("created_at") or "")
            agent = escape_json_for_sql(r.get("agent", ""))
            values_sql_parts.append(f"('{input}', '{answer}', '{created_at}', '{agent}')")
        if not values_sql_parts:
            continue
        values_sql = ",".join(values_sql_parts)
        print(created_at)
        sql_statement = f"""
            MERGE INTO {full_table} AS target
            USING (
                SELECT col1 AS input, col2 AS answer, to_timestamp(col3) AS created_at, col4 AS agent
                FROM (VALUES {values_sql}) AS temp(col1, col2, col3, col4)
            ) AS source
            ON target.input = source.input
            WHEN MATCHED AND source.created_at > target.created_at THEN
              UPDATE SET answer = source.answer, created_at = source.created_at, agent = source.agent
            WHEN NOT MATCHED THEN
              INSERT (input, answer, created_at, agent) VALUES (source.input, source.answer, source.created_at, source.agent)
        """
        dbg(f"MERGE SQL ({len(sql_statement)} chars)")
        payload = {"statement": textwrap.dedent(sql_statement), "warehouse_id": warehouse_id}
        if catalog:
            payload["catalog"] = catalog
        if schema:
            payload["schema"] = schema
        dbg(f"POST MERGE to {api_base} with payload keys: {list(payload.keys())}")
        # Submit
        async with session.post(api_base, headers=headers, json=payload, timeout=120) as resp:
            status_code = resp.status
            text_body = await resp.text()
            dbg(f"POST MERGE status={status_code}, body_len={len(text_body)}")
            try:
                result = json.loads(text_body)
            except Exception:
                result = {"raw_text": text_body, "status_code": status_code}
        # Fast-fail on HTTP error to avoid polling with statement_id=None
        if status_code >= 400:
            snippet = text_body[:400] if 'text_body' in locals() else str(result)[:400]
            print(f"❌ [Databricks] MERGE HTTP {status_code}. Body: {snippet}", file=sys.stderr)
            return
        state = (result.get("status", {}) or {}).get("state")
        if state in ("SUCCEEDED", "FAILED"):
            poll_result = result
        else:
            statement_id = result.get("statement_id")
            poll_result = result
            for _ in range(max_poll_attempts):
                await asyncio.sleep(poll_interval)
                async with session.get(f"{api_base}/{statement_id}", headers=headers, timeout=120) as poll_resp:
                    poll_result = await poll_resp.json()
                state = (poll_result.get("status", {}) or {}).get("state")
                if state not in ("PENDING", "RUNNING"):
                    break
        dbg(f"Final state for batch: {state}")
        if state != "SUCCEEDED":
            err_msg = (poll_result.get("status", {}).get("error", {}) or {}).get("message", "Erro desconhecido")
            details = (poll_result.get("status", {}).get("error", {}) or {}).get("details", "")
            dbg(f"Error payload snippet: {str(poll_result)[:1200]}")
            print(f"❌ [Databricks] Sub-lote falhou: {err_msg}. {details[:300]}", file=sys.stderr)
        else:
            print(f"✅ [Databricks] Sub-lote enviado ({len(sub)} linhas).")

# ------------------------------
# Pending manager (thread-safe/async-safe)
# ------------------------------

class PendingManager:
    def __init__(self, csv_path: Path, initial_rows: List[Dict[str, str]], inputs: List[str]):
        self.csv_path = csv_path
        self._all_rows = initial_rows
        self._pending = set(inputs)
        self._lock = asyncio.Lock()
        self._writes = 0

    def is_pending(self, input: str) -> bool:
        return input in self._pending

    async def mark_done_and_flush(self, input: str):
        async with self._lock:
            if input in self._pending:
                self._pending.remove(input)
                # Reescreve o CSV a cada operação para ser à prova de queda/interrupção.
                # (Se preferir reduzir IO, troque para a cada N operações.)
                rewrite_pending_csv(self.csv_path, self._all_rows, self._pending)
                self._writes += 1

# ------------------------------
# HTTP logic
# ------------------------------

class ToqanClient:
    def __init__(self, api_key: str, session: aiohttp.ClientSession, poll_interval: int = 30):
        self.api_key = api_key
        self.session = session
        self.poll_interval = poll_interval

    def _headers(self) -> Dict[str, str]:
        return {
            "X-Api-Key": self.api_key,
            "accept": "*/*",
            "content-type": "application/json",
        }

    async def create_conversation(self, input: str, retries: int = 3, backoff: float = 1.5) -> Dict:
        payload = {"user_message": input}
        url = f"{API_BASE}/create_conversation"
        last_err: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                async with self.session.post(url, headers=self._headers(), json=payload, timeout=30) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        text = await resp.text()
                        raise RuntimeError(f"HTTP {resp.status}: {text}")
            except Exception as e:
                last_err = e
                if attempt < retries:
                    await asyncio.sleep(backoff ** attempt)
        raise RuntimeError(f"Falha ao criar conversa para Input {input}: {last_err}")

    async def poll_answer_until_ready(self, conversation_id: str, request_id: str) -> Dict:
        """
        Faz GET em /get_answer até status != 'in_progress'.
        Salva e retorna o JSON final (independente de status final ser 'done' ou 'error').
        """
        url = f"{API_BASE}/get_answer"
        params = {"conversation_id": conversation_id, "request_id": request_id}

        while True:
            async with self.session.get(url, headers=self._headers(), params=params, timeout=30) as resp:
                data = await resp.json()
                status = (data.get("status") or "").lower()
                if status and status != "in_progress":
                    return data
            print(f"[POLL] Input {conversation_id[:8]}... ainda em progresso, aguardando {self.poll_interval}s")
            await asyncio.sleep(self.poll_interval)

# ------------------------------
# Worker
# ------------------------------

async def process_input(
    input: str,
    client: ToqanClient,
    out_dir: Path,
    pending_mgr: PendingManager,
    semaphore: asyncio.Semaphore,
    agent_name: str
):
    row_result = None
    # Se o arquivo já existe, pulamos e removemos do pending
    out_path = out_dir / f"{safe_filename(input)}.json"
    if out_path.exists():
        print(f"[SKIP] Já existe saída para {input}, removendo do CSV pendente.")
        await pending_mgr.mark_done_and_flush(input)
        return None

    print(f"[INI] Iniciando processamento do Input {input}")

    async with semaphore:
        try:
            max_attempts = 3
            attempt = 1
            final_answer = None
            while attempt <= max_attempts:
                try:
                    create_resp = await client.create_conversation(input)
                    conversation_id = create_resp.get("conversation_id")
                    request_id = create_resp.get("request_id")
                    if not conversation_id or not request_id:
                        raise RuntimeError(f"Resposta inesperada em create_conversation para {input}: {create_resp}")

                    print(f"[POST] Input {input}: conversa criada (conversation_id={conversation_id}, request_id={request_id}) [tentativa {attempt}/{max_attempts}]")

                    answer = await client.poll_answer_until_ready(conversation_id, request_id)
                    status = (answer.get("status") or "").lower()

                    if status == "error":
                        print(f"[RETENTAR] Input {input}: status=error na tentativa {attempt}. Retentando...")
                        attempt += 1
                        continue

                    # sucesso (status diferente de 'error')
                    final_answer = answer
                    break
                except Exception as inner_e:
                    # falha técnica na tentativa atual -> retentar
                    print(f"[RETENTAR] Input {input}: falha na tentativa {attempt}: {inner_e}", file=sys.stderr)
                    attempt += 1

            if final_answer is None:
                # Após 3 tentativas com status=error ou falhas técnicas, NÃO remover do CSV
                print(f"[ERRO] Input {input}: não foi possível obter resposta válida após {max_attempts} tentativas. Mantendo no CSV pendente.", file=sys.stderr)
                row_result = None
            else:
                print(f"[DONE] Input {input}: resposta concluída, salvando...")

                # Salva JSON bruto
                out_dir.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(final_answer, f, ensure_ascii=False, indent=2)

                await pending_mgr.mark_done_and_flush(input)
                try:
                    parsed_answer = extract_json_from_answer(final_answer.get("answer"))
                except Exception:
                    parsed_answer = final_answer.get("answer")
                row_result = {
                    "input": input,
                    "answer": parsed_answer,
                    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "agent": agent_name
                }
                print(f"[OK] {input} salvo em {out_path}")
        except Exception as e:
            print(f"[ERRO] Input {input}: {e}", file=sys.stderr)
            row_result = None
    return row_result

# ------------------------------
# Main
# ------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Batch Toqan Input worker (50 concorrentes, polling 30s).")
    parser.add_argument("--csv", required=True, help="Caminho do CSV.")
    parser.add_argument("--api-key", default=os.getenv("TOQAN_API_KEY"), help="X-Api-Key (ou defina TOQAN_API_KEY).")
    parser.add_argument("--concurrency", type=int, default=50, help="Concorrência (padrão: 50).")
    parser.add_argument("--poll-interval", type=int, default=30, help="Intervalo do polling em segundos (padrão: 30).")
    parser.add_argument("--outdir", default=None, help="Pasta de saída. Se omitido, usa outputs/<timestamp>/.")
    parser.add_argument("--collect", action="store_true", help="Ao final, roda toqan_collect_answers.py para consolidar JSONs em CSV.")
    parser.add_argument("--collect-out", default=None, help="Caminho do CSV de saída da consolidação (padrão: outputs/<timestamp>/answers.csv).")
    parser.add_argument("--collect-pattern", default="*.json", help="Glob para arquivos a serem consolidados (padrão: *.json).")
    parser.add_argument("--collect-sep", default=".", help="Separador para chaves achatadas (padrão: .).")
    parser.add_argument("--dbricks", action="store_true", help="Se definido, grava também no Databricks via SQL Statements API.")
    parser.add_argument("--dbricks-host", default=os.getenv("DATABRICKS_HOST"), help="Host do Databricks (ex.: https://dbc-xxxx.cloud.databricks.com).")
    parser.add_argument("--dbricks-token", default=os.getenv("DATABRICKS_TOKEN"), help="Token do Databricks.")
    parser.add_argument("--dbricks-warehouse-id", default=os.getenv("DATABRICKS_WAREHOUSE_ID"), help="Warehouse ID.")
    parser.add_argument("--dbricks-table", default=os.getenv("DATABRICKS_TABLE"), help="Tabela de destino (schema.tabela).")
    parser.add_argument("--dbricks-batch-size", type=int, default=1000, help="Tamanho do sublote para envio ao Databricks.")
    parser.add_argument("--agent", default=None, help="Nome do agente. Se não for informado, será derivado da api-key mascarada.")
    parser.add_argument("--dbricks-debug", action="store_true", help="Log detalhado das chamadas ao Databricks (SQL, payload e resposta).")
    parser.add_argument("--dbricks-skip-ddl", action="store_true", help="Não tentar criar schema/tabela no Databricks; apenas MERGE.")
    parser.add_argument("--dbricks-test", action="store_true", help="Executa SELECT 1 no contexto informado (catalog/schema) para checar permissões.")
    args = parser.parse_args()

    global DBRICKS_DEBUG
    DBRICKS_DEBUG = bool(args.dbricks_debug)
    if DBRICKS_DEBUG:
        print("[DBRICKS][DEBUG] Debug de Databricks habilitado.")

    if not args.api_key:
        print("Defina --api-key ou a variável de ambiente TOQAN_API_KEY.", file=sys.stderr)
        sys.exit(1)

    if args.agent:
        agent_name = args.agent
    else:
        key = args.api_key or ""
        if len(key) >= 7:
            agent_name = f"{key[:4]}{'*' * 24}{key[-3:]}"
        else:
            agent_name = "unknown_agent"

    csv_path = Path(args.csv).expanduser().resolve()
    # Lê os inputs a partir da **primeira coluna** do CSV (com ou sem cabeçalho)
    inputs = load_inputs(csv_path)
    # Mantém representação canônica para reescrita do CSV pendente
    all_rows: List[Dict[str, str]] = [{"input": i} for i in inputs]
    if not inputs:
        print("Nenhum Input encontrado no CSV.")
        return

    # Pasta de saída
    out_dir = Path(args.outdir).expanduser().resolve() if args.outdir else (Path("outputs") / now_stamp())
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove inputs já processados (se existir JSON no out_dir)
    inputs = [i for i in inputs if not (out_dir / f"{safe_filename(i)}.json").exists()]
    if not inputs:
        print("Todos os inputs já foram processados na pasta de saída. Pulando para --collect e --dbricks (se habilitados).")

    timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=120)
    connector = aiohttp.TCPConnector(limit_per_host=args.concurrency)  # ajuda a não estoura r conexões
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        client = ToqanClient(api_key=args.api_key, session=session, poll_interval=args.poll_interval)

        if inputs:
            pending_mgr = PendingManager(csv_path=csv_path, initial_rows=all_rows, inputs=inputs)
            semaphore = asyncio.Semaphore(args.concurrency)
            tasks = [asyncio.create_task(process_input(input, client, out_dir, pending_mgr, semaphore, agent_name)) for input in inputs]
            results = await asyncio.gather(*tasks)
        else:
            results = []

        # (Opcional) Consolidação automática inline (sem subprocess)
        if args.collect:
            try:
                collect_out = Path(args.collect_out).expanduser().resolve() if args.collect_out else (out_dir / "answers.csv")
                consolidate_answers(indir=out_dir, out_csv=collect_out, pattern=args.collect_pattern, sep=args.collect_sep)
            except Exception as e:
                print(f"[COLLECT][ERRO] Falha ao consolidar respostas: {e}", file=sys.stderr)

        # (Opcional) Envio ao Databricks
        if args.dbricks:
            missing = [k for k in ("dbricks_host","dbricks_token","dbricks_warehouse_id") if not getattr(args, k)]
            if missing:
                print(f"[DBRICKS][ERRO] Faltam parâmetros: {missing}", file=sys.stderr)
            else:
                # (Opcional) Smoke test de permissões/contexto
                if args.dbricks_test:
                    parts = _parse_schema_table(args.dbricks_table or "")
                    test_sql = "SELECT 1"
                    dbg(f"Executando smoke test (SELECT 1) em {parts.get('catalog') or '(default)'} . {parts.get('schema')}")
                    test_res = await dbricks_exec_sql(
                        session=session,
                        host=args.dbricks_host,
                        token=args.dbricks_token,
                        warehouse_id=args.dbricks_warehouse_id,
                        sql=test_sql,
                        catalog=parts.get("catalog", ""),
                        schema=parts.get("schema", "")
                    )
                    test_state = (test_res.get('status', {}) or {}).get('state')
                    if test_state != "SUCCEEDED":
                        msg = (test_res.get("status", {}).get("error", {}) or {}).get("message", "Falha no SELECT 1")
                        details = (test_res.get("status", {}).get("error", {}) or {}).get("details", "")
                        print(f"❌ [Databricks] Smoke test falhou: {msg}. {details[:300]}", file=sys.stderr)
                        return
                    print("✅ [Databricks] Smoke test OK (SELECT 1).")
                # Garante que a tabela exista (cria schema/tabela se necessário)
                if not args.dbricks_skip_ddl:
                    try:
                        print(args)
                        await dbricks_ensure_table_exists(
                            session=session,
                            host=args.dbricks_host,
                            token=args.dbricks_token,
                            warehouse_id=args.dbricks_warehouse_id,
                            table_ident=args.dbricks_table,
                        )
                    except Exception as e:
                        print(f"[DBRICKS][AVISO] Não foi possível garantir a existência da tabela: {e}", file=sys.stderr)
                else:
                    dbg("Pulando DDL (--dbricks-skip-ddl habilitado).")
                # Monta linhas a partir dos JSONs existentes no out_dir (mesma fonte do --collect)
                rows = []
                files = sorted(out_dir.glob(args.collect_pattern))
                for fp in files:
                    stem = fp.stem
                    m = re.match(r"^(.*)_[0-9a-fA-F]{8}$", stem)
                    input_nohash = m.group(1) if m else stem
                    try:
                        with fp.open("r", encoding="utf-8") as f:
                            data = json.load(f)
                    except Exception as e:
                        print(f"[DBRICKS][AVISO] Falha ao ler {fp}: {e}", file=sys.stderr)
                        continue
                    try:
                        parsed_answer = extract_json_from_answer(data.get("answer"))
                    except Exception as e:
                        print(f"[DBRICKS][AVISO] {fp.name}: não foi possível extrair JSON de 'answer' ({e}), enviando bruto.", file=sys.stderr)
                        parsed_answer = data.get("answer")
                    try:
                        created_at = datetime.fromtimestamp(fp.stat().st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                    rows.append({
                        "input": input_nohash,
                        "answer": parsed_answer,
                        "created_at": created_at,
                        "agent": agent_name
                    })

                try:
                    await dbricks_merge_rows(
                        session=session,
                        host=args.dbricks_host,
                        token=args.dbricks_token,
                        warehouse_id=args.dbricks_warehouse_id,
                        table=args.dbricks_table,
                        rows=rows,
                        batch_size=args.dbricks_batch_size
                    )
                except Exception as e:
                    print(f"[DBRICKS][ERRO] Falha ao enviar resultados ao Databricks: {e}", file=sys.stderr)

    print(f"\nConcluído. Saídas em: {out_dir}")

if __name__ == "__main__":
    # Opcional: carregar .env automaticamente, se existir
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception:
        pass
    asyncio.run(main())