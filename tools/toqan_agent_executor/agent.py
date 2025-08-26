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
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

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

def atomic_write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    tmp.replace(path)

def rewrite_pending_csv(csv_path: Path, all_rows: List[Dict[str, str]], pending_inputs: set) -> None:
    # Mantém apenas Inputs pendentes
    fieldnames = list(all_rows[0].keys()) if all_rows else ["input"]
    remaining_rows = [r for r in all_rows if (r.get("input") or "").strip() in pending_inputs]
    atomic_write_csv(csv_path, fieldnames, remaining_rows)

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
        flat["__ean_from_filename"] = fp.stem
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
async def dbricks_merge_rows(session: aiohttp.ClientSession, host: str, token: str, warehouse_id: str, table: str, rows: List[Dict[str, str]], batch_size: int = 1000, poll_interval: float = 2.0, max_poll_attempts: int = 60) -> None:
    if not rows:
        return
    api_base = host.rstrip("/") + "/api/2.0/sql/statements"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    # Envia em sublotes
    for i in range(0, len(rows), batch_size):
        sub = rows[i:i+batch_size]
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
        sql_statement = f"""
            MERGE INTO {table} AS target
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
        payload = {"statement": textwrap.dedent(sql_statement), "warehouse_id": warehouse_id}
        # Submit
        async with session.post(api_base, headers=headers, json=payload, timeout=120) as resp:
            result = await resp.json()
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
        if state == "FAILED":
            err_msg = (poll_result.get("status", {}).get("error", {}) or {}).get("message", "Erro desconhecido")
            print(f"❌ [Databricks] Sub-lote falhou: {err_msg}", file=sys.stderr)
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
            create_resp = await client.create_conversation(input)
            conversation_id = create_resp.get("conversation_id")
            request_id = create_resp.get("request_id")
            if not conversation_id or not request_id:
                raise RuntimeError(f"Resposta inesperada em create_conversation para {input}: {create_resp}")

            print(f"[POST] Input {input}: conversa criada (conversation_id={conversation_id}, request_id={request_id})")

            answer = await client.poll_answer_until_ready(conversation_id, request_id)

            print(f"[DONE] Input {input}: resposta concluída, salvando...")

            # Salva JSON bruto
            out_dir.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(answer, f, ensure_ascii=False, indent=2)

            await pending_mgr.mark_done_and_flush(input)
            row_result = {
                "input": input,
                "answer": answer,
                "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
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
    args = parser.parse_args()

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
        print("Todos os inputs já foram processados na pasta de saída.")
        return

    timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=120)
    connector = aiohttp.TCPConnector(limit_per_host=args.concurrency)  # ajuda a não estoura r conexões
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        client = ToqanClient(api_key=args.api_key, session=session, poll_interval=args.poll_interval)
        pending_mgr = PendingManager(csv_path=csv_path, initial_rows=all_rows, inputs=inputs)
        semaphore = asyncio.Semaphore(args.concurrency)
        tasks = [asyncio.create_task(process_input(input, client, out_dir, pending_mgr, semaphore, agent_name)) for input in inputs]
        results = await asyncio.gather(*tasks)

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
                rows = [r for r in results if isinstance(r, dict) and r.get("input")]
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