import asyncio
import json
from typing import Any, Dict, Optional, Tuple, Union
import pandas as pd
import streamlit as st
import httpx
import regex as re
from dbricks_writer import DatabricksWriter 
from datetime import datetime

agent_product_searcher_key = ''
agent_product_select_key = ''
agent_product_dimetris_dimension_key = ''
agent_fraldete_key = ''
agent_taxo_key = ''
agent_homer_key = ''

API_BASE = "https://api.coco.prod.toqan.ai"
STEP_TIMEOUT_SECONDS = 30

# =========================
# Databricks – init (usa env ou st.secrets)
# =========================
def _init_dbricks_writer():
    # tenta pegar de st.secrets primeiro, se existir
    #host = st.secrets.get("DATABRICKS_HOST") if hasattr(st, "secrets") else None
    #token = st.secrets.get("DATABRICKS_TOKEN") if hasattr(st, "secrets") else None
    #wh = st.secrets.get("DATABRICKS_WAREHOUSE_ID") if hasattr(st, "secrets") else None
    #table = st.secrets.get("DATABRICKS_TABLE") if hasattr(st, "secrets") else None
    return DatabricksWriter()

dbricks = _init_dbricks_writer()

async def save_to_dbricks(step_name: str, user_input: str, raw_answer: Any, agent_name: str):
    """
    Salva 1 linha no Databricks. Executa mesmo se 'raw_answer' não for JSON (serializa como string).
    """
    if not dbricks.is_enabled():
        # Opcional: log informativo no console para sabermos por quê não está ativo
        if dbricks.missing_vars():
            print(f"[DBRICKS] Não configurado. Faltando: {dbricks.missing_vars()}")
        return

    # serializa raw_answer (sempre salvar, inclusive erro/texto)
    try:
        if isinstance(raw_answer, (dict, list)):
            answer_str = json.dumps(raw_answer, ensure_ascii=False)
        else:
            answer_str = str(raw_answer)
    except Exception as e:
        answer_str = f"<<serialize_error:{e}>>"

    row = {
        "input": f"{user_input}",
        "answer": answer_str,
        "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "agent": agent_name,
    }
    try:
        await dbricks.merge_rows([row], batch_size=1)
    except Exception as e:
        print(f"[DBRICKS][ERRO_SAVE] {e}")

# =========================
# Utils – parsing
# =========================

def clean_code_fence(s: str) -> str:
    if not isinstance(s, str):
        return s
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.DOTALL | re.IGNORECASE)

def _try_json(text: str) -> Optional[Union[dict, list]]:
    try:
        return json.loads(text)
    except Exception:
        return None

def extract_json_from_text(s: Union[str, Dict, list]) -> Union[Dict[str, Any], list, str]:
    if isinstance(s, (dict, list)):
        return s
    if not isinstance(s, str):
        return s

    raw = s.strip()
    direct = _try_json(clean_code_fence(raw))
    if direct is not None:
        return direct

    m = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        parsed = _try_json(candidate)
        if parsed is not None:
            return parsed

    arr = re.search(r"(\[[\s\S]*\])", raw)
    if arr:
        parsed = _try_json(arr.group(1))
        if parsed is not None:
            return parsed

    obj = re.search(r"(\{[\s\S]*\})", raw)
    if obj:
        parsed = _try_json(obj.group(1))
        if parsed is not None:
            return parsed

    return raw

def ensure_json_serializable(obj: Any) -> Any:
    if isinstance(obj, (dict, list, str, int, float, bool)) or obj is None:
        return obj
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)

def run_asyncio(task):
    try:
        return asyncio.run(task)
    except RuntimeError as e:
        if "asyncio.run() cannot be called" in str(e):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(task)
        raise

# =========================
# Cliente Toqan (create/get/continue)
# =========================

async def start_agent_async(message: str, api_key: str, client: httpx.AsyncClient) -> Tuple[Optional[str], Optional[str]]:
    payload = {"user_message": message}
    headers = {"X-Api-Key": api_key}
    r = await client.post(f"{API_BASE}/api/create_conversation", json=payload, headers=headers)
    r.raise_for_status()
    data = r.json()
    return data.get("conversation_id"), data.get("request_id")

async def continue_agent_async(conversation_id: str, message: str, api_key: str, client: httpx.AsyncClient) -> Optional[str]:
    headers = {"X-Api-Key": api_key}
    r = await client.post(
        f"{API_BASE}/api/continue_conversation",
        json={"conversation_id": conversation_id, "user_message": message},
        headers=headers,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("request_id")

async def get_agent_answer_async(conversation_id: str, request_id: str, api_key: str, client: httpx.AsyncClient) -> str:
    headers = {"X-Api-Key": api_key}
    while True:
        r = await client.get(
            f"{API_BASE}/api/get_answer",
            params={"conversation_id": conversation_id, "request_id": request_id},
            headers=headers,
        )
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "finished":
                return data.get("answer", "")
            await asyncio.sleep(0.5)
        elif r.status_code in (404, 500):
            return r.text
        else:
            await asyncio.sleep(0.5)

# =========================
# Conversas persistentes por api_key
# =========================

def _init_state():
    if "conversations" not in st.session_state:
        st.session_state["conversations"] = {}
    if "results" not in st.session_state:
        st.session_state["results"] = {}
    if "placeholders" not in st.session_state:
        st.session_state["placeholders"] = {}

_init_state()

async def start_or_continue_agent_async(message: Any, api_key: str, client: httpx.AsyncClient) -> str:
    if isinstance(message, (dict, list)):
        user_message = json.dumps(message, ensure_ascii=False)
    else:
        user_message = str(message)

    conv_state = st.session_state["conversations"].get(api_key)
    headers = {"X-Api-Key": api_key}

    if not conv_state or conv_state.get("count", 0) >= 10:
        r = await client.post(f"{API_BASE}/api/create_conversation", json={"user_message": user_message}, headers=headers)
        r.raise_for_status()
        data = r.json()
        conversation_id = data.get("conversation_id")
        request_id = data.get("request_id")
        st.session_state["conversations"][api_key] = {"conversation_id": conversation_id, "count": 1}
    else:
        conversation_id = conv_state["conversation_id"]
        request_id = await continue_agent_async(conversation_id, user_message, api_key, client)
        conv_state["count"] += 1
        st.session_state["conversations"][api_key] = conv_state

    return await get_agent_answer_async(conversation_id, request_id, api_key, client)

# =========================
# Helpers para UI com timeout + SAVE
# =========================

async def run_with_timeout_and_update(step_name: str, agent_name: str, input_for_key: str, placeholder: st.delta_generator.DeltaGenerator, coro):
    """
    Executa uma coroutine do step, exibe parcial com timeout e salva no Databricks ao terminar.
    """
    task = asyncio.create_task(coro)
    try:
        result_raw = await asyncio.wait_for(asyncio.shield(task), timeout=STEP_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        placeholder.info(f"Ainda executando o {step_name}…")
        result_raw = await task  # aguarda terminar

    parsed = extract_json_from_text(result_raw)

    # exibição
    placeholder.empty()
    placeholder.subheader(step_name)
    if isinstance(parsed, (dict, list)):
        placeholder.json(parsed)
    else:
        placeholder.write(str(parsed))

    # SAVE (sempre grava, inclusive se a resposta for texto/erro)
    await save_to_dbricks(step_name, input_for_key, result_raw, agent_name)

    return result_raw, parsed

# =========================
# Fluxo base (parcial): 1→2→3
# =========================

async def workflow_base_streaming(user_input: str) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
    timeout = httpx.Timeout(30.0)

    ph1 = st.session_state["placeholders"].setdefault("step1", st.empty())
    ph2 = st.session_state["placeholders"].setdefault("step2", st.empty())
    ph3 = st.session_state["placeholders"].setdefault("step3", st.empty())

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        # 1) querio
        raw1, parsed1 = await run_with_timeout_and_update(
            "querio_product_searcher",
            "querio_product_searcher",
            user_input,
            ph1,
            start_or_continue_agent_async(user_input, agent_product_searcher_key, client),
        )
        results["querio_product_searcher_raw"] = raw1
        results["querio_product_searcher"] = parsed1

        # 2) eligio
        step2_input = ensure_json_serializable(parsed1)
        raw2, parsed2 = await run_with_timeout_and_update(
            "eligio_product_selector",
            "eligio_product_selector",
            user_input,
            ph2,
            start_or_continue_agent_async(step2_input, agent_product_select_key, client),
        )
        results["eligio_product_selector_raw"] = raw2
        results["eligio_product_selector"] = parsed2

        # 3) dimetris
        step3_input = ensure_json_serializable(parsed2)
        raw3, parsed3 = await run_with_timeout_and_update(
            "dimetris_product_search_dimension",
            "dimetris_product_search_dimension",
            user_input,
            ph3,
            start_or_continue_agent_async(step3_input, agent_product_dimetris_dimension_key, client),
        )
        results["dimetris_product_search_dimension_raw"] = raw3
        results["dimetris_product_search_dimension"] = parsed3

    return results

# =========================
# Botões: taxo, fraldete, homer (com SAVE)
# =========================

async def run_taxo_from_cache_stream(user_input: str) -> Dict[str, Any]:
    results = st.session_state.get("results", {})
    step2_parsed = results.get("eligio_product_selector")
    if not agent_taxo_key:
        return {"error": "agent_taxo_key não configurada."}
    if not step2_parsed:
        return {"error": "Resultado do passo 2 (eligio) não encontrado no cache."}

    ph4 = st.session_state["placeholders"].setdefault("step4", st.empty())

    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
    timeout = httpx.Timeout(30.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        raw4, parsed4 = await run_with_timeout_and_update(
            "taxo_analyzer",
            "taxo_analyzer",
            user_input,
            ph4,
            start_or_continue_agent_async(ensure_json_serializable(step2_parsed), agent_taxo_key, client),
        )
        results["taxo_analyzer_raw"] = raw4
        results["taxo_analyzer"] = parsed4
        st.session_state["results"] = results
        return {"ok": True}

async def run_fraldete_from_cache_stream(user_input: str) -> Dict[str, Any]:
    results = st.session_state.get("results", {})
    step2_parsed = results.get("eligio_product_selector")
    step3_parsed = results.get("dimetris_product_search_dimension")
    step4_parsed = results.get("taxo_analyzer")  # opcional

    if not step2_parsed or not step3_parsed:
        return {"error": "É necessário ter os resultados de eligio (2) e dimetris (3) no cache."}
    if not agent_fraldete_key:
        return {"error": "agent_fraldete_key não configurada."}

    payload = {
        "eligio_product_selector": step2_parsed,
        "dimetris_product_search_dimension": step3_parsed,
        "taxo_analyzer": step4_parsed,
    }

    ph5 = st.session_state["placeholders"].setdefault("step5F", st.empty())

    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
    timeout = httpx.Timeout(30.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        raw5, parsed5 = await run_with_timeout_and_update(
            "fraldete_analyzer",
            "fraldete_analyzer",
            user_input,
            ph5,
            start_or_continue_agent_async(payload, agent_fraldete_key, client),
        )
        results["fraldete_analyzer_raw"] = raw5
        results["fraldete_analyzer"] = parsed5
        st.session_state["results"] = results
        return {"ok": True}

async def run_homer_from_cache_stream(user_input: str) -> Dict[str, Any]:
    results = st.session_state.get("results", {})
    step2_parsed = results.get("eligio_product_selector")
    step3_parsed = results.get("dimetris_product_search_dimension")
    step4_parsed = results.get("taxo_analyzer")  # opcional

    if not agent_homer_key:
        return {"error": "agent_homer_key não configurada."}
    if not step2_parsed or not step3_parsed:
        return {"error": "É necessário ter os resultados de eligio (2) e dimetris (3) no cache."}

    payload = {
        "eligio_product_selector": step2_parsed,
        "dimetris_product_search_dimension": step3_parsed,
        "taxo_analyzer": step4_parsed,
    }

    ph5h = st.session_state["placeholders"].setdefault("step5H", st.empty())

    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
    timeout = httpx.Timeout(30.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        rawH, parsedH = await run_with_timeout_and_update(
            "homer_analyzer",
            "homer_analyzer",
            user_input,
            ph5h,
            start_or_continue_agent_async(payload, agent_homer_key, client),
        )
        results["homer_analyzer_raw"] = rawH
        results["homer_analyzer"] = parsedH
        st.session_state["results"] = results
        return {"ok": True}

# =========================
# Streamlit UI
# =========================

def main():
    st.set_page_config(page_title="Toqan + Databricks", page_icon="🤖", layout="wide")
    st.title("Toqan – Exibição Parcial + Timeout + Conversas Persistentes + Save Databricks")

    if not dbricks.is_enabled():
        if dbricks.missing_vars():
            st.warning(f"Databricks não configurado. Faltando: {', '.join(dbricks.missing_vars())}")

    user_input = st.text_area("Entrada do Passo 1 (ex.: EAN, nome do produto, etc.)", "")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        run_base_btn = st.button("Executar base (1→2→3)")
    with c2:
        taxo_btn = st.button("Rodar taxo_analyzer")
    with c3:
        fraldete_btn = st.button("Rodar fraldete_analyzer")
    with c4:
        homer_btn = st.button("Rodar homer_analyzer")
    with c5:
        reset_conv_btn = st.button("Resetar conversas")

    c6, c7 = st.columns(2)
    with c6:
        clear_res_btn = st.button("Limpar resultados")
    with c7:
        clear_ui_btn = st.button("Limpar UI (placeholders)")

    if reset_conv_btn:
        st.session_state["conversations"] = {}
        st.success("Conversas resetadas para todos os agentes.")

    if clear_res_btn:
        st.session_state["results"] = {}
        st.success("Resultados limpos.")

    if clear_ui_btn:
        st.session_state["placeholders"] = {}
        st.success("Placeholders limpos.")

    if run_base_btn:
        if not user_input.strip():
            st.warning("Informe a entrada do passo 1.")
        else:
            st.session_state["placeholders"]["step1"] = st.empty()
            st.session_state["placeholders"]["step2"] = st.empty()
            st.session_state["placeholders"]["step3"] = st.empty()

            with st.spinner("Executando fluxo base (1→2→3)…"):
                try:
                    base_results = run_asyncio(workflow_base_streaming(user_input.strip()))
                    st.session_state["results"] = base_results
                    st.success("Base concluída (querio, eligio, dimetris).")
                except Exception as e:
                    st.error(f"Falha no fluxo base: {e}")

    if taxo_btn:
        with st.spinner("Rodando taxo_analyzer…"):
            out = run_asyncio(run_taxo_from_cache_stream(user_input.strip()))
            if "error" in out:
                st.error(out["error"])
            else:
                st.success("taxo_analyzer concluído.")

    if fraldete_btn:
        with st.spinner("Rodando fraldete_analyzer…"):
            out = run_asyncio(run_fraldete_from_cache_stream(user_input.strip()))
            if "error" in out:
                st.error(out["error"])
            else:
                st.success("fraldete_analyzer concluído.")

    if homer_btn:
        with st.spinner("Rodando homer_analyzer…"):
            out = run_asyncio(run_homer_from_cache_stream(user_input.strip()))
            if "error" in out:
                st.error(out["error"])
            else:
                st.success("homer_analyzer concluído.")

    if st.session_state.get("results"):
        st.subheader("📦 Snapshot completo (cache)")
        st.json(st.session_state["results"])

if __name__ == "__main__":
    main()