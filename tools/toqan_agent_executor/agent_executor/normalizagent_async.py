import asyncio
import json
from typing import Any, Dict, Optional, Tuple, Union
import pandas as pd
import streamlit as st
import httpx
import regex as re
from dbricks_writer import DatabricksWriter 
from datetime import datetime
import unicodedata
import uuid

agent_product_searcher_key = ''
agent_product_select_key = ''
agent_product_dimetris_dimension_key = ''
agent_fraldete_key = ''
agent_taxo_key = ''
agent_homer_key = ''
agent_medison_key = ''
agent_norma_key = ''

API_BASE = "https://api.coco.prod.toqan.ai"
STEP_TIMEOUT_SECONDS = 30

def _norm_label(s: Optional[str]) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.strip().lower()

def _pick_auto_agent_from_taxo(taxo_parsed: Any) -> Optional[str]:
    """
    Lê metadata.current.global_taxonomy_level1 do retorno do taxo e escolhe qual agent disparar.
      - 'cerveja' -> 'homer'
      - 'medicamento' ou 'remedio' -> 'medison'
      - 'fralda'/'fraldas' -> 'fraldete'
    """
    try:
        level1 = taxo_parsed.get("metadata", {}).get("current", {}).get("global_taxonomy_level1")
    except Exception:
        level1 = None

    lbl = _norm_label(level1)
    if not lbl:
        return None

    # singulariza trivialmente (remove 's' final)
    if lbl.endswith("s"):
        lbl = lbl[:-1]

    if lbl == "cerveja":
        return "homer"
    if lbl in ("medicamento", "remedio"):
        return "medison"
    if lbl == "fralda":
        return "fraldete"

    return None

def _get_ph(name: str) -> st.delta_generator.DeltaGenerator:
    phs = st.session_state.setdefault("placeholders", {})
    if name not in phs or phs[name] is None:
        # container persistente (não apaga entre execuções)
        phs[name] = st.container()
    return phs[name]

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

async def save_to_dbricks(
    step_name: str,
    user_input: str,
    raw_answer: Any,
    agent_name: str,
    unique_key: str
):
    if not dbricks.is_enabled():
        if dbricks.missing_vars():
            print(f"[DBRICKS] Não configurado. Faltando: {dbricks.missing_vars()}")
        return

    try:
        answer_str = json.dumps(raw_answer, ensure_ascii=False) if isinstance(raw_answer, (dict, list)) else str(raw_answer)
    except Exception as e:
        answer_str = f"<<serialize_error:{e}>>"

    # SEMPRE concatena: user_input + UUID + agent + step
    input_key = f"{user_input}.{agent_name}.{unique_key}"

    row = {
        "input": input_key,
        "answer": answer_str,
        "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "agent": agent_name,
    }

    print(f"[DBRICKS] saving input={row['input']}")
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

async def run_with_timeout_and_update(
    step_name: str,
    agent_name: str,
    input_for_key: str,
    placeholder: st.delta_generator.DeltaGenerator,
    coro,
    unique_key: str,   # << OBRIGATÓRIO
):
    task = asyncio.create_task(coro)
    try:
        result_raw = await asyncio.wait_for(asyncio.shield(task), timeout=STEP_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        placeholder.container().info(f"Ainda executando o {step_name}…")
        result_raw = await task

    parsed = extract_json_from_text(result_raw)

    box = placeholder.container()
    box.markdown(f"### {step_name}")
    box.json(parsed) if isinstance(parsed, (dict, list)) else box.write(str(parsed))

    await save_to_dbricks(step_name, input_for_key, result_raw, agent_name, unique_key=unique_key)
    return result_raw, parsed

# =========================
# Fluxo base (parcial): 1→2→3
# =========================
async def workflow_base_with_parallel_tail(user_input: str) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
    timeout = httpx.Timeout(30.0)

    run_uuid = st.session_state.get("last_product_uuid")
    if not run_uuid:
        # se quiser que seja determinístico por EAN, troque por: uuid.uuid5(uuid.NAMESPACE_URL, user_input)
        run_uuid = str(uuid.uuid4())
    st.session_state["last_product_uuid"] = run_uuid
    
    # --- Placeholders (agora sempre definidos) ---
    ph1  = _get_ph("step1")   # querio
    ph2  = _get_ph("step2")   # eligio
    ph3  = _get_ph("step3")   # dimetris
    ph5t = _get_ph("step5T")  # taxo
    ph5n = _get_ph("step5N")  # norma
    # autos
    ph5h = _get_ph("step5H")  # homer (auto)
    ph5m = _get_ph("step5M")  # medison (auto)
    ph5f = _get_ph("step5F")  # fraldete (auto)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        # 1) querio
        raw1, parsed1 = await run_with_timeout_and_update(
            "querio_product_searcher",
            "querio_product_searcher",
            user_input,
            ph1,
            start_or_continue_agent_async(user_input, agent_product_searcher_key, client),
            unique_key=run_uuid,
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
            unique_key=run_uuid,
        )
        results["eligio_product_selector_raw"] = raw2
        results["eligio_product_selector"] = parsed2

        # 3) paralelos: dimetris + taxo + norma
        step3_input = ensure_json_serializable(parsed2)

        dimetris_task = asyncio.create_task(
            run_with_timeout_and_update(
                "dimetris_product_search_dimension",
                "dimetris_product_search_dimension",
                user_input,
                ph3,
                start_or_continue_agent_async(step3_input, agent_product_dimetris_dimension_key, client),
                unique_key=run_uuid,
            )
        )
        taxo_task = asyncio.create_task(
            run_with_timeout_and_update(
                "taxo_analyzer",
                "taxo_analyzer",
                user_input,
                ph5t,
                start_or_continue_agent_async({"eligio_product_selector": parsed2}, agent_taxo_key, client),
                unique_key=run_uuid,
            )
        )
        norma_task = asyncio.create_task(
            run_with_timeout_and_update(
                "norma_analyzer",
                "norma_analyzer",
                user_input,
                ph5n,
                start_or_continue_agent_async({"eligio_product_selector": parsed2}, agent_norma_key, client),
                unique_key=run_uuid,
            )
        )

        # Espera o TAXO para decidir disparo automático
        rawT, parsedT = await taxo_task
        results["taxo_analyzer_raw"] = rawT
        results["taxo_analyzer"] = parsedT

        which = _pick_auto_agent_from_taxo(parsedT)  # cerveja/medicamento/remédio/fraldas
        auto_task = None
        if which == "homer" and agent_homer_key:
            payload = {"eligio_product_selector": parsed2, "taxo_analyzer": parsedT}
            auto_task = asyncio.create_task(
                run_with_timeout_and_update(
                    "homer_analyzer (auto)",
                    "homer_analyzer",
                    user_input,
                    ph5h,
                    start_or_continue_agent_async(ensure_json_serializable(payload), agent_homer_key, client),
                    unique_key=run_uuid,
                )
            )
        elif which == "medison" and agent_medison_key:
            payload = {"eligio_product_selector": parsed2, "taxo_analyzer": parsedT}
            auto_task = asyncio.create_task(
                run_with_timeout_and_update(
                    "medison_analyzer (auto)",
                    "medison_analyzer",
                    user_input,
                    ph5m,
                    start_or_continue_agent_async(ensure_json_serializable(payload), agent_medison_key, client),
                    unique_key=run_uuid,
                )
            )
        elif which == "fraldete" and agent_fraldete_key:
            payload = {"eligio_product_selector": parsed2, "taxo_analyzer": parsedT}
            auto_task = asyncio.create_task(
                run_with_timeout_and_update(
                    "fraldete_analyzer (auto)",
                    "fraldete_analyzer",
                    user_input,
                    ph5f,
                    start_or_continue_agent_async(ensure_json_serializable(payload), agent_fraldete_key, client),
                    unique_key=run_uuid,
                )
            )

        # Aguarda os outros paralelos
        raw3, parsed3 = await dimetris_task
        results["dimetris_product_search_dimension_raw"] = raw3
        results["dimetris_product_search_dimension"] = parsed3

        rawN, parsedN = await norma_task
        results["norma_analyzer_raw"] = rawN
        results["norma_analyzer"] = parsedN

        if auto_task:
            rawA, parsedA = await auto_task
            if which == "homer":
                results["homer_analyzer_raw_auto"] = rawA
                results["homer_analyzer_auto"] = parsedA
            elif which == "medison":
                results["medison_analyzer_raw_auto"] = rawA
                results["medison_analyzer_auto"] = parsedA
            elif which == "fraldete":
                results["fraldete_analyzer_raw_auto"] = rawA
                results["fraldete_analyzer_auto"] = parsedA

    return results

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

async def run_medison_from_cache_stream(user_input: str, unique_key: Optional[str] = None) -> Dict[str, Any]:
    results = st.session_state.get("results", {})
    step2_parsed = results.get("eligio_product_selector")
    step3_parsed = results.get("dimetris_product_search_dimension")
    step4_parsed = results.get("taxo_analyzer")  # opcional

    if not agent_medison_key:
        return {"error": "agent_medison_key não configurada."}
    if not step2_parsed or not step3_parsed:
        return {"error": "É necessário ter os resultados de eligio (2) e dimetris (3) no cache."}

    payload = {
        "eligio_product_selector": step2_parsed,
        "dimetris_product_search_dimension": step3_parsed,
        "taxo_analyzer": step4_parsed,
    }

    ph5m = st.session_state["placeholders"].setdefault("step5M", st.container())

    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
    timeout = httpx.Timeout(30.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        rawM, parsedM = await run_with_timeout_and_update(
            "medison_analyzer",
            "medison_analyzer",
            user_input,
            ph5m,
            start_or_continue_agent_async(payload, agent_medison_key, client),
            unique_key=unique_key,  # << repassa
        )
        results["medison_analyzer_raw"] = rawM
        results["medison_analyzer"] = parsedM
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

    user_input = st.text_area("Entrada do Passo 1 (ex.: EAN, nome do produto, etc.)", "", key="txt_user_input")

    # ---- Botões (agora todos com key únicos)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        run_base_btn = st.button("Executar base (1→2→ paralelos)", key="btn_run_base")
    with c2:
        # Taxo saiu do botão e foi para a execução paralela.
        # Aqui entra o MEDISON (no lugar do antigo taxo)
        medison_btn = st.button("Rodar medison_analyzer", key="btn_medison")
    with c3:
        fraldete_btn = st.button("Rodar fraldete_analyzer", key="btn_fraldete")
    with c4:
        homer_btn = st.button("Rodar homer_analyzer", key="btn_homer")
    with c5:
        reset_conv_btn = st.button("Resetar conversas", key="btn_reset_convs")
    with c6:
        clear_res_btn = st.button("Limpar resultados", key="btn_clear_results")

    clear_ui_btn = st.button("Limpar UI (placeholders)", key="btn_clear_ui")

    # ---- Ações dos botões
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
            # zera placeholders relevantes do fluxo paralelo (dimetris + taxo + norma)
            st.session_state["placeholders"]["step1"] = st.empty()
            st.session_state["placeholders"]["step2"] = st.empty()
            st.session_state["placeholders"]["step3"] = st.empty()
            st.session_state["placeholders"]["step5T"] = st.empty()  # taxo
            st.session_state["placeholders"]["step5N"] = st.empty()  # norma

            with st.spinner("Executando fluxo: 1 → 2 → (dimetris + taxo + norma)…"):
                try:
                    base_results = run_asyncio(workflow_base_with_parallel_tail(user_input.strip()))
                    st.session_state["results"] = base_results
                    st.success("Fluxo concluído: querio, eligio, (dimetris + taxo + norma em paralelo).")
                except Exception as e:
                    st.error(f"Falha no fluxo: {e}")

    if medison_btn:
        run_uuid = st.session_state.get("last_run_uuid") or str(uuid.uuid4())
        with st.spinner("Rodando medison_analyzer…"):
            out = run_asyncio(run_medison_from_cache_stream(user_input.strip(), unique_key=run_uuid))
            if "error" in out:
                st.error(out["error"])
            else:
                st.success("medison_analyzer concluído.")

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