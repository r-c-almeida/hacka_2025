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
agent_xande_key = ''

API_BASE = "https://api.coco.prod.toqan.ai"
STEP_TIMEOUT_SECONDS = 30

def _norm_label(s: Optional[str]) -> str:
    if not s:
        return ""
    import unicodedata
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

    if "cerveja" in lbl:
        return "homer"
    if lbl in ("medicamento", "remedio"):
        return "medison"
    if "fralda" in lbl:
        return "fraldete"

    return None


def _taxo_bucket(taxo_parsed: Any) -> Optional[str]:
    """
    Retorna 'homer'|'medison'|'fraldete' conforme TAXO, ou None para outras categorias.
    """
    try:
        level1 = taxo_parsed.get("metadata", {}).get("current", {}).get("global_taxonomy_level1")
    except Exception:
        level1 = None
    lbl = _norm_label(level1) if level1 else ""
    if lbl.endswith("s"):
        lbl = lbl[:-1]
    if lbl == "cerveja":
        return "homer"
    if lbl in ("medicamento", "remedio"):
        return "medison"
    if lbl == "fralda":
        return "fraldete"
    return None


def _get_ph(name: str, scope: str = "global") -> st.delta_generator.DeltaGenerator:
    all_scopes = st.session_state.setdefault("placeholders", {})
    scope_dict = all_scopes.setdefault(scope, {})
    if name not in scope_dict or scope_dict[name] is None:
        scope_dict[name] = st.container()
    return scope_dict[name]


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



def _init_dbricks_writer():
    # tenta pegar de st.secrets primeiro, se existir
    # host = st.secrets.get("DATABRICKS_HOST") if hasattr(st, "secrets") else None
    # token = st.secrets.get("DATABRICKS_TOKEN") if hasattr(st, "secrets") else None
    # wh = st.secrets.get("DATABRICKS_WAREHOUSE_ID") if hasattr(st, "secrets") else None
    # table = st.secrets.get("DATABRICKS_TABLE") if hasattr(st, "secrets") else None
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

    row = {
        "id": str(uuid.uuid4()),
        "input": user_input,
        "answer": answer_str,
        "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "agent": agent_name,
        "orchestration_id": unique_key
    }

    print(f"[DBRICKS] saving input={row['input']}")
    try:
        await dbricks.merge_rows([row], batch_size=1)
    except Exception as e:
        print(f"[DBRICKS][ERRO_SAVE] {e}")


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
    if "agents_enabled" not in st.session_state:
        st.session_state["agents_enabled"] = False
    if "last_product_uuid" not in st.session_state:
        st.session_state["last_product_uuid"] = None


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
    unique_key: Optional[str] = None,
):
    # header + status imediato
    parent = placeholder.container()
    parent.markdown(f"### {step_name}")
    status_ph = parent.empty()
    status_ph.info("Iniciando…")

    # dispara a task
    task = asyncio.create_task(coro)

    # mostra feedback de execução enquanto aguarda
    running = True
    try:
        # mostra “Executando…” logo de cara (sem esperar timeout)
        status_ph.info("⏳ Executando…")
        result_raw = await asyncio.wait_for(asyncio.shield(task), timeout=STEP_TIMEOUT_SECONDS)
        running = False
    except asyncio.TimeoutError:
        # se estourar o timeout, mantém aviso e segue esperando terminar
        status_ph.warning("⏳ Ainda executando… (demorando mais que o normal)")
        result_raw = await task
        running = False
    finally:
        # limpa o status quando terminar
        if not running:
            status_ph.empty()

    parsed = extract_json_from_text(result_raw)

    # conteúdo final no mesmo container
    if isinstance(parsed, (dict, list)):
        parent.json(parsed)
    else:
        parent.write(str(parsed))

    # persistência
    run_uuid = unique_key or st.session_state.get("last_product_uuid") or str(uuid.uuid4())
    await save_to_dbricks(step_name, input_for_key, result_raw, agent_name, unique_key=run_uuid)
    return result_raw, parsed


# =========================
# Fluxos
# =========================
async def workflow_base_with_parallel_tail(user_input: str, scope: Optional[str] = None) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
    timeout = httpx.Timeout(30.0)

    # escopo visual (por EAN); fallback para o próprio user_input
    scope = scope or (user_input.strip() or "global")

    run_uuid = st.session_state.get("last_product_uuid")
    if not run_uuid:
        run_uuid = str(uuid.uuid4())
    st.session_state["last_product_uuid"] = run_uuid

    # Um container “pai” por EAN, para deixar organizado
    with st.expander(f"🔎 EAN: {scope}", expanded=True):
        # Placeholders namespaced
        ph1  = _get_ph("step1", scope)   # querio
        ph2  = _get_ph("step2", scope)   # eligio
        ph3  = _get_ph("step3", scope)   # dimetris
        ph5t = _get_ph("step5T", scope)  # taxo
        ph5n = _get_ph("step5N", scope)  # norma
        # autos
        ph5h = _get_ph("step5H", scope)  # homer (auto)
        ph5m = _get_ph("step5M", scope)  # medison (auto)
        ph5f = _get_ph("step5F", scope)  # fraldete (auto)
        # xande
        ph6x = _get_ph("step6X", scope)  # xande (final)

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
                    "oliveira_analyzer",
                    "oliveira_analyzer",
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
            results["oliveira_analyzer_raw"] = rawT
            results["oliveira_analyzer"] = parsedT

            which = _pick_auto_agent_from_taxo(parsedT)  # cerveja/medicamento/remédio/fraldas
            auto_task = None
            if which == "homer" and agent_homer_key:
                payload = {"eligio_product_selector": parsed2, "oliveira_analyzer": parsedT}
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
                payload = {"eligio_product_selector": parsed2, "oliveira_analyzer": parsedT}
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
                payload = {"eligio_product_selector": parsed2, "oliveira_analyzer": parsedT}
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

            # Xande automático (idêntico às suas regras)
            try:
                # para manter o mesmo container do EAN:
                # reusa o run_xande_from_cache_stream mas sem sobrescrever scope global
                await run_xande_from_cache_stream(user_input, unique_key=run_uuid)
            except Exception as e:
                print(f"[XANDE][AUTO] Falha ao executar: {e}")

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
# Runners extras (cache)
# =========================
async def run_taxo_from_cache_stream(user_input: str) -> Dict[str, Any]:
    results = st.session_state.get("results", {})
    step2_parsed = results.get("eligio_product_selector")
    if not agent_taxo_key:
        return {"error": "agent_taxo_key não configurada."}
    if not step2_parsed:
        return {"error": "Resultado do Eligio não encontrado no cache."}

    ph4 = st.session_state["placeholders"].setdefault("step4", st.empty())

    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
    timeout = httpx.Timeout(30.0)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        raw4, parsed4 = await run_with_timeout_and_update(
            "oliveira_analyzer",
            "oliveira_analyzer",
            user_input,
            ph4,
            start_or_continue_agent_async(ensure_json_serializable(step2_parsed), agent_taxo_key, client),
        )
        results["oliveira_analyzer_raw"] = raw4
        results["oliveira_analyzer"] = parsed4
        st.session_state["results"] = results
        return {"ok": True}


async def run_fraldete_from_cache_stream(user_input: str) -> Dict[str, Any]:
    results = st.session_state.get("results", {})
    step2_parsed = results.get("eligio_product_selector")
    step3_parsed = results.get("dimetris_product_search_dimension")
    step4_parsed = results.get("oliveira_analyzer")  # opcional

    if not step2_parsed or not step3_parsed:
        return {"error": "É necessário ter os resultados de Eligio e Dimetris no cache."}
    if not agent_fraldete_key:
        return {"error": "agent_fraldete_key não configurada."}

    payload = {
        "eligio_product_selector": step2_parsed,
        "dimetris_product_search_dimension": step3_parsed,
        "oliveira_analyzer": step4_parsed,
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
    step4_parsed = results.get("oliveira_analyzer")  # opcional

    if not agent_homer_key:
        return {"error": "agent_homer_key não configurada."}
    if not step2_parsed or not step3_parsed:
        return {"error": "É necessário ter os resultados de Eligio e Dimetris no cache."}

    payload = {
        "eligio_product_selector": step2_parsed,
        "dimetris_product_search_dimension": step3_parsed,
        "oliveira_analyzer": step4_parsed,
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
    step4_parsed = results.get("oliveira_analyzer")  # opcional

    if not agent_medison_key:
        return {"error": "agent_medison_key não configurada."}
    if not step2_parsed or not step3_parsed:
        return {"error": "É necessário ter os resultados de Eligio e Dimetris no cache."}

    payload = {
        "eligio_product_selector": step2_parsed,
        "dimetris_product_search_dimension": step3_parsed,
        "oliveira_analyzer": step4_parsed,
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
            unique_key=unique_key,
        )
        results["medison_analyzer_raw"] = rawM
        results["medison_analyzer"] = parsedM
        st.session_state["results"] = results
        return {"ok": True}


async def run_xande_from_cache_stream(user_input: str, unique_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Executa o agent Xande recebendo APENAS o EAN (user_input).
    Não depende das respostas dos steps anteriores.
    """
    if not agent_xande_key:
        return {"error": "agent_xande_key não configurada."}

    ean = (user_input or "").strip()
    if not ean:
        return {"error": "Xande: EAN vazio."}

    # placeholder
    ph6x = st.session_state["placeholders"].setdefault("step6X", st.container())

    # chamado apenas com o EAN
    payload = f' Faça a análise completa do ean: {ean}. Não faça novas perguntas.'

    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
    timeout = httpx.Timeout(30.0)
    run_uuid = unique_key or st.session_state.get("last_product_uuid") or str(uuid.uuid4())

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        rawX, parsedX = await run_with_timeout_and_update(
            "xande_aggregator",
            "xande_aggregator",
            user_input,              # para logging/DB
            ph6x,
            start_or_continue_agent_async(payload, agent_xande_key, client),
            unique_key=run_uuid,
        )

    # salva como antes
    results = st.session_state.get("results", {})
    results["xande_aggregator_raw"] = rawX
    results["xande_aggregator"] = parsedX
    st.session_state["results"] = results
    return {"ok": True}

async def run_xande_general_assessment(unique_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Executa o agent Xande passando APENAS a string 'Todos' (avaliação geral),
    sem depender do EAN nem de steps anteriores.
    """
    if not agent_xande_key:
        return {"error": "agent_xande_key não configurada."}

    ph6g = st.session_state["placeholders"].setdefault("step6G", st.container())

    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
    timeout = httpx.Timeout(30.0)
    run_uuid = unique_key or st.session_state.get("last_product_uuid") or str(uuid.uuid4())

    payload = "Todos"  # <- ponto central da mudança

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        rawX, parsedX = await run_with_timeout_and_update(
            "xande_aggregator (Avaliação Geral)",
            "xande_aggregator",
            "Todos",  # input_for_key para logging/DB
            ph6g,
            start_or_continue_agent_async(payload, agent_xande_key, client),
            unique_key=run_uuid,
        )

    results = st.session_state.get("results", {})
    results["xande_aggregator_general_raw"] = rawX
    results["xande_aggregator_general"] = parsedX
    st.session_state["results"] = results
    return {"ok": True}

async def _run_one_ean(ean: str, sem: asyncio.Semaphore) -> Tuple[str, Dict[str, Any]]:
    """
    Executa o workflow completo para 1 EAN respeitando o semáforo (até 10 concorrentes).
    O 'scope' é o próprio EAN (para namespacing de UI).
    """
    ean = (ean or "").strip()
    if not ean:
        return ean, {"error": "EAN vazio"}
    async with sem:
        res = await workflow_base_with_parallel_tail(ean, scope=ean)
        return ean, res


def _parse_eans(raw: str) -> list[str]:
    """
    Converte a string do textarea (vírgula, quebra de linha ou espaço) numa lista de EANs.
    """
    if not raw:
        return []
    # aceita vírgula, quebra de linha e ponto-e-vírgula
    parts = re.split(r"[,\n;]+", raw)
    eans = [p.strip() for p in parts if p.strip()]
    # remove duplicatas mantendo ordem
    seen = set()
    unique = []
    for e in eans:
        if e not in seen:
            seen.add(e)
            unique.append(e)
    return unique

# =========================
# Streamlit UI
# =========================
def main():
    st.set_page_config(page_title="iPadroniza", page_icon="🤖", layout="wide")
    st.title("iFood - iPadroniza")

    if not dbricks.is_enabled():
        if dbricks.missing_vars():
            st.warning(f"Databricks não configurado. Faltando: {', '.join(dbricks.missing_vars())}")

    user_input = st.text_area("Ean do produto", "", key="txt_user_input")

    # ---- Botões (linha principal) – agora com Xande
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        run_base_btn = st.button("Pesquisar Produto", key="btn_run_base")

    analyzers_disabled = not st.session_state.get("agents_enabled", False)

    with c2:
        medison_btn = st.button("Medison", key="btn_medison", disabled=analyzers_disabled)
    with c3:
        fraldete_btn = st.button("Fraldete", key="btn_fraldete", disabled=analyzers_disabled)
    with c4:
        homer_btn = st.button("Homer", key="btn_homer", disabled=analyzers_disabled)
    with c5:
        xande_btn = st.button("Xande", key="btn_xande", disabled=False)  # sempre habilitado
    with c6:
        geral_btn = st.button("Avaliação Geral", key="btn_xande_geral", disabled=False)
    # ---- Linha 2: Reset/Limpeza ----
    c6, c7, c8 = st.columns(3)
    with c6:
        reset_conv_btn = st.button("Resetar conversas", key="btn_reset_convs")
    with c7:
        clear_res_btn = st.button("Limpar resultados", key="btn_clear_results")
    with c8:
        clear_ui_btn = st.button("Limpar UI (placeholders)", key="btn_clear_ui")

    # ---- Ações dos botões utilitários
    if reset_conv_btn:
        st.session_state["conversations"] = {}
        st.session_state["agents_enabled"] = False
        st.success("Conversas resetadas para todos os agentes.")

    if clear_res_btn:
        st.session_state["results"] = {}
        st.session_state["agents_enabled"] = False
        st.success("Resultados limpos.")

    if clear_ui_btn:
        st.session_state["placeholders"] = {}
        st.session_state["agents_enabled"] = False
        st.success("Placeholders limpos.")

    # ---- Execução do fluxo base
    # ---- Execução do fluxo base
    if run_base_btn:
        raw = user_input.strip()
        if not raw:
            st.warning("Informe o(s) código(s) de barras do produto (separe por vírgula ou quebra de linha).")
        else:
            eans = _parse_eans(raw)
            if not eans:
                st.warning("Nenhum EAN válido encontrado.")
            else:
                try:
                    if len(eans) == 1:
                        # Caso 1 EAN – mantém exatamente o fluxo anterior
                        base_results = run_asyncio(workflow_base_with_parallel_tail(eans[0], scope=eans[0]))
                        st.session_state["results"] = base_results
                        st.session_state["agents_enabled"] = True
                        st.success("Fluxo concluído.")
                    else:
                        # Caso múltiplos EANs – executa em paralelo com limite de 10 concorrentes
                        async def _run_many(ean_list: list[str]) -> Dict[str, Any]:
                            sem = asyncio.Semaphore(10)  # <= limite de concorrência
                            tasks = [asyncio.create_task(_run_one_ean(e, sem)) for e in ean_list]
                            out_pairs = await asyncio.gather(*tasks, return_exceptions=False)
                            # monta um dicionário: {ean: resultados}
                            return {k: v for (k, v) in out_pairs}

                        with st.spinner(f"Processando {len(eans)} EANs em paralelo (até 10 simultâneos)…"):
                            all_results = run_asyncio(_run_many(eans))

                        # salva um snapshot geral (mantém também o comportamento anterior para 'results')
                        st.session_state["results"] = all_results
                        st.session_state["agents_enabled"] = True
                        st.success("Fluxo concluído para todos os EANs.")
                except Exception as e:
                    st.session_state["agents_enabled"] = False
                    st.error(f"Falha no fluxo: {e}")

    # ---- Handlers dos analisadores
    if medison_btn:
        run_uuid = st.session_state.get("last_product_uuid") or str(uuid.uuid4())
        with st.spinner("Medison Analyzer - Pesquisando medicamento"):
            out = run_asyncio(run_medison_from_cache_stream(user_input.strip(), unique_key=run_uuid))
            if "error" in out:
                st.error(out["error"])
            else:
                st.success("Medison Analyzer concluído.")
                # tenta Xande se possível
                out_x = run_asyncio(run_xande_from_cache_stream(user_input.strip(), unique_key=run_uuid))
                if isinstance(out_x, dict) and "error" in out_x:
                    print(f"[XANDE][TRY_AFTER_MEDISON] {out_x['error']}")

    if fraldete_btn:
        with st.spinner("Fraldete Analyzer - Pesquisando fraldas"):
            out = run_asyncio(run_fraldete_from_cache_stream(user_input.strip()))
            if "error" in out:
                st.error(out["error"])
            else:
                st.success("Fraldete Analyzer concluído.")
                run_uuid = st.session_state.get("last_product_uuid") or str(uuid.uuid4())
                out_x = run_asyncio(run_xande_from_cache_stream(user_input.strip(), unique_key=run_uuid))
                if isinstance(out_x, dict) and "error" in out_x:
                    print(f"[XANDE][TRY_AFTER_FRALDETE] {out_x['error']}")

    if homer_btn:
        with st.spinner("Homer Analyzer - Pesquisando cerveja"):
            out = run_asyncio(run_homer_from_cache_stream(user_input.strip()))
            if "error" in out:
                st.error(out["error"])
            else:
                st.success("Homer Analyzer concluído.")
                run_uuid = st.session_state.get("last_product_uuid") or str(uuid.uuid4())
                out_x = run_asyncio(run_xande_from_cache_stream(user_input.strip(), unique_key=run_uuid))
                if isinstance(out_x, dict) and "error" in out_x:
                    print(f"[XANDE][TRY_AFTER_HOMER] {out_x['error']}")

    if xande_btn:
        run_uuid = st.session_state.get("last_product_uuid") or str(uuid.uuid4())
        with st.spinner("Xande - Agregando resultados finais"):
            out = run_asyncio(run_xande_from_cache_stream(user_input.strip(), unique_key=run_uuid))
            if isinstance(out, dict) and "error" in out:
                st.error(out["error"])
            else:
                st.success("Xande concluído.")
    if geral_btn:
        run_uuid = st.session_state.get("last_product_uuid") or str(uuid.uuid4())
        with st.spinner("Xande - Avaliação Geral (Todos)"):
            out = run_asyncio(run_xande_general_assessment(unique_key=run_uuid))
            if isinstance(out, dict) and "error" in out:
                st.error(out["error"])
            else:
                st.success("Avaliação Geral concluída.")

    # ---- Snapshot
    if st.session_state.get("results"):
        st.subheader("📦 Snapshot completo (cache)")
        res = st.session_state["results"]
        if isinstance(res, dict) and all(isinstance(v, dict) for v in res.values()):
            for ean, payload in res.items():
                with st.expander(f"📦 EAN {ean}", expanded=False):
                    st.json(payload)
        else:
            st.json(res)


if __name__ == "__main__":
    main()