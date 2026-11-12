"""Demo Streamlit do SQLChef: pergunta em português -> SQL -> resultado -> resposta.

    streamlit run app.py

Usa agent.sql_agent.answer(). Requer o Ollama rodando e data/olist.db gerado.
"""

import json
import time
from pathlib import Path

import altair as alt
import ollama
import pandas as pd
import streamlit as st

from agent.config import VERSIONS, get_settings
from agent.db import DB_PATH
from agent.sql_agent import PREVIEW_ROWS, answer

VERSION_INFO = {
    "v1": "uma chamada, sem exemplos",
    "v2": "+ exemplos few-shot",
    "v3": "+ autocorreção com retry",
    "v4": "+ dica de schema na correção",
    "v5": "+ notas de schema e checagem de formato",
    "v6": "+ revisão da SQL e do resultado",
}

QUESTIONS_PATH = Path(__file__).resolve().parent / "eval" / "questions.json"
LEVELS = {"facil": "Fácil", "media": "Média", "dificil": "Difícil"}
# Ids do conjunto de avaliação. Com qwen2.5-coder:7b/v3: q48 precisa de retry e q19 falha
# mesmo com 3 tentativas (mostra a limitação). A pergunta livre no fim gera gráfico de linha.
EXAMPLE_IDS = ["q02", "q05", "q13", "q48", "q47", "q19"]
EXTRA_EXAMPLES = [("Livre", "Quantos pedidos foram feitos em cada mês de 2017? Retorne o mês (AAAA-MM) e a quantidade.")]


def load_examples() -> list[tuple[str, str]]:
    try:
        data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return EXTRA_EXAMPLES
    by_id = {q["id"]: q for q in (data if isinstance(data, list) else data.get("questions", []))}
    picked = [(LEVELS.get(by_id[i]["difficulty"], ""), by_id[i]["question"]) for i in EXAMPLE_IDS if i in by_id]
    return picked + EXTRA_EXAMPLES


EXAMPLES = load_examples()

REPO_URL = "https://github.com/maheluy/SQLChef"
DATASET_URL = "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"

st.set_page_config(page_title="SQLChef", page_icon="🍳", layout="wide")


# ---------- verificações e cache ----------

@st.cache_data(ttl=60, show_spinner=False)
def available_models(host: str) -> list[str] | None:
    """Modelos qwen2.5-coder instalados no Ollama; None se o Ollama não responde."""
    try:
        models = [m.model for m in ollama.Client(host=host).list().models]
    except Exception:  # noqa: BLE001 - qualquer falha de conexão vira mensagem de setup
        return None
    return sorted(m for m in models if m.startswith("qwen2.5-coder")) or sorted(models)


@st.cache_data(show_spinner=False, max_entries=200)
def cached_answer(question: str, version: str, model: str) -> tuple[dict, float]:
    """Temperatura 0: a mesma pergunta com a mesma configuração dá a mesma resposta."""
    start = time.perf_counter()
    result = answer(question, version=version, model=model)
    return result, time.perf_counter() - start


def chart_for(df: pd.DataFrame):
    """Barra ou linha quando o resultado é rótulo + número; senão None."""
    if len(df) < 2 or df.shape[1] != 2:
        return None
    label, value = df.columns
    if pd.api.types.is_numeric_dtype(df[label]) or not pd.api.types.is_numeric_dtype(df[value]):
        return None
    data = df.rename(columns={label: "rotulo", value: "valor"}).assign(rotulo=lambda d: d["rotulo"].astype(str))
    is_time = data["rotulo"].str.fullmatch(r"\d{4}(-\d{2}){0,2}").all()
    base = alt.Chart(data)
    if is_time:
        chart = base.mark_line(point=True).encode(
            x=alt.X("rotulo:O", title=label), y=alt.Y("valor:Q", title=value)
        )
    else:
        chart = base.mark_bar().encode(
            y=alt.Y("rotulo:N", sort=None, title=label), x=alt.X("valor:Q", title=value)
        )
    return chart.encode(tooltip=[alt.Tooltip("rotulo", title=label), alt.Tooltip("valor", title=value)])


# ---------- barra lateral ----------

settings = get_settings()
models = available_models(settings.host)

with st.sidebar:
    st.title("SQLChef")
    st.caption("Agente text-to-SQL sobre dados reais de marketplace (Olist), com LLM local no Ollama.")

    version = st.radio(
        "Versão do agente",
        VERSIONS,
        index=VERSIONS.index(settings.version),
        format_func=lambda v: f"{v}: {VERSION_INFO[v]}",
    )

    if models:
        default = models.index(settings.model) if settings.model in models else 0
        model = st.selectbox("Modelo", models, index=default, disabled=len(models) == 1)
        if len(models) == 1:
            st.caption("Outros modelos (ex.: 14b) só rodando localmente.")
    else:
        model = settings.model

    st.subheader("Perguntas de exemplo")
    for i, (level, q) in enumerate(EXAMPLES):
        if st.button(f"{level}: {q}", key=f"ex{i}", width="stretch"):
            st.session_state.pending = q

    if st.button("Limpar conversa", type="secondary"):
        st.session_state.history = []

    st.divider()
    st.caption(
        f"Dados: [Olist Brazilian E-Commerce]({DATASET_URL}), CC BY-NC-SA 4.0. "
        f"Avaliação e código: [repositório]({REPO_URL})."
    )


# ---------- pré-requisitos ----------

if models is None:
    st.error(
        f"Não consegui falar com o Ollama em `{settings.host}`.\n\n"
        f"Inicie com `ollama serve` e baixe o modelo: `ollama pull {settings.model}`."
    )
    st.stop()
if not DB_PATH.exists():
    st.error(f"Banco não encontrado em `{DB_PATH}`. Gere com `python data/build_db.py`.")
    st.stop()


# ---------- conversa ----------

def render(turn: dict) -> None:
    result, latency = turn["result"], turn["latency"]
    failed = bool(result["error"]) or not result["rows"]

    if failed:
        st.warning(result["text"])
    else:
        st.markdown(result["text"])
        if result.get("review"):
            note = " A SQL revisada falhou; mantida a anterior." if result.get("review_reverted") else ""
            st.caption(f"Revisão: {result['review']}{note}")
        if result.get("suspect"):
            st.warning(f"Resultado com formato suspeito mesmo após as tentativas: {result['suspect']}")

    cols = st.columns(4)
    cols[0].metric("Tentativas", result["attempts"])
    cols[1].metric("Latência", f"{latency:.1f} s")
    cols[2].metric("Tokens (entrada / saída)", f"{result['input_tokens']:,} / {result['output_tokens']:,}".replace(",", "."))
    cols[3].metric("Configuração", result["version"], help=result["model"])

    with st.expander("SQL usada" + (" (última tentativa)" if failed else ""), expanded=failed):
        st.code(result["sql"] or "-- nenhuma SQL gerada", language="sql")

    if result["columns"] and result["rows"]:
        df = pd.DataFrame(result["rows"], columns=result["columns"])
        chart = chart_for(df)
        if chart is not None:
            st.altair_chart(chart, width="stretch")
        with st.expander("Resultado", expanded=chart is None and len(df) > 1):
            st.dataframe(df, width="stretch", hide_index=True)
            if result["truncated"]:
                st.caption(f"Mostrando só as primeiras {PREVIEW_ROWS} linhas.")


st.session_state.setdefault("history", [])

if not st.session_state.history:
    st.markdown(
        "### Pergunte sobre os pedidos, clientes, vendedores e pagamentos da Olist\n"
        "O agente lê o schema, gera a SQL, executa em um SQLite **somente leitura** e, se der erro "
        "ou resultado vazio, devolve o erro ao modelo para corrigir. Escolha um exemplo ao lado ou escreva abaixo."
    )

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        render(turn)

question = st.chat_input("Ex.: Qual estado tem mais clientes?") or st.session_state.pop("pending", None)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner(f"Gerando e executando SQL com {model} ({version})…"):
            try:
                result, latency = cached_answer(question, version, model)
            except Exception as exc:  # noqa: BLE001 - mostra a falha em vez de quebrar a página
                st.error(f"Falha ao chamar o agente: {type(exc).__name__}: {exc}")
                st.stop()
        turn = {"question": question, "result": result, "latency": latency}
        render(turn)
    st.session_state.history.append(turn)
