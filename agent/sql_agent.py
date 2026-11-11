"""Agente text-to-SQL como grafo LangGraph: gerar -> executar -> (corrigir e gerar de novo).

Versões (para a tabela comparativa):
    v1  uma chamada, sem exemplos
    v2  + few-shot
    v3  + few-shot + autocorreção (erro ou resultado vazio volta ao LLM, até MAX_ATTEMPTS)
    v4  v3 + dica de schema na correção (em quais tabelas está a coluna que não existe)

O LLM roda localmente no Ollama; o modelo vem de OLLAMA_MODEL no ambiente/.env.

Avaliação:
    python eval/run_eval.py --agent agent.sql_agent:generate_sql_v1 --label qwen7b_v1
Linha de comando:
    python -m agent.sql_agent "Quantos pedidos foram cancelados?"
"""

import operator
import sqlite3
import sys
from functools import lru_cache
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from agent import prompts
from agent.config import Settings, get_settings
from agent.db import UnsafeSQL, run_select
from agent.llm import get_llm

PREVIEW_ROWS = 50


class AgentState(TypedDict, total=False):
    question: str
    messages: Annotated[list, operator.add]  # histórico enviado ao LLM (só cresce)
    sql: str
    error: str
    attempts: int
    columns: list
    rows: list
    truncated: bool
    input_tokens: int
    output_tokens: int


def _build_graph(settings: Settings):
    llm = get_llm(settings)
    few_shot = settings.version in ("v2", "v3", "v4")
    retry = settings.version in ("v3", "v4")
    hints = settings.version == "v4"
    max_attempts = settings.max_attempts if retry else 1
    system = prompts.system_prompt(few_shot)

    def generate(state: AgentState) -> dict:
        response = llm.complete(system, state["messages"])
        return {
            "messages": [{"role": "assistant", "content": response.text}],
            "sql": prompts.extract_sql(response.text),
            "attempts": state.get("attempts", 0) + 1,
            "input_tokens": state.get("input_tokens", 0) + response.input_tokens,
            "output_tokens": state.get("output_tokens", 0) + response.output_tokens,
        }

    def execute(state: AgentState) -> dict:
        try:
            result = run_select(state["sql"], max_rows=PREVIEW_ROWS)
        except (UnsafeSQL, sqlite3.Error) as exc:
            return {"error": f"{type(exc).__name__}: {exc}", "columns": [], "rows": []}
        error = ""
        if not result.rows:
            error = "a consulta não devolveu nenhuma linha; confira filtros, valores e joins"
        elif all(all(v is None for v in row) for row in result.rows):
            error = "a consulta devolveu apenas NULL; confira filtros, valores e joins"
        return {
            "error": error,
            "columns": result.columns,
            "rows": result.rows,
            "truncated": result.truncated,
        }

    def repair(state: AgentState) -> dict:
        hint = prompts.schema_hint(state["error"]) if hints else ""
        return {"messages": [prompts.repair_message(state["sql"], state["error"], hint)]}

    def should_retry(state: AgentState) -> str:
        if state.get("error") and state["attempts"] < max_attempts:
            return "repair"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("generate", generate)
    graph.add_node("execute", execute)
    graph.add_node("repair", repair)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "execute")
    graph.add_conditional_edges("execute", should_retry, ["repair", END])
    graph.add_edge("repair", "generate")
    return graph.compile(), llm


@lru_cache(maxsize=None)
def _graph_for(settings: Settings):
    return _build_graph(settings)


def _settings(version: str | None = None, model: str | None = None) -> Settings:
    """Configuração do ambiente com versão e/ou modelo substituídos."""
    settings = get_settings()
    overrides = {k: v for k, v in (("version", version), ("model", model)) if v}
    return Settings(**{**settings.__dict__, **overrides}) if overrides else settings


def run(question: str, version: str | None = None, model: str | None = None) -> AgentState:
    """Executa o grafo e devolve o estado final completo."""
    graph, _ = _graph_for(_settings(version, model))
    return graph.invoke({"question": question, "messages": [prompts.question_message(question)]})


def _for_eval(state: AgentState) -> dict:
    return {
        "sql": state.get("sql", ""),
        "input_tokens": state.get("input_tokens", 0),
        "output_tokens": state.get("output_tokens", 0),
        "attempts": state.get("attempts", 0),
    }


def generate_sql(question: str) -> dict:
    """Usa AGENT_VERSION do ambiente. Devolve o dict que eval/run_eval.py espera."""
    return _for_eval(run(question))


def generate_sql_v1(question: str) -> dict:
    return _for_eval(run(question, "v1"))


def generate_sql_v2(question: str) -> dict:
    return _for_eval(run(question, "v2"))


def generate_sql_v3(question: str) -> dict:
    return _for_eval(run(question, "v3"))


def generate_sql_v4(question: str) -> dict:
    return _for_eval(run(question, "v4"))


def answer(question: str, version: str | None = None, model: str | None = None) -> dict:
    """Para a demo: SQL, resultado e uma resposta curta em português.

    Os tokens incluem a chamada que redige a resposta, então são maiores que os da avaliação.
    """
    settings = _settings(version, model)
    state = run(question, settings.version, settings.model)
    result = {
        "sql": state.get("sql", ""),
        "columns": state.get("columns", []),
        "rows": state.get("rows", []),
        "truncated": state.get("truncated", False),
        "attempts": state.get("attempts", 0),
        "error": state.get("error", ""),
        "input_tokens": state.get("input_tokens", 0),
        "output_tokens": state.get("output_tokens", 0),
        "version": settings.version,
        "model": settings.model,
    }
    if result["rows"] and not result["error"]:
        _, llm = _graph_for(settings)
        msg = prompts.answer_message(question, result["columns"], result["rows"], result["truncated"])
        response = llm.complete(prompts.ANSWER_SYSTEM, [msg])
        result["text"] = response.text.strip()
        result["input_tokens"] += response.input_tokens
        result["output_tokens"] += response.output_tokens
    else:
        result["text"] = f"Não consegui responder: {result['error'] or 'sem resultado'}"
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit('uso: python -m agent.sql_agent "sua pergunta"')
    s = get_settings()
    out = answer(" ".join(sys.argv[1:]))
    print(f"[ollama / {s.model} / {s.version}] tentativas: {out['attempts']}")
    print(f"\nSQL:\n{out['sql']}\n")
    if out["columns"]:
        print(" | ".join(out["columns"]))
        for row in out["rows"][:10]:
            print(" | ".join(str(v) for v in row))
    print(f"\n{out['text']}")
