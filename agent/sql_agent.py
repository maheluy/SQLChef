"""Agente text-to-SQL como grafo LangGraph: gerar -> executar -> (corrigir e gerar de novo).

Versões (para a tabela comparativa):
    v1  uma chamada, sem exemplos
    v2  + few-shot
    v3  + few-shot + autocorreção (erro ou resultado vazio volta ao LLM, até MAX_ATTEMPTS)
    v4  v3 + dica de schema na correção (em quais tabelas está a coluna que não existe)
    v5  v4 + notas de semântica do schema no prompt + checagem de formato do resultado
        (pergunta pede um valor e voltou uma linha por grupo -> correção)
    v6  v5 + revisão: com a SQL rodando, o LLM (em contexto separado) confere pergunta, SQL e
        resultado; se apontar um problema, ele vai para a correção (uma revisão por pergunta;
        se a SQL revisada falhar, volta ao resultado anterior)

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
    suspect: str  # v5: resultado roda, mas tem formato suspeito (dispara correção, não bloqueia)
    attempts: int
    columns: list
    rows: list
    truncated: bool
    input_tokens: int
    output_tokens: int
    # v6
    reviewed: bool
    review: str  # "" sem revisão, "OK" ou o problema apontado
    review_bonus: int  # geração extra concedida pela revisão (não conta em MAX_ATTEMPTS)
    review_reverted: bool  # a SQL revisada falhou e o resultado anterior foi restaurado
    pre_review_sql: str
    pre_review_columns: list
    pre_review_rows: list
    pre_review_truncated: bool


def _build_graph(settings: Settings):
    llm = get_llm(settings)
    few_shot = settings.version in ("v2", "v3", "v4", "v5", "v6")
    retry = settings.version in ("v3", "v4", "v5", "v6")
    hints = settings.version in ("v4", "v5", "v6")
    notes = shape = settings.version in ("v5", "v6")
    review = settings.version == "v6"
    max_attempts = settings.max_attempts if retry else 1
    system = prompts.system_prompt(few_shot, notes)

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
            return {"error": f"{type(exc).__name__}: {exc}", "suspect": "", "columns": [], "rows": []}
        error = suspect = ""
        if not result.rows:
            error = "a consulta não devolveu nenhuma linha; confira filtros, valores e joins"
        elif all(all(v is None for v in row) for row in result.rows):
            error = "a consulta devolveu apenas NULL; confira filtros, valores e joins"
        elif shape:
            suspect = prompts.shape_problem(
                state["question"], state["sql"], result.columns, result.rows, result.truncated
            )
        return {
            "error": error,
            "suspect": suspect,
            "columns": result.columns,
            "rows": result.rows,
            "truncated": result.truncated,
        }

    def repair(state: AgentState) -> dict:
        problem = state.get("error") or state.get("suspect", "")
        hint = prompts.schema_hint(problem) if hints else ""
        return {"messages": [prompts.repair_message(state["sql"], problem, hint)]}

    def route(state: AgentState) -> str:
        limit = max_attempts + state.get("review_bonus", 0)
        if (state.get("error") or state.get("suspect")) and state["attempts"] < limit:
            return "repair"
        if review and not state.get("reviewed") and state.get("rows") and not state.get("error"):
            return "review"
        return "finalize"

    def review_node(state: AgentState) -> dict:
        msg = prompts.review_message(
            state["question"], state["sql"], state["columns"], state["rows"], state.get("truncated", False)
        )
        response = llm.complete(prompts.review_system(), [msg])
        problem = prompts.parse_review(response.text)
        return {
            "reviewed": True,
            "review": problem or "OK",
            "suspect": problem,
            "review_bonus": 1 if problem else 0,
            "pre_review_sql": state["sql"],
            "pre_review_columns": state["columns"],
            "pre_review_rows": state["rows"],
            "pre_review_truncated": state.get("truncated", False),
            "input_tokens": state.get("input_tokens", 0) + response.input_tokens,
            "output_tokens": state.get("output_tokens", 0) + response.output_tokens,
        }

    def after_review(state: AgentState) -> str:
        return "repair" if state.get("suspect") else "finalize"

    def finalize(state: AgentState) -> dict:
        """A revisão nunca troca uma resposta que roda por um erro ou por um resultado vazio."""
        revised_failed = state.get("error") or not state.get("rows")
        if state.get("review_bonus") and state.get("pre_review_sql") and revised_failed:
            return {
                "sql": state["pre_review_sql"],
                "columns": state["pre_review_columns"],
                "rows": state["pre_review_rows"],
                "truncated": state.get("pre_review_truncated", False),
                "error": "",
                "suspect": "",
                "review_reverted": True,
            }
        return {}

    graph = StateGraph(AgentState)
    graph.add_node("generate", generate)
    graph.add_node("execute", execute)
    graph.add_node("repair", repair)
    graph.add_node("review", review_node)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "execute")
    graph.add_conditional_edges("execute", route, ["repair", "review", "finalize"])
    graph.add_conditional_edges("review", after_review, ["repair", "finalize"])
    graph.add_edge("repair", "generate")
    graph.add_edge("finalize", END)
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
        "review": state.get("review", ""),
        "review_reverted": state.get("review_reverted", False),
        "pre_review_sql": state.get("pre_review_sql", ""),
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


def generate_sql_v5(question: str) -> dict:
    return _for_eval(run(question, "v5"))


def generate_sql_v6(question: str) -> dict:
    return _for_eval(run(question, "v6"))


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
        "suspect": state.get("suspect", ""),
        "review": state.get("review", ""),
        "review_reverted": state.get("review_reverted", False),
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
