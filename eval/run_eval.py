"""Avaliação por execution accuracy: a SQL gerada devolve o mesmo resultado que a SQL de referência?

Uso (a partir da raiz do repositório):

    # 1) checar se as SQLs de referência rodam e não vêm vazias (rode após montar o banco)
    python eval/run_eval.py --validate-gold

    # 2) avaliar um agente. A função recebe a pergunta e devolve a SQL
    #    (str) ou um dict {"sql": ..., "input_tokens": ..., "output_tokens": ..., "attempts": ...}
    python eval/run_eval.py --agent agent.sql_agent:generate_sql --label v1_sem_exemplos

Regras de comparação (documente no README):
  - nomes de colunas são ignorados;
  - floats são arredondados a 2 casas;
  - perguntas "ordered" comparam a ordem das linhas; as demais comparam como conjunto
    (multiconjunto) de linhas;
  - por padrão a ordem das colunas é ignorada; perguntas com "strict_columns" exigem a ordem.
  - colunas a mais ou a menos contam como erro: a pergunta diz o que retornar.
"""

import argparse
import csv
import importlib
import json
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from agent.db import UnsafeSQL, connect_ro, run_select  # noqa: E402

QUERY_TIMEOUT_S = 30.0


def load_questions(path, difficulty=None, limit=None, lote=None):
    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [q["id"] for q in questions]
    dupes = [i for i, n in Counter(ids).items() if n > 1]
    if dupes:
        raise SystemExit(f"ids repetidos em {path}: {dupes}")
    if difficulty:
        questions = [q for q in questions if q["difficulty"] == difficulty]
    if lote:
        questions = [q for q in questions if q.get("lote") == lote]
    return questions[:limit] if limit else questions


def normalize_value(value):
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, str):
        return value.strip()
    return value


def normalize_rows(rows, strict_columns):
    out = []
    for row in rows:
        values = [normalize_value(v) for v in row]
        if not strict_columns:
            values = sorted(values, key=repr)
        out.append(tuple(values))
    return out


def same_result(gold_rows, pred_rows, ordered, strict_columns):
    gold = normalize_rows(gold_rows, strict_columns)
    pred = normalize_rows(pred_rows, strict_columns)
    if ordered:
        return gold == pred
    return Counter(gold) == Counter(pred)


def execute(conn, sql):
    return run_select(sql, max_rows=None, timeout_s=QUERY_TIMEOUT_S, conn=conn)


def validate_gold(questions):
    conn = connect_ro()
    problems = 0
    for q in questions:
        try:
            result = execute(conn, q["gold_sql"])
        except (UnsafeSQL, sqlite3.Error) as exc:
            print(f"[ERRO ] {q['id']}: {exc}")
            problems += 1
            continue
        n = len(result.rows)
        preview = result.rows[0] if n else "-"
        all_null = n > 0 and all(all(v is None for v in r) for r in result.rows)
        flag = "OK   "
        if n == 0 or all_null:
            flag = "VAZIO"
            problems += 1
        print(f"[{flag}] {q['id']} ({q['difficulty']}): {n} linha(s); 1ª = {preview}")
    conn.close()
    print(f"\n{len(questions) - problems}/{len(questions)} sem problemas.")
    return problems == 0


def load_agent(spec):
    module_name, _, func_name = spec.partition(":")
    if not func_name:
        raise SystemExit("--agent deve ter o formato modulo:funcao, ex.: agent.sql_agent:generate_sql")
    return getattr(importlib.import_module(module_name), func_name)


def evaluate(questions, agent_fn, label):
    conn = connect_ro()
    records = []
    for q in questions:
        record = {
            "id": q["id"],
            "difficulty": q["difficulty"],
            "lote": q.get("lote", ""),
            "correct": False,
            "error": "",
            "latency_s": "",
            "input_tokens": "",
            "output_tokens": "",
            "attempts": "",
            "generated_sql": "",
            "question": q["question"],
        }
        start = time.perf_counter()
        try:
            answer = agent_fn(q["question"])
        except Exception as exc:  # o agente falhou: conta como erro, segue em frente
            record["latency_s"] = round(time.perf_counter() - start, 3)
            record["error"] = f"agente: {type(exc).__name__}: {exc}"
            records.append(record)
            print(f"[ERRO ] {q['id']}: {record['error']}")
            continue
        record["latency_s"] = round(time.perf_counter() - start, 3)

        if isinstance(answer, dict):
            sql = answer.get("sql", "")
            for key in ("input_tokens", "output_tokens", "attempts"):
                record[key] = answer.get(key, "")
        else:
            sql = answer
        record["generated_sql"] = " ".join((sql or "").split())

        try:
            gold = execute(conn, q["gold_sql"])
            pred = execute(conn, sql or "")
            record["correct"] = same_result(
                gold.rows, pred.rows, q.get("ordered", False), q.get("strict_columns", False)
            )
        except (UnsafeSQL, sqlite3.Error) as exc:
            record["error"] = f"sql: {exc}"
        records.append(record)
        print(f"[{'OK   ' if record['correct'] else 'FALHA'}] {q['id']} ({q['difficulty']})")
    conn.close()
    return records


def _num(value):
    return float(value) if value not in ("", None) else None


def summarize(records):
    by_diff = defaultdict(list)
    for r in records:
        by_diff[r["difficulty"]].append(r)

    def block(rows):
        latencies = [_num(r["latency_s"]) for r in rows if _num(r["latency_s"]) is not None]
        tokens = [
            _num(r["input_tokens"]) + _num(r["output_tokens"])
            for r in rows
            if _num(r["input_tokens"]) is not None and _num(r["output_tokens"]) is not None
        ]
        return {
            "n": len(rows),
            "acertos": sum(1 for r in rows if r["correct"]),
            "acuracia": round(sum(1 for r in rows if r["correct"]) / len(rows), 4) if rows else 0,
            "latencia_media_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "tokens_medios": round(sum(tokens) / len(tokens)) if tokens else None,
        }

    summary = {"geral": block(records)}
    for diff in ("facil", "media", "dificil"):
        if diff in by_diff:
            summary[diff] = block(by_diff[diff])
    # lote 1 = q01-q22 (usadas para desenhar v2-v4); lote 2 = q23-q60 (inéditas até a v4);
    # lote 3 = q61-q80 (escritas antes da v5, para medi-la)
    for lote in sorted({r["lote"] for r in records if r["lote"] != ""}):
        summary[f"lote{lote}"] = block([r for r in records if r["lote"] == lote])
    return summary


def print_summary(label, summary):
    print(f"\n=== {label} ===")
    print(f"{'grupo':<8} {'acertos':>8} {'acurácia':>9} {'lat. média (s)':>15} {'tokens médios':>14}")
    for name, s in summary.items():
        lat = "-" if s["latencia_media_s"] is None else s["latencia_media_s"]
        tok = "-" if s["tokens_medios"] is None else s["tokens_medios"]
        print(f"{name:<8} {s['acertos']:>3}/{s['n']:<4} {s['acuracia']:>9.1%} {lat!s:>15} {tok!s:>14}")


def run_config(agent_spec):
    """Provedor/modelo/versão desta rodada, para a tabela comparativa saber de onde veio cada uma."""
    config = {"agent": agent_spec, "data": datetime.now().isoformat(timespec="seconds")}
    try:
        from agent.config import get_settings

        s = get_settings()
        m = re.search(r"_(v\d+)$", agent_spec)
        version = m.group(1) if m else s.version
        config.update(provider="ollama", model=s.model, version=version, num_ctx=s.num_ctx)
    except Exception:  # agente externo sem agent.config: registra só o spec
        pass
    return config


def save(label, records, summary):
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"{label}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    (out_dir / f"{label}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nResultados em {csv_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--questions", default=str(HERE / "questions.json"))
    parser.add_argument("--validate-gold", action="store_true")
    parser.add_argument("--agent", help="modulo:funcao, ex.: agent.sql_agent:generate_sql")
    parser.add_argument("--label", help="nome desta rodada (padrão: função + data/hora)")
    parser.add_argument("--difficulty", choices=["facil", "media", "dificil"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--lote", type=int, help="1 = q01-q22, 2 = q23-q60, 3 = q61-q80")
    args = parser.parse_args()

    questions = load_questions(args.questions, args.difficulty, args.limit, args.lote)

    if args.validate_gold:
        sys.exit(0 if validate_gold(questions) else 1)

    if not args.agent:
        parser.error("informe --agent modulo:funcao ou use --validate-gold")

    agent_fn = load_agent(args.agent)
    label = args.label or f"{agent_fn.__name__}_{datetime.now():%Y%m%d_%H%M%S}"
    records = evaluate(questions, agent_fn, label)
    summary = summarize(records)
    summary["config"] = run_config(args.agent)
    print_summary(label, {k: v for k, v in summary.items() if k != "config"})
    save(label, records, summary)


if __name__ == "__main__":
    main()
