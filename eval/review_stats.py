"""Efeito do revisor (v6): compara a SQL antes e depois da revisão com a SQL de referência.

    python eval/review_stats.py eval/results/qwen7b_v6_100.csv

Para cada pergunta em que o revisor apontou um problema, classifica:
    salvou    errada antes da revisão, certa depois
    estragou  certa antes, errada depois
    manteve   certa antes e depois
    não ajudou  errada antes e depois
"""

import csv
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from run_eval import execute, same_result  # noqa: E402

from agent.db import UnsafeSQL, connect_ro  # noqa: E402


def correct(conn, q, sql):
    try:
        gold = execute(conn, q["gold_sql"])
        pred = execute(conn, sql)
    except (UnsafeSQL, sqlite3.Error):
        return False
    return same_result(gold.rows, pred.rows, q.get("ordered", False), q.get("strict_columns", False))


def main(path, questions_path=HERE / "questions.json"):
    questions = {q["id"]: q for q in json.loads(Path(questions_path).read_text(encoding="utf-8"))}
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    conn = connect_ro()
    reviewed = [r for r in rows if r.get("review")]
    flagged = [r for r in reviewed if r["review"] != "OK"]
    outcome = Counter()
    by_lote = Counter()
    details = []
    for r in flagged:
        before = correct(conn, questions[r["id"]], r["pre_review_sql"])
        after = r["correct"] == "True"
        kind = {(False, True): "salvou", (True, False): "estragou",
                (True, True): "manteve", (False, False): "não ajudou"}[(before, after)]
        outcome[kind] += 1
        by_lote[(r["lote"], kind)] += 1
        details.append((r["id"], r["lote"], kind, r.get("review_reverted") == "True", r["review"][:110]))
    conn.close()

    print(f"{path}: {len(rows)} perguntas, {len(reviewed)} revisadas, {len(flagged)} com problema apontado")
    for kind in ("salvou", "estragou", "manteve", "não ajudou"):
        lotes = ", ".join(f"lote {l}: {n}" for (l, k), n in sorted(by_lote.items()) if k == kind)
        print(f"  {kind:<11} {outcome[kind]:>3}   ({lotes})")
    reverted = sum(1 for d in details if d[3])
    print(f"  SQL revisada falhou e voltou à anterior: {reverted}")
    print()
    for qid, lote, kind, rev, text in details:
        print(f"  {qid:<5} lote {lote}  {kind:<11}{' (revertida)' if rev else ''}  {text}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("uso: python eval/review_stats.py eval/results/<label>.csv")
    main(sys.argv[1])
