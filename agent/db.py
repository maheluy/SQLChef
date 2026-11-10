"""Acesso ao SQLite: conexão somente leitura, execução segura e schema para o prompt.

Usado tanto pelo agente quanto pela avaliação.
"""

import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

DB_PATH = Path(os.environ.get("OLIST_DB", Path(__file__).resolve().parents[1] / "data" / "olist.db"))


class UnsafeSQL(ValueError):
    """A consulta não é um único SELECT."""


@dataclass
class QueryResult:
    columns: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    truncated: bool = False


def connect_ro(path=None):
    """Abre o banco em modo somente leitura (o SQLite recusa qualquer escrita)."""
    db = Path(path or DB_PATH).resolve()
    if not db.exists():
        raise FileNotFoundError(f"Banco não encontrado: {db}. Rode data/build_db.py primeiro.")
    conn = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def validate_select(sql):
    """Aceita só um único SELECT (ou WITH ... SELECT). Devolve a SQL sem ';' final.

    É uma defesa em profundidade: a proteção de verdade é a conexão somente leitura.
    """
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise UnsafeSQL("SQL vazia")
    if ";" in cleaned:
        raise UnsafeSQL("Apenas uma instrução por vez")
    if not re.match(r"(?is)^(select|with)\b", cleaned):
        raise UnsafeSQL("Apenas consultas SELECT são permitidas")
    return cleaned


def run_select(sql, *, max_rows=200, timeout_s=10.0, conn=None):
    """Valida e executa um SELECT.

    max_rows: limite de linhas devolvidas (None = todas). Não reescreve a SQL;
              apenas corta o resultado e marca `truncated`.
    timeout_s: aborta consultas lentas (ex.: produto cartesiano acidental).
    Levanta UnsafeSQL ou sqlite3.Error; o agente deve ler a mensagem e tentar corrigir.
    """
    cleaned = validate_select(sql)
    own_conn = conn is None
    conn = conn or connect_ro()
    try:
        deadline = time.monotonic() + timeout_s
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        cur = conn.execute(cleaned)
        columns = [d[0] for d in cur.description] if cur.description else []
        if max_rows is None:
            return QueryResult(columns, cur.fetchall(), False)
        rows = cur.fetchmany(max_rows + 1)
        return QueryResult(columns, rows[:max_rows], len(rows) > max_rows)
    finally:
        conn.set_progress_handler(None, 0)
        if own_conn:
            conn.close()


def schema_prompt(sample_rows=3, conn=None):
    """Texto com CREATE TABLE e algumas linhas de exemplo de cada tabela."""
    own_conn = conn is None
    conn = conn or connect_ro()
    try:
        tables = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        parts = []
        for name, ddl in tables:
            parts.append(ddl + ";")
            if sample_rows:
                cur = conn.execute(f"SELECT * FROM {name} LIMIT {int(sample_rows)}")
                cols = [d[0] for d in cur.description]
                parts.append(f"-- exemplos de {name} ({', '.join(cols)}):")
                for row in cur.fetchall():
                    parts.append("-- " + " | ".join("NULL" if v is None else str(v)[:40] for v in row))
            parts.append("")
        return "\n".join(parts)
    finally:
        if own_conn:
            conn.close()
