"""Prompts do agente: instruções, schema, exemplos few-shot e mensagem de correção.

Os exemplos few-shot NÃO repetem perguntas de eval/questions.json, para não inflar a acurácia.
"""

import re
from functools import lru_cache

from agent.db import connect_ro, schema_prompt

INSTRUCTIONS = """Você é um analista de dados que escreve consultas SQL para um banco SQLite \
com dados públicos da Olist, um marketplace brasileiro de e-commerce.

Regras:
- Use apenas as tabelas e colunas do schema abaixo. O dialeto é SQLite.
- Escreva uma única consulta SELECT (ou WITH ... SELECT). Nunca altere dados.
- Retorne exatamente as colunas que a pergunta pede, na ordem em que aparecem na pergunta; \
não acrescente colunas extras.
- Datas são texto no formato 'AAAA-MM-DD HH:MM:SS'. Use strftime para extrair ano/mês e \
julianday para diferenças em dias.
- Valores de status e tipo de pagamento estão em inglês (ex.: 'delivered', 'canceled', 'credit_card', 'boleto').
- Um cliente real é identificado por customers.customer_unique_id; customer_id muda a cada pedido.
- Nomes de categoria em inglês estão em products.product_category_name_english.
- Use ORDER BY quando a pergunta pedir ranking ou "os N maiores", e LIMIT nesses casos.
- Responda somente com a consulta dentro de um bloco ```sql ... ```, sem explicações.

Schema (com uma linha de exemplo por tabela):
"""

FEW_SHOT = [
    (
        "Quantos produtos existem na categoria perfumery?",
        "SELECT COUNT(*) FROM products WHERE product_category_name_english = 'perfumery'",
    ),
    (
        "Qual cidade tem mais vendedores? Retorne a cidade e a quantidade de vendedores.",
        "SELECT seller_city, COUNT(*) AS qtd FROM sellers "
        "GROUP BY seller_city ORDER BY qtd DESC LIMIT 1",
    ),
    (
        "Quantos pedidos foram feitos em cada ano? Retorne o ano e a quantidade, ordenado por ano.",
        "SELECT strftime('%Y', order_purchase_timestamp) AS ano, COUNT(*) AS qtd FROM orders "
        "GROUP BY ano ORDER BY ano",
    ),
    (
        "Qual é o frete médio por item em pedidos de clientes do estado do RJ?",
        "SELECT ROUND(AVG(oi.freight_value), 2) FROM order_items oi "
        "JOIN orders o ON o.order_id = oi.order_id "
        "JOIN customers c ON c.customer_id = o.customer_id "
        "WHERE c.customer_state = 'RJ'",
    ),
    (
        "Quantos pedidos pagos com boleto receberam nota 1 na avaliação?",
        "SELECT COUNT(DISTINCT p.order_id) FROM order_payments p "
        "JOIN order_reviews r ON r.order_id = p.order_id "
        "WHERE p.payment_type = 'boleto' AND r.review_score = 1",
    ),
    (
        "Em média, quantos dias se passam entre a aprovação do pedido e a entrega à transportadora, "
        "considerando pedidos com as duas datas registradas?",
        "SELECT ROUND(AVG(julianday(order_delivered_carrier_date) - julianday(order_approved_at)), 2) "
        "FROM orders WHERE order_approved_at IS NOT NULL AND order_delivered_carrier_date IS NOT NULL",
    ),
    (
        "Quais são as 3 categorias de produto (nome em inglês) com maior peso médio, ignorando "
        "produtos sem categoria? Retorne a categoria e o peso médio em gramas.",
        "SELECT product_category_name_english, ROUND(AVG(product_weight_g), 2) AS peso_medio "
        "FROM products WHERE product_category_name_english IS NOT NULL "
        "GROUP BY product_category_name_english ORDER BY peso_medio DESC LIMIT 3",
    ),
]


@lru_cache(maxsize=None)
def system_prompt(few_shot: bool) -> str:
    """Prompt do sistema. Calculado uma vez por variante (o schema não muda durante a execução)."""
    parts = [INSTRUCTIONS, schema_prompt(sample_rows=1)]
    if few_shot:
        parts.append("Exemplos de perguntas e consultas corretas:\n")
        for question, sql in FEW_SHOT:
            parts.append(f"Pergunta: {question}\n```sql\n{sql}\n```\n")
    return "\n".join(parts)


def question_message(question: str) -> dict:
    return {"role": "user", "content": f"Pergunta: {question}"}


@lru_cache(maxsize=None)
def _table_columns() -> dict:
    """{tabela: [colunas]} lido do banco (ignora tabelas internas do SQLite)."""
    conn = connect_ro()
    try:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        return {t: [r[1] for r in conn.execute(f"PRAGMA table_info({t})")] for t in tables}
    finally:
        conn.close()


_NO_COLUMN = re.compile(r"no such column: (?:[\w]+\.)?(\w+)", re.IGNORECASE)
_NO_TABLE = re.compile(r"no such table: (\w+)", re.IGNORECASE)


def schema_hint(error: str) -> str:
    """Dica de schema para erros de ligação (v4): onde a coluna existe e com quais colunas.

    Ver as colunas das tabelas certas deixa visível a chave de JOIN (ex.: customer_id),
    que a mensagem crua do SQLite não mostra.
    """
    tables = _table_columns()
    if m := _NO_COLUMN.search(error or ""):
        column = m.group(1)
        owners = [t for t, cols in tables.items() if column in cols]
        if not owners:
            return f"Dica: a coluna {column} não existe em nenhuma tabela; confira o schema."
        lines = [f"Dica: a coluna {column} existe apenas em: {', '.join(owners)}."]
        lines += [f"  {t}({', '.join(tables[t])})" for t in owners]
        lines.append("Se precisar dela, faça JOIN com essa tabela por uma coluna em comum.")
        return "\n".join(lines)
    if m := _NO_TABLE.search(error or ""):
        return f"Dica: a tabela {m.group(1)} não existe. Tabelas disponíveis: {', '.join(tables)}."
    return ""


def repair_message(sql: str, error: str, hint: str = "") -> dict:
    return {
        "role": "user",
        "content": (
            "A consulta abaixo falhou ou devolveu um resultado suspeito.\n"
            f"```sql\n{sql}\n```\n"
            f"Problema: {error}\n"
            + (f"{hint}\n" if hint else "")
            + "Corrija a consulta para responder à pergunta original. "
            "Responda somente com a consulta corrigida em um bloco ```sql ... ```."
        ),
    }


ANSWER_SYSTEM = (
    "Você responde em português, em uma ou duas frases, a pergunta do usuário usando apenas "
    "o resultado da consulta fornecido. Não invente números. Se o resultado estiver truncado, diga isso."
)


def answer_message(question: str, columns: list, rows: list, truncated: bool) -> dict:
    preview = "\n".join(" | ".join(str(v) for v in row) for row in rows[:20])
    note = "\n(resultado truncado)" if truncated else ""
    return {
        "role": "user",
        "content": f"Pergunta: {question}\nColunas: {', '.join(columns)}\nResultado:\n{preview}{note}",
    }


_SQL_BLOCK = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_sql(text: str) -> str:
    """Pega a última consulta em bloco ``` da resposta; sem bloco, usa o texto inteiro."""
    blocks = [b.strip() for b in _SQL_BLOCK.findall(text or "") if b.strip()]
    sql = blocks[-1] if blocks else (text or "").strip()
    return sql.strip().rstrip(";").strip()
