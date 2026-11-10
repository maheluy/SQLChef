"""Baixa o dataset Olist (Kaggle) e monta o banco SQLite em data/olist.db.

Uso (a partir da raiz do repositório):

    python data/build_db.py                 # usa data/raw/ se já existir, senão baixa
    python data/build_db.py --raw-dir DIR   # usa CSVs já baixados em DIR
    python data/build_db.py --db-path X.db  # grava o banco em outro lugar

Download automático: requer `pip install kagglehub`. Se falhar, baixe o zip em
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce, extraia os CSVs em
data/raw/ e rode o script de novo.

Só usa a biblioteca padrão para montar o banco (csv + sqlite3).

Licença do dataset: confira na página do Kaggle (uso não comercial com atribuição,
se a memória não falha). Por isso data/raw/ e o .db ficam fora do git.

Mudanças em relação aos CSVs originais:
  - products: colunas com erro de grafia renomeadas (name_lenght -> name_length,
    description_lenght -> description_length) e coluna extra
    product_category_name_english, vinda do arquivo de tradução.
  - geolocation não é carregada (1M de linhas, não ajuda nas perguntas).
  - strings vazias viram NULL. Datas ficam como TEXT 'AAAA-MM-DD HH:MM:SS'.
"""

import argparse
import csv
import os
import shutil
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RAW = HERE / "raw"
DEFAULT_DB = Path(os.environ.get("OLIST_DB", HERE / "olist.db"))
KAGGLE_SLUG = "olistbr/brazilian-ecommerce"
KAGGLE_URL = "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"
TRANSLATION_FILE = "product_category_name_translation.csv"


def col(name, sqltype="TEXT", src=None):
    """(nome no banco, tipo, nome no CSV)."""
    return (name, sqltype, src or name)


TABLES = {
    "customers": {
        "file": "olist_customers_dataset.csv",
        "cols": [
            col("customer_id"),
            col("customer_unique_id"),
            col("customer_zip_code_prefix"),
            col("customer_city"),
            col("customer_state"),
        ],
    },
    "sellers": {
        "file": "olist_sellers_dataset.csv",
        "cols": [
            col("seller_id"),
            col("seller_zip_code_prefix"),
            col("seller_city"),
            col("seller_state"),
        ],
    },
    "products": {
        "file": "olist_products_dataset.csv",
        "cols": [
            col("product_id"),
            col("product_category_name"),
            col("product_name_length", "INTEGER", "product_name_lenght"),
            col("product_description_length", "INTEGER", "product_description_lenght"),
            col("product_photos_qty", "INTEGER"),
            col("product_weight_g", "REAL"),
            col("product_length_cm", "REAL"),
            col("product_height_cm", "REAL"),
            col("product_width_cm", "REAL"),
        ],
        "extra": [("product_category_name_english", "TEXT")],
    },
    "orders": {
        "file": "olist_orders_dataset.csv",
        "cols": [
            col("order_id"),
            col("customer_id"),
            col("order_status"),
            col("order_purchase_timestamp"),
            col("order_approved_at"),
            col("order_delivered_carrier_date"),
            col("order_delivered_customer_date"),
            col("order_estimated_delivery_date"),
        ],
    },
    "order_items": {
        "file": "olist_order_items_dataset.csv",
        "cols": [
            col("order_id"),
            col("order_item_id", "INTEGER"),
            col("product_id"),
            col("seller_id"),
            col("shipping_limit_date"),
            col("price", "REAL"),
            col("freight_value", "REAL"),
        ],
    },
    "order_payments": {
        "file": "olist_order_payments_dataset.csv",
        "cols": [
            col("order_id"),
            col("payment_sequential", "INTEGER"),
            col("payment_type"),
            col("payment_installments", "INTEGER"),
            col("payment_value", "REAL"),
        ],
    },
    "order_reviews": {
        "file": "olist_order_reviews_dataset.csv",
        "cols": [
            col("review_id"),
            col("order_id"),
            col("review_score", "INTEGER"),
            col("review_comment_title"),
            col("review_comment_message"),
            col("review_creation_date"),
            col("review_answer_timestamp"),
        ],
    },
}

# Sem PRIMARY KEY de propósito: o Olist tem review_id repetido, e não queremos
# que o build quebre por isso. Índices cobrem os joins.
INDEXES = [
    ("customers", "customer_id"),
    ("customers", "customer_unique_id"),
    ("sellers", "seller_id"),
    ("products", "product_id"),
    ("orders", "order_id"),
    ("orders", "customer_id"),
    ("order_items", "order_id"),
    ("order_items", "product_id"),
    ("order_items", "seller_id"),
    ("order_payments", "order_id"),
    ("order_reviews", "order_id"),
]


def needed_files():
    return [spec["file"] for spec in TABLES.values()] + [TRANSLATION_FILE]


def missing_files(raw_dir):
    return [f for f in needed_files() if not (raw_dir / f).exists()]


def manual_instructions(raw_dir):
    return (
        "\nNão consegui obter os CSVs automaticamente. Faça manualmente:\n"
        f"  1. Baixe o dataset em {KAGGLE_URL}\n"
        f"  2. Extraia os CSVs em {raw_dir}\n"
        "  3. Rode este script de novo.\n"
    )


def ensure_raw(raw_dir):
    """Garante que os CSVs existem em raw_dir, baixando via kagglehub se preciso."""
    if not missing_files(raw_dir):
        print(f"CSVs encontrados em {raw_dir}")
        return

    print("CSVs não encontrados; tentando baixar via kagglehub...")
    try:
        import kagglehub  # type: ignore
    except ImportError:
        print("kagglehub não instalado (pip install kagglehub).")
        print(manual_instructions(raw_dir))
        sys.exit(1)

    try:
        downloaded = Path(kagglehub.dataset_download(KAGGLE_SLUG))
    except Exception as exc:  # erro de rede, credenciais, etc.
        print(f"Falha no download: {exc}")
        print(manual_instructions(raw_dir))
        sys.exit(1)

    raw_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in downloaded.rglob("*.csv"):
        shutil.copy(csv_path, raw_dir / csv_path.name)

    still_missing = missing_files(raw_dir)
    if still_missing:
        print(f"Faltam arquivos após o download: {still_missing}")
        print(manual_instructions(raw_dir))
        sys.exit(1)


def read_csv(path):
    # utf-8-sig remove BOM, se houver
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield row


def load_translation(raw_dir):
    return {
        row["product_category_name"]: row["product_category_name_english"]
        for row in read_csv(raw_dir / TRANSLATION_FILE)
    }


def create_table(conn, name, spec):
    columns = [(c[0], c[1]) for c in spec["cols"]] + spec.get("extra", [])
    ddl_cols = ", ".join(f"{n} {t}" for n, t in columns)
    conn.execute(f"CREATE TABLE {name} ({ddl_cols})")
    return [n for n, _ in columns]


def load_table(conn, name, spec, raw_dir, translation):
    db_cols = create_table(conn, name, spec)
    placeholders = ", ".join("?" for _ in db_cols)
    insert = f"INSERT INTO {name} ({', '.join(db_cols)}) VALUES ({placeholders})"

    path = raw_dir / spec["file"]
    batch, total = [], 0
    for row in read_csv(path):
        try:
            values = [row[src] or None for _, _, src in spec["cols"]]
        except KeyError as exc:
            raise SystemExit(
                f"Coluna {exc} não encontrada em {path.name}. "
                "O formato do dataset mudou?"
            )
        if name == "products":
            values.append(translation.get(row["product_category_name"]))
        batch.append(values)
        if len(batch) >= 10_000:
            conn.executemany(insert, batch)
            total += len(batch)
            batch = []
    if batch:
        conn.executemany(insert, batch)
        total += len(batch)
    print(f"  {name:<15} {total:>8} linhas")


def build(raw_dir, db_path):
    ensure_raw(raw_dir)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    translation = load_translation(raw_dir)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")
        print(f"Montando {db_path}")
        for name, spec in TABLES.items():
            load_table(conn, name, spec, raw_dir, translation)
        for table, column in INDEXES:
            conn.execute(f"CREATE INDEX idx_{table}_{column} ON {table} ({column})")
        conn.commit()
        conn.execute("ANALYZE")
        conn.commit()
    finally:
        conn.close()
    print("Pronto.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    build(args.raw_dir.resolve(), args.db_path.resolve())


if __name__ == "__main__":
    main()
