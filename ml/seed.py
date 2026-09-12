"""Seed the SQLite database from the raw CSV dataset.

Re-running the script reloads the database from the CSV, so the stored
rows always reflect the latest data source before a re-training.
"""
import argparse

import pandas as pd

from db import connect, create_schema

COLUMNS = [
    "CustomerID",
    "Age",
    "Gender",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
    "Last Interaction",
    "Churn",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Load CSV data into the churn database.")
    parser.add_argument("--data", required=True, help="Path to the CSV dataset.")
    parser.add_argument("--db", required=True, help="Path to the SQLite database file.")
    parser.add_argument("--max-rows", type=int, default=2000, help="Max rows to load.")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    if args.max_rows:
        data = data.head(args.max_rows)

    conn = connect(args.db)
    create_schema(conn)
    conn.execute("DELETE FROM customers")
    quoted = ", ".join(f'"{col}"' for col in COLUMNS)
    conn.executemany(
        f"INSERT INTO customers ({quoted}) VALUES ({', '.join('?' for _ in COLUMNS)})",
        data[COLUMNS].itertuples(index=False, name=None),
    )
    conn.commit()

    rows = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    print(f"Loaded {rows} rows into {args.db}")
    conn.close()


if __name__ == "__main__":
    main()