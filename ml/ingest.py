"""Simulate a live stream of new customers into the churn database.

Every interval (default 5 minutes) the next unseen CSV row is inserted,
emulating organic growth of the customer base between training windows.
The seed script loads the first rows; this process appends the rest.
"""
import argparse
import numbers
import time

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


def next_customer(data: pd.DataFrame, conn) -> pd.Series | None:
    """Return the first CSV row whose CustomerID is not yet in the database."""
    max_id = conn.execute(
        "SELECT COALESCE(MAX(CustomerID), 0) FROM customers"
    ).fetchone()[0]
    pending = data[data["CustomerID"] > max_id]
    return pending.iloc[0] if not pending.empty else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream one new customer into the DB every N seconds."
    )
    parser.add_argument("--data", required=True, help="Path to the CSV dataset.")
    parser.add_argument("--db", required=True, help="Path to the SQLite database file.")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between new customers.")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    conn = connect(args.db)
    create_schema(conn)
    quoted = ", ".join(f'"{col}"' for col in COLUMNS)

    print(f"Ingesting one customer every {args.interval}s")
    while True:
        row = next_customer(data, conn)
        if row is None:
            print("No more customers left in the CSV source; waiting anyway.")
        else:
            conn.execute(
                f"INSERT INTO customers ({quoted}) VALUES ({', '.join('?' for _ in COLUMNS)})",
                tuple(
                    int(value) if isinstance(value, numbers.Integral) else value
                    for value in row[COLUMNS].tolist()
                ),
            )
            conn.commit()
            print(f"Inserted CustomerID {int(row.CustomerID)}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()