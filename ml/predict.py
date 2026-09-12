"""Batch-predict churn for every customer and persist it in the database.

Loads the active (versioned) model, runs predict_proba over all customer rows
in one shot and stores the results in the predictions table. Predictions are
kept per model version so the churn history of each customer can be traced
across versions.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from db import connect, create_schema
from features import FEATURES


def load_active_model(model_dir: Path) -> tuple[object, int]:
    """Load the model pointed to by current.txt and return it with its version."""
    current = model_dir / "current.txt"
    if not current.exists():
        raise SystemExit(f"No current.txt found in {model_dir}")
    version = int(current.read_text(encoding="utf-8").strip())
    return joblib.load(model_dir / str(version) / "model.pkl"), version


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-predict churn for all customers.")
    parser.add_argument("--db", required=True, help="Path to the SQLite database file.")
    parser.add_argument("--model-dir", required=True, help="Root directory of versioned models.")
    args = parser.parse_args()

    conn = connect(args.db)
    create_schema(conn)
    data = pd.read_sql_query("SELECT * FROM customers", conn)

    if data.empty:
        print("No customers to predict.")
        conn.close()
        return

    model, version = load_active_model(Path(args.model_dir))
    probability = model.predict_proba(data[FEATURES])[:, 1]

    predicted_at = datetime.now(timezone.utc).isoformat()
    conn.execute("DELETE FROM predictions WHERE ModelVersion = ?", (version,))
    conn.executemany(
        "INSERT INTO predictions (CustomerID, ChurnProbability, ModelVersion, PredictedAt) VALUES (?, ?, ?, ?)",
        [
            (int(row.CustomerID), float(prob), version, predicted_at)
            for row, prob in zip(data.itertuples(index=False), probability)
        ],
    )
    conn.commit()

    rows = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE ModelVersion = ?", (version,)
    ).fetchone()[0]
    print(f"Predicted {rows} customers with model v{version} into {args.db}")
    conn.close()


if __name__ == "__main__":
    main()