"""Shared database helpers for the ML pipeline."""
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path("/db/churn.db")


def connect(db_path: str | None = None) -> sqlite3.Connection:
    """Open a connection to the churn database, creating parent dirs if needed."""
    path = Path(db_path or DEFAULT_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def create_schema(conn: sqlite3.Connection) -> None:
    """Create the customers and predictions tables if they do not exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS customers (
            CustomerID        INTEGER PRIMARY KEY,
            Age               INTEGER,
            Gender            TEXT,
            Tenure            INTEGER,
            "Usage Frequency" INTEGER,
            "Support Calls"   INTEGER,
            "Payment Delay"   INTEGER,
            "Subscription Type" TEXT,
            "Contract Length" TEXT,
            "Total Spend"     REAL,
            "Last Interaction" INTEGER,
            Churn             INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            CustomerID        INTEGER NOT NULL,
            ChurnProbability  REAL,
            ModelVersion      INTEGER NOT NULL,
            PredictedAt       TEXT,
            PRIMARY KEY (CustomerID, ModelVersion)
        )
        """
    )
    conn.commit()