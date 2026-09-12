"""Read persisted customer predictions from the database."""
import sqlite3
from collections import Counter
from pathlib import Path

from api.schemas import (
    CustomerChurnRow,
    CustomerHistoryResponse,
    CustomerListResponse,
    DashboardSummary,
    GroupStat,
    RiskCount,
    VersionPrediction,
)
from services.risk import classify_risk


class CustomerRepository:
    """Serve paginated customer rows with their stored churn prediction."""

    def __init__(self, db_path: Path | None) -> None:
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection | None:
        if self.db_path is None or not self.db_path.exists():
            return None
        return sqlite3.connect(self.db_path)

    def _latest_version(self, conn: sqlite3.Connection) -> int | None:
        """Return the newest model version present in the predictions table."""
        row = conn.execute("SELECT MAX(ModelVersion) FROM predictions").fetchone()
        return row[0] if row and row[0] is not None else None

    def list_paginated(self, page: int, page_size: int) -> CustomerListResponse:
        """Return one page of customers ordered by churn probability (desc)."""
        conn = self._connect()
        if conn is None:
            return CustomerListResponse(items=[], page=page, page_size=page_size, total=0, total_pages=0)

        version = self._latest_version(conn)
        if version is None:
            conn.close()
            return CustomerListResponse(items=[], page=page, page_size=page_size, total=0, total_pages=0)

        total = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE ModelVersion = ?", (version,)
        ).fetchone()[0]
        offset = (page - 1) * page_size
        rows = conn.execute(
            """
            SELECT c.CustomerID, c.Age, c.Gender, c.Tenure, c."Usage Frequency",
                   c."Support Calls", c."Payment Delay", c."Subscription Type",
                   c."Contract Length", c."Total Spend", c."Last Interaction",
                   p.ChurnProbability, p.ModelVersion
            FROM customers c
            JOIN predictions p ON c.CustomerID = p.CustomerID
            WHERE p.ModelVersion = ?
            ORDER BY p.ChurnProbability DESC
            LIMIT ? OFFSET ?
            """,
            (version, page_size, offset),
        ).fetchall()
        conn.close()

        items = [
            CustomerChurnRow(
                CustomerID=str(r[0]),
                Age=r[1],
                Gender=r[2],
                Tenure=r[3],
                **{"Usage Frequency": r[4], "Support Calls": r[5], "Payment Delay": r[6]},
                **{"Subscription Type": r[7], "Contract Length": r[8]},
                **{"Total Spend": r[9], "Last Interaction": r[10]},
                ChurnProbability=r[11],
                ModelVersion=r[12],
                RiskLevel=classify_risk(r[11]).value,
            )
            for r in rows
        ]
        return CustomerListResponse(
            items=items,
            page=page,
            page_size=page_size,
            total=total,
            total_pages=(total + page_size - 1) // page_size,
        )

    def summary(self) -> DashboardSummary:
        """Compute the aggregates displayed in the dashboard."""
        conn = self._connect()
        if conn is None:
            return DashboardSummary(
                total_customers=0,
                avg_churn_probability=0.0,
                risk_counts=[],
                by_subscription=[],
                by_contract=[],
            )

        version = self._latest_version(conn)
        if version is None:
            conn.close()
            return DashboardSummary(
                total_customers=0,
                avg_churn_probability=0.0,
                risk_counts=[],
                by_subscription=[],
                by_contract=[],
            )

        total, avg_churn = conn.execute(
            "SELECT COUNT(*), AVG(ChurnProbability) FROM predictions WHERE ModelVersion = ?",
            (version,),
        ).fetchone()
        probabilities = [
            p
            for (p,) in conn.execute(
                "SELECT ChurnProbability FROM predictions WHERE ModelVersion = ?", (version,)
            ).fetchall()
        ]
        buckets = Counter(classify_risk(p).value for p in probabilities)
        risk_counts = [
            RiskCount(level=level, count=buckets.get(level, 0), percentage=round(100 * buckets.get(level, 0) / total, 1))
            for level in ("Low", "Medium", "High")
        ]

        def groups(column: str) -> list[GroupStat]:
            rows = conn.execute(
                f"""
                SELECT c.{column}, COUNT(*), AVG(p.ChurnProbability)
                FROM customers c
                JOIN predictions p ON c.CustomerID = p.CustomerID
                WHERE p.ModelVersion = ?
                GROUP BY c.{column}
                ORDER BY AVG(p.ChurnProbability) DESC
                """,
                (version,),
            ).fetchall()
            return [GroupStat(name=name, customers=count, avg_churn_probability=round(avg, 4)) for name, count, avg in rows]

        by_subscription = groups('"Subscription Type"')
        by_contract = groups('"Contract Length"')
        conn.close()

        return DashboardSummary(
            total_customers=total,
            avg_churn_probability=round(avg_churn, 4),
            risk_counts=risk_counts,
            by_subscription=by_subscription,
            by_contract=by_contract,
        )

    def history(self, customer_id: str) -> CustomerHistoryResponse | None:
        """Return the churn history across versions for one customer."""
        conn = self._connect()
        if conn is None:
            return None
        churn = conn.execute(
            "SELECT Churn FROM customers WHERE CustomerID = ?", (customer_id,)
        ).fetchone()
        rows = conn.execute(
            """
            SELECT ModelVersion, ChurnProbability, PredictedAt
            FROM predictions
            WHERE CustomerID = ?
            ORDER BY ModelVersion
            """,
            (customer_id,),
        ).fetchall()
        conn.close()
        if not rows:
            return None

        history: list[VersionPrediction] = []
        for index, (version, probability, predicted_at) in enumerate(rows):
            delta = (
                probability - rows[index - 1][1]
                if index > 0 else None
            )
            history.append(
                VersionPrediction(
                    version=version,
                    churn_probability=round(probability, 4),
                    risk_level=classify_risk(probability),
                    predicted_at=predicted_at,
                    delta=round(delta, 4) if delta is not None else None,
                )
            )
        return CustomerHistoryResponse(
            customer_id=customer_id,
            churn_label=int(churn[0]) if churn is not None else None,
            history=history,
        )