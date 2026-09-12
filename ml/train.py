"""Churn model training script.

Reads the customer rows from the SQLite database, applies preprocessing and
trains a lightweight classifier (no GPU needed). Models are versioned: each
training run that sees changed data increments the version and writes a
versioned folder (model/<version>/) containing the artifact plus its own
manifest.json. A current.txt pointer at the root marks the active version.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from db import connect, create_schema
from features import CATEGORICAL, FEATURES, NUMERIC, TARGET
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MANIFEST_NAME = "manifest.json"
CURRENT_NAME = "current.txt"


def current_version(artifacts_dir: Path) -> int | None:
    """Read the active version from current.txt, or None if absent."""
    path = artifacts_dir / CURRENT_NAME
    if not path.exists():
        return None
    return int(path.read_text(encoding="utf-8").strip())


def load_manifest(version_dir: Path) -> dict | None:
    """Read the manifest of a version folder, or None if it does not exist."""
    path = version_dir / MANIFEST_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def data_hash(frame: pd.DataFrame) -> str:
    """Stable identifier of the exact training rows (used to detect changes)."""
    return hashlib.md5(
        frame[FEATURES + [TARGET]].to_csv(index=False).encode("utf-8")
    ).hexdigest()


def next_version(artifacts_dir: Path) -> int:
    """Return the next model version (monotonic, starts at 1)."""
    version_dirs = [p for p in artifacts_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not version_dirs:
        return 1
    return max(int(p.name) for p in version_dirs) + 1


def build_pipeline() -> Pipeline:
    """Return a lightweight sklearn pipeline ready to be trained."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=42,
            )),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a churn classifier.")
    parser.add_argument("--db", required=True, help="Path to the SQLite database file.")
    parser.add_argument("--output", required=True, help="Directory where versioned artifacts are stored.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on rows to train on (default: all).")
    args = parser.parse_args()

    conn = connect(args.db)
    create_schema(conn)
    data = pd.read_sql_query("SELECT * FROM customers", conn)
    conn.close()
    if args.max_rows:
        data = data.head(args.max_rows)

    for feature in FEATURES + [TARGET]:
        if feature not in data.columns:
            raise ValueError(f"Missing expected column: {feature}")

    artifacts_dir = Path(args.output)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    version = current_version(artifacts_dir)
    current_manifest = (
        load_manifest(artifacts_dir / str(version)) if version is not None else None
    )
    current_hash = data_hash(data)

    if current_manifest is not None and current_manifest["data_hash"] == current_hash:
        print(f"Data unchanged (hash {current_hash[:8]}), keeping version {version}")
        return

    version = next_version(artifacts_dir)
    version_dir = artifacts_dir / str(version)
    version_dir.mkdir(parents=True, exist_ok=True)
    model_path = version_dir / "model.pkl"

    x = data[FEATURES]
    y = data[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    y_proba = pipeline.predict_proba(x_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    accuracy = float(accuracy_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_proba))
    print(f"Rows trained: {len(x_train)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC:  {roc_auc:.4f}")

    joblib.dump(pipeline, model_path)

    manifest = {
        "version": version,
        "model_path": model_path.name,
        "data_hash": current_hash,
        "rows": len(x_train),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {"accuracy": accuracy, "roc_auc": roc_auc},
    }
    (version_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (artifacts_dir / CURRENT_NAME).write_text(str(version), encoding="utf-8")
    print(f"Model v{version} saved to {model_path}")


if __name__ == "__main__":
    main()