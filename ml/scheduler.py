"""Periodic retraining loop.

Runs the training script followed immediately by a batch predict every
interval (default 12 hours). The first cycle runs at startup so the app has
a trained model and predictions right away.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain and predict on a schedule.")
    parser.add_argument("--db", required=True, help="Path to the SQLite database file.")
    parser.add_argument(
        "--model-dir", required=True, help="Directory where versioned artifacts are stored."
    )
    parser.add_argument("--interval", type=int, default=12 * 3600, help="Seconds between cycles.")
    args = parser.parse_args()
    db = Path(args.db).resolve()
    model_dir = Path(args.model_dir).resolve()

    while True:
        run([sys.executable, str(HERE / "train.py"), "--db", str(db), "--output", str(model_dir)])
        run([sys.executable, str(HERE / "predict.py"), "--db", str(db), "--model-dir", str(model_dir)])
        print(f"Next training cycle in {args.interval}s", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()