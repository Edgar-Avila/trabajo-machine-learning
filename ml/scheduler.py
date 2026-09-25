"""Periodic retraining loop with MLOps Quality Gatekeeper.

Runs the training script with --no-promote, triggers the MLOps gatekeeper
to evaluate the candidate model against production, and only runs batch
prediction if the model is approved or already active.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CURRENT_NAME = "current.txt"


def run(cmd: list[str]) -> bool:
    print(f"$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    return result.returncode == 0


def get_latest_version(model_dir: Path) -> int | None:
    version_dirs = [p for p in model_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not version_dirs:
        return None
    return max(int(p.name) for p in version_dirs)


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
        print("=== Iniciando ciclo de reentrenamiento MLOps ===", flush=True)
        # 1. Entrenar sin promover automáticamente
        train_ok = run([
            sys.executable,
            str(HERE / "train.py"),
            "--db", str(db),
            "--output", str(model_dir),
            "--no-promote"
        ])

        if train_ok:
            candidate_ver = get_latest_version(model_dir)
            if candidate_ver is not None:
                # 2. Gatekeeper: Evaluar si el nuevo candidato supera al modelo actual
                gate_ok = run([
                    sys.executable,
                    str(HERE / "promote_model.py"),
                    "--model-dir", str(model_dir),
                    "--candidate-version", str(candidate_ver),
                    "--metric", "roc_auc",
                    "--min-threshold", "0.70"
                ])

                if not gate_ok:
                    print(f"⚠️ Alerta MLOps: Modelo v{candidate_ver} rechazado. Producción protegida.", flush=True)

        # 3. Predicciones en lote con el modelo activo en current.txt
        run([sys.executable, str(HERE / "predict.py"), "--db", str(db), "--model-dir", str(model_dir)])
        print(f"Next training cycle in {args.interval}s", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
