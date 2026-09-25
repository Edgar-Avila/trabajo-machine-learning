import argparse
import json
import sys
from pathlib import Path

MANIFEST_NAME = "manifest.json"
CURRENT_NAME = "current.txt"

def load_manifest(version_dir: Path) -> dict:
    manifest_path = version_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"No se encontró manifest en {version_dir}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))

def main():
    parser = argparse.ArgumentParser(description="Validar y promover un nuevo modelo a producción.")
    parser.add_argument("--model-dir", required=True, help="Directorio raíz de modelos (donde está current.txt)")
    parser.add_argument("--candidate-version", type=int, required=True, help="Versión del modelo candidato recién entrenado")
    parser.add_argument("--metric", default="roc_auc", choices=["roc_auc", "accuracy"], help="Métrica de comparación")
    parser.add_argument("--min-threshold", type=float, default=0.70, help="Umbral mínimo aceptable de la métrica")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    current_file = model_dir / CURRENT_NAME
    candidate_dir = model_dir / str(args.candidate_version)

    candidate_manifest = load_manifest(candidate_dir)
    candidate_score = candidate_manifest.get("metrics", {}).get(args.metric, 0.0)

    print("=== MLOPS GATEKEEPER ===")
    print(f"Evaluando Modelo Candidato v{args.candidate_version}...")
    print(f"Métrica objetivo ({args.metric}): {candidate_score:.4f} (Umbral mínimo: {args.min_threshold})")

    # Regla 1: Superar el umbral mínimo absoluto
    if candidate_score < args.min_threshold:
        print(f"RECHAZADO: La métrica {candidate_score:.4f} está por debajo del umbral mínimo {args.min_threshold}.")
        sys.exit(1)

    # Regla 2: Comparar contra el modelo en producción (si existe)
    if current_file.exists():
        current_version_str = current_file.read_text(encoding="utf-8").strip()
        if current_version_str:
            current_version = int(current_version_str)
            if current_version == args.candidate_version:
                print(f"El modelo v{args.candidate_version} ya es el modelo activo en producción.")
                sys.exit(0)

            current_dir = model_dir / str(current_version)
            current_manifest = load_manifest(current_dir)
            current_score = current_manifest.get("metrics", {}).get(args.metric, 0.0)
            print(f"Modelo Actual en Producción v{current_version} ({args.metric}): {current_score:.4f}")

            if candidate_score < current_score:
                print(f"RECHAZADO: El modelo candidato ({candidate_score:.4f}) es inferior al modelo actual ({current_score:.4f}).")
                print("Producción protegida: No se modificó current.txt.")
                sys.exit(1)

    # Si pasa todas las validaciones -> Promover a producción
    current_file.write_text(str(args.candidate_version), encoding="utf-8")
    print(f"APROBADO: Modelo v{args.candidate_version} promovido exitosamente como el nuevo modelo activo en {current_file.name}.")
    sys.exit(0)

if __name__ == "__main__":
    main()
