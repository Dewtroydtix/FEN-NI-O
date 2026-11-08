"""CLI: python -m ninoprob_ia.cli entrenar [--archivo archivo_noaa.ascii]."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from ninoprob_ia.core import NOAA_URL, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento científico de NIÑOPROB-IA.")
    parser.add_argument("comando", choices=["entrenar"])
    parser.add_argument("--directorio", default=".", help="Carpeta de datos/ y resultados/.")
    parser.add_argument("--archivo", default=None, help="Archivo ERSSTv5 descargado de NOAA.")
    parser.add_argument("--sin-descarga", action="store_true", help="Usar solo archivo existente.")
    args = parser.parse_args()
    try:
        metrics = run_pipeline(
            Path(args.directorio), input_file=args.archivo, allow_download=not args.sin_descarga
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\nFuente oficial: {NOAA_URL}\n")
    print(json.dumps({
        "estado": "entrenado", "mejor_C": metrics["mejor_C"],
        "resultados_prueba": metrics["prueba"], "conclusion": metrics["conclusion"]
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
