"""Carga los resultados (summary.json) de SIA y competidores en tablas uniformes.

Estructura esperada (generada por los runners):
    results/SIA_online/<algo>/<familia>/<func>/dim_<d>/run_<r>/summary.json
    results/competitors/<algo>/<familia>/<func>/dim_<d>/run_<r>/summary.json

Devuelve listas de dicts con: algorithm, family, benchmark, dim, run_id,
best_fitness, error_to_optimum, y la curva de convergencia si se pide.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import RESULTS  # noqa: E402  (raíz única; ver paths.py)


def _metric(summary: dict, metric: str) -> float:
    """Devuelve la métrica pedida; cae a best_fitness si no hay error_to_optimum."""
    if metric == "error_to_optimum":
        val = summary.get("error_to_optimum")
        if val is None:  # BBOB no tiene f*: usar fitness crudo
            val = summary.get("best_fitness")
        return float(val)
    return float(summary.get("best_fitness"))


def load_summaries(root: Path) -> list[dict]:
    """Lee recursivamente todos los summary.json bajo `root`."""
    out = []
    for path in root.rglob("summary.json"):
        try:
            out.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def collect(algorithms: dict[str, Path], metric: str = "error_to_optimum") -> list[dict]:
    """Carga y unifica resultados de varios algoritmos.

    algorithms: {nombre_a_mostrar: carpeta_raíz_de_ese_algoritmo}.
    Devuelve filas: {algorithm, family, benchmark, dim, run_id, value}.
    """
    rows = []
    for algo_name, root in algorithms.items():
        for s in load_summaries(Path(root)):
            rows.append({
                "algorithm": algo_name,
                "family": s.get("family"),
                "benchmark": s.get("benchmark"),
                "dim": s.get("dim"),
                "run_id": s.get("run_id"),
                "value": _metric(s, metric),
            })
    return rows


def load_convergence(run_dir: Path) -> list[tuple[float, float]]:
    """Lee convergence.csv de una corrida. Soporta columnas (iteration|fe, best_fitness)."""
    f = Path(run_dir) / "convergence.csv"
    if not f.exists():
        return []
    out = []
    with f.open() as fh:
        reader = csv.DictReader(fh)
        xcol = "fe" if "fe" in reader.fieldnames else "iteration"
        for r in reader:
            out.append((float(r[xcol]), float(r["best_fitness"])))
    return out
