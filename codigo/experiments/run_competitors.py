"""Ejecuta los algoritmos competidores sobre los mismos benchmarks y presupuesto
que el SIA online, para una comparación justa.

Presupuesto: FEs = pop_size * max_iter (por defecto 30*100 = 3000), idéntico al
SIA. Cada competidor gestiona su propia estructura interna pero está limitado por
el contador de FEs compartido (algorithms/competitors/base.py).

Salida por corrida (misma estructura de carpetas que el SIA):
    results/competitors/<COMP>/<familia>/<func>/dim_<d>/run_<r>/
        convergence.csv   (fe, best_fitness)
        summary.json

Uso:
    python -m experiments.run_competitors --family cec2017 --benchmarks F1 F5 --dims 10 20 50
    python -m experiments.run_competitors --competitors ASGK EA4eig --family bbob --benchmarks f01 --dims 10 20
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.competitors import COMPETITORS, get_competitor
from benchmarks.registry import get_problem, supported_dims, FAMILIES

from paths import RESULTS  # noqa: E402  (raíz única; ver paths.py)

RESULTS_ROOT = RESULTS / "competitors"


def run_one(comp_name, family, benchmark, dim, run_id, max_fes, seed, out_dir) -> dict:
    objective, lower, upper, info = get_problem(family, benchmark, dim, run_id)
    comp = get_competitor(comp_name)
    result = comp.optimize(objective, dim, lower, upper, max_fes=max_fes, seed=seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    # Curva de convergencia re-muestreada a 100 puntos (comparable entre algoritmos).
    conv = result.convergence_at(100)
    with (out_dir / "convergence.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fe", "best_fitness"])
        w.writerows(conv)

    f_global = info.get("f_global")
    summary = {
        "algorithm": comp_name,
        "family": info.get("family", family),
        "benchmark": benchmark, "dim": dim, "run_id": run_id,
        "best_fitness": result.best_fitness,
        "f_global": f_global,
        "error_to_optimum": (result.best_fitness - f_global) if f_global is not None else None,
        "function_evaluations": result.fes_used,
        "max_fes": max_fes,
        "seed": seed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Competidores con presupuesto comparable al SIA")
    p.add_argument("--competitors", nargs="+", default=COMPETITORS,
                   help=f"Subconjunto de: {COMPETITORS}")
    p.add_argument("--family", default="cec2017", choices=FAMILIES)
    p.add_argument("--benchmarks", nargs="+", default=["F1", "F3", "F5"])
    p.add_argument("--dims", nargs="+", type=int, default=[10, 20, 50])
    p.add_argument("--runs", type=int, default=30)
    # Presupuesto idéntico al SIA: pop_size * max_iter.
    p.add_argument("--pop-size", type=int, default=30)
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--max-fes", type=int, default=None,
                   help="Sobrescribe el presupuesto (por defecto pop_size*max_iter).")
    p.add_argument("--base-seed", type=int, default=0)
    args = p.parse_args()

    max_fes = args.max_fes if args.max_fes is not None else args.pop_size * args.max_iter
    dims = supported_dims(args.family, args.dims)
    skipped = [d for d in args.dims if d not in dims]
    if skipped:
        print(f"[aviso] {args.family} no soporta dim(es) {skipped}; se omiten. Usadas: {dims}")
    if not dims:
        print(f"[error] Ninguna dimensión válida para {args.family}.")
        return

    all_summaries = []
    for comp_name in args.competitors:
        for benchmark in args.benchmarks:
            for dim in dims:
                bests = []
                for run_id in range(1, args.runs + 1):
                    out_dir = (RESULTS_ROOT / comp_name / args.family
                               / benchmark / f"dim_{dim}" / f"run_{run_id:02d}")
                    seed = args.base_seed + dim * 1000 + run_id
                    s = run_one(comp_name, args.family, benchmark, dim, run_id,
                                max_fes, seed, out_dir)
                    bests.append(s["best_fitness"])
                    all_summaries.append(s)
                arr = np.array(bests)
                print(f"[{comp_name} {args.family} {benchmark} d={dim}] "
                      f"best={arr.min():.4e} mean={arr.mean():.4e} "
                      f"std={arr.std():.4e} median={np.median(arr):.4e}")

    agg = RESULTS_ROOT / f"all_summaries_{args.family}.json"
    agg.parent.mkdir(parents=True, exist_ok=True)
    agg.write_text(json.dumps(all_summaries, indent=2))
    print(f"\nResumen agregado: {agg}  (FEs={max_fes})")


if __name__ == "__main__":
    main()
