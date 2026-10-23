"""Métricas derivadas para las figuras-plantilla (estilo del estudio GA/PSO/...).

Mapeo acordado (memoria tesis-figuras-estilo):
  - "scale" (eje X) -> dimensión del problema, facetando por familia.
  - "optimality gap":
      classical/CEC (con óptimo): error_to_opt = best - f_opt  (ya viene en summary).
      BBOB (sin óptimo útil): gap relativo al mejor empírico entre métodos.
  - "feasibility" -> tasa de éxito = fracción de corridas con error < tau.
  - celdas "x" -> combinaciones sin datos o sin ninguna corrida bajo tau.

Estas funciones consumen las filas de results_loader.collect (cada fila tiene
algorithm, family, benchmark, dim, run_id, value) y producen tablas agregadas
listas para graficar.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def per_run_gap(rows: list[dict], family: str) -> list[dict]:
    """Devuelve filas con 'gap' por corrida.

    Para classical/CEC el 'value' ya es error_to_opt (si se cargó con esa métrica).
    Para BBOB se calcula el gap relativo al mejor empírico (best_global) por
    (benchmark, dim) sobre TODOS los métodos.
    """
    is_bbob = family.lower().startswith("bbob")
    if not is_bbob:
        return [dict(r, gap=float(r["value"])) for r in rows]

    # BBOB: best_global por (benchmark, dim) entre todos los métodos/corridas.
    best_global = defaultdict(lambda: np.inf)
    for r in rows:
        key = (r["benchmark"], r["dim"])
        best_global[key] = min(best_global[key], float(r["value"]))
    out = []
    for r in rows:
        bg = best_global[(r["benchmark"], r["dim"])]
        gap = (float(r["value"]) - bg) / (abs(bg) + 1e-12)
        out.append(dict(r, gap=gap))
    return out


def aggregate(rows_with_gap: list[dict], tau: float = 1e-2) -> dict:
    """Agrega por (algorithm, dim): mediana del gap, tasa de éxito, nº corridas.

    tau: umbral de "éxito" (sustituto de factibilidad). Una corrida es exitosa si
    su gap < tau. La tasa de éxito agrega sobre función×corrida de esa dimensión.

    Devuelve {(algorithm, dim): {median_gap, success_rate, n_runs}}.
    """
    buckets = defaultdict(list)
    for r in rows_with_gap:
        buckets[(r["algorithm"], r["dim"])].append(r["gap"])
    out = {}
    for (algo, dim), gaps in buckets.items():
        g = np.array(gaps, dtype=float)
        out[(algo, dim)] = {
            "median_gap": float(np.median(g)),
            "mean_gap": float(np.mean(g)),
            "success_rate": float(np.mean(g < tau)),
            "n_runs": int(len(g)),
        }
    return out


def algorithms_and_dims(agg: dict, algo_order: list[str] | None = None):
    """Extrae listas ordenadas de algoritmos y dimensiones presentes en `agg`."""
    algos = sorted({a for a, _ in agg})
    if algo_order:
        algos = [a for a in algo_order if a in algos] + [a for a in algos if a not in algo_order]
    dims = sorted({d for _, d in agg})
    return algos, dims


def gap_matrix(agg: dict, algos: list[str], dims: list[int], field: str = "median_gap"):
    """Matriz (algos x dims) del campo pedido; NaN donde no hay datos (celda ✗)."""
    M = np.full((len(algos), len(dims)), np.nan)
    for i, a in enumerate(algos):
        for j, d in enumerate(dims):
            if (a, d) in agg:
                M[i, j] = agg[(a, d)][field]
    return M
