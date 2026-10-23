"""Pruebas estadísticas no paramétricas para comparar metaheurísticas.

Convención estándar en optimización (Derrac et al., 2011):
  - Wilcoxon signed-rank: comparación PAREADA de DOS algoritmos (p.ej. A vs A').
    Se emparejan por (función, dim) usando la mediana de las corridas.
  - Friedman: comparación de >=3 algoritmos sobre múltiples problemas; produce un
    ranking medio por algoritmo. Si rechaza H0, se hace post-hoc (Wilcoxon por pares).

Métrica menor = mejor (error al óptimo). Todas las funciones devuelven dicts
serializables para guardar en JSON/CSV.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import stats


def _aggregate_by_problem(rows: list[dict], agg=np.median) -> dict:
    """{(algorithm): {(family,benchmark,dim): valor_agregado_sobre_corridas}}."""
    buckets = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["family"], r["benchmark"], r["dim"])
        v = r["value"]
        if v is not None and np.isfinite(v):  # descarta NaN/inf de funciones rotas
            buckets[r["algorithm"]][key].append(v)
    out = {}
    for algo, probs in buckets.items():
        out[algo] = {k: float(agg(v)) for k, v in probs.items() if len(v) > 0}
    return out


def wilcoxon_pair(rows: list[dict], algo_a: str, algo_b: str,
                  agg=np.median, scale: str = "relative") -> dict:
    """Wilcoxon signed-rank pareado entre dos algoritmos sobre problemas comunes.

    H0: las medianas de las diferencias son 0 (no hay diferencia).
    Empareja por (familia, función, dim). Reporta también cuántas veces gana cada uno.

    scale='relative' (por defecto) aplica el test sobre la diferencia RELATIVA
    (b - a) / max(|a|, |b|) en lugar de la diferencia cruda. Es necesario cuando las
    celdas de una familia viven en escalas distintas (CEC 2022 va de 1e-2 a 1e6;
    microhubs de 1e3 a 1e5): con diferencias crudas los rangos del signed-rank los
    fijan solo las celdas de escala grande y el test deja de medir consistencia.
    scale='raw' conserva el comportamiento anterior.
    """
    agg_data = _aggregate_by_problem(rows, agg)
    if algo_a not in agg_data or algo_b not in agg_data:
        raise KeyError(f"Faltan datos de {algo_a} o {algo_b}.")
    common = sorted(set(agg_data[algo_a]) & set(agg_data[algo_b]))
    if not common:
        raise ValueError("No hay problemas en común entre los dos algoritmos.")

    a = np.array([agg_data[algo_a][k] for k in common])
    b = np.array([agg_data[algo_b][k] for k in common])
    diff = a - b
    wins_a = int(np.sum(a < b))   # a mejor (menor error)
    wins_b = int(np.sum(b < a))
    ties = int(np.sum(a == b))

    # Diferencia relativa pareada: adimensional y comparable entre celdas.
    denom = np.maximum(np.abs(a), np.abs(b))
    denom[denom == 0] = 1.0
    rel = (b - a) / denom

    def _test(d: np.ndarray) -> tuple[float, float]:
        # Wilcoxon necesita al menos una diferencia no nula.
        if np.all(d == 0):
            return float("nan"), 1.0
        s, pv = stats.wilcoxon(d, zero_method="wilcox", correction=False)
        return float(s), float(pv)

    stat_raw, p_raw = _test(diff)
    stat_rel, p_rel = _test(rel)
    stat, p = (stat_rel, p_rel) if scale == "relative" else (stat_raw, p_raw)

    median_rel = float(np.median(rel))
    # 'better' se decide por el signo del efecto pareado, no por conteo de victorias:
    # contar celdas ignora la magnitud y puede contradecir al propio p-value.
    if np.isnan(stat) or p >= 0.05:
        better = algo_a if median_rel > 0 else (algo_b if median_rel < 0 else "tie")
    else:
        better = algo_b if median_rel < 0 else algo_a

    return {
        "test": "wilcoxon_signed_rank",
        "algo_a": algo_a, "algo_b": algo_b,
        "n_problems": len(common),
        "statistic": stat, "p_value": p,
        "significant_0.05": bool(p < 0.05),
        "scale_used": scale,
        "p_value_raw": p_raw, "p_value_relative": p_rel,
        "median_rel_diff": median_rel,
        "wins_a": wins_a, "wins_b": wins_b, "ties": ties,
        "median_a": float(np.median(a)), "median_b": float(np.median(b)),
        "better": better,
    }


def friedman_test(rows: list[dict], agg=np.median) -> dict:
    """Friedman sobre todos los algoritmos y problemas comunes.

    H0: todos los algoritmos tienen el mismo rendimiento (mismo ranking medio).
    Devuelve estadístico, p-valor y ranking medio por algoritmo (menor = mejor).
    """
    agg_data = _aggregate_by_problem(rows, agg)
    algos = sorted(agg_data.keys())
    if len(algos) < 3:
        raise ValueError("Friedman requiere >=3 algoritmos. Para 2 usa Wilcoxon.")

    # Problemas comunes a TODOS los algoritmos.
    common = set(agg_data[algos[0]])
    for a in algos[1:]:
        common &= set(agg_data[a])
    common = sorted(common)
    if len(common) < 2:
        raise ValueError("Se necesitan >=2 problemas comunes a todos los algoritmos.")

    # Matriz problemas x algoritmos.
    M = np.array([[agg_data[a][k] for a in algos] for k in common])
    stat, p = stats.friedmanchisquare(*[M[:, j] for j in range(len(algos))])

    # Ranking por problema (1 = mejor = menor valor), luego ranking medio.
    ranks = np.array([stats.rankdata(row) for row in M])  # menor valor -> rango 1
    mean_ranks = ranks.mean(axis=0)
    ranking = sorted(zip(algos, mean_ranks), key=lambda t: t[1])

    return {
        "test": "friedman",
        "n_algorithms": len(algos),
        "n_problems": len(common),
        "statistic": float(stat), "p_value": float(p),
        "significant_0.05": bool(p < 0.05),
        "mean_ranks": {a: float(r) for a, r in zip(algos, mean_ranks)},
        "ranking_best_to_worst": [a for a, _ in ranking],
    }


def holm_correction(pairs: list[dict]) -> list[dict]:
    """Corrección de Holm-Bonferroni para múltiples comparaciones por pares.

    Recibe una lista de resultados de wilcoxon_pair y añade 'p_holm' y
    'significant_holm'. Ordena por p ascendente y ajusta.
    """
    ordered = sorted(pairs, key=lambda d: d["p_value"])
    m = len(ordered)
    prev = 0.0
    for i, res in enumerate(ordered):
        adj = min(1.0, (m - i) * res["p_value"])
        adj = max(adj, prev)  # monotonía de Holm
        prev = adj
        res["p_holm"] = adj
        res["significant_holm_0.05"] = bool(adj < 0.05)
    return ordered
