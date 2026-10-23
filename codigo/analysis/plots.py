"""Gráficos para la comparación de metaheurísticas (A = base vs A' = mejorada).

Genera:
  - boxplot_comparison: boxplot de la métrica (error al óptimo) por función,
    A vs A', usando las 30 corridas.
  - convergence_comparison: curva de convergencia mediana ± rango intercuartílico
    de A vs A' en una función/dim.
  - surface_3d: superficie 3D del paisaje de una función benchmark (estilo de las
    figuras "Convex Optimization Problem"), evaluando la función en 2D.

Usa el backend Agg (no requiere ventana). Guarda PNG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def boxplot_comparison(rows: list[dict], algo_a: str, algo_b: str,
                       dim: int, out_path: Path, metric_name: str = "Error al óptimo"):
    """Boxplot A vs A' por función, para una dimensión dada.

    rows: filas de results_loader.collect (con 'value' = métrica por corrida).
    """
    funcs = sorted({r["benchmark"] for r in rows if r["dim"] == dim})
    data_a, data_b = [], []
    for fn in funcs:
        data_a.append([r["value"] for r in rows
                       if r["algorithm"] == algo_a and r["benchmark"] == fn and r["dim"] == dim])
        data_b.append([r["value"] for r in rows
                       if r["algorithm"] == algo_b and r["benchmark"] == fn and r["dim"] == dim])

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(funcs)), 5))
    n = len(funcs)
    pos = np.arange(n)
    w = 0.35
    bp_a = ax.boxplot(data_a, positions=pos - w/2, widths=w, patch_artist=True,
                      manage_ticks=False)
    bp_b = ax.boxplot(data_b, positions=pos + w/2, widths=w, patch_artist=True,
                      manage_ticks=False)
    for b in bp_a["boxes"]:
        b.set_facecolor("#4C72B0"); b.set_alpha(0.8)
    for b in bp_b["boxes"]:
        b.set_facecolor("#DD8452"); b.set_alpha(0.8)

    ax.set_xticks(pos)
    ax.set_xticklabels(funcs, rotation=0)
    ax.set_ylabel(metric_name)
    # Log si todos los valores son positivos; symlog si hay ceros/negativos
    # (p.ej. error exactamente 0 al alcanzar el óptimo, o fitness BBOB).
    all_vals = [v for grp in (data_a + data_b) for v in grp]
    if all_vals and min(all_vals) > 0:
        ax.set_yscale("log")
    elif all_vals:
        ax.set_yscale("symlog")
    ax.set_title(f"{algo_a} vs {algo_b} — dim {dim}")
    ax.legend([bp_a["boxes"][0], bp_b["boxes"][0]], [algo_a, algo_b], loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


def convergence_comparison(curves_a: list[list[tuple[float, float]]],
                           curves_b: list[list[tuple[float, float]]],
                           algo_a: str, algo_b: str, title: str, out_path: Path,
                           xlabel: str = "Iteración"):
    """Convergencia mediana ± IQR de A vs A' a partir de varias corridas cada uno.

    curves_*: lista de curvas [(x, best), ...] (una por corrida). Se re-muestrean a
    una rejilla común de x para poder promediar.
    xlabel: 'Iteración' para el SIA (convergence.csv indexa por iteración) o
            'Evaluaciones de función (FEs)' para competidores.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for curves, name, color in [(curves_a, algo_a, "#4C72B0"),
                                (curves_b, algo_b, "#DD8452")]:
        grid, med, q1, q3 = _aggregate_curves(curves)
        if grid is None:
            continue
        ax.plot(grid, med, label=name, color=color, lw=2)
        ax.fill_between(grid, q1, q3, color=color, alpha=0.2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mejor fitness")
    # symlog tolera valores <=0 (fitness BBOB puede ser negativo).
    all_curves = [c for c in (curves_a + curves_b) if c]
    min_v = min((v for c in all_curves for _, v in c), default=1.0)
    ax.set_yscale("log" if min_v > 0 else "symlog")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


def surface_3d(objective, lower: float, upper: float, out_path: Path,
               title: str = "Paisaje de la función objetivo", grid: int = 120):
    """Superficie 3D del paisaje de una función 2D (estilo figuras CEC).

    Evalúa `objective` sobre una rejilla 2D en [lower, upper]^2.
    """
    xs = np.linspace(lower, upper, grid)
    ys = np.linspace(lower, upper, grid)
    X, Y = np.meshgrid(xs, ys)
    Z = np.empty_like(X)
    for i in range(grid):
        for j in range(grid):
            Z[i, j] = objective(np.array([X[i, j], Y[i, j]]))

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Objective Function")
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=12)
    fig.tight_layout()
    _save(fig, out_path)


# ----------------------------------------------------------------- internos
def _aggregate_curves(curves: list[list[tuple[float, float]]]):
    """Re-muestrea una lista de curvas a una rejilla común y devuelve
    (grid, mediana, q1, q3). Mantiene el mejor-hasta-ahora (monótono no creciente)."""
    curves = [c for c in curves if c]
    if not curves:
        return None, None, None, None
    x_max = max(c[-1][0] for c in curves)
    grid = np.linspace(1, x_max, 100)
    resampled = []
    for c in curves:
        cx = np.array([p[0] for p in c])
        cy = np.array([p[1] for p in c])
        # best-so-far en cada punto de la rejilla (búsqueda por la izquierda).
        idx = np.searchsorted(cx, grid, side="right") - 1
        idx = np.clip(idx, 0, len(cy) - 1)
        resampled.append(cy[idx])
    R = np.vstack(resampled)
    return grid, np.median(R, axis=0), np.percentile(R, 25, axis=0), np.percentile(R, 75, axis=0)


def _save(fig, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
