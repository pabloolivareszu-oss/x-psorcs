"""Las 7 figuras-plantilla del estudio de referencia, adaptadas a los datos del SIA.

Estilos reproducidos (eje X = dimensión, una FAMILIA por figura):
  1. heatmap          gap mediano por método x dimensión, con celdas ✗ sin datos.
  2. gap_lines        líneas gap-vs-dimensión con banda IQR.
  3. bubble           tamaño=tasa de éxito, color=gap mediano.
  4. contour          gap interpolado sobre método x dimensión (ilustrativo).
  5. convergence_facets  convergencia log con banda IQR, una faceta por dimensión.
  6. critical_difference Nemenyi sobre el ranking medio del gap por problema.
  7. radar            calidad x éxito (1=mejor) por dimensión.

Convenciones acordadas (memoria tesis-figuras-estilo): "feasibility" -> tasa de
éxito (gap<tau); celdas ✗ -> sin datos; contour anotado "ilustrativo".

Todas guardan PNG (backend Agg). Reciben tablas de analysis.metrics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analysis.metrics import gap_matrix

_GREEN_RED = "RdYlGn_r"   # verde=bajo gap (mejor), rojo=alto
_VIRIDIS = "viridis"


def _save(fig, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _dim_labels(dims: list[int]) -> list[str]:
    return [f"dim {d}" for d in dims]


# --------------------------------------------------------------- 1. HEATMAP
def heatmap(agg: dict, algos: list[str], dims: list[int], family: str,
            out_path: Path, gap_label: str = "error al óptimo"):
    """Heatmap gap mediano (método x dimensión). Celdas sin datos -> ✗.

    El color usa escala LOG porque el error CEC abarca muchos órdenes de magnitud
    (si no, dim baja se aplasta y no se distinguen métodos).
    """
    import matplotlib.colors as mcolors
    M = gap_matrix(agg, algos, dims, "median_gap")

    # Norma log si todos los valores positivos; si hay <=0, lineal.
    finite = M[np.isfinite(M)]
    use_log = finite.size > 0 and np.nanmin(finite) > 0
    norm = (mcolors.LogNorm(vmin=np.nanmin(finite), vmax=np.nanmax(finite))
            if use_log else None)

    fig, ax = plt.subplots(figsize=(1.4 * len(dims) + 3, 0.7 * len(algos) + 2))
    im = ax.imshow(M, aspect="auto", cmap=_VIRIDIS, norm=norm)
    ax.set_xticks(range(len(dims)), _dim_labels(dims))
    ax.set_yticks(range(len(algos)), algos)
    ax.set_xlabel("escala (dimensión)")
    ax.set_title(f"Gap por método y escala — {family}  (✗ = sin corrida)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{gap_label} (mediano, escala log)" if use_log
                   else f"{gap_label} (mediano)")

    thr = np.nanmedian(finite) if finite.size else 0
    for i in range(len(algos)):
        for j in range(len(dims)):
            if np.isnan(M[i, j]):
                ax.text(j, i, "✗", ha="center", va="center", color="gray", fontsize=12)
            else:
                val = M[i, j]
                txt = f"{val:.3g}"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if val < thr else "black", fontsize=8)
    _save(fig, out_path)


# ----------------------------------------------------------- 2. GAP LINES
def gap_lines(rows_with_gap: list[dict], algos: list[str], dims: list[int],
              family: str, out_path: Path, gap_label: str = "error al óptimo"):
    """Líneas gap mediano vs dimensión, con banda intercuartílica por método.

    Eje Y en escala log (el error CEC abarca muchos órdenes de magnitud)."""
    from collections import defaultdict
    by = defaultdict(lambda: defaultdict(list))
    for r in rows_with_gap:
        by[r["algorithm"]][r["dim"]].append(r["gap"])

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    all_positive = True
    for k, a in enumerate(algos):
        xs, med, q1, q3 = [], [], [], []
        for d in dims:
            vals = by[a].get(d)
            if not vals:
                continue
            v = np.array(vals)
            xs.append(d); med.append(np.median(v))
            q1.append(np.percentile(v, 25)); q3.append(np.percentile(v, 75))
            if np.min(v) <= 0:
                all_positive = False
        if not xs:
            continue
        c = cmap(k % 10)
        ax.plot(xs, med, "-o", color=c, label=a, lw=2)
        ax.fill_between(xs, q1, q3, color=c, alpha=0.18)

    ax.set_xlabel("escala (dimensión)")
    ax.set_ylabel(gap_label)
    if all_positive:
        ax.set_yscale("log")
    ax.set_xticks(dims)
    ax.set_title(f"Gap vs escala — {family}")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    _save(fig, out_path)


# -------------------------------------------------------------- 3. BUBBLE
def bubble(agg: dict, algos: list[str], dims: list[int], family: str,
           out_path: Path, gap_label: str = "error al óptimo"):
    """Bubble: tamaño = tasa de éxito, color = gap mediano (log). ✗ sin datos."""
    import matplotlib.colors as mcolors
    fig, ax = plt.subplots(figsize=(1.3 * len(dims) + 3, 0.7 * len(algos) + 2))

    gaps = [agg[(a, d)]["median_gap"] for a in algos for d in dims if (a, d) in agg]
    pos = [g for g in gaps if g > 0]
    use_log = len(pos) == len(gaps) and len(gaps) > 0
    cmap = plt.get_cmap(_GREEN_RED)
    if use_log:
        norm = mcolors.LogNorm(vmin=min(pos), vmax=max(pos))
    else:
        norm = plt.Normalize(min(gaps) if gaps else 0, max(gaps) if gaps else 1)

    for i, a in enumerate(algos):
        for j, d in enumerate(dims):
            if (a, d) not in agg:
                ax.text(j, i, "✗", ha="center", va="center", color="gray", fontsize=12)
                continue
            cell = agg[(a, d)]
            size = 200 + 1800 * cell["success_rate"]  # tasa de éxito -> área
            gval = cell["median_gap"]
            ax.scatter(j, i, s=size, color=cmap(norm(gval)),
                       edgecolors="black", linewidths=0.6, zorder=3)
            ax.text(j, i, f"{gval:.2g}", ha="center", va="center", fontsize=7, zorder=4)

    ax.set_xticks(range(len(dims)), _dim_labels(dims))
    ax.set_yticks(range(len(algos)), algos)
    ax.set_xlabel("escala (dimensión)")
    ax.set_ylim(-0.5, len(algos) - 0.5)
    ax.invert_yaxis()
    ax.set_title(f"Rendimiento por escala — {family}\n"
                 f"(tamaño = tasa de éxito, color = {gap_label} mediano)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label=f"{gap_label} mediano" + (" (log)" if use_log else ""))
    _save(fig, out_path)


# ------------------------------------------------------------- 4. CONTOUR
def contour(agg: dict, algos: list[str], dims: list[int], family: str,
            out_path: Path, gap_label: str = "error al óptimo"):
    """Contour interpolado del gap sobre método x dimensión (ILUSTRATIVO).

    Trabaja en log10 del gap (el error CEC abarca muchos órdenes de magnitud).
    Marca con ✗ las celdas sin datos; el relleno se interpola sobre las que sí hay.

    Requiere al menos 2 dimensiones y 2 algoritmos (una rejilla 2x2); si no, se
    omite con aviso (el contorno no tiene sentido con una sola fila/columna).
    """
    if len(dims) < 2 or len(algos) < 2:
        print(f"[contour] omitido: requiere >=2 dims y >=2 algoritmos "
              f"(hay {len(dims)} dims, {len(algos)} algoritmos).")
        return
    M = gap_matrix(agg, algos, dims, "median_gap")
    finite = M[np.isfinite(M)]
    use_log = finite.size > 0 and np.nanmin(finite) > 0
    disp = np.log10(M) if use_log else M

    # Relleno de NaN por media de columna, solo para el contorno; ✗ encima.
    filled = disp.copy()
    col_mean = np.nanmean(disp, axis=0)
    global_mean = np.nanmean(disp)
    for j in range(filled.shape[1]):
        nanrows = np.isnan(filled[:, j])
        filled[nanrows, j] = col_mean[j] if not np.isnan(col_mean[j]) else global_mean

    fig, ax = plt.subplots(figsize=(1.4 * len(dims) + 3, 0.7 * len(algos) + 2))
    X, Y = np.meshgrid(range(len(dims)), range(len(algos)))
    cs = ax.contourf(X, Y, filled, levels=12, cmap=_VIRIDIS)
    ax.set_xticks(range(len(dims)), _dim_labels(dims))
    ax.set_yticks(range(len(algos)), algos)
    ax.set_xlabel("escala (dimensión)")
    for i in range(len(algos)):
        for j in range(len(dims)):
            if np.isnan(M[i, j]):
                ax.text(j, i, "✗", ha="center", va="center", color="white", fontsize=11)
    ax.set_title(f"Contorno de gap sobre escala × método — {family}\n(ilustrativo, interpolado)")
    lbl = f"log10({gap_label} mediano)" if use_log else f"{gap_label} mediano"
    fig.colorbar(cs, ax=ax, label=lbl)
    _save(fig, out_path)


# ------------------------------------------ 5. CONVERGENCE FACETS (por dim)
def convergence_facets(curves_by_dim: dict, algos: list[str], family: str,
                       out_path: Path):
    """Facetas de convergencia (una por dimensión); en cada una, una línea
    mediana ± IQR por algoritmo.

    curves_by_dim: {dim: {algo: [curva_por_corrida, ...]}}.
    """
    dims = sorted(curves_by_dim.keys())
    if not dims:
        return
    fig, axes = plt.subplots(1, len(dims), figsize=(4.2 * len(dims), 4), squeeze=False)
    cmap = plt.get_cmap("tab10")
    for ax, d in zip(axes[0], dims):
        any_curve = False
        for k, a in enumerate(algos):
            curves = curves_by_dim[d].get(a, [])
            grid, med, q1, q3 = _aggregate_curves(curves)
            if grid is None:
                continue
            any_curve = True
            c = cmap(k % 10)
            ax.plot(grid, med, color=c, lw=1.8, label=a)
            ax.fill_between(grid, q1, q3, color=c, alpha=0.15)
        ax.set_title(f"dim {d}")
        ax.set_xlabel("iteración")
        if any_curve:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel("mejor objetivo (log)")
    axes[0][-1].legend(fontsize=8, ncol=2)
    fig.suptitle(f"Convergencia con banda IQR — {family}")
    _save(fig, out_path)


# -------------------------------------------- 6. CRITICAL DIFFERENCE (Nemenyi)
def critical_difference(rows_with_gap: list[dict], family: str, out_path: Path,
                        alpha: float = 0.05):
    """Diagrama de diferencia crítica (Nemenyi) sobre el ranking medio del gap.

    Empareja por (benchmark, dim) usando la mediana del gap; rankea los métodos
    en cada problema (1=mejor) y dibuja el ranking medio con la barra CD.
    """
    from collections import defaultdict
    algos = sorted({r["algorithm"] for r in rows_with_gap})
    by_prob = defaultdict(dict)  # (benchmark,dim) -> {algo: mediana_gap}
    tmp = defaultdict(list)
    for r in rows_with_gap:
        tmp[(r["algorithm"], r["benchmark"], r["dim"])].append(r["gap"])
    for (a, b, d), v in tmp.items():
        by_prob[(b, d)][a] = float(np.median(v))

    # Solo problemas con TODOS los algoritmos.
    problems = [k for k, dd in by_prob.items() if all(a in dd for a in algos)]
    k = len(algos)
    N = len(problems)
    if k < 2 or N < 2:
        return  # insuficiente para CD

    ranks = np.zeros((N, k))
    for i, prob in enumerate(problems):
        vals = np.array([by_prob[prob][a] for a in algos])
        ranks[i] = stats.rankdata(vals)  # menor gap -> rango 1
    mean_ranks = ranks.mean(axis=0)

    # Distancia crítica de Nemenyi: CD = q_alpha * sqrt(k(k+1)/(6N)).
    q_alpha = _nemenyi_q(alpha, k)
    CD = q_alpha * np.sqrt(k * (k + 1) / (6.0 * N))

    order = np.argsort(mean_ranks)
    algos_sorted = [algos[i] for i in order]
    ranks_sorted = mean_ranks[order]

    fig, ax = plt.subplots(figsize=(9, 2.6 + 0.3 * k))
    lo, hi = 1, k
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(0, 1.15)
    ax.axis("off")
    # eje superior
    ax.plot([lo, hi], [0.82, 0.82], "k-", lw=1)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [0.82, 0.84], "k-", lw=1)
        ax.text(r, 0.87, str(r), ha="center", fontsize=9)
    # barra CD
    ax.plot([lo, lo + CD], [0.94, 0.94], "k-", lw=3)
    ax.text(lo + CD / 2, 0.97, f"CD = {CD:.2f}", ha="center", fontsize=9)

    # marcadores: mitad a la izquierda, mitad a la derecha
    n = len(algos_sorted)
    for idx, (a, r) in enumerate(zip(algos_sorted, ranks_sorted)):
        left = idx < n / 2
        y = 0.65 - 0.08 * (idx if left else (n - 1 - idx))
        ax.plot([r, r], [0.85, y], "k-", lw=0.8)
        xend = lo - 0.4 if left else hi + 0.4
        ax.plot([r, xend], [y, y], "k-", lw=0.8)
        ax.text(xend + (-0.05 if left else 0.05), y, f"{a} ({r:.2f})",
                ha="right" if left else "left", va="center", fontsize=9)

    ax.set_title(f"Diagrama de diferencia crítica (Nemenyi, α={alpha}) — {family}\n"
                 f"menor rango = mejor   (n={N} problemas)", fontsize=10)
    _save(fig, out_path)


# --------------------------------------------------------------- 7. RADAR
def radar(agg: dict, algos: list[str], dims: list[int], family: str,
          out_path: Path, max_algos: int = 5):
    """Radar de rendimiento (1=mejor) por dimensión.

    Score combinado por (algo, dim): calidad (1 - gap normalizado) * éxito.
    """
    # Normaliza gap por dimensión al rango [0,1] (peor=1).
    scores = {}
    for d in dims:
        vals = [agg[(a, d)]["median_gap"] for a in algos if (a, d) in agg]
        gmax = max(vals) if vals else 1.0
        gmin = min(vals) if vals else 0.0
        rng = (gmax - gmin) or 1.0
        for a in algos:
            if (a, d) in agg:
                cell = agg[(a, d)]
                quality = 1.0 - (cell["median_gap"] - gmin) / rng
                scores[(a, d)] = max(0.0, quality) * (0.5 + 0.5 * cell["success_rate"])

    # Limita a los mejores `max_algos` por score medio para legibilidad.
    mean_score = {a: np.mean([scores.get((a, d), 0) for d in dims]) for a in algos}
    top = sorted(algos, key=lambda a: mean_score[a], reverse=True)[:max_algos]

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    cmap = plt.get_cmap("tab10")
    for k, a in enumerate(top):
        vals = [scores.get((a, d), 0.0) for d in dims]
        vals += vals[:1]
        c = cmap(k % 10)
        ax.plot(angles, vals, "-o", color=c, lw=2, label=a)
        ax.fill(angles, vals, color=c, alpha=0.12)
    ax.set_xticks(angles[:-1], _dim_labels(dims))
    ax.set_ylim(0, 1)
    ax.set_title(f"Rendimiento por escala (calidad × éxito, 1=mejor) — {family}",
                 pad=28)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), fontsize=8)
    _save(fig, out_path)


# ----------------------------------------------------------------- helpers
def _aggregate_curves(curves):
    curves = [c for c in curves if c]
    if not curves:
        return None, None, None, None
    x_max = max(c[-1][0] for c in curves)
    grid = np.linspace(1, x_max, 100)
    R = []
    for c in curves:
        cx = np.array([p[0] for p in c]); cy = np.array([p[1] for p in c])
        idx = np.clip(np.searchsorted(cx, grid, side="right") - 1, 0, len(cy) - 1)
        R.append(cy[idx])
    R = np.vstack(R)
    return grid, np.median(R, 0), np.percentile(R, 25, 0), np.percentile(R, 75, 0)


# Valores críticos q_alpha del test de Nemenyi (alpha=0.05) para k algoritmos.
_Q005 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
         8: 3.031, 9: 3.102, 10: 3.164}
_Q010 = {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.460, 6: 2.589, 7: 2.693,
         8: 2.780, 9: 2.855, 10: 2.920}


def _nemenyi_q(alpha: float, k: int) -> float:
    table = _Q005 if abs(alpha - 0.05) < 1e-9 else _Q010
    return table.get(k, table[max(table)])
