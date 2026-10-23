"""Tabla resumen comparativa por problema y algoritmo (estilo papers CEC).

Por cada (problema = familia+función+dimensión) y cada algoritmo calcula sobre las
R corridas independientes:
    best   = mejor corrida (mínimo error)
    mean   = media
    worst  = peor corrida (máximo error)
    std    = desviación estándar
    median = mediana
    iqr    = rango intercuartílico (Q3 - Q1)
    success_rate = fracción de corridas con error < tau
    rank   = ranking del algoritmo en ese problema (1 = mejor mean)

Tests:
    - Wilcoxon rank-sum por celda vs el algoritmo de REFERENCIA (SIA-online),
      sobre las R corridas: símbolo +/=/− (referencia mejor / igual / peor).
    - Conteo global W/T/L (wins/ties/losses) de la referencia frente a cada rival.
    - Friedman global sobre los means por problema + ranking medio.

Exporta:
    - CSV largo: una fila por (problema, algoritmo) con todas las columnas.
    - LaTeX ancho: problemas en filas, bloques de columnas por algoritmo
      (mean ± std y símbolo Wilcoxon), fila final con ranking medio de Friedman.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from analysis.statistical_tests import friedman_test


def _problem_key(r: dict) -> tuple:
    return (r["family"], r["benchmark"], r["dim"])


def build_per_run(rows: list[dict]) -> dict:
    """{(family,benchmark,dim): {algorithm: np.array de valores por corrida}}."""
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        out[_problem_key(r)][r["algorithm"]].append(float(r["value"]))
    return {p: {a: np.array(v) for a, v in algos.items()} for p, algos in out.items()}


def compute_stats(per_run: dict, tau: float, reference: str) -> list[dict]:
    """Estadísticos por (problema, algoritmo) + símbolo Wilcoxon vs `reference`."""
    table = []
    for prob, algos in sorted(per_run.items()):
        # Ranking por mean (1 = mejor) dentro del problema.
        means = {a: float(np.mean(v)) for a, v in algos.items()}
        order = sorted(means, key=means.get)
        rank = {a: i + 1 for i, a in enumerate(order)}
        ref_vals = algos.get(reference)

        for a, v in sorted(algos.items()):
            q1, q3 = np.percentile(v, [25, 75])
            symbol = ""
            p_val = None
            if ref_vals is not None and a != reference:
                symbol, p_val = _wilcoxon_symbol(ref_vals, v)
            table.append({
                "family": prob[0], "benchmark": prob[1], "dim": prob[2],
                "algorithm": a,
                "best": float(np.min(v)),
                "mean": float(np.mean(v)),
                "worst": float(np.max(v)),
                "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "median": float(np.median(v)),
                "iqr": float(q3 - q1),
                "success_rate": float(np.mean(v < tau)),
                "n_runs": int(len(v)),
                "rank": rank[a],
                "vs_ref": symbol,           # +/=/− respecto a la referencia
                "wilcoxon_p": p_val,
            })
    return table


def _wilcoxon_symbol(ref: np.ndarray, other: np.ndarray, alpha: float = 0.05):
    """Compara `other` vs `ref` (referencia). Menor = mejor.

    Devuelve (símbolo, p) donde el símbolo describe a la REFERENCIA frente al rival:
        '+'  la referencia es significativamente MEJOR (menor error)
        '−'  la referencia es significativamente PEOR
        '='  sin diferencia significativa
    Usa rank-sum (Mann-Whitney) por ser corridas independientes no pareadas.
    """
    if len(ref) < 2 or len(other) < 2 or (np.all(ref == ref[0]) and np.all(other == other[0])):
        return "=", 1.0
    try:
        _, p = stats.mannwhitneyu(ref, other, alternative="two-sided")
    except ValueError:
        return "=", 1.0
    if p >= alpha:
        return "=", float(p)
    return ("+" if np.median(ref) < np.median(other) else "−"), float(p)


def win_tie_loss(table: list[dict], reference: str) -> dict:
    """Conteo W/T/L de la referencia frente a cada rival (sobre los símbolos)."""
    wtl = defaultdict(lambda: {"win": 0, "tie": 0, "loss": 0})
    for row in table:
        a = row["algorithm"]
        if a == reference or not row["vs_ref"]:
            continue
        if row["vs_ref"] == "+":
            wtl[a]["win"] += 1
        elif row["vs_ref"] == "−":
            wtl[a]["loss"] += 1
        else:
            wtl[a]["tie"] += 1
    return dict(wtl)


def friedman_summary(rows: list[dict]) -> dict | None:
    """Friedman sobre los datos (mean por problema lo hace internamente)."""
    algos = {r["algorithm"] for r in rows}
    if len(algos) < 3:
        return None
    try:
        return friedman_test(rows)
    except ValueError:
        return None


# ----------------------------------------------------------- exportadores
def export_csv(table: list[dict], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["family", "benchmark", "dim", "algorithm", "best", "mean", "worst",
            "std", "median", "iqr", "success_rate", "n_runs", "rank",
            "vs_ref", "wilcoxon_p"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in table:
            w.writerow({k: row.get(k) for k in cols})


def _fmt(x: float) -> str:
    """Formato científico compacto para LaTeX."""
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.2e}".replace("e", r"\mathrm{e}")
    return f"{x:.3g}"


def export_latex(table: list[dict], out_path: Path, reference: str,
                 friedman: dict | None, wtl: dict, metric_label: str) -> None:
    """Tabla LaTeX ancha: problemas en filas, algoritmos en bloques (mean±std + símbolo).

    El mejor mean por problema se resalta en negrita.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Organiza por problema y algoritmo.
    by_prob = defaultdict(dict)
    algos = []
    for row in table:
        by_prob[(row["family"], row["benchmark"], row["dim"])][row["algorithm"]] = row
        if row["algorithm"] not in algos:
            algos.append(row["algorithm"])
    # Referencia primero, resto en orden de aparición.
    algos = ([reference] if reference in algos else []) + [a for a in algos if a != reference]
    problems = sorted(by_prob.keys())

    lines = []
    lines.append("% Tabla generada por analysis/summary_table.py")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(rf"\caption{{Comparación por problema ({metric_label}). "
                 rf"Para cada problema, mean$\pm$std sobre las corridas; "
                 rf"\textbf{{negrita}} = mejor. Símbolo vs.\ {reference}: "
                 rf"$+$/$=$/$-$ = {reference} mejor/igual/peor (Wilcoxon, $\alpha=0.05$).}}")
    col_spec = "ll" + "c" * len(algos)
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\hline")
    header = ["Familia", "Problema"] + [_escape(a) for a in algos]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\hline")

    for prob in problems:
        fam, fn, dim = prob
        cells = by_prob[prob]
        means = {a: cells[a]["mean"] for a in algos if a in cells}
        best_algo = min(means, key=means.get) if means else None
        row_cells = [f"{_escape(fam)}", f"{_escape(fn)}-D{dim}"]
        for a in algos:
            if a not in cells:
                row_cells.append("--")
                continue
            c = cells[a]
            inner = rf"{_fmt(c['mean'])}\pm{_fmt(c['std'])}"
            if a == best_algo:
                inner = rf"\mathbf{{{inner}}}"
            sym = c["vs_ref"]
            sup = ""
            if sym in ("+", "−", "="):
                sup = rf"^{{{sym.replace('−', '-')}}}"
            # Todo en UN bloque math: evita el doble $$ y la negrita math se aplica.
            row_cells.append(rf"${inner}{sup}$")
        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\hline")
    # Fila W/T/L de la referencia vs cada rival.
    wtl_row = [r"\multicolumn{2}{l}{W/T/L vs " + _escape(reference) + "}"]
    for a in algos:
        if a == reference:
            wtl_row.append("--")
        elif a in wtl:
            w = wtl[a]
            wtl_row.append(f"{w['win']}/{w['tie']}/{w['loss']}")
        else:
            wtl_row.append("--")
    lines.append(" & ".join(wtl_row) + r" \\")

    # Fila de ranking medio de Friedman.
    if friedman:
        mr = friedman["mean_ranks"]
        rank_row = [r"\multicolumn{2}{l}{Rank medio (Friedman)}"]
        for a in algos:
            rank_row.append(f"{mr.get(a, float('nan')):.2f}" if a in mr else "--")
        lines.append(r"\hline")
        lines.append(" & ".join(rank_row) + r" \\")
        lines.append(r"\multicolumn{" + str(2 + len(algos)) + r"}{l}{"
                     rf"Friedman: $p={friedman['p_value']:.3g}$, "
                     rf"{'significativo' if friedman['significant_0.05'] else 'no significativo'} "
                     rf"($\alpha=0.05$)}} \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")
