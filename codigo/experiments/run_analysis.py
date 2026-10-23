"""Orquesta el análisis estadístico y los gráficos de la comparación A vs A'.

A  = SIA-base   (PSO puro, results/SIA_online/SIA-base/...)
A' = SIA-online (pipeline completo, results/SIA_online/SIA-online/...)

Genera en results/analysis/<familia>/:
  - wilcoxon_A_vs_Aprime.json     (test pareado base vs online)
  - friedman.json + post-hoc      (si se incluyen competidores)
  - boxplots por dimensión
  - convergencia A vs A' por (función, dim)
  - superficies 3D del paisaje de cada función

Uso:
    python -m experiments.run_analysis --family cec2017 --benchmarks F1 F5 --dims 10 20 50
    python -m experiments.run_analysis --family cec2017 --benchmarks F1 --dims 10 --with-competitors ASGK EA4eig
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.results_loader import collect, load_convergence, RESULTS
from analysis.statistical_tests import wilcoxon_pair, friedman_test, holm_correction
from analysis import plots, template_plots, summary_table
from analysis.metrics import per_run_gap, aggregate, algorithms_and_dims
from benchmarks.registry import get_problem, supported_dims, FAMILIES

SIA_ROOT = RESULTS / "SIA_online"
COMP_ROOT = RESULTS / "competitors"
ANALYSIS_ROOT = RESULTS / "analysis"


def _algo_sources(family: str, with_competitors: list[str]) -> dict[str, Path]:
    """Mapa {nombre: carpeta_raíz} para el cargador, filtrado por familia."""
    src = {
        "SIA-base": SIA_ROOT / "SIA-base" / family,
        "SIA-online": SIA_ROOT / "SIA-online" / family,
    }
    for comp in with_competitors:
        src[comp] = COMP_ROOT / comp / family
    return src


def main() -> None:
    p = argparse.ArgumentParser(description="Análisis estadístico y gráficos A vs A'")
    p.add_argument("--family", default="cec2017", choices=FAMILIES)
    p.add_argument("--benchmarks", nargs="+", default=["F1", "F3", "F5"])
    p.add_argument("--dims", nargs="+", type=int, default=[10, 20, 50])
    p.add_argument("--metric", default="error_to_optimum",
                   choices=["error_to_optimum", "best_fitness"])
    p.add_argument("--with-competitors", nargs="*", default=[],
                   help="Incluye competidores para el test de Friedman (>=3 algoritmos).")
    p.add_argument("--no-surfaces", action="store_true", help="Omite las superficies 3D.")
    p.add_argument("--no-template", action="store_true",
                   help="Omite las 7 figuras-plantilla.")
    p.add_argument("--no-table", action="store_true",
                   help="Omite la tabla resumen CSV/LaTeX.")
    p.add_argument("--tau", type=float, default=1e-2,
                   help="Umbral de exito (gap<tau) para tasa de exito y celdas sin datos.")
    args = p.parse_args()

    family = args.family
    dims = supported_dims(family, args.dims)
    out_dir = ANALYSIS_ROOT / family
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = _algo_sources(family, args.with_competitors)
    rows = collect(sources, metric=args.metric)
    if not rows:
        print("[error] No se encontraron resultados. ¿Ejecutaste los runners (base y online)?")
        return

    # --- Wilcoxon pareado: A (SIA-base) vs A' (SIA-online). ---------------
    try:
        wil = wilcoxon_pair(rows, "SIA-base", "SIA-online")
        (out_dir / "wilcoxon_A_vs_Aprime.json").write_text(json.dumps(wil, indent=2))
        print(f"[Wilcoxon] base vs online: p={wil['p_value']:.4g} "
              f"sig={wil['significant_0.05']} mejor={wil['better']} "
              f"(wins base={wil['wins_a']}, online={wil['wins_b']}, n={wil['n_problems']})")
    except (KeyError, ValueError) as e:
        print(f"[Wilcoxon] omitido: {e}")

    # --- Friedman (si hay >=3 algoritmos). --------------------------------
    algos_present = {r["algorithm"] for r in rows}
    if len(algos_present) >= 3:
        try:
            fr = friedman_test(rows)
            (out_dir / "friedman.json").write_text(json.dumps(fr, indent=2))
            print(f"[Friedman] p={fr['p_value']:.4g} sig={fr['significant_0.05']} "
                  f"ranking={fr['ranking_best_to_worst']}")
            # Post-hoc: online vs cada otro, con corrección de Holm.
            pairs = []
            for other in sorted(algos_present - {"SIA-online"}):
                try:
                    pairs.append(wilcoxon_pair(rows, "SIA-online", other))
                except (KeyError, ValueError):
                    pass
            pairs = holm_correction(pairs)
            (out_dir / "posthoc_holm.json").write_text(json.dumps(pairs, indent=2))
        except ValueError as e:
            print(f"[Friedman] omitido: {e}")

    # --- Tabla resumen por problema y algoritmo (CSV + LaTeX). ------------
    if not args.no_table:
        _summary_table(rows, family, args, out_dir)

    # --- Boxplots por dimensión: base vs online. --------------------------
    for dim in dims:
        if any(r["dim"] == dim for r in rows):
            plots.boxplot_comparison(
                rows, "SIA-base", "SIA-online", dim,
                out_dir / f"boxplot_dim_{dim}.png")
    print(f"[plots] boxplots -> {out_dir}")

    # --- Convergencia base vs online por (función, dim). ------------------
    for benchmark in args.benchmarks:
        for dim in dims:
            curves_a = _load_curves(SIA_ROOT / "SIA-base" / family / benchmark / f"dim_{dim}")
            curves_b = _load_curves(SIA_ROOT / "SIA-online" / family / benchmark / f"dim_{dim}")
            if curves_a or curves_b:
                plots.convergence_comparison(
                    curves_a, curves_b, "SIA-base", "SIA-online",
                    f"Convergencia {family} {benchmark} dim {dim}",
                    out_dir / f"convergence_{benchmark}_dim_{dim}.png")
    print(f"[plots] convergencia -> {out_dir}")

    # --- Superficies 3D del paisaje (corte 2D) de cada función. -----------
    # Las funciones CEC/BBOB existen en dims fijas (10/20/...), no en 2D. Para
    # visualizar el paisaje hacemos un CORTE 2D: variamos x0,x1 y fijamos el resto
    # de coordenadas en el centro del dominio.
    if not args.no_surfaces:
        surf_dim = dims[0] if dims else 10
        for benchmark in args.benchmarks:
            try:
                objD, lo, hi, info = get_problem(family, benchmark, surf_dim, run_id=1)
                center = 0.5 * (lo + hi)

                def slice_2d(xy, _objD=objD, _D=surf_dim, _c=center):
                    full = np.full(_D, _c)
                    full[0], full[1] = xy[0], xy[1]
                    return _objD(full)

                plots.surface_3d(
                    slice_2d, lo, hi, out_dir / f"surface_{benchmark}.png",
                    title=f"{info.get('family', family)} {benchmark} — corte 2D (dim {surf_dim})")
            except Exception as e:
                print(f"[surface] {benchmark} omitida: {e}")
        print(f"[plots] superficies 3D -> {out_dir}")

    # --- Las 7 figuras-plantilla (estilo del estudio de referencia). ------
    if not args.no_template:
        _template_figures(rows, family, args, dims, out_dir / "template")

    print(f"\nAnálisis completo en: {out_dir}")


def _summary_table(rows, family, args, out_dir, reference="SIA-online"):
    """Genera la tabla resumen (CSV largo + LaTeX ancho) por problema y algoritmo."""
    per_run = summary_table.build_per_run(rows)
    table = summary_table.compute_stats(per_run, tau=args.tau, reference=reference)
    if not table:
        print("[tabla] sin datos; omitida.")
        return
    wtl = summary_table.win_tie_loss(table, reference)
    fr = summary_table.friedman_summary(rows)

    summary_table.export_csv(table, out_dir / "summary_table.csv")
    summary_table.export_latex(
        table, out_dir / "summary_table.tex", reference=reference,
        friedman=fr, wtl=wtl, metric_label=args.metric.replace("_", " "))
    n_prob = len({(r["family"], r["benchmark"], r["dim"]) for r in table})
    print(f"[tabla] resumen ({n_prob} problemas) -> {out_dir}/summary_table.(csv|tex)")


def _template_figures(rows, family, args, dims, tdir):
    """Genera las 7 figuras-plantilla a partir de las filas cargadas."""
    rows_gap = per_run_gap(rows, family)
    agg = aggregate(rows_gap, tau=args.tau)
    algos, dims_present = algorithms_and_dims(agg, algo_order=["SIA-online", "SIA-base"])
    if not algos or not dims_present:
        print("[template] sin datos suficientes; omitido.")
        return

    template_plots.heatmap(agg, algos, dims_present, family, tdir / "1_heatmap.png")
    template_plots.gap_lines(rows_gap, algos, dims_present, family, tdir / "2_gap_lines.png")
    template_plots.bubble(agg, algos, dims_present, family, tdir / "3_bubble.png")
    template_plots.contour(agg, algos, dims_present, family, tdir / "4_contour.png")

    # Convergencia facetada por dimensión: {dim: {algo: [curvas]}}.
    curves_by_dim = {}
    for dim in dims_present:
        per_algo = {}
        for algo in algos:
            root = (SIA_ROOT / algo / family if algo.startswith("SIA")
                    else COMP_ROOT / algo / family)
            cset = []
            for benchmark in args.benchmarks:
                cset += _load_curves(root / benchmark / f"dim_{dim}")
            if cset:
                per_algo[algo] = cset
        if per_algo:
            curves_by_dim[dim] = per_algo
    if curves_by_dim:
        template_plots.convergence_facets(curves_by_dim, algos, family,
                                          tdir / "5_convergence_facets.png")

    template_plots.critical_difference(rows_gap, family, tdir / "6_critical_difference.png")
    template_plots.radar(agg, algos, dims_present, family, tdir / "7_radar.png")
    print(f"[template] 7 figuras -> {tdir}")


def _load_curves(dim_dir: Path) -> list[list[tuple[float, float]]]:
    """Carga las curvas de convergencia de todas las corridas bajo dim_dir."""
    if not dim_dir.exists():
        return []
    curves = []
    for run_dir in sorted(dim_dir.glob("run_*")):
        c = load_convergence(run_dir)
        if c:
            curves.append(c)
    return curves


if __name__ == "__main__":
    main()
