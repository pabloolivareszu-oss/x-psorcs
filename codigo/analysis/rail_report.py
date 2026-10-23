"""Reporte del caso ferroviario (Sec. 8 de la guía CPB).

Con 3 instancias no hay base para un Friedman sobre instancias: el análisis es
POR INSTANCIA sobre las semillas pareadas (run_id):
  - RPD = 100 (f - BKS) / BKS: media, mediana, desviación estándar y mejor.
  - Wilcoxon pareado SIA-base vs SIA-online, y SIA-online vs cada competidor,
    con control de Finner por instancia sobre la familia de comparaciones.
  - Â de Vargha–Delaney: P(A' mejor que X) + 0.5 P(empate) (minimización).

Salida en results/analysis/rail/: rail_summary.csv, rail_tests.csv,
rail_table.tex (formato de la plantilla de la guía) y rail_report.json.

Uso (desde RecursosPaper/codigo/):
    python -m analysis.rail_report                 # las 3 instancias
    python -m analysis.rail_report rail516         # una sola
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.results_loader import RESULTS, load_summaries  # noqa: E402
from benchmarks.rail import INSTANCES, _load_bks  # noqa: E402

OUT = RESULTS / "analysis" / "rail"
REF = "SIA-online"
EXPECTED_RUNS = 31
EXPECTED_FES = 20000
T10_V2ELITR = {"rail507": 7.100, "rail516": 1.648, "rail582": 7.491}  # guía, Tabla T10


def _collect() -> dict:
    """{algo: {instancia: {run_id: summary}}} para la familia rail."""
    data: dict = defaultdict(lambda: defaultdict(dict))
    roots = [(p.name, p / "rail") for p in (RESULTS / "SIA_online").glob("*")]
    roots += [(p.name, p / "rail") for p in (RESULTS / "competitors").glob("*")]
    for algo, root in roots:
        if root.exists():
            for s in load_summaries(root):
                data[algo][s["benchmark"]][int(s["run_id"])] = s
    return data


def finner(pvals: list[float]) -> list[float]:
    """Ajuste de Finner (step-down): p_adj(i) = 1 - (1 - p_(i))^(k/i), monótono."""
    k = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(k)
    running = 0.0
    for rank, idx in enumerate(order, start=1):
        running = max(running, 1 - (1 - pvals[idx]) ** (k / rank))
        adj[idx] = min(1.0, running)
    return adj.tolist()


def vargha_delaney(a: np.ndarray, b: np.ndarray) -> float:
    """Â = P(a < b) + 0.5 P(a == b): probabilidad de que `a` sea mejor (minimización)."""
    gt = (a[:, None] < b[None, :]).mean()
    eq = (a[:, None] == b[None, :]).mean()
    return float(gt + 0.5 * eq)


def main(instances: list[str] | None = None) -> dict:
    instances = list(instances or INSTANCES)
    bks = _load_bks()
    data = _collect()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"warnings": [], "instances": {}}

    unverified = [i for i in instances if not bks[i].get("verified")]
    if unverified:
        report["warnings"].append(f"BKS sin verificar para {unverified}: RPD provisional.")

    summary_rows, test_rows = [], []
    for inst in instances:
        f_bks = float(bks[inst]["bks"])
        per_algo = {a: runs[inst] for a, runs in data.items() if runs.get(inst)}
        if not per_algo:
            report["warnings"].append(f"{inst}: sin resultados.")
            continue
        rpd = {}
        for algo, runs in per_algo.items():
            n_runs = len(runs)
            fes = {s.get("function_evaluations") for s in runs.values()}
            if n_runs != EXPECTED_RUNS:
                report["warnings"].append(f"{inst}/{algo}: {n_runs} corridas (se esperaban {EXPECTED_RUNS}).")
            if fes != {EXPECTED_FES}:
                report["warnings"].append(f"{inst}/{algo}: FEs {sorted(fes)} (se esperaba {EXPECTED_FES}).")
            vals = {r: 100 * (float(s["best_fitness"]) - f_bks) / f_bks for r, s in runs.items()}
            if min(vals.values()) < -1e-9:
                report["warnings"].append(f"{inst}/{algo}: RPD negativo -> el BKS está mal. DETENERSE.")
            rpd[algo] = vals
            v = np.array(list(vals.values()))
            summary_rows.append({"instance": inst, "algorithm": algo, "runs": n_runs,
                                 "rpd_mean": v.mean(), "rpd_median": float(np.median(v)),
                                 "rpd_std": v.std(ddof=1) if len(v) > 1 else 0.0,
                                 "rpd_best": v.min(), "rpd_worst": v.max(),
                                 "time_mean_s": float(np.mean([s.get("elapsed_seconds") or np.nan
                                                               for s in runs.values()]))})

        # Comparaciones pareadas por run_id.
        pairs = []
        if "SIA-base" in rpd and REF in rpd:
            pairs.append(("SIA-base", REF))
        pairs += [(REF, other) for other in sorted(rpd) if other not in ("SIA-base", REF) and REF in rpd]
        tests = []
        for a_name, b_name in pairs:
            common = sorted(set(rpd[a_name]) & set(rpd[b_name]))
            a = np.array([rpd[a_name][r] for r in common])
            b = np.array([rpd[b_name][r] for r in common])
            d = a - b
            p = 1.0 if np.all(d == 0) else float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
            ref_vals, other_vals = (b, a) if b_name == REF else (a, b)
            tests.append({"instance": inst, "a": a_name, "b": b_name, "n_pairs": len(common),
                          "better_ref": int((ref_vals < other_vals).sum()),
                          "equal": int((ref_vals == other_vals).sum()),
                          "worse_ref": int((ref_vals > other_vals).sum()),
                          "p": p, "A_ref": vargha_delaney(ref_vals, other_vals)})
        for t, pa in zip(tests, finner([t["p"] for t in tests])):
            t["p_finner"] = pa
        test_rows += tests
        report["instances"][inst] = {
            "bks": f_bks, "bks_verified": bool(bks[inst].get("verified")),
            "t10_v2elitr_rpd": T10_V2ELITR.get(inst),
            "best_worst": {a: [min(v.values()), max(v.values())] for a, v in rpd.items()},
        }

    _write_csv(OUT / "rail_summary.csv", summary_rows)
    _write_csv(OUT / "rail_tests.csv", test_rows)
    _write_latex(OUT / "rail_table.tex", summary_rows, test_rows)
    (OUT / "rail_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for w in report["warnings"]:
        print("[aviso]", w)
    print(f"[rail] reporte en {OUT}")
    return report


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _write_latex(path: Path, summary: list[dict], tests: list[dict]) -> None:
    """Tabla en el formato de la plantilla de la guía (tab:plantilla)."""
    a_vs = {(t["instance"], t["b"] if t["a"] == REF else t["a"]): t["A_ref"] for t in tests}
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Instancia & Método & RPD medio & Mediana & Desv. est. & Mejor & $\hat{A}$ de SIA-online \\",
             r"\midrule"]
    by_inst = defaultdict(list)
    for r in summary:
        by_inst[r["instance"]].append(r)
    for inst, rows in by_inst.items():
        for r in sorted(rows, key=lambda r: r["rpd_mean"]):
            a = a_vs.get((inst, r["algorithm"]))
            a_txt = "--" if a is None else f"{a:.2f}"
            lines.append(f"{inst} & {r['algorithm']} & {r['rpd_mean']:.3f} & {r['rpd_median']:.3f} & "
                         f"{r['rpd_std']:.3f} & {r['rpd_best']:.3f} & {a_txt} \\\\")
        if inst in T10_V2ELITR:
            lines.append(f"{inst} & V2-ELITR (T10, CPB) & {T10_V2ELITR[inst]:.3f} & & & & -- \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
