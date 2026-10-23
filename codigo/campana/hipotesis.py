"""Evalúa las tres hipótesis de trabajo de la tesis sobre los resultados de la v3.

  H1  El estado de la búsqueda contiene señal predictiva sobre la mejora futura.
      Por corrida de SIA-online: AUC y Brier del modelo online frente a la etiqueta
      observada (¿mejora el mejor fitness en las siguientes 3 iteraciones?), sobre
      las predicciones emitidas con el modelo ya entrenado. Referencias: AUC 0,5 =
      azar. El Brier se compara con el del predictor de TASA BASE de la corrida (que
      siempre predice la frecuencia observada de mejora), no con 0,25: cuando la mejora
      es rara, un predictor constante ya obtiene un Brier bajo sin discriminar nada.
      Brier Skill Score = 1 - Brier / Brier_tasa_base; > 0 indica que el modelo aporta.
  H2  A′ mejora a A en desempeño final y en comportamiento anytime.
      Final: el contraste del análisis por familia (resultados/analysis/). Anytime:
      área bajo la curva de convergencia (media del mejor fitness por iteración),
      con el mismo procedimiento que el final: mediana por celda y Wilcoxon sobre
      diferencias relativas en las familias; Wilcoxon pareado sobre las 31 semillas
      por instancia en rail.
  H3  Las decisiones del controlador quedan registradas y explicadas.
      Explicaciones XAI y adaptaciones registradas, con su motivo, por corrida.

Escribe resultados/analysis/hipotesis.json.

    python campana/hipotesis.py   (desde RecursosPaper/codigo)
"""

from __future__ import annotations

import collections
import csv
import glob
import json
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

AQUI = Path(__file__).resolve().parent
R = AQUI.parents[1] / "resultados"
FAMILIAS = ["classical", "cec2014", "cec2017", "cec2022", "bbob", "fir", "rail"]
HORIZONTE = 3


def _leer(path: Path, col: str) -> dict[int, float]:
    with open(path) as fh:
        return {int(r["iteration"]): float(r[col]) for r in csv.DictReader(fh)}


def h1_corrida(run_dir: str) -> dict | None:
    """AUC y Brier de una corrida de SIA-online."""
    from sklearn.metrics import roc_auc_score
    d = Path(run_dir)
    try:
        pred = _leer(d / "predictions.csv", "prob_improvement")
        best = _leer(d / "traces.csv", "best_fitness")
    except (OSError, KeyError, ValueError):
        return None
    ys, ps = [], []
    for t, p in pred.items():
        if p == 0.5 or (t + HORIZONTE) not in best or t not in best:
            continue                       # modelo aún sin entrenar, o etiqueta no observable
        ys.append(1 if best[t + HORIZONTE] < best[t] - 1e-12 else 0)
        ps.append(p)
    if len(ys) < 10:
        return {"auc": None, "brier": None, "bss": None, "tasa": None}
    ys_a, ps_a = np.array(ys), np.array(ps)
    brier = float(np.mean((ps_a - ys_a) ** 2))
    tasa = float(ys_a.mean())
    ref = tasa * (1 - tasa)
    bss = float(1 - brier / ref) if ref > 0 else None
    auc = float(roc_auc_score(ys_a, ps_a)) if len(set(ys)) == 2 else None
    return {"auc": auc, "brier": brier, "bss": bss, "tasa": tasa}


def area(run_dir: Path) -> float:
    with open(run_dir / "convergence.csv") as fh:
        v = [float(r["best_fitness"]) for r in csv.DictReader(fh)]
    return float(np.mean(v))


def h2_anytime(fam: str) -> dict:
    celdas = collections.defaultdict(dict)
    for algo in ("SIA-base", "SIA-online"):
        for p in glob.glob(str(R / "SIA_online" / algo / fam / "*/*/run_*")):
            p = Path(p)
            celdas[(p.parts[-3], p.parts[-2])].setdefault(algo, {})[p.name] = area(p)
    if fam == "rail":                       # por instancia, pareado sobre semillas
        out = []
        for (inst, _dim), d in sorted(celdas.items()):
            comunes = sorted(set(d["SIA-base"]) & set(d["SIA-online"]))
            a = np.array([d["SIA-base"][r] for r in comunes])
            b = np.array([d["SIA-online"][r] for r in comunes])
            p = 1.0 if np.all(a == b) else float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
            out.append({"instancia": inst, "corridas": len(comunes), "gana_A2": int((b < a).sum()),
                        "gana_A": int((a < b).sum()), "p": p,
                        "mejora_mediana_pct": float(100 * np.median((a - b) / np.maximum(np.abs(a), np.abs(b))))})
        return {"por_instancia": out}
    A, B = [], []
    for d in celdas.values():
        if "SIA-base" in d and "SIA-online" in d:
            A.append(np.median(list(d["SIA-base"].values())))
            B.append(np.median(list(d["SIA-online"].values())))
    A, B = np.array(A), np.array(B)
    den = np.maximum(np.abs(A), np.abs(B)); den[den == 0] = 1
    rel = (B - A) / den
    p = float(stats.wilcoxon(rel, zero_method="wilcox").pvalue) if np.any(rel != 0) else 1.0
    return {"celdas": len(A), "gana_A2": int((B < A).sum()), "gana_A": int((A < B).sum()),
            "p": p, "mejora_mediana_pct": float(-100 * np.median(rel))}


def h3(fam: str) -> dict:
    expl, adapt, sin_expl, sin_adapt, n = 0, 0, 0, 0, 0
    motivos = collections.Counter()
    for p in glob.glob(str(R / "SIA_online/SIA-online" / fam / "*/*/run_*/summary.json")):
        d = json.loads(Path(p).read_text())
        n += 1
        expl += d.get("n_xai_explanations", 0); adapt += d.get("n_adaptations", 0)
        sin_expl += int(d.get("n_xai_explanations", 0) == 0)
        sin_adapt += int(d.get("n_adaptations", 0) == 0)
        f = Path(p).parent / "adaptations.csv"
        if f.exists():
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    motivos[r.get("reason") or "sin motivo"] += 1
    return {"corridas": n, "explicaciones": expl, "adaptaciones": adapt,
            "corridas_sin_explicacion": sin_expl, "corridas_sin_adaptacion": sin_adapt,
            "adaptaciones_sin_motivo": motivos.get("sin motivo", 0),
            "motivos": dict(motivos.most_common())}


def h2_final(fam: str) -> dict | None:
    if fam == "rail":
        tests = list(csv.DictReader(open(R / "analysis/rail/rail_tests.csv")))
        return {"por_instancia": [{"instancia": t["instance"], "gana_A2": int(t["better_ref"]),
                                   "iguales": int(t["equal"]), "gana_A": int(t["worse_ref"]),
                                   "p": float(t["p"]), "p_finner": float(t["p_finner"]),
                                   "A": float(t["A_ref"])}
                                  for t in tests if t["a"] == "SIA-base"]}
    w = json.loads((R / "analysis" / fam / "wilcoxon_A_vs_Aprime.json").read_text())
    return {"celdas": w["n_problems"], "gana_A2": w["wins_b"], "gana_A": w["wins_a"],
            "p": w["p_value"], "mejora_mediana_pct": -100 * w["median_rel_diff"]}


def main() -> None:
    out = {}
    runs = {f: glob.glob(str(R / "SIA_online/SIA-online" / f / "*/*/run_*")) for f in FAMILIAS}
    with ProcessPoolExecutor(max_workers=20) as ex:
        h1_res = {f: list(ex.map(h1_corrida, runs[f], chunksize=64)) for f in FAMILIAS}
    for f in FAMILIAS:
        res = [r for r in h1_res[f] if r]
        aucs = [r["auc"] for r in res if r["auc"] is not None]
        briers = [r["brier"] for r in res if r["brier"] is not None]
        out[f] = {
            "H1": {"corridas": len(res), "corridas_con_auc": len(aucs),
                   "auc_mediana": st.median(aucs) if aucs else None,
                   "auc_p25": float(np.percentile(aucs, 25)) if aucs else None,
                   "auc_p75": float(np.percentile(aucs, 75)) if aucs else None,
                   "corridas_auc_sobre_05": sum(1 for a in aucs if a > 0.5),
                   "brier_mediano": st.median(briers) if briers else None,
                   "bss_mediano": st.median([r["bss"] for r in res if r["bss"] is not None]),
                   "tasa_mejora_mediana": st.median([r["tasa"] for r in res if r["tasa"] is not None]),
                   # ¿la AUC mediana supera a 0,5 de forma significativa? (Wilcoxon de una muestra)
                   "p_auc_sobre_05": float(stats.wilcoxon(np.array(aucs) - 0.5, alternative="greater").pvalue)
                   if len(aucs) > 10 else None},
            "H2_final": h2_final(f),
            "H2_anytime": h2_anytime(f),
            "H3": h3(f),
        }
        print(f"{f:9} AUC {out[f]['H1']['auc_mediana']:.3f}  Brier {out[f]['H1']['brier_mediano']:.3f}  "
              f"anytime {json.dumps(out[f]['H2_anytime'])[:110]}")
    (R / "analysis" / "hipotesis.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nhipotesis.json en {R / 'analysis'}")


if __name__ == "__main__":
    main()
