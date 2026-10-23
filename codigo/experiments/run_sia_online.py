"""Ejecuta el SIA online explicable y adaptativo sobre benchmarks.

Flujo online dentro del ciclo iterativo (NO offline):
    SIA(t) -> traza -> etiqueta(horizonte k) -> modelo online -> XAI -> adaptación -> SIA(t+1)

Soporta tres familias de benchmarks (--family): classical, cec, bbob.

Uso:
    python -m experiments.run_sia_online --family cec --benchmarks F1 F3 --dims 10 20 50
    python -m experiments.run_sia_online --family bbob --benchmarks f01 f06 --dims 10 20
    python -m experiments.run_sia_online --family classical --benchmarks sphere --dims 10
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

# Permite ejecutar como script o como módulo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptation.adaptation_controller import AdaptationController
from algorithms.sia_online import OnlineExplainableSIA
from benchmarks.registry import get_problem, supported_dims, FAMILIES
from explainability.xai_module import XAIModule
from online_learning.feature_builder import FeatureBuilder
from online_learning.label_builder import LabelBuilder
from online_learning.online_model import OnlineModel
from online_learning.trace_recorder import TraceRecorder

from paths import RESULTS  # noqa: E402  (raíz única; ver paths.py)

RESULTS_ROOT = RESULTS / "SIA_online"


def run_single(
    family: str,
    benchmark: str,
    dim: int,
    run_id: int,
    pop_size: int,
    max_iter: int,
    horizon: int,
    update_interval: int,
    xai_interval: int,
    window_size: int,
    seed: int,
    out_dir: Path,
    mode: str = "online",
    controller_kwargs: dict | None = None,
) -> dict:
    """Ejecuta una corrida completa del SIA y vuelca CSVs + summary.json.

    mode='online': pipeline completo (modelo + XAI + adaptación) = A' (mejorada).
    mode='base':   PSO base con parámetros fijos, sin online/XAI/adaptación = A.
                   Sirve de baseline para la comparación A vs A'.
    """
    online = (mode == "online")
    objective, lower, upper, prob_info = get_problem(family, benchmark, dim, run_id)

    sia = OnlineExplainableSIA(
        objective=objective, dim=dim, lower=lower, upper=upper,
        pop_size=pop_size, max_iter=max_iter, seed=seed,
    )
    recorder = TraceRecorder()
    feats = FeatureBuilder()
    labeler = LabelBuilder()
    model = OnlineModel(window_size=window_size, update_interval=update_interval, seed=seed)
    xai = XAIModule(feature_names=feats.feature_names, seed=seed)
    controller = AdaptationController(**(controller_kwargs or {}))

    convergence_rows, trace_rows, pred_rows, expl_rows, adapt_rows = [], [], [], [], []
    # Trazas pendientes de etiquetar: (iteration, X, best_fitness_en_t).
    pending: list[tuple[int, np.ndarray, float]] = []
    last_explanation = None
    t_start = time.perf_counter()

    for t in range(1, max_iter + 1):
        # 1. Una iteración del SIA.
        state = sia.step()

        # 2-3. Traza + features.
        trace = recorder.capture(state)
        X_t = feats.build(trace)
        pending.append((t, X_t, state["best_fitness"]))

        if online:
            # 4. Etiquetar trazas cuyo horizonte ya venció y alimentar el modelo.
            while pending and pending[0][0] + horizon <= t:
                it0, X0, best0 = pending.pop(0)
                y0 = labeler.build(best0, state["best_fitness"])
                model.add_sample(X0, y0)
            model.maybe_update(t)

            # 5. Predicción.
            prob_imp = model.predict_proba_improvement(X_t) if model.is_ready() else 0.5

            # 6. XAI cada xai_interval (sobre la ventana actual del buffer).
            if model.is_ready() and t % xai_interval == 0 and model.buffer_size >= 10:
                X_win = np.array(model._X)
                y_win = np.array(model._y)
                expl = xai.explain(model.model, X_win, y_win)
                if expl is not None:
                    last_explanation = expl
                    tf = (expl["top_features"] + ["", "", ""])[:3]
                    eff = expl["feature_effects"]
                    expl_rows.append({
                        "iteration": t,
                        "top_feature_1": tf[0], "top_feature_2": tf[1], "top_feature_3": tf[2],
                        "effect_1": round(eff.get(tf[0], 0.0), 5),
                        "effect_2": round(eff.get(tf[1], 0.0), 5),
                        "effect_3": round(eff.get(tf[2], 0.0), 5),
                    })
            # La explicación más reciente guía la adaptación entre intervalos.
            explanation_for_adapt = last_explanation if (t % xai_interval == 0) else None

            # 7. Adaptar.
            adaptations = controller.adapt(state, prob_imp, explanation_for_adapt)
            for a in adaptations:
                adapt_rows.append({"iteration": t, **a})
        else:
            # Modo base: PSO puro, sin modelo/XAI/adaptación. Predicción neutra.
            prob_imp = 0.5

        prob_stag = 1.0 - prob_imp
        pred_rows.append({
            "iteration": t, "prob_improvement": round(prob_imp, 5),
            "prob_stagnation": round(prob_stag, 5),
        })

        # 8. Logging por iteración.
        convergence_rows.append({
            "iteration": t, "best_fitness": state["best_fitness"],
            "mean_fitness": state["mean_fitness"], "std_fitness": state["std_fitness"],
        })
        trace_rows.append({
            "iteration": t, "best_fitness": trace["best_fitness"],
            "mean_fitness": trace["mean_fitness"], "diversity": round(trace["diversity"], 6),
            "stagnation_counter": trace["stagnation_counter"],
            "operator_used": trace["operator_used"],
            "exploration_rate": round(trace["exploration_rate"], 4),
            "exploitation_rate": round(trace["exploitation_rate"], 4),
        })

    elapsed = time.perf_counter() - t_start

    # --- Escribir CSVs ----------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "convergence.csv", convergence_rows)
    _write_csv(out_dir / "traces.csv", trace_rows)
    _write_csv(out_dir / "predictions.csv", pred_rows)
    _write_csv(out_dir / "explanations.csv", expl_rows,
               header=["iteration", "top_feature_1", "top_feature_2", "top_feature_3",
                       "effect_1", "effect_2", "effect_3"])
    _write_csv(out_dir / "adaptations.csv", adapt_rows,
               header=["iteration", "reason", "adaptation_type", "key", "old_value", "new_value"])

    f_global = prob_info.get("f_global")
    error_to_optimum = (
        sia.gbest_fitness - f_global if f_global is not None else None
    )
    summary = {
        "mode": mode,
        "algorithm": "SIA-online" if online else "SIA-base",
        "family": prob_info.get("family", family),
        "benchmark": benchmark, "dim": dim, "run_id": run_id,
        "pop_size": pop_size, "max_iter": max_iter,
        "best_fitness": sia.gbest_fitness,
        "f_global": f_global,
        "error_to_optimum": error_to_optimum,
        "function_evaluations": sia.fe,
        "n_model_updates": model.n_updates,
        "n_adaptations": len(adapt_rows),
        "n_xai_explanations": len(expl_rows),
        "final_exploration_rate": sia.state["exploration_rate"],
        "final_exploitation_rate": sia.state["exploitation_rate"],
        "final_operator_probabilities": sia.state["operator_probabilities"],
        "elapsed_seconds": round(elapsed, 3),
        "seed": seed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _write_csv(path: Path, rows: list[dict], header: list[str] | None = None) -> None:
    fieldnames = header or (list(rows[0].keys()) if rows else [])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    p = argparse.ArgumentParser(description="SIA online explicable y adaptativo")
    p.add_argument("--family", default="cec2017", choices=FAMILIES,
                   help="Familia de benchmarks: classical, cec2014, cec2017, cec2022, bbob.")
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--benchmarks", nargs="+", default=["F1", "F3", "F5"],
                   help="Funciones de la familia (p.ej. F1 F3 para CEC, f01 f06 para BBOB).")
    p.add_argument("--dims", nargs="+", type=int, default=[10, 20, 50],
                   help="Dimensiones a barrer. Se filtran a las soportadas por la familia.")
    p.add_argument("--pop-size", type=int, default=30)
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--update-interval", type=int, default=5)
    p.add_argument("--xai-interval", type=int, default=10)
    p.add_argument("--window-size", type=int, default=200)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--mode", default="online", choices=["online", "base"],
                   help="online = pipeline completo (A'); base = PSO puro (A).")
    args = p.parse_args()

    family = args.family
    mode = args.mode
    # Filtra dims a las soportadas por la familia (p.ej. BBOB sin 50, CEC2022 solo 10/20).
    dims = supported_dims(family, args.dims)
    skipped = [d for d in args.dims if d not in dims]
    if skipped:
        print(f"[aviso] {family} no soporta dim(es) {skipped}; se omiten. "
              f"Dimensiones usadas: {dims}")
    if not dims:
        print(f"[error] Ninguna de las dimensiones {args.dims} es válida para {family}.")
        return

    # Carpeta por modo para no pisar A (base) y A' (online).
    algo_dir = "SIA-online" if mode == "online" else "SIA-base"
    out_root = RESULTS_ROOT / algo_dir / family
    all_summaries = []
    # Barrido completo: benchmark x dimensión x corrida (sección 13 del spec).
    for benchmark in args.benchmarks:
        for dim in dims:
            bests = []
            for run_id in range(1, args.runs + 1):
                out_dir = out_root / f"{benchmark}" / f"dim_{dim}" / f"run_{run_id:02d}"
                # Semilla única por (dim, run) para reproducibilidad.
                seed = args.base_seed + dim * 1000 + run_id
                summary = run_single(
                    family=family, benchmark=benchmark, dim=dim, run_id=run_id,
                    pop_size=args.pop_size, max_iter=args.max_iter,
                    horizon=args.horizon, update_interval=args.update_interval,
                    xai_interval=args.xai_interval, window_size=args.window_size,
                    seed=seed, out_dir=out_dir, mode=mode,
                )
                bests.append(summary["best_fitness"])
                all_summaries.append(summary)
            arr = np.array(bests)
            print(f"[{family} {benchmark} d={dim}] runs={args.runs} "
                  f"best={arr.min():.4e} mean={arr.mean():.4e} std={arr.std():.4e} "
                  f"median={np.median(arr):.4e}")

    # Resumen agregado.
    agg_path = out_root / "all_summaries.json"
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    agg_path.write_text(json.dumps(all_summaries, indent=2))
    print(f"\nResumen agregado: {agg_path}")


if __name__ == "__main__":
    main()
