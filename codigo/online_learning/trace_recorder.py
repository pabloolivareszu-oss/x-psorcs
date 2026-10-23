"""Registro de trazas online: captura el estado de la búsqueda en cada iteración."""

from __future__ import annotations

import copy


class TraceRecorder:
    """Acumula una traza por iteración a partir del `state` del SIA."""

    def __init__(self):
        self.traces: list[dict] = []

    def capture(self, state: dict) -> dict:
        params = state.get("parameters", {})
        trace = {
            "iteration": state["iteration"],
            "best_fitness": state["best_fitness"],
            "previous_best_fitness": state["previous_best_fitness"],
            "mean_fitness": state["mean_fitness"],
            "std_fitness": state["std_fitness"],
            "diversity": state["diversity"],
            "stagnation_counter": state["stagnation_counter"],
            "n_improved_agents": state["n_improved_agents"],
            "exploration_rate": state["exploration_rate"],
            "exploitation_rate": state["exploitation_rate"],
            "operator_used": state["operator_used"],
            "w": params.get("w", 0.0),
            "c1": params.get("c1", 0.0),
            "c2": params.get("c2", 0.0),
        }
        # improvement_rate: mejora relativa del best respecto al anterior.
        prev = trace["previous_best_fitness"]
        denom = abs(prev) + 1e-12
        trace["improvement_rate"] = (prev - trace["best_fitness"]) / denom

        self.traces.append(copy.deepcopy(trace))
        return trace
