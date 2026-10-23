"""Controlador adaptativo basado en reglas.

Lee predicción + explicación XAI + estado y modifica en sitio:
  - exploration_rate / exploitation_rate
  - operator_probabilities
Tras adaptar, normaliza para mantener valores válidos.

Devuelve la lista de adaptaciones aplicadas en la iteración (para logging).
"""

from __future__ import annotations

import numpy as np


class AdaptationController:
    def __init__(
        self,
        prob_low: float = 0.25,
        prob_high: float = 0.70,
        diversity_min: float = 0.10,
        stagnation_limit: int = 8,
        decay: float = 0.02,
        op_decay: float = 0.05,
        p_min: float = 0.05,
        div_cooldown: int = 5,
    ):
        self.prob_low = prob_low
        self.prob_high = prob_high
        self.diversity_min = diversity_min
        self.stagnation_limit = stagnation_limit
        # Homeostasis: las tasas decaen suavemente hacia 0.5 cada iteración para
        # evitar saturación monotónica; las reglas las empujan según el régimen actual.
        self.decay = decay
        # Corrección del estado absorbente: sin estos tres mecanismos la probabilidad
        # de un operador podía caer a 0 y ninguna regla la devolvía.
        #   op_decay:     las probabilidades de operador decaen hacia su prior inicial.
        #   p_min:        piso por operador (probability matching, selección adaptativa
        #                 de operadores); ningún operador queda inalcanzable.
        #   div_cooldown: iteraciones sin volver a activar diversificación tras hacerlo.
        self.op_decay = op_decay
        self.p_min = p_min
        self.div_cooldown = div_cooldown
        self._prior: dict | None = None
        self._last_div_iter: int | None = None

    def adapt(self, state: dict, prob_improvement: float, explanation: dict | None) -> list[dict]:
        adaptations: list[dict] = []

        if self._prior is None:
            self._prior = dict(state["operator_probabilities"])

        # Decaimiento hacia el balance neutro 0.5 antes de aplicar reglas.
        for key in ("exploration_rate", "exploitation_rate"):
            state[key] += (0.5 - state[key]) * self.decay
        # Decaimiento de las probabilidades de operador hacia su prior.
        probs = state["operator_probabilities"]
        for k in probs:
            probs[k] += (self._prior[k] - probs[k]) * self.op_decay

        def record(reason, kind, key, old, new):
            adaptations.append({
                "reason": reason, "adaptation_type": kind,
                "key": key, "old_value": round(float(old), 5),
                "new_value": round(float(new), 5),
            })

        # --- Reglas basadas en la predicción ---------------------------------
        if prob_improvement < self.prob_low:
            if state["diversity"] < self.diversity_min:
                old_e = state["exploration_rate"]
                state["exploration_rate"] += 0.10
                state["exploitation_rate"] -= 0.10
                record("low_prob_low_diversity", "increase_exploration",
                       "exploration_rate", old_e, state["exploration_rate"])
            in_cooldown = (
                self._last_div_iter is not None
                and state["iteration"] - self._last_div_iter < self.div_cooldown
            )
            if state["stagnation_counter"] > self.stagnation_limit and not in_cooldown:
                self._last_div_iter = state["iteration"]
                old_d = state["operator_probabilities"]["diversification"]
                state["operator_probabilities"]["diversification"] += 0.15
                state["operator_probabilities"]["intensification"] = max(
                    0.0, state["operator_probabilities"]["intensification"] - 0.10
                )
                record("prolonged_stagnation", "activate_diversification",
                       "op_diversification", old_d,
                       state["operator_probabilities"]["diversification"])

        elif prob_improvement > self.prob_high:
            old_x = state["exploitation_rate"]
            state["exploitation_rate"] += 0.05
            state["exploration_rate"] -= 0.05
            record("high_prob_improvement", "increase_exploitation",
                   "exploitation_rate", old_x, state["exploitation_rate"])

        # --- Reglas guiadas por XAI ------------------------------------------
        if explanation is not None:
            top = explanation["top_features"]
            eff = explanation["feature_effects"]

            if "diversity" in top and eff.get("diversity", 0.0) < 0:
                old_e = state["exploration_rate"]
                state["exploration_rate"] += 0.05
                record("xai_diversity_negative", "xai_guided_exploration",
                       "exploration_rate", old_e, state["exploration_rate"])

            # Si la diversidad ayuda (efecto positivo) y estamos estancados, explorar.
            if "stagnation_counter" in top and eff.get("stagnation_counter", 0.0) < 0:
                old_e = state["exploration_rate"]
                state["exploration_rate"] += 0.05
                record("xai_stagnation_harmful", "xai_guided_exploration",
                       "exploration_rate", old_e, state["exploration_rate"])

            # Operador actual reforzado si su uso se asocia positivamente.
            op = state["operator_used"]
            op_feat = f"op_{op}"
            if op_feat in top and eff.get(op_feat, 0.0) > 0:
                old_p = state["operator_probabilities"][op]
                state["operator_probabilities"][op] += 0.10
                record("xai_operator_positive", "reinforce_operator",
                       op_feat, old_p, state["operator_probabilities"][op])

        self._normalize(state)
        self._apply_floor(state["operator_probabilities"])
        state["last_adaptation"] = adaptations[-1]["adaptation_type"] if adaptations else None
        return adaptations

    def _apply_floor(self, probs: dict) -> None:
        """Eleva al piso p_min los operadores por debajo y reescala el resto para
        que la suma siga en 1. No altera nada si todos están sobre el piso."""
        floored: set = set()
        while True:
            low = {k for k in probs if k not in floored and probs[k] < self.p_min}
            if not low:
                return
            floored |= low
            free = [k for k in probs if k not in floored]
            mass = 1.0 - self.p_min * len(floored)
            total_free = sum(probs[k] for k in free)
            for k in floored:
                probs[k] = self.p_min
            for k in free:
                probs[k] = probs[k] / total_free * mass if total_free > 0 else mass / len(free)

    @staticmethod
    def _normalize(state: dict) -> None:
        state["exploration_rate"] = float(np.clip(state["exploration_rate"], 0.0, 1.0))
        state["exploitation_rate"] = float(np.clip(state["exploitation_rate"], 0.0, 1.0))
        probs = state["operator_probabilities"]
        for k in probs:
            probs[k] = max(0.0, probs[k])
        total = sum(probs.values())
        if total <= 0:
            n = len(probs)
            for k in probs:
                probs[k] = 1.0 / n
        else:
            for k in probs:
                probs[k] /= total
