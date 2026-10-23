"""Convierte una traza en un vector numérico de características para el modelo.

El orden y los nombres de las características son FIJOS para que el modelo y el
módulo XAI compartan el mismo espacio de features.
"""

from __future__ import annotations

import numpy as np

# Características numéricas (operator_used se codifica como one-hot aparte).
NUMERIC_FEATURES = [
    "best_fitness",
    "mean_fitness",
    "std_fitness",
    "diversity",
    "stagnation_counter",
    "n_improved_agents",
    "improvement_rate",
    "exploration_rate",
    "exploitation_rate",
    "w",
    "c1",
    "c2",
]

OPERATORS = ["standard", "diversification", "intensification"]
OPERATOR_FEATURES = [f"op_{o}" for o in OPERATORS]

FEATURE_NAMES = NUMERIC_FEATURES + OPERATOR_FEATURES


class FeatureBuilder:
    """Traza -> vector de características (numpy 1D)."""

    feature_names = FEATURE_NAMES

    def build(self, trace: dict) -> np.ndarray:
        vec = [float(trace.get(f, 0.0)) for f in NUMERIC_FEATURES]
        op = trace.get("operator_used", "standard")
        vec.extend(1.0 if op == o else 0.0 for o in OPERATORS)
        return np.asarray(vec, dtype=float)
