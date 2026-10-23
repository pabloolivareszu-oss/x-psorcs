"""XAI online mediante Permutation Importance sobre la ventana reciente.

Devuelve, para la predicción actual del modelo:
  - top_features: las features más importantes (por permutación).
  - feature_effects: efecto firmado de cada top-feature sobre P(mejora).
      signo > 0  -> valores altos de la feature se asocian a mayor P(mejora).
      signo < 0  -> valores altos de la feature se asocian a menor P(mejora).
  - interpretation: texto legible.

El efecto firmado se estima por correlación de Pearson entre la feature y la
probabilidad predicha de la clase positiva en la ventana. Es barato y suficiente
para guiar reglas adaptativas. La magnitud proviene de Permutation Importance.
"""

from __future__ import annotations

import numpy as np
from sklearn.inspection import permutation_importance


class XAIModule:
    def __init__(self, feature_names: list[str], top_k: int = 3, seed: int | None = None):
        self.feature_names = feature_names
        self.top_k = top_k
        self.seed = seed

    def explain(self, model, X_window: np.ndarray, y_window: np.ndarray) -> dict | None:
        """Explica el modelo sobre la ventana. Devuelve None si no es posible."""
        if model is None or X_window is None or len(X_window) < 10:
            return None
        if len(np.unique(y_window)) < 2:
            return None

        try:
            result = permutation_importance(
                model, X_window, y_window,
                n_repeats=5, random_state=self.seed, n_jobs=1,
            )
        except Exception:
            return None

        importances = result.importances_mean
        order = np.argsort(importances)[::-1][: self.top_k]

        # Signo del efecto: correlación feature vs. P(clase positiva) en la ventana.
        proba = self._positive_proba(model, X_window)
        effects: dict[str, float] = {}
        top_features: list[str] = []
        for idx in order:
            name = self.feature_names[idx]
            top_features.append(name)
            col = X_window[:, idx]
            if np.std(col) < 1e-12 or np.std(proba) < 1e-12:
                sign = 0.0
            else:
                sign = float(np.corrcoef(col, proba)[0, 1])
            # Magnitud por importancia, dirección por correlación.
            effects[name] = float(np.sign(sign) * importances[idx])

        return {
            "top_features": top_features,
            "feature_effects": effects,
            "interpretation": self._interpret(top_features, effects),
        }

    @staticmethod
    def _positive_proba(model, X: np.ndarray) -> np.ndarray:
        classes = list(model.classes_)
        proba = model.predict_proba(X)
        if 1 in classes:
            return proba[:, classes.index(1)]
        return np.zeros(len(X))

    @staticmethod
    def _interpret(top_features: list[str], effects: dict[str, float]) -> str:
        parts = []
        for f in top_features:
            direction = "aumenta" if effects[f] > 0 else "reduce"
            parts.append(f"{f} ({direction} P(mejora))")
        return "Variables clave: " + ", ".join(parts) + "."
