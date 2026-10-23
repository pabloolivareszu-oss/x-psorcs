"""Modelo probabilístico online por ventana deslizante (Random Forest).

No es online "puro": mantiene un buffer de las últimas `window_size` muestras
(X, y) y reentrena el bosque cada `update_interval` iteraciones cuando hay
suficientes muestras y al menos dos clases. Esto se considera online por
reentrenamiento incremental en ventana DURANTE la ejecución del SIA.

Entrega P(mejora futura | estado actual).
"""

from __future__ import annotations

from collections import deque

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class OnlineModel:
    def __init__(
        self,
        window_size: int = 200,
        update_interval: int = 5,
        min_samples: int = 30,
        n_estimators: int = 100,
        seed: int | None = None,
    ):
        self.window_size = window_size
        self.update_interval = update_interval
        self.min_samples = min_samples
        self.seed = seed

        self._X: deque = deque(maxlen=window_size)
        self._y: deque = deque(maxlen=window_size)
        self.model: RandomForestClassifier | None = None
        self._n_updates = 0
        self.last_fit_iteration: int | None = None

    def add_sample(self, x: np.ndarray, y: int) -> None:
        """Agrega un par (X(t), y(t)) ya etiquetado al buffer de ventana."""
        self._X.append(np.asarray(x, dtype=float))
        self._y.append(int(y))

    def maybe_update(self, iteration: int) -> bool:
        """Reentrena si toca según el intervalo y hay datos suficientes.
        Devuelve True si se reentrenó."""
        if iteration % self.update_interval != 0:
            return False
        if len(self._y) < self.min_samples:
            return False
        y = np.array(self._y)
        if len(np.unique(y)) < 2:
            return False  # RF necesita al menos dos clases

        X = np.array(self._X)
        self.model = RandomForestClassifier(
            n_estimators=100 if self.model is None else self.model.n_estimators,
            random_state=self.seed,
            n_jobs=1,
        )
        self.model.fit(X, y)
        self._n_updates += 1
        self.last_fit_iteration = iteration
        return True

    def is_ready(self) -> bool:
        return self.model is not None

    def predict_proba_improvement(self, x: np.ndarray) -> float:
        """P(mejora futura) = P(y=1 | x). Si no hay modelo, devuelve 0.5 (neutro)."""
        if self.model is None:
            return 0.5
        x = np.asarray(x, dtype=float).reshape(1, -1)
        classes = list(self.model.classes_)
        proba = self.model.predict_proba(x)[0]
        if 1 in classes:
            return float(proba[classes.index(1)])
        return 0.0  # el modelo solo vio la clase 0

    @property
    def n_updates(self) -> int:
        return self._n_updates

    @property
    def buffer_size(self) -> int:
        return len(self._y)
