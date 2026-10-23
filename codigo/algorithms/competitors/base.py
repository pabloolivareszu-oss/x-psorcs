"""Arnés común para los algoritmos competidores.

Todos los competidores comparten:
  - La MISMA interfaz: optimize(objective, dim, lower, upper, max_fes, seed) -> Result.
  - El MISMO presupuesto: se mide en evaluaciones de función (FEs), no en iteraciones,
    para que la comparación contra el SIA online (3000 FEs por defecto) sea justa
    independientemente del tamaño de población interno de cada algoritmo.
  - La MISMA salida: historial de convergencia por FE-checkpoint + best final.

El contador de FEs es la autoridad del presupuesto: cada llamada al objetivo pasa
por `FEvalCounter`, que detiene la optimización en cuanto se agota el presupuesto.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


class BudgetExhausted(Exception):
    """Se lanza internamente cuando se agotan los FEs (control de flujo)."""


class FEvalCounter:
    """Envuelve la función objetivo y contabiliza/limita las evaluaciones.

    Registra la curva de convergencia: cada vez que mejora el best, guarda
    (fe, best). Así todos los algoritmos producen convergencia comparable aunque
    tengan estructuras de iteración distintas.
    """

    def __init__(self, objective: Callable[[np.ndarray], float], max_fes: int):
        self._objective = objective
        self.max_fes = max_fes
        self.fes = 0
        self.best_fitness = np.inf
        self.best_x: np.ndarray | None = None
        # Curva de convergencia comprimida: lista de (fe, best_fitness).
        self.history: list[tuple[int, float]] = []

    def __call__(self, x: np.ndarray) -> float:
        if self.fes >= self.max_fes:
            raise BudgetExhausted()
        f = float(self._objective(np.asarray(x, dtype=float)))
        self.fes += 1
        if f < self.best_fitness:
            self.best_fitness = f
            self.best_x = np.array(x, dtype=float)
            self.history.append((self.fes, f))
        return f

    def eval_population(self, X: np.ndarray) -> np.ndarray:
        """Evalúa una matriz (n, dim). Respeta el presupuesto fila a fila."""
        out = np.empty(len(X))
        for i, row in enumerate(X):
            out[i] = self(row)  # puede lanzar BudgetExhausted
        return out

    @property
    def remaining(self) -> int:
        return max(0, self.max_fes - self.fes)


@dataclass
class Result:
    """Resultado uniforme de una corrida de un competidor."""
    name: str
    best_fitness: float
    best_x: np.ndarray
    fes_used: int
    history: list[tuple[int, float]] = field(default_factory=list)

    def convergence_at(self, n_points: int) -> list[tuple[int, float]]:
        """Re-muestrea la curva de convergencia a n_points uniformes en FEs,
        para tablas/gráficas comparables entre algoritmos."""
        if not self.history:
            return []
        max_fe = self.history[-1][0]
        targets = np.linspace(1, max_fe, n_points).astype(int)
        out, j = [], 0
        cur = self.history[0][1]
        for t in targets:
            while j < len(self.history) and self.history[j][0] <= t:
                cur = self.history[j][1]
                j += 1
            out.append((int(t), float(cur)))
        return out


class Competitor:
    """Clase base. Cada competidor implementa `_run(fcounter, dim, lower, upper, rng)`."""

    name: str = "base"

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        dim: int,
        lower: float,
        upper: float,
        max_fes: int,
        seed: int | None = None,
    ) -> Result:
        rng = np.random.default_rng(seed)
        fcounter = FEvalCounter(objective, max_fes)
        try:
            self._run(fcounter, dim, lower, upper, rng)
        except BudgetExhausted:
            pass  # fin normal: se agotó el presupuesto
        # Garantiza un best aunque el algoritmo no evaluara nada útil.
        if fcounter.best_x is None:
            x0 = rng.uniform(lower, upper, size=dim)
            try:
                fcounter(x0)
            except BudgetExhausted:
                fcounter.best_x = x0
                fcounter.best_fitness = float(objective(x0))
        return Result(
            name=self.name,
            best_fitness=fcounter.best_fitness,
            best_x=fcounter.best_x,
            fes_used=fcounter.fes,
            history=fcounter.history,
        )

    def _run(self, fcounter: FEvalCounter, dim: int, lower: float,
             upper: float, rng: np.random.Generator) -> None:
        raise NotImplementedError


def clip(x: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.clip(x, lower, upper)


def reflect(x: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Rebote en los límites (boundary handling habitual en DE de competición)."""
    span = upper - lower
    if span <= 0:
        return np.clip(x, lower, upper)
    y = x.copy()
    over = y > upper
    under = y < lower
    y[over] = upper - ((y[over] - upper) % span)
    y[under] = lower + ((lower - y[under]) % span)
    return np.clip(y, lower, upper)
