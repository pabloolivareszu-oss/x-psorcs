"""BKA: Black-winged Kite Algorithm.

Referencia:
    J. Wang, W. Wang, et al., "Black-winged kite algorithm: a nature-inspired
    meta-heuristic for solving benchmark functions and engineering problems",
    Artificial Intelligence Review, 2024.

Mecanismo en dos comportamientos:
  - Attacking (ataque/caza): el milano ajusta el ángulo de ataque; con prob. p
    planea (cos) o se lanza en picado, modulado por n=0.05*exp(-2*(t/T)^2).
  - Migrating (migración): guiado por líder; si el individuo es peor que uno
    aleatorio, migra hacia el mejor con un término de Cauchy; si no, explora.
"""

from __future__ import annotations

import numpy as np

from algorithms.competitors.base import Competitor, FEvalCounter, clip


def _cauchy(rng, dim):
    return np.tan((rng.random(dim) - 0.5) * np.pi)


class BKA(Competitor):
    name = "BKA"

    def __init__(self, pop_size: int = 30, p: float = 0.9):
        self.pop_size = pop_size
        self.p = p

    def _run(self, fc: FEvalCounter, dim, lower, upper, rng):
        N = self.pop_size
        pop = rng.uniform(lower, upper, size=(N, dim))
        fit = fc.eval_population(pop)
        max_iter = max(1, fc.max_fes // N)

        for t in range(max_iter):
            leader_i = np.argmin(fit)
            leader = pop[leader_i].copy()
            n = 0.05 * np.exp(-2 * (t / max_iter) ** 2)

            for i in range(N):
                xi = pop[i].copy()
                # --- Attacking behaviour ---
                r = rng.random()
                if self.p < r:
                    new_a = xi + n * (1 + np.sin(r)) * xi
                else:
                    new_a = xi * (n * (2 * rng.random(dim) - 1) + 1)
                new_a = clip(new_a, lower, upper)
                f = fc(new_a)
                if f < fit[i]:
                    pop[i], fit[i] = new_a, f
                    xi = new_a

                # --- Migrating behaviour ---
                m = 2 * np.sin(r + np.pi / 2)
                ri = rng.integers(N)
                if fit[i] < fit[ri]:
                    new_m = xi + m * (leader - xi)  # bien posicionado: refina hacia líder cautamente
                else:
                    new_m = leader + m * _cauchy(rng, dim) * (leader - xi)
                new_m = clip(new_m, lower, upper)
                f = fc(new_m)
                if f < fit[i]:
                    pop[i], fit[i] = new_m, f
