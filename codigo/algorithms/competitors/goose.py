"""GOOSE: Goose Algorithm.

Referencia:
    R. M. Hamad, M. A. Sahib, "The Goose Algorithm: a novel metaheuristic
    optimization algorithm for engineering applications", 2024 (familia GOOSE).

Mecanismo inspirado en el comportamiento de los gansos: una parte de la bandada
vigila (guarding) mientras otra descansa/se alimenta. El balance exploración-
explotación se rige por un coeficiente que depende de un "peso" aleatorio y de la
velocidad de aproximación al mejor. Con prob. >=0.5 la bandada explota (rápida
aproximación al mejor, modelada con un término de tiempo de llegada); en otro caso
explora con saltos alrededor del mejor y de posiciones aleatorias.
"""

from __future__ import annotations

import numpy as np

from algorithms.competitors.base import Competitor, FEvalCounter, clip


class GOOSE(Competitor):
    name = "GOOSE"

    def __init__(self, pop_size: int = 30):
        self.pop_size = pop_size

    def _run(self, fc: FEvalCounter, dim, lower, upper, rng):
        N = self.pop_size
        pop = rng.uniform(lower, upper, size=(N, dim))
        fit = fc.eval_population(pop)
        max_iter = max(1, fc.max_fes // N)

        for t in range(max_iter):
            best = pop[np.argmin(fit)].copy()
            coe = 1 - t / max_iter  # coeficiente de exploración decreciente
            for i in range(N):
                xi = pop[i].copy()
                if rng.random() >= 0.5:
                    # --- Explotación: aproximación rápida al mejor ---
                    weight = rng.random()
                    # "tiempo de llegada" simulado: distancia escalada
                    speed = rng.random() * coe
                    arrival = np.sqrt(np.sum((best - xi) ** 2)) + 1e-12
                    new = best + weight * speed * (best - xi) / arrival * (upper - lower) * 0.1
                else:
                    # --- Exploración: vigilancia/salto alrededor del mejor ---
                    alpha = rng.random() * 2 * np.pi
                    rand = pop[rng.integers(N)]
                    new = best + coe * np.cos(alpha) * (rand - xi) \
                        + coe * rng.normal(size=dim) * (upper - lower) * 0.05
                new = clip(new, lower, upper)
                f = fc(new)
                if f < fit[i]:
                    pop[i], fit[i] = new, f
