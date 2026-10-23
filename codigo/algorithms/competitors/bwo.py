"""BWO: Beluga Whale Optimization.

Referencia:
    C. Zhong, G. Li, Z. Meng, "Beluga whale optimization: A novel nature-inspired
    metaheuristic algorithm", Knowledge-Based Systems, 251, 2022.

Mecanismo con tres fases gobernadas por el factor de balance B0:
  - Exploración (nado en pares): las belugas nadan reflejando posiciones por pares
    de dimensiones (espejo).
  - Explotación (prey): se acercan a la mejor con vuelo de Lévy.
  - Whale fall (caída de ballena): probabilidad Wf de reinicializar con paso
    aleatorio, simulando muerte/renovación de población.
El factor B = B0*(1 - t/2T) decide exploración (B>0.5) vs explotación.
"""

from __future__ import annotations

import numpy as np

from algorithms.competitors.base import Competitor, FEvalCounter, clip
from algorithms.competitors.hho import levy


class BWO(Competitor):
    name = "BWO"

    def __init__(self, pop_size: int = 30):
        self.pop_size = pop_size

    def _run(self, fc: FEvalCounter, dim, lower, upper, rng):
        N = self.pop_size
        pop = rng.uniform(lower, upper, size=(N, dim))
        fit = fc.eval_population(pop)
        max_iter = max(1, fc.max_fes // N)

        for t in range(max_iter):
            best = pop[np.argmin(fit)].copy()
            B0 = rng.random()
            B = B0 * (1 - t / (2 * max_iter))           # factor de balance
            Wf = 0.1 - 0.05 * t / max_iter              # prob. whale fall

            for i in range(N):
                xi = pop[i].copy()
                if B > 0.5:
                    # --- Exploración: nado en pares (espejo por dimensiones) ---
                    new = xi.copy()
                    r = pop[rng.integers(N)]
                    pj = rng.integers(dim)
                    for d in range(dim):
                        if d % 2 == 0:
                            new[d] = xi[pj] + (r[pj] - xi[d]) * (1 + rng.random()) * np.sin(2 * np.pi * rng.random())
                        else:
                            new[d] = xi[pj] + (r[pj] - xi[d]) * (1 + rng.random()) * np.cos(2 * np.pi * rng.random())
                else:
                    # --- Explotación: aproximación con Lévy ---
                    r1, r2 = rng.random(), rng.random()
                    C1 = 2 * r2 * (1 - t / max_iter)
                    rj = pop[rng.integers(N)]
                    LF = levy(dim, rng)
                    new = r1 * best - r2 * xi + C1 * LF * (rj - xi)

                new = clip(new, lower, upper)
                f = fc(new)
                if f < fit[i]:
                    pop[i], fit[i] = new, f

            # --- Whale fall: renovación de algunos individuos ---
            for i in range(N):
                if rng.random() < Wf:
                    r1, r2 = rng.random(), rng.random()
                    C2 = 2 * Wf * N / max(1, N)
                    step = (upper - lower) * np.exp(-C2 * t / max_iter)
                    new = r1 * pop[i] - r2 * pop[rng.integers(N)] + step * (rng.random(dim) - 0.5)
                    new = clip(new, lower, upper)
                    f = fc(new)
                    if f < fit[i]:
                        pop[i], fit[i] = new, f
