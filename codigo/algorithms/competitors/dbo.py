"""DBO: Dung Beetle Optimizer.

Referencia:
    J. Xue, B. Shen, "Dung beetle optimizer: a new meta-heuristic algorithm for
    global optimization", The Journal of Supercomputing, 79, 2023.

Mecanismo: la población se divide en cuatro roles de escarabajo pelotero:
  - rolling (rodadores): ruedan la bola recta o bailan (desviación) si hay obstáculo.
  - breeding (reproductores): ponen huevos en una región límite alrededor del mejor.
  - foraging (forrajeadores pequeños): buscan comida en región límite del óptimo.
  - stealing (ladrones): roban bolas cerca del mejor global.
Las proporciones típicas: 6/6/7/11 de 30 individuos.
"""

from __future__ import annotations

import numpy as np

from algorithms.competitors.base import Competitor, FEvalCounter, clip


class DBO(Competitor):
    name = "DBO"

    def __init__(self, pop_size: int = 30):
        self.pop_size = pop_size

    def _run(self, fc: FEvalCounter, dim, lower, upper, rng):
        N = self.pop_size
        pop = rng.uniform(lower, upper, size=(N, dim))
        fit = fc.eval_population(pop)
        max_iter = max(1, fc.max_fes // N)

        # Particiona roles (proporciones del paper, escaladas a N).
        n_roll = max(1, int(0.2 * N))
        n_breed = max(1, int(0.2 * N))
        n_forage = max(1, int(0.25 * N))
        # el resto: ladrones
        idx = np.arange(N)

        for t in range(max_iter):
            gbest_i = np.argmin(fit)
            gbest = pop[gbest_i].copy()
            gworst = pop[np.argmax(fit)].copy()
            R = 1 - t / max_iter

            for i in idx:
                xi = pop[i].copy()
                if i < n_roll:
                    # Rodador: rodar recto o bailar.
                    if rng.random() < 0.9:
                        b = 0.3; k = 0.1
                        alpha = 1 if rng.random() > 0.1 else -1
                        new = xi + alpha * k * pop[i] + b * np.abs(xi - gworst)
                    else:
                        theta = rng.choice([0, np.pi / 2, np.pi])  # ángulo de baile
                        if theta in (0, np.pi):
                            new = xi
                        else:
                            new = xi + np.tan(theta) * np.abs(xi - pop[(i + 1) % N])
                elif i < n_roll + n_breed:
                    # Reproductor: región límite alrededor del mejor.
                    lo_b = clip(gbest * (1 - R), lower, upper)
                    hi_b = clip(gbest * (1 + R), lower, upper)
                    new = gbest + rng.random(dim) * (xi - lo_b) + rng.random(dim) * (xi - hi_b)
                elif i < n_roll + n_breed + n_forage:
                    # Forrajeador: región alrededor del óptimo global.
                    lo_b = clip(gbest * (1 - R), lower, upper)
                    hi_b = clip(gbest * (1 + R), lower, upper)
                    C1 = rng.normal(size=dim)
                    C2 = rng.random(dim)
                    new = xi + C1 * (xi - lo_b) + C2 * (xi - hi_b)
                else:
                    # Ladrón: roba cerca del mejor global.
                    S = 0.5
                    g = rng.normal(size=dim)
                    new = gbest + S * g * (np.abs(xi - gbest) + np.abs(xi - pop[gbest_i]))

                new = clip(new, lower, upper)
                f = fc(new)
                if f < fit[i]:
                    pop[i], fit[i] = new, f
