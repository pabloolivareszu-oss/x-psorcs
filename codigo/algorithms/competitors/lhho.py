"""LHHO: Lévy-flight enhanced Harris Hawks Optimization.

Referencia (familia de variantes Lévy-HHO):
    Variantes de HHO que incorporan vuelos de Lévy en la fase de EXPLORACIÓN (no
    solo en los rapid dives), para mejorar la diversidad y escapar de óptimos
    locales. Basado en el HHO de Heidari et al. (2019) con el operador de Lévy
    extendido a la exploración y un paso de aprendizaje por oposición ocasional.

Diferencia clave vs HHO: en exploración los halcones se desplazan con paso de Lévy
alrededor de la presa/mejor, lo que da saltos de cola pesada que aceleran la
cobertura del espacio en las primeras iteraciones.
"""

from __future__ import annotations

import numpy as np

from algorithms.competitors.base import Competitor, FEvalCounter, clip
from algorithms.competitors.hho import levy


class LHHO(Competitor):
    name = "LHHO"

    def __init__(self, pop_size: int = 30):
        self.pop_size = pop_size

    def _run(self, fc: FEvalCounter, dim, lower, upper, rng):
        N = self.pop_size
        pop = rng.uniform(lower, upper, size=(N, dim))
        fit = fc.eval_population(pop)
        rabbit = pop[np.argmin(fit)].copy()
        rabbit_f = float(np.min(fit))
        max_iter = max(1, fc.max_fes // N)

        for t in range(max_iter):
            E1 = 2 * (1 - t / max_iter)
            mean_pos = pop.mean(axis=0)
            for i in range(N):
                E = E1 * (2 * rng.random() - 1)
                if abs(E) >= 1:
                    # --- Exploración con vuelo de Lévy (mejora LHHO) ---
                    q = rng.random()
                    if q >= 0.5:
                        rand = pop[rng.integers(N)]
                        new = rand - rng.random() * np.abs(rand - 2 * rng.random() * pop[i]) \
                            + 0.01 * levy(dim, rng) * (rand - pop[i])
                    else:
                        new = (rabbit - mean_pos) + 0.01 * levy(dim, rng) * (rabbit - pop[i])
                else:
                    # --- Explotación (igual estructura que HHO) ---
                    r = rng.random()
                    J = 2 * (1 - rng.random())
                    if r >= 0.5 and abs(E) >= 0.5:
                        new = (rabbit - pop[i]) - E * np.abs(J * rabbit - pop[i])
                    elif r >= 0.5:
                        new = rabbit - E * np.abs(rabbit - pop[i])
                    else:
                        base = mean_pos if abs(E) < 0.5 else pop[i]
                        Y = rabbit - E * np.abs(J * rabbit - base)
                        Y = clip(Y, lower, upper)
                        fY = fc(Y)
                        Z = Y + rng.random(dim) * levy(dim, rng)
                        Z = clip(Z, lower, upper)
                        fZ = fc(Z)
                        new = Y if fY < fZ else Z

                new = clip(new, lower, upper)
                f = fc(new)
                if f < fit[i]:
                    pop[i], fit[i] = new, f
                    if f < rabbit_f:
                        rabbit, rabbit_f = new.copy(), f
