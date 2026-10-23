"""HHO: Harris Hawks Optimization.

Referencia:
    A. A. Heidari, S. Mirjalili, H. Faris, I. Aljarah, M. Mafarja, H. Chen,
    "Harris hawks optimization: Algorithm and applications", Future Generation
    Computer Systems, 97, 2019.

Mecanismo: halcones de Harris cazan cooperativamente. Energía de la presa E decrece
con el tiempo y decide exploración (|E|>=1) vs explotación (|E|<1). En explotación,
cuatro estrategias según E y la probabilidad de escape r: soft/hard besiege con y
sin picados progresivos (rapid dives con vuelos de Lévy).
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma

from algorithms.competitors.base import Competitor, FEvalCounter, clip


def levy(dim, rng, beta=1.5):
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, sigma, dim)
    v = rng.normal(0, 1, dim)
    return u / (np.abs(v) ** (1 / beta))


class HHO(Competitor):
    name = "HHO"

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
            E1 = 2 * (1 - t / max_iter)  # energía decreciente
            mean_pos = pop.mean(axis=0)
            for i in range(N):
                E0 = 2 * rng.random() - 1
                E = E1 * E0
                if abs(E) >= 1:
                    # --- Exploración ---
                    q = rng.random()
                    if q >= 0.5:
                        rand = pop[rng.integers(N)]
                        new = rand - rng.random() * np.abs(rand - 2 * rng.random() * pop[i])
                    else:
                        new = (rabbit - mean_pos) - rng.random() * (
                            lower + rng.random() * (upper - lower))
                else:
                    # --- Explotación ---
                    r = rng.random()
                    J = 2 * (1 - rng.random())  # fuerza de salto
                    if r >= 0.5 and abs(E) >= 0.5:        # soft besiege
                        new = (rabbit - pop[i]) - E * np.abs(J * rabbit - pop[i])
                    elif r >= 0.5 and abs(E) < 0.5:       # hard besiege
                        new = rabbit - E * np.abs(rabbit - pop[i])
                    elif r < 0.5 and abs(E) >= 0.5:       # soft + rapid dives
                        Y = rabbit - E * np.abs(J * rabbit - pop[i])
                        Y = clip(Y, lower, upper)
                        fY = fc(Y)
                        Z = Y + rng.random(dim) * levy(dim, rng)
                        Z = clip(Z, lower, upper)
                        fZ = fc(Z)
                        new = Y if fY < fZ else Z
                    else:                                  # hard + rapid dives
                        Y = rabbit - E * np.abs(J * rabbit - mean_pos)
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
