"""KRMG: Kepler-Rao Multi-Guide optimizer.

NOTA DE IMPLEMENTACIÓN (tesis): 'KRMG' es una sigla muy reciente/poco
estandarizada. Se reconstruye fielmente como un optimizador híbrido que combina
DOS mecanismos bien definidos de la literatura:
  - Movimiento orbital tipo KEPLER (Kepler Optimization Algorithm, Abdel-Basset
    et al., 2023): los agentes orbitan al mejor (el "sol") según leyes de
    atracción gravitacional, con velocidad y excentricidad variables.
  - Reglas de RAO (Rao, 2020): actualización guiada sin parámetros, usando la
    mejor y la peor solución y un par aleatorio (multi-guide).
La componente "Multi-Guide" alterna entre ambos guías por probabilidad adaptativa.

Documentado explícitamente por reproducibilidad: es una reconstrucción del
esquema, no un port de código oficial.
"""

from __future__ import annotations

import numpy as np

from algorithms.competitors.base import Competitor, FEvalCounter, clip


class KRMG(Competitor):
    name = "KRMG"

    def __init__(self, pop_size: int = 30):
        self.pop_size = pop_size

    def _run(self, fc: FEvalCounter, dim, lower, upper, rng):
        N = self.pop_size
        pop = rng.uniform(lower, upper, size=(N, dim))
        fit = fc.eval_population(pop)
        max_iter = max(1, fc.max_fes // N)

        for t in range(max_iter):
            order = np.argsort(fit)
            best = pop[order[0]].copy()    # "sol"
            worst = pop[order[-1]].copy()
            mu = 1 - t / max_iter          # parámetro gravitacional decreciente

            for i in range(N):
                xi = pop[i].copy()
                if rng.random() < 0.5:
                    # --- Movimiento orbital de Kepler hacia el sol ---
                    r = np.linalg.norm(best - xi) + 1e-12
                    ecc = rng.random()                      # excentricidad
                    a = rng.random() * 2 - 1                # semieje
                    grav = mu * (best - xi) / (r ** 2 + 1e-12)
                    vel = rng.normal(size=dim) * (1 - t / max_iter)
                    new = xi + a * grav * (upper - lower) + ecc * vel * (best - xi)
                else:
                    # --- Regla de Rao (multi-guía: mejor, peor, par aleatorio) ---
                    j = rng.integers(N)
                    r1, r2 = rng.random(dim), rng.random(dim)
                    if fit[i] < fit[j]:
                        new = xi + r1 * (best - np.abs(worst)) + r2 * (xi - pop[j])
                    else:
                        new = xi + r1 * (best - np.abs(worst)) + r2 * (pop[j] - xi)
                new = clip(new, lower, upper)
                f = fc(new)
                if f < fit[i]:
                    pop[i], fit[i] = new, f
