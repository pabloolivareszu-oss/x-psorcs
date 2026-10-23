"""FGIN: Fractional-order Gradient-Inspired Navigation optimizer.

NOTA DE IMPLEMENTACIÓN (tesis): 'FGIN' es una sigla muy reciente/poco
estandarizada. Se reconstruye fielmente como un optimizador inspirado en gradiente
con memoria FRACCIONARIA, que combina dos ideas bien fundamentadas:
  - Gradient-based search (Gradient-Based Optimizer, Ahmadianfar et al., 2020):
    dirección de mejora estimada por diferencias entre el mejor y el peor (regla
    de Newton aproximada sin derivadas analíticas).
  - Memoria de orden fraccionario: el paso incorpora una combinación ponderada de
    los desplazamientos previos (coeficientes binomiales de la derivada fraccional
    de Grünwald-Letnikov), dando "navegación" con memoria larga.

Documentado por reproducibilidad: reconstrucción del esquema, no port oficial.
"""

from __future__ import annotations

import numpy as np

from algorithms.competitors.base import Competitor, FEvalCounter, clip


class FGIN(Competitor):
    name = "FGIN"

    def __init__(self, pop_size: int = 30, frac_order: float = 0.9, memory: int = 3):
        self.pop_size = pop_size
        self.alpha = frac_order
        self.memory = memory

    def _frac_weights(self, m):
        """Coeficientes de Grünwald-Letnikov para la derivada fraccional de orden alpha."""
        w = [1.0]
        for j in range(1, m):
            w.append(w[-1] * (1 - (self.alpha + 1) / j))
        return np.array(w)

    def _run(self, fc: FEvalCounter, dim, lower, upper, rng):
        N = self.pop_size
        pop = rng.uniform(lower, upper, size=(N, dim))
        fit = fc.eval_population(pop)
        max_iter = max(1, fc.max_fes // N)
        # historial de desplazamientos por individuo (memoria fraccionaria)
        history = [np.zeros((0, dim)) for _ in range(N)]
        fw = self._frac_weights(self.memory)

        for t in range(max_iter):
            order = np.argsort(fit)
            best = pop[order[0]].copy()
            worst = pop[order[-1]].copy()
            rate = 1 - t / max_iter

            for i in range(N):
                xi = pop[i].copy()
                # --- Dirección inspirada en gradiente (best - worst) ---
                denom = (best - worst)
                denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
                grad_dir = (best - xi) * np.abs(best - worst) / np.abs(denom)
                rand_term = rng.normal(size=dim) * rate * (upper - lower) * 0.1
                step = rate * rng.random() * grad_dir + rand_term

                # --- Memoria fraccionaria: pondera desplazamientos previos ---
                if len(history[i]) > 0:
                    m = min(len(history[i]), self.memory)
                    frac = np.zeros(dim)
                    for j in range(m):
                        frac += fw[j] * history[i][-1 - j]
                    step = step + 0.3 * frac

                new = clip(xi + step, lower, upper)
                f = fc(new)
                disp = new - xi
                if f < fit[i]:
                    pop[i], fit[i] = new, f
                    history[i] = np.vstack([history[i], disp])[-self.memory:]
