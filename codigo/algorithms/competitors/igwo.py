"""IGWO: Improved Grey Wolf Optimizer (con Dimension Learning-based Hunting).

Referencia:
    M. H. Nadimi-Shahraki, S. Taghian, S. Mirjalili, "An improved grey wolf
    optimizer for solving engineering problems", Expert Systems with Applications,
    166, 2021.

Mejora sobre GWO: además del candidato GWO clásico (X_GWO), genera un candidato
por "dimension learning" (DLH): cada individuo aprende de un vecino aleatorio del
radio de vecindad. Se selecciona el mejor entre X_GWO y X_DLH (selección greedy).
Esto mejora diversidad y balance respecto al GWO estándar.
"""

from __future__ import annotations

import numpy as np

from algorithms.competitors.base import Competitor, FEvalCounter, clip


class IGWO(Competitor):
    name = "IGWO"

    def __init__(self, pop_size: int = 30):
        self.pop_size = pop_size

    def _run(self, fc: FEvalCounter, dim, lower, upper, rng):
        N = self.pop_size
        pop = rng.uniform(lower, upper, size=(N, dim))
        fit = fc.eval_population(pop)
        max_iter = max(1, fc.max_fes // (2 * N))  # ~2 evals por individuo/iter

        for t in range(max_iter):
            order = np.argsort(fit)
            alpha, beta, delta = pop[order[0]], pop[order[1]], pop[order[2]]
            a = 2 - 2 * t / max_iter

            for i in range(N):
                # --- Candidato GWO clásico ---
                r1 = rng.random(dim); r2 = rng.random(dim)
                A1 = 2 * a * r1 - a; C1 = 2 * r2
                X1 = alpha - A1 * np.abs(C1 * alpha - pop[i])
                r1 = rng.random(dim); r2 = rng.random(dim)
                A2 = 2 * a * r1 - a; C2 = 2 * r2
                X2 = beta - A2 * np.abs(C2 * beta - pop[i])
                r1 = rng.random(dim); r2 = rng.random(dim)
                A3 = 2 * a * r1 - a; C3 = 2 * r2
                X3 = delta - A3 * np.abs(C3 * delta - pop[i])
                X_gwo = clip((X1 + X2 + X3) / 3.0, lower, upper)

                # --- Candidato DLH (dimension learning-based hunting) ---
                R = np.linalg.norm(pop[i] - X_gwo)  # radio de vecindad
                dists = np.linalg.norm(pop - pop[i], axis=1)
                neighbors = np.where(dists <= R)[0]
                if len(neighbors) == 0:
                    neighbors = np.array([rng.integers(N)])
                X_dlh = pop[i].copy()
                for d in range(dim):
                    n_idx = neighbors[rng.integers(len(neighbors))]
                    r_idx = rng.integers(N)
                    X_dlh[d] = pop[i, d] + rng.random() * (pop[n_idx, d] - pop[r_idx, d])
                X_dlh = clip(X_dlh, lower, upper)

                # --- Selección greedy entre los dos candidatos ---
                f_gwo = fc(X_gwo)
                f_dlh = fc(X_dlh)
                if f_gwo < f_dlh:
                    cand, fcand = X_gwo, f_gwo
                else:
                    cand, fcand = X_dlh, f_dlh
                if fcand < fit[i]:
                    pop[i], fit[i] = cand, fcand
