"""LSHADE: Success-History based Adaptive DE with Linear Population Size Reduction.

Referencia:
    R. Tanabe, A. S. Fukunaga, "Improving the search performance of SHADE using
    linear population size reduction", IEEE CEC 2014.

Extiende SHADE (memoria histórica de F y CR) con LPSR: el tamaño de población
decrece linealmente de N_init a N_min conforme se consumen FEs, concentrando el
presupuesto en explotación final. Mutación current-to-pbest/1 con archivo externo.
Es la base de toda la familia L-SHADE (incluida L-SHADE-EpSin).
"""

from __future__ import annotations

import numpy as np

from algorithms.competitors.base import Competitor, FEvalCounter, reflect


class LSHADE(Competitor):
    name = "LSHADE"

    def __init__(self, n_init_factor: int = 18, n_min: int = 4,
                 memory_size: int = 6, p_best: float = 0.11, arc_rate: float = 1.4):
        self.n_init_factor = n_init_factor   # N_init = factor * dim
        self.n_min = n_min
        self.H = memory_size
        self.p_best = p_best
        self.arc_rate = arc_rate

    def _run(self, fc: FEvalCounter, dim, lower, upper, rng):
        N_init = self.n_init_factor * dim
        N = N_init
        pop = rng.uniform(lower, upper, size=(N, dim))
        fit = fc.eval_population(pop)

        archive = np.empty((0, dim))
        M_F = np.full(self.H, 0.5)
        M_CR = np.full(self.H, 0.5)
        k = 0  # puntero de memoria

        while True:
            S_F, S_CR, S_w = [], [], []
            order = np.argsort(fit)
            p_num = max(2, int(self.p_best * N))
            pbest_pool = order[:p_num]

            new_pop = pop.copy()
            new_fit = fit.copy()
            for i in range(N):
                ri = rng.integers(self.H)
                # CR ~ N(M_CR, 0.1); F ~ Cauchy(M_F, 0.1)
                CR = 0.0 if M_CR[ri] < 0 else float(np.clip(rng.normal(M_CR[ri], 0.1), 0, 1))
                F = -1.0
                while F <= 0:
                    F = M_F[ri] + 0.1 * np.tan(np.pi * (rng.random() - 0.5))
                F = min(F, 1.0)

                pbest = pop[rng.choice(pbest_pool)]
                # r1 de población, r2 de población ∪ archivo (ambos != i)
                r1 = _pick(rng, N, {i})
                P_un = np.vstack([pop, archive]) if len(archive) else pop
                r2 = _pick(rng, len(P_un), {i, r1})
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - P_un[r2])
                v = reflect(v, lower, upper)

                # cruce binomial
                u = pop[i].copy()
                jr = rng.integers(dim)
                mask = rng.random(dim) < CR
                mask[jr] = True
                u[mask] = v[mask]

                fu = fc(u)
                if fu < fit[i]:
                    new_pop[i] = u
                    new_fit[i] = fu
                    # archivo recibe el padre reemplazado
                    if len(archive) < int(self.arc_rate * N):
                        archive = np.vstack([archive, pop[i]])
                    else:
                        archive[rng.integers(len(archive))] = pop[i]
                    S_F.append(F); S_CR.append(CR)
                    S_w.append(fit[i] - fu)

            pop, fit = new_pop, new_fit

            # --- Actualiza memorias (media de Lehmer ponderada por mejora) ---
            if S_F:
                w = np.array(S_w); w = w / (w.sum() + 1e-30)
                sf = np.array(S_F); scr = np.array(S_CR)
                M_F[k] = np.sum(w * sf ** 2) / (np.sum(w * sf) + 1e-30)
                if max(scr) == 0:
                    M_CR[k] = -1.0  # terminal value: CR fijado a 0
                else:
                    M_CR[k] = np.sum(w * scr ** 2) / (np.sum(w * scr) + 1e-30)
                k = (k + 1) % self.H

            # --- LPSR: reduce el tamaño de población según FEs consumidos ---
            ratio = fc.fes / fc.max_fes
            N_new = round(N_init + (self.n_min - N_init) * ratio)
            N_new = max(self.n_min, N_new)
            if N_new < N:
                keep = np.argsort(fit)[:N_new]
                pop, fit = pop[keep], fit[keep]
                N = N_new
                maxA = int(self.arc_rate * N)
                if len(archive) > maxA:
                    sel = rng.choice(len(archive), maxA, replace=False)
                    archive = archive[sel]


def _pick(rng, n, exclude):
    while True:
        r = int(rng.integers(n))
        if r not in exclude:
            return r
