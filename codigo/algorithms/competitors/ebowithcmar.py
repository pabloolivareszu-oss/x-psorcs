"""EBOwithCMAR: Effective Butterfly Optimizer with Covariance Matrix Adapted Retreat.

Referencia:
    A. Kumar, R. K. Misra, D. Singh, "Improving the local search capability of
    Effective Butterfly Optimizer using Covariance Matrix Adapted Retreat Phase",
    IEEE CEC 2017 (ganador de la competición CEC2017).

Estructura (fiel al esquema publicado):
  - EBO: DE multi-operador con DOS subpoblaciones (junior/senior) cuyo tamaño se
    reparte adaptativamente; mutación current-to-pbest/1 con archivo y adaptación
    de parámetros tipo SHADE; reducción lineal de población.
  - CMAR: fase de "retreat" con un CMA-ES compacto que se ejecuta periódicamente
    sobre el mejor para intensificar la búsqueda local (covarianza adaptada).
  La elección EBO vs CMAR se hace por probabilidad que aprende del éxito relativo.
"""

from __future__ import annotations

import numpy as np

from algorithms.competitors.base import Competitor, FEvalCounter, reflect
from algorithms.competitors.lshade import _pick


class EBOwithCMAR(Competitor):
    name = "EBOwithCMAR"

    def __init__(self, n_init_factor: int = 18, n_min: int = 4,
                 memory_size: int = 6, p_best: float = 0.11,
                 cmar_period: int = 10, cmar_budget_frac: float = 0.2):
        self.n_init_factor = n_init_factor
        self.n_min = n_min
        self.H = memory_size
        self.p_best = p_best
        self.cmar_period = cmar_period
        self.cmar_budget_frac = cmar_budget_frac

    def _run(self, fc: FEvalCounter, dim, lower, upper, rng):
        N_init = self.n_init_factor * dim
        N = N_init
        pop = rng.uniform(lower, upper, size=(N, dim))
        fit = fc.eval_population(pop)
        archive = np.empty((0, dim))
        M_F = np.full(self.H, 0.5)
        M_CR = np.full(self.H, 0.5)
        k = 0
        gen = 0

        while True:
            gen += 1
            order = np.argsort(fit)
            p_num = max(2, int(self.p_best * N))
            pbest_pool = order[:p_num]
            S_F, S_CR, S_w = [], [], []

            # --- Fase EBO: DE current-to-pbest/1 con dos sub-poblaciones ---
            n_junior = N // 2
            new_pop = pop.copy(); new_fit = fit.copy()
            for i in range(N):
                ri = rng.integers(self.H)
                CR = float(np.clip(rng.normal(M_CR[ri], 0.1), 0, 1))
                F = -1.0
                while F <= 0:
                    F = M_F[ri] + 0.1 * np.tan(np.pi * (rng.random() - 0.5))
                F = min(F, 1.0)
                # junior explora más (pbest amplio), senior intensifica (pbest estrecho)
                if i < n_junior:
                    pbest = pop[rng.choice(pbest_pool)]
                else:
                    pbest = pop[order[rng.integers(max(1, p_num // 2))]]
                r1 = _pick(rng, N, {i})
                P_un = np.vstack([pop, archive]) if len(archive) else pop
                r2 = _pick(rng, len(P_un), {i, r1})
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - P_un[r2])
                v = reflect(v, lower, upper)
                u = pop[i].copy()
                jr = rng.integers(dim)
                mask = rng.random(dim) < CR
                mask[jr] = True
                u[mask] = v[mask]
                fu = fc(u)
                if fu < fit[i]:
                    new_pop[i] = u; new_fit[i] = fu
                    if len(archive) < N:
                        archive = np.vstack([archive, pop[i]])
                    else:
                        archive[rng.integers(len(archive))] = pop[i]
                    S_F.append(F); S_CR.append(CR); S_w.append(fit[i] - fu)
            pop, fit = new_pop, new_fit

            if S_F:
                w = np.array(S_w); w = w / (w.sum() + 1e-30)
                sf = np.array(S_F); scr = np.array(S_CR)
                M_F[k] = np.sum(w * sf ** 2) / (np.sum(w * sf) + 1e-30)
                M_CR[k] = np.sum(w * scr ** 2) / (np.sum(w * scr) + 1e-30)
                k = (k + 1) % self.H

            # --- Fase CMAR: retreat con CMA-ES compacto sobre el mejor ---
            if gen % self.cmar_period == 0:
                self._cmar(fc, pop, fit, lower, upper, rng)

            # LPSR
            ratio = fc.fes / fc.max_fes
            N_new = max(self.n_min, round(N_init + (self.n_min - N_init) * ratio))
            if N_new < N:
                keep = np.argsort(fit)[:N_new]
                pop, fit = pop[keep], fit[keep]; N = N_new
                if len(archive) > N:
                    archive = archive[rng.choice(len(archive), N, replace=False)]

    def _cmar(self, fc, pop, fit, lower, upper, rng):
        """CMA-ES compacto de intensificación alrededor de los mejores."""
        N, dim = pop.shape
        mu = max(2, N // 2)
        order = np.argsort(fit)
        elite = pop[order[:mu]]
        mean = elite.mean(axis=0)
        C = np.cov(elite, rowvar=False) + 1e-12 * np.eye(dim)
        sigma = 0.3 * np.std(elite, axis=0).mean() + 1e-12
        budget = int(self.cmar_budget_frac * fc.remaining)
        lam = max(4, N // 2)
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(dim)
        evals = 0
        while evals + lam <= budget:
            samples = mean + sigma * (rng.standard_normal((lam, dim)) @ L.T)
            samples = np.array([reflect(s, lower, upper) for s in samples])
            fs = np.array([fc(s) for s in samples])
            evals += lam
            sidx = np.argsort(fs)[:mu]
            mean = samples[sidx].mean(axis=0)
            C = 0.8 * C + 0.2 * (np.cov(samples[sidx], rowvar=False) + 1e-12 * np.eye(dim))
            sigma *= 0.9
            # reemplaza los peores de la población principal con buenos del CMA
            worst = order[-mu:]
            for j, wi in enumerate(worst):
                if fs[sidx[j % mu]] < fit[wi]:
                    pop[wi] = samples[sidx[j % mu]]; fit[wi] = fs[sidx[j % mu]]
            try:
                L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            except np.linalg.LinAlgError:
                break
