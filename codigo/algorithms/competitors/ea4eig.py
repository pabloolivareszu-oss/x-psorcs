"""EA4eig: Ensemble de algoritmos con cruce por eigenvectores (CEC 2022 winner).

Referencia:
    P. Bujok, R. Poláková, "Eigen Crossover in Cooperative Model of Evolutionary
    Algorithms Applied to CEC 2022 Single Objective Numerical Optimisation",
    IEEE CEC 2022.

Idea central (fiel al esquema publicado):
  - Ensemble cooperativo de varios EAs que comparten UNA población. Aquí incluimos:
      h1: CoBiDE-like  (DE bin con cruce ocasional en base de eigenvectores)
      h2: DE current-to-pbest/1 con archivo (componente tipo jSO/SHADE)
      h3: CMA-ES compacto (muestreo gaussiano con covarianza adaptada)
  - Selección competitiva: cada EA tiene una probabilidad de uso que se actualiza
    según su éxito reciente (mejora de fitness por evaluación). Empiezan equiprob.
  - Eigen crossover: con probabilidad pe, el cruce DE se realiza en la base de
    eigenvectores de la matriz de covarianza de los mejores individuos (rotación
    invariante), que es la marca distintiva de EA4eig.
"""

from __future__ import annotations

import numpy as np

from algorithms.competitors.base import Competitor, FEvalCounter, reflect


class EA4eig(Competitor):
    name = "EA4eig"

    def __init__(self, pop_size: int = 30, pe: float = 0.4, n_top: int = None):
        self.pop_size = pop_size
        self.pe = pe          # probabilidad de cruce en base de eigenvectores
        self.n_top = n_top    # nº de individuos para estimar la covarianza

    def _run(self, fc: FEvalCounter, dim: int, lower: float, upper: float,
             rng: np.random.Generator) -> None:
        NP, D = self.pop_size, dim
        pop = rng.uniform(lower, upper, size=(NP, D))
        fit = fc.eval_population(pop)
        archive: list[np.ndarray] = []

        n_top = self.n_top or max(D + 1, NP // 2)
        heuristics = ["cobide", "ctpbest", "cmaes"]
        n_h = len(heuristics)
        success = np.full(n_h, 1.0)   # éxito acumulado (suavizado)
        usage = np.full(n_h, 1.0)
        # Estado CMA-ES compacto.
        cma_mean = pop[np.argmin(fit)].copy()
        cma_sigma = 0.3 * (upper - lower)
        C = np.eye(D)

        while True:
            # Base de eigenvectores de la covarianza de los mejores (eigen crossover).
            top = pop[np.argsort(fit)[:n_top]]
            try:
                cov = np.cov(top, rowvar=False) + 1e-12 * np.eye(D)
                _eigvals, B = np.linalg.eigh(cov)
                B = np.real(B)
            except np.linalg.LinAlgError:
                B = np.eye(D)

            probs = success / success.sum()
            h = int(rng.choice(n_h, p=probs))
            usage[h] += NP

            order = np.argsort(fit)
            p_num = max(2, int(0.1 * NP))
            best_pool = order[:p_num]
            gain = 0.0

            if heuristics[h] == "cmaes":
                gain += self._cmaes_generation(
                    fc, pop, fit, cma_mean, cma_sigma, C, lower, upper, rng)
                # Reajusta mean/sigma a partir de los nuevos mejores.
                order = np.argsort(fit)
                mu = max(1, NP // 2)
                cma_mean = pop[order[:mu]].mean(axis=0)
                cma_sigma *= 0.98
                C = np.cov(pop[order[:mu]], rowvar=False) + 1e-12 * np.eye(D)
            else:
                for i in range(NP):
                    F = float(np.clip(rng.standard_cauchy() * 0.1 + 0.5, 0.1, 1.0))
                    CR = float(np.clip(rng.normal(0.5, 0.1), 0.0, 1.0))
                    r1 = self._pick(rng, NP, {i})
                    P_union = np.vstack([pop, np.array(archive)]) if archive else pop
                    r2 = self._pick(rng, len(P_union), {i, r1})
                    x = pop[i]
                    if heuristics[h] == "ctpbest":
                        pbest = pop[rng.choice(best_pool)]
                        v = x + F * (pbest - x) + F * (pop[r1] - P_union[r2])
                    else:  # cobide: rand/1
                        r3 = self._pick(rng, NP, {i, r1})
                        v = pop[r1] + F * (pop[r3] - P_union[r2])
                    v = reflect(v, lower, upper)

                    # Cruce: binomial estándar o en base de eigenvectores (eigen xover).
                    if rng.random() < self.pe:
                        xb = B.T @ x
                        vb = B.T @ v
                        ub = xb.copy()
                        mask = rng.random(D) < CR
                        mask[rng.integers(D)] = True
                        ub[mask] = vb[mask]
                        u = B @ ub
                    else:
                        u = x.copy()
                        mask = rng.random(D) < CR
                        mask[rng.integers(D)] = True
                        u[mask] = v[mask]
                    u = reflect(u, lower, upper)

                    fu = fc(u)
                    if fu < fit[i]:
                        gain += fit[i] - fu
                        if len(archive) < NP:
                            archive.append(x.copy())
                        else:
                            archive[rng.integers(NP)] = x.copy()
                        pop[i] = u
                        fit[i] = fu

            # Actualiza éxito del heurístico (mejora por evaluación, suavizada).
            success[h] = 0.9 * success[h] + 0.1 * (gain / max(1.0, NP)) + 1e-12

    def _cmaes_generation(self, fc, pop, fit, mean, sigma, C,
                          lower, upper, rng) -> float:
        """Una generación de muestreo gaussiano tipo CMA-ES sobre la covarianza C."""
        NP, D = pop.shape
        try:
            L = np.linalg.cholesky(C + 1e-12 * np.eye(D))
        except np.linalg.LinAlgError:
            L = np.eye(D)
        gain = 0.0
        worst_order = np.argsort(fit)[::-1]  # reemplaza los peores
        for k in range(NP):
            z = rng.standard_normal(D)
            cand = mean + sigma * (L @ z)
            cand = reflect(cand, lower, upper)
            fc_val = fc(cand)
            j = worst_order[k % NP]
            if fc_val < fit[j]:
                gain += fit[j] - fc_val
                pop[j] = cand
                fit[j] = fc_val
        return gain

    @staticmethod
    def _pick(rng, n, exclude):
        while True:
            r = int(rng.integers(n))
            if r not in exclude:
                return r
