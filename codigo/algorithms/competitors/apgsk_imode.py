"""APGSK-IMODE: combinación de APGSK e IMODE (CEC 2021).

Referencias:
    - A. W. Mohamed et al., "Gaining-Sharing Knowledge Based Algorithm (GSK)",
      Int. J. Mach. Learn. & Cyber., 2020.
    - A. W. Mohamed et al., "Evaluating the Performance of Adaptive Gaining-Sharing
      Knowledge Based Algorithm on CEC 2020", 2020 (APGSK).
    - K. M. Sallam et al., "Improved Multi-operator Differential Evolution Algorithm
      for Solving Unconstrained Problems" (IMODE), CEC 2020.
    - Híbrido APGSK-IMODE presentado en la competición CEC 2021.

Estructura del híbrido (fiel al esquema publicado):
  La población total se reparte entre dos subalgoritmos —APGSK e IMODE— y el
  reparto se adapta según cuál genera mayor mejora normalizada de fitness
  (diversificación de presupuesto guiada por calidad).

Componentes:
  APGSK: dos fases — Junior (gaining/sharing local con vecinos ordenados) y
         Senior (gaining/sharing con mejores/medios/peores). Probabilidad de
         conocimiento kf, kr; ratio junior->senior según progreso.
  IMODE: 3 estrategias DE (ctb1, ctr1, ord) con asignación adaptativa de población
         por mejora, parámetros F/CR muestreados de pools.
"""

from __future__ import annotations

import numpy as np

from algorithms.competitors.base import Competitor, FEvalCounter, reflect


class APGSK_IMODE(Competitor):
    name = "APGSK-IMODE"

    def __init__(self, pop_size: int = 30):
        self.pop_size = pop_size

    def _run(self, fc: FEvalCounter, dim: int, lower: float, upper: float,
             rng: np.random.Generator) -> None:
        NP = self.pop_size
        pop = rng.uniform(lower, upper, size=(NP, dim))
        fit = fc.eval_population(pop)

        # Reparto adaptativo de población entre los dos subalgoritmos.
        # improvement[k] acumula la mejora normalizada lograda por el subalg k.
        alloc = np.array([0.5, 0.5])  # [APGSK, IMODE]
        improvement = np.array([1e-12, 1e-12])

        gen = 0
        while True:
            gen += 1
            # Particiona índices según asignación actual.
            n_gsk = int(np.clip(round(alloc[0] * NP), 1, NP - 1))
            order = rng.permutation(NP)
            idx_gsk, idx_imode = order[:n_gsk], order[n_gsk:]

            gain_gsk = self._apgsk_step(fc, pop, fit, idx_gsk, lower, upper, rng, gen)
            gain_imode = self._imode_step(fc, pop, fit, idx_imode, lower, upper, rng)

            # Actualiza mejora normalizada y reasigna presupuesto poblacional.
            improvement[0] = 0.5 * improvement[0] + 0.5 * max(gain_gsk, 0.0)
            improvement[1] = 0.5 * improvement[1] + 0.5 * max(gain_imode, 0.0)
            total = improvement.sum()
            if total > 0:
                alloc = 0.8 * alloc + 0.2 * (improvement / total)
                alloc = np.clip(alloc, 0.1, 0.9)
                alloc /= alloc.sum()

    # ------------------------------------------------------------------ APGSK
    def _apgsk_step(self, fc, pop, fit, idx, lower, upper, rng, gen) -> float:
        """Una actualización GSK adaptativa sobre los índices `idx`.
        Devuelve la mejora total de fitness lograda (>=0)."""
        if len(idx) == 0:
            return 0.0
        D = pop.shape[1]
        NP = pop.shape[0]

        # Parámetros de conocimiento (valores típicos del paper APGSK).
        kf = rng.choice([0.1, 1.0, 0.5, 1.0], size=len(idx))  # knowledge factor
        kr = rng.choice([0.2, 0.1, 0.9, 0.9], size=len(idx))  # knowledge ratio

        # Ratio dimensiones junior vs senior: decrece con el progreso (k=10 típico).
        order = np.argsort(fit)
        rank = np.empty(NP, dtype=int)
        rank[order] = np.arange(NP)

        gain = 0.0
        for local_j, i in enumerate(idx):
            # Número de dimensiones en fase junior (resto, senior).
            ratio = max(0.0, 1.0 - gen / 100.0)  # decae el componente junior
            d_junior = int(round(D * ratio))
            dims = rng.permutation(D)
            junior_dims, senior_dims = dims[:d_junior], dims[d_junior:]

            xi = pop[i].copy()
            new = xi.copy()

            # ---- Junior gaining-sharing: vecinos por ranking (mejor/peor cercano).
            r = rank[i]
            better = order[max(0, r - 1)]
            worse = order[min(NP - 1, r + 1)]
            rand_idx = int(rng.integers(NP))
            for d in junior_dims:
                if rng.random() <= kr[local_j]:
                    if fit[i] > fit[rand_idx]:
                        new[d] = xi[d] + kf[local_j] * (
                            (pop[better, d] - pop[worse, d]) + (pop[rand_idx, d] - xi[d]))
                    else:
                        new[d] = xi[d] + kf[local_j] * (
                            (pop[better, d] - pop[worse, d]) + (xi[d] - pop[rand_idx, d]))

            # ---- Senior gaining-sharing: mejores(p), medios(m), peores(p).
            p = max(1, int(0.1 * NP))
            best_pool = order[:p]
            worst_pool = order[-p:]
            mid_pool = order[p:NP - p] if NP - 2 * p > 0 else order
            pb = pop[rng.choice(best_pool)]
            pm = pop[rng.choice(mid_pool)]
            pw = pop[rng.choice(worst_pool)]
            for d in senior_dims:
                if rng.random() <= kr[local_j]:
                    if fit[i] > self._f_of(fit, mid_pool, rng):
                        new[d] = xi[d] + kf[local_j] * (
                            (pb[d] - pm[d]) + (pm[d] - xi[d]))
                    else:
                        new[d] = xi[d] + kf[local_j] * (
                            (pb[d] - pm[d]) + (xi[d] - pw[d]))

            new = reflect(new, lower, upper)
            fnew = fc(new)
            if fnew < fit[i]:
                gain += fit[i] - fnew
                pop[i] = new
                fit[i] = fnew
        return gain

    # ------------------------------------------------------------------ IMODE
    def _imode_step(self, fc, pop, fit, idx, lower, upper, rng) -> float:
        """Una actualización IMODE (multi-operador DE) sobre `idx`."""
        if len(idx) == 0:
            return 0.0
        NP, D = pop.shape
        order = np.argsort(fit)
        p_num = max(2, int(0.1 * NP))
        best_pool = order[:p_num]

        # Pools de parámetros (IMODE muestrea F y CR de distribuciones).
        gain = 0.0
        for i in idx:
            strat = int(rng.integers(3))  # 0:ctb1, 1:ctr1, 2:weighted-rand-to-pbest
            F = float(np.clip(rng.standard_cauchy() * 0.1 + 0.5, 0.1, 1.0))
            CR = float(np.clip(rng.normal(0.5, 0.1), 0.0, 1.0))

            r1 = self._pick(rng, NP, {i})
            r2 = self._pick(rng, NP, {i, r1})
            pbest = pop[rng.choice(best_pool)]
            x = pop[i]
            if strat == 0:    # current-to-pbest/1
                v = x + F * (pbest - x) + F * (pop[r1] - pop[r2])
            elif strat == 1:  # current-to-rand/1 (rotación-invariante)
                r3 = self._pick(rng, NP, {i, r1, r2})
                K = rng.random()
                v = x + K * (pop[r3] - x) + F * (pop[r1] - pop[r2])
            else:             # weighted rand-to-pbest
                v = pop[r1] + F * (pbest - pop[r1]) + F * (pop[r2] - x)
            v = reflect(v, lower, upper)

            u = x.copy()
            jrand = rng.integers(D)
            cross = rng.random(D) < CR
            cross[jrand] = True
            u[cross] = v[cross]

            fu = fc(u)
            if fu < fit[i]:
                gain += fit[i] - fu
                pop[i] = u
                fit[i] = fu
        return gain

    @staticmethod
    def _f_of(fit, pool, rng):
        return fit[int(rng.choice(pool))]

    @staticmethod
    def _pick(rng, n, exclude):
        while True:
            r = int(rng.integers(n))
            if r not in exclude:
                return r
