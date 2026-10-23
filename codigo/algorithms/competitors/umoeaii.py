"""UMOEAII: United Multi-Operator Evolutionary Algorithms II (CEC 2016).

Referencia:
    S. M. Elsayed, R. A. Sarker, D. L. Essam, N. M. Hamza, "United Multi-Operator
    Evolutionary Algorithms", IEEE CEC 2014; y su versión mejorada UMOEAs-II,
    CEC 2016.

Estructura (fiel al esquema publicado):
  - DOS sub-poblaciones, cada una gestionada por un Multi-Operator EA (MOEA):
      MOEA-1: multi-operador de Evolución Diferencial (varias estrategias DE).
      MOEA-2: multi-operador tipo Algoritmo Genético (cruces SBX/aritmético + mutación).
  - El presupuesto poblacional se reparte adaptativamente entre las dos según su
    mejora de fitness reciente (diversificación de recursos guiada por calidad).
  - Intercambio de información: periódicamente la peor sub-población recibe los
    mejores individuos de la otra.
  - Dentro de cada MOEA, los operadores compiten por probabilidad de uso según éxito.
"""

from __future__ import annotations

import numpy as np

from algorithms.competitors.base import Competitor, FEvalCounter, reflect


class UMOEAII(Competitor):
    name = "UMOEAII"

    def __init__(self, pop_size: int = 30, exchange_every: int = 10):
        self.pop_size = pop_size
        self.exchange_every = exchange_every

    def _run(self, fc: FEvalCounter, dim: int, lower: float, upper: float,
             rng: np.random.Generator) -> None:
        NP, D = self.pop_size, dim
        # Reparto adaptativo entre las dos sub-poblaciones (DE-MOEA, GA-MOEA).
        alloc = np.array([0.5, 0.5])
        improvement = np.array([1e-12, 1e-12])

        pop = rng.uniform(lower, upper, size=(NP, D))
        fit = fc.eval_population(pop)

        # Probabilidades de operador dentro de cada MOEA.
        de_op_succ = np.ones(3)   # ctpbest, rand1, ctr1
        ga_op_succ = np.ones(2)   # sbx, arithmetic

        gen = 0
        while True:
            gen += 1
            n1 = int(np.clip(round(alloc[0] * NP), 2, NP - 2))
            order = rng.permutation(NP)
            idx1, idx2 = order[:n1], order[n1:]

            g1 = self._de_moea(fc, pop, fit, idx1, lower, upper, rng, de_op_succ)
            g2 = self._ga_moea(fc, pop, fit, idx2, lower, upper, rng, ga_op_succ)

            improvement[0] = 0.5 * improvement[0] + 0.5 * max(g1, 0.0)
            improvement[1] = 0.5 * improvement[1] + 0.5 * max(g2, 0.0)
            tot = improvement.sum()
            if tot > 0:
                alloc = 0.8 * alloc + 0.2 * (improvement / tot)
                alloc = np.clip(alloc, 0.15, 0.85)
                alloc /= alloc.sum()

            # Intercambio de información entre sub-poblaciones.
            if gen % self.exchange_every == 0:
                order_f = np.argsort(fit)
                best = pop[order_f[0]].copy()
                worst_i = order_f[-1]
                if fc.best_fitness < fit[worst_i]:
                    pop[worst_i] = best  # propaga el mejor a la peor ranura

    # ------------------------------------------------------------- DE-MOEA
    def _de_moea(self, fc, pop, fit, idx, lower, upper, rng, op_succ) -> float:
        if len(idx) == 0:
            return 0.0
        NP, D = pop.shape
        order = np.argsort(fit)
        best_pool = order[:max(2, int(0.1 * NP))]
        probs = op_succ / op_succ.sum()
        gain = 0.0
        for i in idx:
            op = int(rng.choice(3, p=probs))
            F = float(np.clip(rng.standard_cauchy() * 0.1 + 0.5, 0.1, 1.0))
            CR = float(np.clip(rng.normal(0.5, 0.1), 0.0, 1.0))
            r1 = self._pick(rng, NP, {i})
            r2 = self._pick(rng, NP, {i, r1})
            x = pop[i]
            if op == 0:
                pbest = pop[rng.choice(best_pool)]
                v = x + F * (pbest - x) + F * (pop[r1] - pop[r2])
            elif op == 1:
                r3 = self._pick(rng, NP, {i, r1, r2})
                v = pop[r1] + F * (pop[r2] - pop[r3])
            else:
                r3 = self._pick(rng, NP, {i, r1, r2})
                K = rng.random()
                v = x + K * (pop[r3] - x) + F * (pop[r1] - pop[r2])
            v = reflect(v, lower, upper)
            u = x.copy()
            mask = rng.random(D) < CR
            mask[rng.integers(D)] = True
            u[mask] = v[mask]
            fu = fc(u)
            if fu < fit[i]:
                op_succ[op] = 0.9 * op_succ[op] + 0.1 * (fit[i] - fu) + 1e-9
                gain += fit[i] - fu
                pop[i] = u
                fit[i] = fu
        return gain

    # ------------------------------------------------------------- GA-MOEA
    def _ga_moea(self, fc, pop, fit, idx, lower, upper, rng, op_succ) -> float:
        if len(idx) == 0:
            return 0.0
        NP, D = pop.shape
        probs = op_succ / op_succ.sum()
        eta = 15.0  # índice de distribución SBX
        gain = 0.0
        for i in idx:
            op = int(rng.choice(2, p=probs))
            r1 = self._pick(rng, NP, {i})
            x1, x2 = pop[i], pop[r1]
            if op == 0:  # SBX
                u = rng.random(D)
                beta = np.where(u <= 0.5,
                                (2 * u) ** (1 / (eta + 1)),
                                (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
                child = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
            else:        # cruce aritmético
                a = rng.random()
                child = a * x1 + (1 - a) * x2
            # Mutación gaussiana ligera.
            if rng.random() < 0.5:
                child = child + rng.normal(0, 0.05 * (upper - lower), D)
            child = reflect(child, lower, upper)
            fc_val = fc(child)
            if fc_val < fit[i]:
                op_succ[op] = 0.9 * op_succ[op] + 0.1 * (fit[i] - fc_val) + 1e-9
                gain += fit[i] - fc_val
                pop[i] = child
                fit[i] = fc_val
        return gain

    @staticmethod
    def _pick(rng, n, exclude):
        while True:
            r = int(rng.integers(n))
            if r not in exclude:
                return r
