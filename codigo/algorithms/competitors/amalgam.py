"""AMALGAM: A Multialgorithm Genetically Adaptive Method.

Referencia:
    J. A. Vrugt, B. A. Robinson, "Improved evolutionary optimization from
    genetically adaptive multimethod search", PNAS 104(3), 2007.

Idea central (fiel al esquema publicado):
  - Varios algoritmos de búsqueda corren SIMULTÁNEAMENTE sobre una población común.
  - Cada generación, cada algoritmo k genera N_k descendientes; se combinan padres e
    hijos y se aplica SELECCIÓN ELITISTA global (los mejores NP sobreviven).
  - "Genéticamente adaptativo": N_k de la siguiente generación es proporcional a la
    fracción de descendientes del algoritmo k que SOBREVIVIERON la selección,
    normalizada por cuántos generó. Algoritmos productivos reciben más descendientes.
  - Aquí el cóctel de algoritmos es: DE (current-to-pbest/1), PSO y GA (SBX+mutación).
"""

from __future__ import annotations

import numpy as np

from algorithms.competitors.base import Competitor, FEvalCounter, reflect


class AMALGAM(Competitor):
    name = "AMALGAM"

    def __init__(self, pop_size: int = 30, n_min_frac: float = 0.1):
        self.pop_size = pop_size
        self.n_min_frac = n_min_frac  # fracción mínima de descendientes por algoritmo

    def _run(self, fc: FEvalCounter, dim: int, lower: float, upper: float,
             rng: np.random.Generator) -> None:
        NP, D = self.pop_size, dim
        pop = rng.uniform(lower, upper, size=(NP, D))
        fit = fc.eval_population(pop)

        # PSO necesita velocidad y memoria personal.
        vel = rng.uniform(-(upper - lower), upper - lower, size=(NP, D)) * 0.1
        pbest = pop.copy()
        pbest_fit = fit.copy()

        algos = ["de", "pso", "ga"]
        K = len(algos)
        N_k = np.full(K, NP // K)            # nº descendientes por algoritmo
        N_k[: NP - N_k.sum()] += 1           # reparte el resto
        n_min = max(1, int(self.n_min_frac * NP))

        while True:
            order = np.argsort(fit)
            gbest = pop[order[0]]
            best_pool = order[:max(2, int(0.1 * NP))]

            offspring = []
            origin = []  # algoritmo que generó cada hijo
            for k, algo in enumerate(algos):
                for _ in range(int(N_k[k])):
                    i = int(rng.integers(NP))
                    if algo == "de":
                        F = float(np.clip(rng.standard_cauchy() * 0.1 + 0.5, 0.1, 1.0))
                        CR = float(np.clip(rng.normal(0.5, 0.1), 0.0, 1.0))
                        r1 = self._pick(rng, NP, {i})
                        r2 = self._pick(rng, NP, {i, r1})
                        pb = pop[rng.choice(best_pool)]
                        v = pop[i] + F * (pb - pop[i]) + F * (pop[r1] - pop[r2])
                        child = pop[i].copy()
                        mask = rng.random(D) < CR
                        mask[rng.integers(D)] = True
                        child[mask] = v[mask]
                    elif algo == "pso":
                        w, c1, c2 = 0.7, 1.5, 1.5
                        r1v, r2v = rng.random(D), rng.random(D)
                        vel[i] = (w * vel[i] + c1 * r1v * (pbest[i] - pop[i])
                                  + c2 * r2v * (gbest - pop[i]))
                        child = pop[i] + vel[i]
                    else:  # ga: SBX + mutación gaussiana
                        r1 = self._pick(rng, NP, {i})
                        eta = 15.0
                        u = rng.random(D)
                        beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)),
                                        (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
                        child = 0.5 * ((1 + beta) * pop[i] + (1 - beta) * pop[r1])
                        if rng.random() < 0.5:
                            child = child + rng.normal(0, 0.05 * (upper - lower), D)

                    child = reflect(child, lower, upper)
                    offspring.append(child)
                    origin.append(k)

            offspring = np.array(offspring)
            off_fit = fc.eval_population(offspring)  # respeta presupuesto fila a fila
            origin = np.array(origin)

            # Combina padres + hijos y selección elitista global de NP.
            comb = np.vstack([pop, offspring])
            comb_fit = np.concatenate([fit, off_fit])
            # marca origen: padres = -1, hijos = origin[k]
            comb_origin = np.concatenate([np.full(NP, -1), origin])
            sel = np.argsort(comb_fit)[:NP]

            pop, fit = comb[sel], comb_fit[sel]
            # Tras la selección elitista, pbest = nueva población (ya son los mejores).
            pbest = pop.copy()
            pbest_fit = fit.copy()
            # vel tiene tamaño NP (corresponde a las ranuras de la población padre);
            # se reusa para la siguiente generación de PSO.

            # --- Adaptación genética: supervivientes por algoritmo.
            survived_origin = comb_origin[sel]
            survivors = np.array([(survived_origin == k).sum() for k in range(K)])
            produced = np.array([max(1, int(N_k[k])) for k in range(K)])
            ratio = survivors / produced
            if ratio.sum() > 0:
                share = ratio / ratio.sum()
            else:
                share = np.full(K, 1.0 / K)
            N_new = np.maximum(n_min, np.round(share * NP).astype(int))
            # Ajusta para que sumen NP.
            while N_new.sum() > NP:
                N_new[np.argmax(N_new)] -= 1
            while N_new.sum() < NP:
                N_new[np.argmin(N_new)] += 1
            N_k = N_new

    @staticmethod
    def _pick(rng, n, exclude):
        while True:
            r = int(rng.integers(n))
            if r not in exclude:
                return r
