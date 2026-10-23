"""JADE: Adaptive Differential Evolution with Optional External Archive.

Referencia:
    J. Zhang, A. C. Sanderson, "JADE: Adaptive Differential Evolution With
    Optional External Archive", IEEE Trans. Evol. Comput., 13(5), 2009.

JADE es la base de la familia SHADE/L-SHADE y de los componentes DE de varios
competidores CEC modernos (APGSK-IMODE, EA4eig). Lo implementamos fielmente como
competidor por derecho propio y como bloque reutilizable.

Componentes clave (del paper):
  - Mutación "DE/current-to-pbest/1" con conjunto de los p% mejores.
  - Archivo externo opcional A de soluciones padre reemplazadas.
  - Adaptación de parámetros: mu_CR (media aritmética) y mu_F (media de Lehmer),
    actualizadas con los CR_i / F_i de los individuos exitosos, con tasa c.
"""

from __future__ import annotations

import numpy as np

from algorithms.competitors.base import Competitor, FEvalCounter, reflect


class JADE(Competitor):
    name = "JADE"

    def __init__(self, pop_size: int = 30, p: float = 0.1, c: float = 0.1,
                 use_archive: bool = True):
        # p: fracción superior para current-to-pbest. c: tasa de adaptación.
        self.pop_size = pop_size
        self.p = p
        self.c = c
        self.use_archive = use_archive

    def _run(self, fc: FEvalCounter, dim: int, lower: float, upper: float,
             rng: np.random.Generator) -> None:
        NP = self.pop_size
        pop = rng.uniform(lower, upper, size=(NP, dim))
        fitness = fc.eval_population(pop)  # consume NP FEs

        archive: list[np.ndarray] = []
        mu_F, mu_CR = 0.5, 0.5
        p_num = max(2, int(round(self.p * NP)))

        while True:  # termina por BudgetExhausted dentro de las evaluaciones
            S_F, S_CR = [], []
            # Genera CR_i ~ N(mu_CR, 0.1), F_i ~ Cauchy(mu_F, 0.1).
            CR = np.clip(rng.normal(mu_CR, 0.1, NP), 0.0, 1.0)
            F = self._cauchy(rng, mu_F, 0.1, NP)
            F = np.clip(F, 1e-6, 1.0)  # F<=0 se regenera implícitamente al recortar

            # Unión población + archivo para la diferencia (JADE).
            if self.use_archive and archive:
                P_union = np.vstack([pop, np.array(archive)])
            else:
                P_union = pop

            best_idx = np.argsort(fitness)[:p_num]

            new_pop = pop.copy()
            new_fit = fitness.copy()
            for i in range(NP):
                # x_pbest: uno aleatorio del p% mejor.
                pbest = pop[rng.choice(best_idx)]
                # r1 de la población (!= i), r2 de la unión (!= i, != r1).
                r1 = self._pick(rng, NP, exclude={i})
                r2 = self._pick(rng, len(P_union), exclude={i, r1})
                x = pop[i]
                v = (x + F[i] * (pbest - x)
                     + F[i] * (pop[r1] - P_union[r2]))
                v = reflect(v, lower, upper)

                # Cruce binomial.
                u = x.copy()
                jrand = rng.integers(dim)
                cross = rng.random(dim) < CR[i]
                cross[jrand] = True
                u[cross] = v[cross]

                fu = fc(u)  # puede lanzar BudgetExhausted -> termina la corrida
                if fu <= fitness[i]:
                    new_pop[i] = u
                    new_fit[i] = fu
                    if fu < fitness[i]:
                        if self.use_archive:
                            archive.append(x.copy())
                        S_F.append(F[i])
                        S_CR.append(CR[i])

            pop, fitness = new_pop, new_fit

            # Poda del archivo a tamaño NP.
            if self.use_archive and len(archive) > NP:
                idx = rng.choice(len(archive), NP, replace=False)
                archive = [archive[k] for k in idx]

            # Adaptación de mu_CR (media aritmética) y mu_F (media de Lehmer).
            if S_CR:
                mu_CR = (1 - self.c) * mu_CR + self.c * np.mean(S_CR)
            if S_F:
                sf = np.array(S_F)
                lehmer = np.sum(sf ** 2) / np.sum(sf)
                mu_F = (1 - self.c) * mu_F + self.c * lehmer

    @staticmethod
    def _cauchy(rng, loc, scale, n):
        return loc + scale * np.tan(np.pi * (rng.random(n) - 0.5))

    @staticmethod
    def _pick(rng, n, exclude):
        while True:
            r = int(rng.integers(n))
            if r not in exclude:
                return r
