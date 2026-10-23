"""ASGK: Adaptive Gaining-Sharing Knowledge con memoria histórica y LPSR.

Referencia:
    Familia GSK — A. W. Mohamed et al., "Gaining-Sharing Knowledge Based Algorithm
    for Solving Optimization Problems (GSK)", Int. J. Mach. Learn. & Cyber., 2020.
    Variantes adaptativas (APGSK / ASGK) introducen adaptación de los parámetros de
    conocimiento (kf, kr) mediante memorias históricas tipo SHADE y reducción lineal
    de población (LPSR).

Diferencia respecto al componente APGSK del híbrido anterior:
  - Adaptación de kf y kr por MEMORIA HISTÓRICA (éxitos ponderados por mejora,
    media de Lehmer), no por muestreo de pools fijos.
  - Reducción Lineal del Tamaño de Población (LPSR): NP decrece de NP_init a NP_min
    en función de los FEs consumidos, concentrando el presupuesto al final.
  - Transición junior->senior controlada por el progreso de FEs.
"""

from __future__ import annotations

import numpy as np

from algorithms.competitors.base import Competitor, FEvalCounter, reflect


class ASGK(Competitor):
    name = "ASGK"

    def __init__(self, pop_init: int = 30, pop_min: int = 8, memory_size: int = 5):
        self.pop_init = pop_init
        self.pop_min = pop_min
        self.H = memory_size

    def _run(self, fc: FEvalCounter, dim: int, lower: float, upper: float,
             rng: np.random.Generator) -> None:
        D = dim
        NP = self.pop_init
        pop = rng.uniform(lower, upper, size=(NP, D))
        fit = fc.eval_population(pop)

        # Memorias históricas de kf y kr (estilo SHADE).
        M_kf = np.full(self.H, 0.5)
        M_kr = np.full(self.H, 0.9)
        mem_pos = 0

        while True:
            # --- LPSR: tamaño de población objetivo según progreso de FEs.
            ratio = fc.fes / fc.max_fes
            NP_target = int(round(self.pop_init + (self.pop_min - self.pop_init) * ratio))
            NP_target = max(self.pop_min, NP_target)
            if NP_target < len(pop):
                keep = np.argsort(fit)[:NP_target]
                pop, fit = pop[keep], fit[keep]
            NP = len(pop)
            if NP < 4:
                NP = 4  # mínimo operativo

            order = np.argsort(fit)
            rank = np.empty(NP, dtype=int)
            rank[order] = np.arange(NP)

            # Muestrea kf (Cauchy) y kr (Normal) de la memoria histórica.
            ridx = rng.integers(self.H, size=NP)
            kf = np.clip(M_kf[ridx] + 0.1 * np.tan(np.pi * (rng.random(NP) - 0.5)), 0.01, 1.0)
            kr = np.clip(M_kr[ridx] + 0.1 * rng.standard_normal(NP), 0.0, 1.0)

            S_kf, S_kr, S_dw = [], [], []  # éxitos y mejora asociada (peso)

            ratio_junior = max(0.0, 1.0 - ratio)  # componente junior decrece con FEs
            p = max(1, int(0.1 * NP))
            best_pool = order[:p]
            worst_pool = order[-p:]
            mid_pool = order[p:NP - p] if NP - 2 * p > 0 else order

            for i in range(NP):
                d_junior = int(round(D * ratio_junior))
                dims = rng.permutation(D)
                junior_dims, senior_dims = dims[:d_junior], dims[d_junior:]
                xi = pop[i].copy()
                new = xi.copy()

                # Junior: vecinos por ranking.
                r = rank[i]
                better = order[max(0, r - 1)]
                worse = order[min(NP - 1, r + 1)]
                rnd = int(rng.integers(NP))
                for d in junior_dims:
                    if rng.random() <= kr[i]:
                        if fit[i] > fit[rnd]:
                            new[d] = xi[d] + kf[i] * ((pop[better, d] - pop[worse, d])
                                                      + (pop[rnd, d] - xi[d]))
                        else:
                            new[d] = xi[d] + kf[i] * ((pop[better, d] - pop[worse, d])
                                                      + (xi[d] - pop[rnd, d]))
                # Senior: mejores/medios/peores.
                pb = pop[rng.choice(best_pool)]
                pm = pop[rng.choice(mid_pool)]
                pw = pop[rng.choice(worst_pool)]
                fmid = fit[int(rng.choice(mid_pool))]
                for d in senior_dims:
                    if rng.random() <= kr[i]:
                        if fit[i] > fmid:
                            new[d] = xi[d] + kf[i] * ((pb[d] - pm[d]) + (pm[d] - xi[d]))
                        else:
                            new[d] = xi[d] + kf[i] * ((pb[d] - pm[d]) + (xi[d] - pw[d]))

                new = reflect(new, lower, upper)
                fnew = fc(new)
                if fnew < fit[i]:
                    S_kf.append(kf[i]); S_kr.append(kr[i]); S_dw.append(fit[i] - fnew)
                    pop[i] = new
                    fit[i] = fnew

            # --- Actualiza memorias con media de Lehmer ponderada por mejora.
            if S_dw:
                w = np.array(S_dw); w = w / w.sum()
                skf = np.array(S_kf); skr = np.array(S_kr)
                M_kf[mem_pos] = np.sum(w * skf ** 2) / np.sum(w * skf)
                M_kr[mem_pos] = np.sum(w * skr ** 2) / max(1e-12, np.sum(w * skr))
                mem_pos = (mem_pos + 1) % self.H
