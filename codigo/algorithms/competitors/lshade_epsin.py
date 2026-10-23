"""LSHADE-EpSin: LSHADE con adaptación sinusoidal de F (Ensemble of Parameter Sin).

Referencia:
    N. H. Awad, M. Z. Ali, P. N. Suganthan, R. G. Reynolds, "An ensemble
    sinusoidal parameter adaptation incorporated with L-SHADE for solving CEC2014
    benchmark problems", IEEE CEC 2016.

Sobre LSHADE, en la PRIMERA mitad de la búsqueda el factor de escala F se genera
con un ENSEMBLE de dos esquemas sinusoidales (adaptativo y no adaptativo), elegidos
por su tasa de éxito; en la segunda mitad se usa la adaptación SHADE estándar.
Esto mejora el balance exploración/explotación temprano.
"""

from __future__ import annotations

import numpy as np

from algorithms.competitors.base import Competitor, FEvalCounter, reflect
from algorithms.competitors.lshade import _pick


class LSHADE_EpSin(Competitor):
    name = "LSHADE-EpSin"

    def __init__(self, n_init_factor: int = 18, n_min: int = 4,
                 memory_size: int = 5, p_best: float = 0.11, arc_rate: float = 1.4,
                 freq: float = 0.5):
        self.n_init_factor = n_init_factor
        self.n_min = n_min
        self.H = memory_size
        self.p_best = p_best
        self.arc_rate = arc_rate
        self.freq0 = freq

    def _run(self, fc: FEvalCounter, dim, lower, upper, rng):
        N_init = self.n_init_factor * dim
        N = N_init
        pop = rng.uniform(lower, upper, size=(N, dim))
        fit = fc.eval_population(pop)
        archive = np.empty((0, dim))
        M_F = np.full(self.H, 0.5)
        M_CR = np.full(self.H, 0.5)
        M_freq = np.full(self.H, self.freq0)
        k = 0

        # probabilidades del ensemble sinusoidal (no-adaptativo vs adaptativo)
        succ = np.ones(2)
        max_iter_est = max(1, fc.max_fes // N_init)
        gen = 0

        while True:
            gen += 1
            half = fc.max_fes / 2.0
            order = np.argsort(fit)
            p_num = max(2, int(self.p_best * N))
            pbest_pool = order[:p_num]
            S_F, S_CR, S_w = [], [], []
            use_sin = fc.fes < half  # ensemble sinusoidal solo en 1ª mitad

            # elige configuración sinusoidal por éxito
            cfg = int(rng.random() < succ[1] / succ.sum()) if use_sin else -1
            cfg_succ = 0.0

            new_pop = pop.copy(); new_fit = fit.copy()
            for i in range(N):
                ri = rng.integers(self.H)
                CR = 0.0 if M_CR[ri] < 0 else float(np.clip(rng.normal(M_CR[ri], 0.1), 0, 1))

                if use_sin:
                    G = gen
                    Gmax = max(gen + 1, max_iter_est)
                    if cfg == 0:
                        # sinusoidal NO adaptativo decreciente
                        F = 0.5 * (np.sin(2 * np.pi * self.freq0 * G) * (Gmax - G) / Gmax + 1.0)
                    else:
                        # sinusoidal adaptativo con frecuencia de memoria
                        fr = M_freq[ri] + 0.1 * np.tan(np.pi * (rng.random() - 0.5))
                        fr = abs(fr) if fr != 0 else self.freq0
                        F = 0.5 * (np.sin(2 * np.pi * fr * G) * G / Gmax + 1.0)
                    F = float(np.clip(F, 0.01, 1.0))
                else:
                    F = -1.0
                    while F <= 0:
                        F = M_F[ri] + 0.1 * np.tan(np.pi * (rng.random() - 0.5))
                    F = min(F, 1.0)

                pbest = pop[rng.choice(pbest_pool)]
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
                    if len(archive) < int(self.arc_rate * N):
                        archive = np.vstack([archive, pop[i]])
                    else:
                        archive[rng.integers(len(archive))] = pop[i]
                    S_F.append(F); S_CR.append(CR); S_w.append(fit[i] - fu)
                    cfg_succ += (fit[i] - fu)

            pop, fit = new_pop, new_fit

            if use_sin and cfg >= 0:
                succ[cfg] = 0.9 * succ[cfg] + 0.1 * cfg_succ + 1e-9

            if S_F:
                w = np.array(S_w); w = w / (w.sum() + 1e-30)
                sf = np.array(S_F); scr = np.array(S_CR)
                M_F[k] = np.sum(w * sf ** 2) / (np.sum(w * sf) + 1e-30)
                M_freq[k] = self.freq0
                if max(scr) == 0:
                    M_CR[k] = -1.0
                else:
                    M_CR[k] = np.sum(w * scr ** 2) / (np.sum(w * scr) + 1e-30)
                k = (k + 1) % self.H

            # LPSR
            ratio = fc.fes / fc.max_fes
            N_new = max(self.n_min, round(N_init + (self.n_min - N_init) * ratio))
            if N_new < N:
                keep = np.argsort(fit)[:N_new]
                pop, fit = pop[keep], fit[keep]; N = N_new
                maxA = int(self.arc_rate * N)
                if len(archive) > maxA:
                    archive = archive[rng.choice(len(archive), maxA, replace=False)]
