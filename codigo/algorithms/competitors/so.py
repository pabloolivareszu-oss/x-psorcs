"""SO: Snake Optimizer.

Referencia:
    F. A. Hashim, A. G. Hussien, "Snake Optimizer: A novel meta-heuristic
    optimization algorithm", Knowledge-Based Systems, 242, 2022.

Mecanismo: la población se divide en machos y hembras. Dos cantidades guían el
comportamiento: temperatura Temp y cantidad de comida Q (ambas función del tiempo).
  - Si Q < umbral: exploración (buscar comida aleatoriamente).
  - Si Temp alta: explotación hacia la comida (mejor).
  - Si Temp baja: modo lucha (fight) o apareamiento (mating) entre machos/hembras,
    seguido de eclosión de huevos (reemplazo de los peores).
"""

from __future__ import annotations

import numpy as np

from algorithms.competitors.base import Competitor, FEvalCounter, clip


class SO(Competitor):
    name = "SO"

    def __init__(self, pop_size: int = 30):
        self.pop_size = pop_size

    def _run(self, fc: FEvalCounter, dim, lower, upper, rng):
        N = self.pop_size
        nm = N // 2          # machos
        nf = N - nm          # hembras
        males = rng.uniform(lower, upper, size=(nm, dim))
        females = rng.uniform(lower, upper, size=(nf, dim))
        fm = fc.eval_population(males)
        ff = fc.eval_population(females)
        max_iter = max(1, fc.max_fes // N)

        C1, C2, C3 = 0.5, 0.05, 2.0
        THRESH_Q, THRESH_T = 0.25, 0.6

        for t in range(max_iter):
            Temp = np.exp(-t / max_iter)
            Q = C1 * np.exp((t - max_iter) / max_iter)

            best_m = males[np.argmin(fm)]; best_m_f = fm.min()
            best_f = females[np.argmin(ff)]; best_f_f = ff.min()
            food = best_m if best_m_f < best_f_f else best_f

            if Q < THRESH_Q:
                # --- Exploración: cada serpiente busca comida aleatoriamente ---
                for i in range(nm):
                    rand = males[rng.integers(nm)]
                    Am = np.exp(-fm[rng.integers(nm)] / (fm[i] + 1e-30))
                    new = clip(rand + C2 * Am * ((upper - lower) * rng.random(dim) + lower),
                               lower, upper)
                    f = fc(new)
                    if f < fm[i]: males[i], fm[i] = new, f
                for i in range(nf):
                    rand = females[rng.integers(nf)]
                    Af = np.exp(-ff[rng.integers(nf)] / (ff[i] + 1e-30))
                    new = clip(rand + C2 * Af * ((upper - lower) * rng.random(dim) + lower),
                               lower, upper)
                    f = fc(new)
                    if f < ff[i]: females[i], ff[i] = new, f
            else:
                if Temp > THRESH_T:
                    # --- Explotación hacia la comida ---
                    for i in range(nm):
                        new = clip(food + C3 * Temp * rng.random(dim) * (food - males[i]),
                                   lower, upper)
                        f = fc(new)
                        if f < fm[i]: males[i], fm[i] = new, f
                    for i in range(nf):
                        new = clip(food + C3 * Temp * rng.random(dim) * (food - females[i]),
                                   lower, upper)
                        f = fc(new)
                        if f < ff[i]: females[i], ff[i] = new, f
                else:
                    # --- Lucha o apareamiento ---
                    if rng.random() < 0.6:  # fight
                        for i in range(nm):
                            FM = np.exp(-best_f_f / (fm[i] + 1e-30))
                            new = clip(males[i] + C3 * FM * rng.random(dim) * (Q * best_f - males[i]),
                                       lower, upper)
                            f = fc(new)
                            if f < fm[i]: males[i], fm[i] = new, f
                        for i in range(nf):
                            FF = np.exp(-best_m_f / (ff[i] + 1e-30))
                            new = clip(females[i] + C3 * FF * rng.random(dim) * (Q * best_m - females[i]),
                                       lower, upper)
                            f = fc(new)
                            if f < ff[i]: females[i], ff[i] = new, f
                    else:                   # mating
                        for i in range(min(nm, nf)):
                            Mm = np.exp(-ff[i] / (fm[i] + 1e-30))
                            new = clip(males[i] + C3 * Mm * rng.random(dim) * (Q * females[i] - males[i]),
                                       lower, upper)
                            f = fc(new)
                            if f < fm[i]: males[i], fm[i] = new, f

        # Reporta el mejor (ya está en fcounter, pero aseguramos evaluación final).
