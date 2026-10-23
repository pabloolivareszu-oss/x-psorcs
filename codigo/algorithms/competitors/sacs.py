"""SACS: Self-Adaptive Cuckoo Search.

Referencia (familia Cuckoo Search auto-adaptativa):
    X.-S. Yang, S. Deb, "Cuckoo Search via Lévy Flights", 2009; con la extensión
    AUTO-ADAPTATIVA de la fracción de abandono pa y el tamaño de paso alpha
    (self-adaptive CS), que ajustan estos parámetros según la tasa de éxito en
    lugar de mantenerlos fijos.

NOTA DE IMPLEMENTACIÓN (tesis): 'SACS' es una sigla poco estandarizada; aquí se
implementa como Cuckoo Search con auto-adaptación de pa y alpha, que es el
significado más extendido. Documentado para reproducibilidad.

Mecanismo: nidos (soluciones) que evolucionan por vuelos de Lévy; una fracción pa
de los peores nidos se abandona y reconstruye. pa y alpha se auto-ajustan según
cuántos huevos nuevos mejoraron en la generación previa.
"""

from __future__ import annotations

import numpy as np

from algorithms.competitors.base import Competitor, FEvalCounter, clip
from algorithms.competitors.hho import levy


class SACS(Competitor):
    name = "SACS"

    def __init__(self, pop_size: int = 30, pa0: float = 0.25, alpha0: float = 0.01):
        self.pop_size = pop_size
        self.pa0 = pa0
        self.alpha0 = alpha0

    def _run(self, fc: FEvalCounter, dim, lower, upper, rng):
        N = self.pop_size
        nests = rng.uniform(lower, upper, size=(N, dim))
        fit = fc.eval_population(nests)
        pa, alpha = self.pa0, self.alpha0
        prev_success = 0.5

        while True:
            best = nests[np.argmin(fit)]
            success = 0

            # --- Vuelos de Lévy: genera huevos nuevos ---
            for i in range(N):
                step = alpha * levy(dim, rng) * (nests[i] - best)
                new = clip(nests[i] + step * rng.normal(size=dim), lower, upper)
                f = fc(new)
                if f < fit[i]:
                    nests[i], fit[i] = new, f
                    success += 1

            # --- Abandono: reemplaza una fracción pa de nidos ---
            K = rng.random((N, dim)) > pa
            stepsize = rng.random((N, 1)) * (
                nests[rng.permutation(N)] - nests[rng.permutation(N)])
            new_nests = clip(nests + stepsize * K, lower, upper)
            for i in range(N):
                f = fc(new_nests[i])
                if f < fit[i]:
                    nests[i], fit[i] = new_nests[i], f
                    success += 1

            # --- Auto-adaptación de pa y alpha según éxito ---
            rate = success / (2 * N)
            if rate > prev_success:       # mejorando: intensifica
                alpha *= 0.95
                pa = max(0.05, pa * 0.98)
            else:                          # estancado: diversifica
                alpha = min(0.5, alpha * 1.05)
                pa = min(0.5, pa * 1.02)
            prev_success = 0.5 * prev_success + 0.5 * rate
