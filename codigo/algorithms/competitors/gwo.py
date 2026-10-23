"""GWO: Grey Wolf Optimizer.

Referencia:
    S. Mirjalili, S. M. Mirjalili, A. Lewis, "Grey Wolf Optimizer",
    Advances in Engineering Software, 69, 2014.

Mecanismo central: jerarquía social de lobos (alpha, beta, delta = 3 mejores) que
guían la caza. Cada lobo actualiza su posición como promedio de su atracción hacia
los tres líderes, con el coeficiente 'a' decreciendo linealmente de 2 a 0 (control
exploración->explotación).
"""

from __future__ import annotations

import numpy as np

from algorithms.competitors.base import Competitor, FEvalCounter, clip


class GWO(Competitor):
    name = "GWO"

    def __init__(self, pop_size: int = 30):
        self.pop_size = pop_size

    def _run(self, fc: FEvalCounter, dim, lower, upper, rng):
        N = self.pop_size
        pop = rng.uniform(lower, upper, size=(N, dim))
        fit = fc.eval_population(pop)

        # Estima nº de iteraciones por el presupuesto restante.
        max_iter = max(1, fc.max_fes // N)
        for t in range(max_iter):
            order = np.argsort(fit)
            alpha, beta, delta = pop[order[0]], pop[order[1]], pop[order[2]]
            a = 2 - 2 * t / max_iter  # decrece 2 -> 0

            for i in range(N):
                # Los tres líderes como columnas: leaders[d, k] = líder k en la dim d.
                # Se reconstruye en cada i porque alpha/beta/delta son VISTAS de filas
                # de pop: si un líder mejora dentro de esta iteración, los individuos
                # siguientes ya ven su valor nuevo, igual que en el bucle original.
                leaders = np.stack((alpha, beta, delta), axis=1)

                # Vectorizado sobre las dimensiones, con resultados IDÉNTICOS bit a bit
                # al bucle escalar original (por cada d, por cada líder: r1, r2 =
                # rng.random(), rng.random(); pos = líder - A*|C*líder - x|; media de
                # los tres). Generator.random(n) produce exactamente la misma secuencia
                # que n llamadas escalares, así que basta con pedirla en ese orden:
                # R[d, k, 0] = r1 y R[d, k, 1] = r2. La media de tres valores coincide
                # con (p0 + p1 + p2) / 3. Con D ≈ 50.000 (caso ferroviario) el bucle
                # escalar tardaba ~0,19 s por evaluación.
                R = rng.random(dim * 6).reshape(dim, 3, 2)
                A = 2 * a * R[:, :, 0] - a
                C = 2 * R[:, :, 1]
                D = np.abs(C * leaders - pop[i][:, None])
                pos = leaders - A * D
                new = clip((pos[:, 0] + pos[:, 1] + pos[:, 2]) / 3, lower, upper)
                f = fc(new)
                if f < fit[i]:
                    pop[i], fit[i] = new, f
