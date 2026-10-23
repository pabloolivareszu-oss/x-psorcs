"""Construcción de etiquetas online con horizonte k.

Etiqueta binaria:
    y = 1 si hubo mejora del best_fitness entre la iteración t y t+k.
    y = 0 en caso contrario.

El emparejamiento (X(t), y(t)) requiere un desfase temporal: la traza de t solo
puede etiquetarse cuando ya se observó el best_fitness de t+k.
"""

from __future__ import annotations


class LabelBuilder:
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def build(self, best_fitness_t: float, best_fitness_t_plus_k: float) -> int:
        """Devuelve 1 si el best mejoró (disminuyó) más que epsilon en el horizonte."""
        return int((best_fitness_t - best_fitness_t_plus_k) > self.epsilon)
