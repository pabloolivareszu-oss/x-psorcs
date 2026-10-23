"""Funciones benchmark clásicas para optimización continua.

Cada función se minimiza y tiene óptimo global conocido (f* = 0) salvo que se
indique lo contrario. Todas operan sobre vectores numpy de dimensión D.
"""

from __future__ import annotations

import numpy as np


def sphere(x: np.ndarray) -> float:
    """Esfera. Óptimo en 0. Unimodal, separable."""
    return float(np.sum(x ** 2))


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock. Óptimo en (1,...,1). Valle no convexo."""
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin. Óptimo en 0. Multimodal, muchos mínimos locales."""
    return float(10 * x.size + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    """Ackley. Óptimo en 0. Multimodal."""
    n = x.size
    s1 = np.sum(x ** 2)
    s2 = np.sum(np.cos(2 * np.pi * x))
    return float(
        -20.0 * np.exp(-0.2 * np.sqrt(s1 / n))
        - np.exp(s2 / n)
        + 20.0
        + np.e
    )


def griewank(x: np.ndarray) -> float:
    """Griewank. Óptimo en 0. Multimodal."""
    s = np.sum(x ** 2) / 4000.0
    p = np.prod(np.cos(x / np.sqrt(np.arange(1, x.size + 1))))
    return float(s - p + 1.0)


# Registro: nombre -> (función, dominio_inferior, dominio_superior)
BENCHMARKS = {
    "sphere": (sphere, -5.12, 5.12),
    "rosenbrock": (rosenbrock, -5.0, 10.0),
    "rastrigin": (rastrigin, -5.12, 5.12),
    "ackley": (ackley, -32.768, 32.768),
    "griewank": (griewank, -600.0, 600.0),
}


def get_benchmark(name: str):
    """Devuelve (func, lower, upper) para el benchmark dado."""
    if name not in BENCHMARKS:
        raise KeyError(f"Benchmark desconocido: {name}. Disponibles: {list(BENCHMARKS)}")
    return BENCHMARKS[name]
