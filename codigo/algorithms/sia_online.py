"""SIA base online y adaptativo basado en PSO (Particle Swarm Optimization).

Diseño clave: la unidad de ejecución es UNA iteración (`step`). El comportamiento
del enjambre se controla mediante un diccionario `state` mutable que los módulos
online (modelo, XAI, controlador adaptativo) pueden modificar entre iteraciones.

Operadores expuestos para adaptación:
  - standard:        actualización PSO clásica (w, c1, c2).
  - diversification: reinicia aleatoriamente las partículas más estancadas.
  - intensification: búsqueda local gaussiana alrededor del mejor global.

El balance exploración/explotación se mapea a los pesos PSO:
  - exploration_rate alto  -> w y c1 altos (búsqueda global).
  - exploitation_rate alto -> c2 alto      (atracción al gbest).
"""

from __future__ import annotations

from typing import Callable

import numpy as np


# Penalización finita para evaluaciones NaN/inf (función mal condicionada en un
# punto extremo). Suficientemente grande para que el algoritmo la rechace.
_PENALTY = 1e30


def _safe_objective(objective):
    """Envuelve una función objetivo para que nunca devuelva NaN/inf.

    Devuelve _PENALTY si la evaluación es no finita (NaN o ±inf) o si lanza una
    excepción numérica. Mantiene el algoritmo y los estadísticos libres de NaN."""
    def wrapped(x):
        try:
            v = float(objective(x))
        except (FloatingPointError, ValueError, OverflowError):
            return _PENALTY
        if not np.isfinite(v):
            return _PENALTY
        return v
    return wrapped


def compute_diversity(population: np.ndarray, lower: float, upper: float) -> float:
    """Diversidad normalizada en [0, 1] (aprox).

    Distancia media de cada partícula al centroide, dividida por la diagonal del
    espacio de búsqueda. Así el umbral `diversity_min` del controlador (p.ej. 0.10)
    es interpretable como fracción de la diagonal, independiente de la dimensión."""
    centroid = population.mean(axis=0)
    dists = np.linalg.norm(population - centroid, axis=1)
    diag = np.sqrt(population.shape[1]) * (upper - lower)
    if diag <= 0:
        return 0.0
    return float(dists.mean() / diag)


class OnlineExplainableSIA:
    """Enjambre PSO con estado mutable adaptable en tiempo real."""

    OPERATORS = ("standard", "diversification", "intensification")

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        dim: int,
        lower: float,
        upper: float,
        pop_size: int = 30,
        max_iter: int = 100,
        seed: int | None = None,
    ):
        # Envuelve el objetivo: algunas funciones CEC devuelven NaN/inf cuando una
        # partícula cae en valores extremos. Se penaliza con un finito muy grande
        # para que NUNCA contamine el best ni los estadísticos posteriores.
        self.objective = _safe_objective(objective)
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)

        span = upper - lower
        self.population = self.rng.uniform(lower, upper, size=(pop_size, dim))
        self.velocity = self.rng.uniform(-span, span, size=(pop_size, dim)) * 0.1

        self.fitness = np.array([self.objective(x) for x in self.population])
        # Memoria personal (pbest) y global (gbest).
        self.pbest = self.population.copy()
        self.pbest_fitness = self.fitness.copy()
        g = int(np.argmin(self.fitness))
        self.gbest = self.population[g].copy()
        self.gbest_fitness = float(self.fitness[g])

        # Evaluaciones de función consumidas (presupuesto).
        self.fe = pop_size

        # Estado mutable: lo leen/escriben los módulos online.
        self.state: dict = {
            "iteration": 0,
            "best_fitness": self.gbest_fitness,
            "previous_best_fitness": self.gbest_fitness,
            "mean_fitness": float(self.fitness.mean()),
            "std_fitness": float(self.fitness.std()),
            "diversity": compute_diversity(self.population, lower, upper),
            "stagnation_counter": 0,
            "exploration_rate": 0.5,
            "exploitation_rate": 0.5,
            "operator_used": "standard",
            "operator_probabilities": {
                "standard": 0.7,
                "diversification": 0.15,
                "intensification": 0.15,
            },
            "parameters": {"w": 0.7, "c1": 1.5, "c2": 1.5},
            "last_adaptation": None,
            "n_improved_agents": 0,
        }

    # ---- utilidades internas --------------------------------------------

    def _clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lower, self.upper)

    def _pso_params_from_rates(self) -> tuple[float, float, float]:
        """Deriva (w, c1, c2) a partir de las tasas de exploración/explotación
        del estado, manteniendo los parámetros base como ancla."""
        p = self.state["parameters"]
        expl = self.state["exploration_rate"]
        expt = self.state["exploitation_rate"]
        # Modula alrededor de los valores base; mezcla suave.
        w = float(np.clip(0.4 + 0.5 * expl, 0.1, 0.95)) * 0.5 + p["w"] * 0.5
        c1 = float(np.clip(0.5 + 2.0 * expl, 0.5, 2.5)) * 0.5 + p["c1"] * 0.5
        c2 = float(np.clip(0.5 + 2.0 * expt, 0.5, 2.5)) * 0.5 + p["c2"] * 0.5
        return w, c1, c2

    # ---- operadores ------------------------------------------------------

    def _op_standard(self) -> None:
        w, c1, c2 = self._pso_params_from_rates()
        self.state["parameters"] = {"w": w, "c1": c1, "c2": c2}
        r1 = self.rng.random((self.pop_size, self.dim))
        r2 = self.rng.random((self.pop_size, self.dim))
        self.velocity = (
            w * self.velocity
            + c1 * r1 * (self.pbest - self.population)
            + c2 * r2 * (self.gbest - self.population)
        )
        self.population = self._clip(self.population + self.velocity)

    def _op_diversification(self) -> None:
        """Reinicia la mitad peor de las partículas en posiciones aleatorias."""
        self._op_standard()  # también avanza el enjambre normalmente
        n_reset = max(1, self.pop_size // 2)
        worst = np.argsort(self.fitness)[-n_reset:]
        span = self.upper - self.lower
        self.population[worst] = self.rng.uniform(
            self.lower, self.upper, size=(n_reset, self.dim)
        )
        self.velocity[worst] = self.rng.uniform(
            -span, span, size=(n_reset, self.dim)
        ) * 0.1

    def _op_intensification(self) -> None:
        """Búsqueda local gaussiana alrededor del gbest para las mejores
        partículas, además del avance estándar."""
        self._op_standard()
        n_local = max(1, self.pop_size // 4)
        best = np.argsort(self.fitness)[:n_local]
        scale = 0.05 * (self.upper - self.lower)
        noise = self.rng.normal(0.0, scale, size=(n_local, self.dim))
        self.population[best] = self._clip(self.gbest + noise)

    def _select_operator(self) -> str:
        probs = self.state["operator_probabilities"]
        ops = list(probs.keys())
        weights = np.array([probs[o] for o in ops], dtype=float)
        weights = weights / weights.sum()
        return str(self.rng.choice(ops, p=weights))

    # ---- una iteración ---------------------------------------------------

    def step(self) -> dict:
        """Ejecuta UNA iteración del SIA y actualiza el estado. Devuelve `state`."""
        prev_best = self.gbest_fitness
        self.state["previous_best_fitness"] = prev_best

        operator = self._select_operator()
        self.state["operator_used"] = operator
        getattr(self, f"_op_{operator}")()

        # Reevaluar fitness.
        new_fitness = np.array([self.objective(x) for x in self.population])
        self.fe += self.pop_size

        # Actualizar pbest.
        improved_mask = new_fitness < self.pbest_fitness
        self.pbest[improved_mask] = self.population[improved_mask]
        self.pbest_fitness[improved_mask] = new_fitness[improved_mask]
        self.fitness = new_fitness

        # Actualizar gbest.
        g = int(np.argmin(self.pbest_fitness))
        if self.pbest_fitness[g] < self.gbest_fitness:
            self.gbest = self.pbest[g].copy()
            self.gbest_fitness = float(self.pbest_fitness[g])

        # Estancamiento.
        if self.gbest_fitness < prev_best - 1e-12:
            self.state["stagnation_counter"] = 0
        else:
            self.state["stagnation_counter"] += 1

        # Refrescar estadísticos del estado.
        self.state["iteration"] += 1
        self.state["best_fitness"] = self.gbest_fitness
        self.state["mean_fitness"] = float(self.fitness.mean())
        self.state["std_fitness"] = float(self.fitness.std())
        self.state["diversity"] = compute_diversity(self.population, self.lower, self.upper)
        self.state["n_improved_agents"] = int(improved_mask.sum())
        return self.state
