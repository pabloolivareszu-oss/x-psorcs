"""Suite BBOB (Black-Box Optimization Benchmarking) vía cocoex (COCO oficial).

BBOB define 24 funciones (f1..f24) ruidosas/no ruidosas y múltiples INSTANCIAS por
función. La práctica estándar de COCO es ejecutar varias instancias en vez de
"corridas con distinta semilla": cada instancia es una transformación afín distinta
del mismo problema base. Por eso `instance` cumple el rol de id de corrida.

Interfaz uniforme:
    get_bbob_problem(func_index, dim, instance) -> (objective, lower, upper, info)

Nota: en BBOB el valor óptimo f_opt NO se conoce a priori (es parte del protocolo
de caja negra). Reportamos el fitness crudo; la distancia al óptimo se obtiene tras
la corrida con problem.best_observed_fvalue1 si se desea.
"""

from __future__ import annotations

import contextlib
import os

import numpy as np
import cocoex

# Dominio BBOB estándar: hipercubo [-5, 5] en todas las dimensiones.
BBOB_LOWER = -5.0
BBOB_UPPER = 5.0
N_FUNCTIONS = 24
N_INSTANCES = 15  # BBOB tiene 15 instancias oficiales por (función, dim)
# Dimensiones oficiales de COCO/BBOB. Pedir otra (p.ej. 50) hace que COCO la
# ajuste SILENCIOSAMENTE a la mayor válida (40), lo que falsearía resultados.
SUPPORTED_DIMS = (2, 3, 5, 10, 20, 40)


@contextlib.contextmanager
def _silence_coco():
    """Silencia los COCO WARNING que cocoex emite por C-stdout/stderr (no por
    el módulo warnings de Python), redirigiendo los fds a devnull."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old1, old2 = os.dup(1), os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old1, 1)
        os.dup2(old2, 2)
        os.close(old1)
        os.close(old2)
        os.close(devnull)


def get_bbob_problem(func_index: int, dim: int, instance: int = 1):
    """Devuelve (objective, lower, upper, info) para una función/instancia BBOB.

    func_index: 1..24
    dim:        cualquiera de las soportadas por COCO (2,3,5,10,20,40); para
                10/20 es directo. Ver nota sobre dim 50 abajo.
    instance:   id de instancia (>=1); se usa como id de corrida.
    """
    if not (1 <= func_index <= N_FUNCTIONS):
        raise ValueError(f"func_index debe estar en 1..{N_FUNCTIONS}, recibido {func_index}")
    if dim not in SUPPORTED_DIMS:
        raise ValueError(
            f"BBOB no soporta dim={dim}. Dimensiones válidas: {SUPPORTED_DIMS}. "
            f"(COCO ajustaría silenciosamente 50->40, lo que falsearía los resultados.)"
        )

    # BBOB tiene 15 instancias oficiales. run_id va hasta 30, así que mapeamos
    # cíclicamente al rango 1..15 (1,2,...,15,1,2,...). Evita el COCO WARNING de
    # 'instance_indices adjusted to be <= 15' y mantiene instancias válidas.
    coco_instance = ((int(instance) - 1) % N_INSTANCES) + 1
    options = (
        f"dimensions:{dim} "
        f"function_indices:{func_index} "
        f"instance_indices:{coco_instance}"
    )
    with _silence_coco():
        suite = cocoex.Suite("bbob", "", options)
    if len(suite) == 0:
        raise ValueError(
            f"COCO no ofrece f{func_index} dim={dim} instancia={instance}. "
            f"Dimensiones BBOB soportadas: 2,3,5,10,20,40."
        )
    problem = suite[0]
    if problem.dimension != dim:
        raise RuntimeError(
            f"COCO devolvió dim={problem.dimension} para una petición de dim={dim}; "
            f"abortando para no falsear resultados."
        )

    lower = float(problem.lower_bounds[0])
    upper = float(problem.upper_bounds[0])

    def objective(x: np.ndarray) -> float:
        return float(problem(np.asarray(x, dtype=float)))

    info = {
        "family": "BBOB",
        "name": f"f{func_index:02d}",
        "func_index": func_index,
        "instance": coco_instance,   # instancia COCO real (1..15)
        "run_id": instance,          # id de corrida original (1..30)
        "dim": dim,
        "problem_id": problem.id,
        "lower": lower,
        "upper": upper,
        "f_global": None,  # desconocido a priori en BBOB
        "_problem": problem,  # referencia viva para leer best_observed tras la corrida
    }
    return objective, lower, upper, info


def list_bbob_functions() -> list[str]:
    return [f"f{i:02d}" for i in range(1, N_FUNCTIONS + 1)]
