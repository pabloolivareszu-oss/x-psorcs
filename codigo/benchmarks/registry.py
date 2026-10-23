"""Acceso unificado a las familias de benchmarks: classical, CEC (2014/2017/2022), BBOB.

El runner no necesita saber de qué familia viene un problema:
    get_problem(family, name, dim, run_id) -> (objective, lower, upper, info)

Familias válidas (--family):
    classical, cec2014, cec2017, cec2022, bbob

Corridas independientes:
  - classical / CEC: problema fijo; las corridas se distinguen por la SEMILLA del SIA.
  - BBOB: cada corrida usa una INSTANCIA distinta (instance=run_id), práctica COCO.

Dimensiones válidas por familia:
  - classical: cualquiera.
  - CEC2014/2017: 10, 20, 30, 50, 100.
  - CEC2022:      10, 20.
  - BBOB:         2, 3, 5, 10, 20, 40 (NO 50).
"""

from __future__ import annotations

from benchmarks import classical as _classical
from benchmarks.cec import (
    get_cec_problem,
    list_cec_functions,
    SUPPORTED_DIMS as CEC_DIMS,
    SUPPORTED_YEARS as CEC_YEARS,
)
from benchmarks.bbob import (
    get_bbob_problem,
    list_bbob_functions,
    SUPPORTED_DIMS as BBOB_DIMS,
)
from benchmarks.fir import (
    get_fir_problem,
    list_fir_functions,
    SUPPORTED_DIMS as FIR_DIMS,
)
from benchmarks.antenna import (
    get_antenna_problem,
    list_antenna_functions,
    case_dim as _antenna_case_dim,
)
from benchmarks.pv import (
    get_pv_problem,
    list_pv_functions,
    case_dim as _pv_case_dim,
)
from benchmarks.microhubs import (
    get_microhubs_problem,
    list_microhubs_functions,
    instance_dim as _microhubs_instance_dim,
)
from benchmarks.rail import (
    get_rail_problem,
    list_rail_functions,
    instance_dim as _rail_instance_dim,
)
from benchmarks.microhubs_cont import (
    get_microhubs_cont_problem,
    list_microhubs_cont_functions,
    instance_dim as _microhubs_cont_instance_dim,
)

# family string -> ("cec", year) | ("classical", None) | ("bbob", None)
#               | ("fir", None) | ("antenna", None) | ("pv", None)
#               | ("microhubs", None) | ("microhubs_cont", None) | ("rail", None)
FAMILIES = (["classical", "bbob", "fir", "antenna", "pv",
             "microhubs", "microhubs_cont", "rail"]
            + [f"cec{y}" for y in CEC_YEARS])


def _parse_family(family: str):
    family = family.lower()
    if family == "classical":
        return "classical", None
    if family == "bbob":
        return "bbob", None
    if family == "fir":
        return "fir", None
    if family == "antenna":
        return "antenna", None
    if family == "pv":
        return "pv", None
    if family == "microhubs":
        return "microhubs", None
    if family == "rail":
        return "rail", None
    if family == "microhubs_cont":
        return "microhubs_cont", None
    if family.startswith("cec"):
        digits = "".join(ch for ch in family if ch.isdigit())
        if not digits:
            raise KeyError(f"Familia CEC sin año: {family!r}. Usa cec2014/cec2017/cec2022.")
        year = int(digits)
        if year not in CEC_YEARS:
            raise KeyError(f"Año CEC no soportado: {year}. Disponibles: {CEC_YEARS}")
        return "cec", year
    raise KeyError(f"Familia desconocida: {family}. Disponibles: {FAMILIES}")


def get_problem(family: str, name: str, dim: int, run_id: int = 1):
    kind, year = _parse_family(family)

    if kind == "classical":
        objective, lower, upper = _classical.get_benchmark(name)
        info = {"family": "classical", "name": name, "dim": dim,
                "lower": lower, "upper": upper, "f_global": 0.0}
        return objective, lower, upper, info

    if kind == "cec":
        return get_cec_problem(name, dim, year=year)

    if kind == "bbob":
        func_index = _bbob_name_to_index(name)
        return get_bbob_problem(func_index, dim, instance=run_id)

    if kind == "fir":
        return get_fir_problem(name, dim)

    if kind == "antenna":
        # Como FIR, la dimensión es intrínseca al caso (N/2 amplitudes), pero
        # varía entre casos (16 o 32), así que se resuelve por función.
        return get_antenna_problem(name, dim=None)

    if kind == "pv":
        return get_pv_problem(name, dim=None)

    if kind == "microhubs":
        # La dimensión de microhubs es intrínseca a cada instancia (m hubs); el
        # `dim` recibido es un placeholder que la propia instancia ignora/valida.
        # `run_id` siembra la binarización estocástica (reproducibilidad).
        return get_microhubs_problem(name, dim=None, seed=run_id)

    if kind == "rail":
        # Como microhubs: dim = n columnas de la instancia; run_id siembra la
        # binarización V2-ELITR de forma explícita (sin hash()).
        return get_rail_problem(name, dim=None, seed=run_id)
    if kind == "microhubs_cont":
        # Igual que microhubs, la dimensión (2p coordenadas) es intrínseca a la
        # instancia. Aquí el objetivo es determinista, así que run_id no siembra
        # nada: las corridas se distinguen solo por la semilla del optimizador.
        return get_microhubs_cont_problem(name, dim=None)

    raise KeyError(f"Familia desconocida: {family}")


def supported_dims(family: str, dims: list[int]) -> list[int]:
    """Filtra la lista de dimensiones a las soportadas por la familia.

    microhubs es un caso especial: la dimensión depende de la INSTANCIA, no de la
    familia, así que aquí no se filtra por dim (devuelve la lista tal cual). El
    emparejamiento instancia<->dim lo resuelve `microhubs_dim_for` en run.py.
    """
    kind, year = _parse_family(family)
    if kind == "bbob":
        return [d for d in dims if d in BBOB_DIMS]
    if kind == "cec":
        return [d for d in dims if d in CEC_DIMS[year]]
    if kind == "fir":
        return [d for d in dims if d in FIR_DIMS]  # solo D=31
    # classical / antenna / pv / microhubs / microhubs_cont / rail: la dim depende
    # de la instancia o del caso, no de la familia; se resuelve por función en
    # run.py.
    return list(dims)


def antenna_dim_for(name: str) -> int:
    """Dimensión (N/2 amplitudes) de un caso antenna concreto."""
    return _antenna_case_dim(name)


def pv_dim_for(name: str) -> int:
    """Dimensión (nº de parámetros del modelo) de un caso pv concreto."""
    return _pv_case_dim(name)


def microhubs_dim_for(name: str) -> int:
    """Dimensión (m = nº de hubs) de una instancia microhubs concreta."""
    return _microhubs_instance_dim(name)


def rail_dim_for(name: str) -> int:
    """Dimensión (n = nº de columnas) de una instancia rail concreta."""
    return _rail_instance_dim(name)


def microhubs_cont_dim_for(name: str) -> int:
    """Dimensión (2p coordenadas) de una instancia microhubs_cont concreta."""
    return _microhubs_cont_instance_dim(name)


def list_functions(family: str) -> list[str]:
    kind, year = _parse_family(family)
    if kind == "classical":
        return list(_classical.BENCHMARKS.keys())
    if kind == "cec":
        return list_cec_functions(year)
    if kind == "bbob":
        return list_bbob_functions()
    if kind == "fir":
        return list_fir_functions()
    if kind == "antenna":
        return list_antenna_functions()
    if kind == "pv":
        return list_pv_functions()
    if kind == "microhubs":
        return list_microhubs_functions()
    if kind == "rail":
        return list_rail_functions()
    if kind == "microhubs_cont":
        return list_microhubs_cont_functions()
    raise KeyError(f"Familia desconocida: {family}")


def _bbob_name_to_index(name: str) -> int:
    digits = "".join(ch for ch in str(name) if ch.isdigit())
    if not digits:
        raise ValueError(f"Nombre BBOB inválido: {name!r} (esperado 'f01'..'f24').")
    return int(digits)
