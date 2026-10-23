"""Benchmark CMLPAP — localización continua de microhubs.

Familia 'microhubs_cont'. Reutiliza las 105 instancias geográficas del MLPAP
original (mismos clientes, demandas, sitios candidatos con sus costes fijos y
radio peatonal) y cambia lo que se decide: en lugar de un vector binario que dice
QUÉ sitios abrir, el vector de decisión son las COORDENADAS de los p hubs.

    id de función = "<instancia>@<nivel>"   p.ej. "L_01@high", "2XL_09@max"

    x = [x1, y1, x2, y2, ..., xp, yp] in R^{2p},  p = Pmax de la instancia

Cinco niveles de restricción —low, medium, high, extra, max— que escalan el radio
peatonal Dmax (ver LEVELS en cmlpap_core). Mismo eje de restricción que la familia
microhubs_tw, pero con el tornillo que tiene sentido físico aquí: cuanto más corto
el radio, más clientes quedan fuera de alcance y más se acoplan las ubicaciones.

Por qué existe esta familia: en los seis tiers se cumple 2*Pmax = m, así que la
dimensión de búsqueda coincide instancia por instancia con la de la familia
binaria. Misma geografía, misma dimensión, mismo presupuesto de evaluaciones — lo
único que cambia es la CODIFICACIÓN. Eso permite contrastar de forma controlada la
hipótesis de que el resultado neutro del SIA-online en microhubs se debe a la
binarización por función de transferencia (mapeo muchos-a-uno, paisaje con
mesetas) y no al dominio de aplicación.

    tier    n      m     p=Pmax   dim = 2p
    L      100     25      12        24
    XL     250     40      20        40
    2XL    500     70      35        70
    3XL   1.000   120      60       120
    4XL   2.000   200     100       200
    5XL   4.000   320     160       320

Como en BBOB/FIR, el óptimo es desconocido a priori (f_global=None).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmarks.microhubs_data import cmlpap_core as _core

_DATA_DIR = Path(__file__).resolve().parent / "microhubs_data"
_MANIFEST = _DATA_DIR / "manifest.json"

LEVELS = tuple(_core.LEVELS)          # ("low", "medium", "high", "extra", "max")

_INSTANCE_INDEX: dict[str, tuple[str, int]] | None = None
_PREPARED: dict[tuple[str, str], object] = {}


def _build_index() -> dict[str, tuple[str, int]]:
    global _INSTANCE_INDEX
    if _INSTANCE_INDEX is not None:
        return _INSTANCE_INDEX
    index: dict[str, tuple[str, int]] = {}
    for tier in json.loads(_MANIFEST.read_text())["tiers"]:
        records = json.loads((_DATA_DIR / tier["file"]).read_text())["instances"]
        for pos, rec in enumerate(records):
            index[rec["id"]] = (tier["file"], pos)
    _INSTANCE_INDEX = index
    return index


def split_name(name: str) -> tuple[str, str]:
    """'2XL_09@high' -> ('2XL_09', 'high'). Sin nivel explícito, 'medium'."""
    if "@" in name:
        inst_id, level = name.split("@", 1)
    else:
        inst_id, level = name, "medium"
    if level not in _core.LEVELS:
        raise ValueError(
            f"Nivel desconocido {level!r} en {name!r}. Usa uno de {list(_core.LEVELS)}."
        )
    return inst_id, level


def _prepare(name: str):
    inst_id, level = split_name(name)
    key = (inst_id, level)
    if key in _PREPARED:
        return _PREPARED[key]
    index = _build_index()
    if inst_id not in index:
        raise KeyError(
            f"Instancia microhubs desconocida: {inst_id!r}. "
            f"Usa una de las 105 ids con nivel, p.ej. 'L_01@high'."
        )
    fname, pos = index[inst_id]
    records = json.loads((_DATA_DIR / fname).read_text())["instances"]
    inst = _core.prepare(records[pos], level=level)
    _PREPARED[key] = inst
    return inst


def list_microhubs_cont_functions(levels: tuple[str, ...] = ("medium",)) -> list[str]:
    """Ids 'instancia@nivel'. Por defecto solo 'medium': es el nivel que da la
    comparación pareada directa contra la familia binaria ya corrida."""
    return [f"{i}@{lv}" for i in _build_index() for lv in levels]


def instance_dim(name: str) -> int:
    """Dimensión de búsqueda: 2p coordenadas. No depende del nivel."""
    return 2 * int(_prepare(name).p)


def supported_dims_for(name: str) -> tuple[int]:
    return (instance_dim(name),)


def get_microhubs_cont_problem(name: str, dim: int | None = None, seed: int = 0):
    """Devuelve (objective, lower, upper, info) para 'instancia@nivel'.

    La objective recibe x de longitud 2p —las coordenadas de los hubs, intercaladas
    (x1,y1,x2,y2,...)— y devuelve el coste total a minimizar. `seed` se acepta por
    compatibilidad con el registry pero no introduce aleatoriedad: f es una función
    pura de x.
    """
    inst = _prepare(name)
    inst_id, level = split_name(name)
    d = 2 * int(inst.p)
    if dim is not None and dim != d:
        raise ValueError(
            f"La instancia {inst_id} ubica p={inst.p} hubs, así que su dimensión es "
            f"2p={d}; pedida dim={dim}."
        )

    def objective(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        total, _base, _out, _served = _core.eval_pop(inst, x[None, :])
        return float(total[0])

    info = {
        "family": "microhubs_cont",
        "name": name,
        "instance": inst_id,
        "level": level,
        "dmax_scale": _core.LEVELS[level],
        "dim": d,
        "n_clients": int(inst.n),
        "n_sites": int(inst.m),
        "p_hubs": int(inst.p),
        "Dmax": float(inst.Dmax),
        "hub_capacity": float(inst.hub_capacity),
        "lower": float(inst.lower),
        "upper": float(inst.upper),
        "f_global": None,
    }
    return objective, float(inst.lower), float(inst.upper), info
