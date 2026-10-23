"""Benchmark MLPAP — Microhub Location and Pedestrian Assignment Problem.

Problema combinatorio binario (familia 'microhubs' en el pipeline): seleccionar
qué microhubs abrir (vector y in {0,1}^m) y, vía un decoder greedy determinista,
asignar cada cliente al hub abierto, alcanzable y con capacidad de menor distancia
ponderada. El objetivo penalizado es

    f~(z) = f_raw(z) + pi * v(z)

con f_raw = coste de instalación (apertura + operación) + coste de asignación
ponderado por prioridad, y v = magnitud agregada de violaciones (asignación,
capacidad, utilización mínima, presupuesto de hubs). Definición completa en el
modelo MLPAP (Olivares et al.) y en benchmarks/microhubs_data/mlpap_core.py.

105 instancias en 6 tiers (manifest.json):
    L (m=25, 20)  XL (m=40, 20)  2XL (m=70, 20)
    3XL (m=120, 15)  4XL (m=200, 15)  5XL (m=320, 15)
Cada instancia es una "función" de la familia, identificada por su id (p.ej.
'L_00', '3XL_07', '5XL_14'). La DIMENSIÓN de búsqueda es m (nº de hubs) de esa
instancia concreta: cada instancia conserva su propia dimensión.

Adaptación a optimizadores CONTINUOS (SIA y competidores):
  El algoritmo busca en x in [LOWER, UPPER]^m (continuo). La función objetivo
  binariza ese vector con un transfer sigmoide ESTOCÁSTICO:  y_j = 1 con
  probabilidad sigmoid(x_j)  (réplica del esquema de mlpap_run.py original). El
  RNG de la binarización se siembra de forma determinista por instancia+semilla,
  de modo que cada corrida es reproducible.

Como en BBOB/FIR, el óptimo f* es desconocido a priori (f_global=None); el
análisis usa el fitness crudo (best_fitness = f~ del mejor candidato).
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np

# Reutilizamos el core numba del paquete original (decoder + evaluación penalizada).
from benchmarks.microhubs_data import mlpap_core as _core

_DATA_DIR = Path(__file__).resolve().parent / "microhubs_data"
_MANIFEST = _DATA_DIR / "manifest.json"

# Dominio continuo de búsqueda para la binarización (igual que el paquete original).
LOWER = -4.0
UPPER = 4.0


def _load_manifest() -> list[dict]:
    return json.loads(_MANIFEST.read_text())["tiers"]


# Índice perezoso: id_instancia -> (archivo_tier, posición). Se construye una vez.
_INSTANCE_INDEX: dict[str, tuple[str, int]] | None = None
# Caché de instancias ya preparadas (core.Instance) por id.
_PREPARED: dict[str, object] = {}


def _build_index() -> dict[str, tuple[str, int]]:
    global _INSTANCE_INDEX
    if _INSTANCE_INDEX is not None:
        return _INSTANCE_INDEX
    index: dict[str, tuple[str, int]] = {}
    for tier in _load_manifest():
        path = _DATA_DIR / tier["file"]
        records = json.loads(path.read_text())["instances"]
        for pos, rec in enumerate(records):
            index[rec["id"]] = (tier["file"], pos)
    _INSTANCE_INDEX = index
    return index


def _prepare_instance(inst_id: str):
    """Devuelve el core.Instance (con arrays numba-ready) para esa id, cacheado."""
    if inst_id in _PREPARED:
        return _PREPARED[inst_id]
    index = _build_index()
    if inst_id not in index:
        raise KeyError(
            f"Instancia microhubs desconocida: {inst_id!r}. "
            f"Usa una de las 105 ids (p.ej. L_00, 3XL_07, 5XL_14)."
        )
    fname, pos = index[inst_id]
    records = json.loads((_DATA_DIR / fname).read_text())["instances"]
    inst = _core.prepare(records[pos])
    _PREPARED[inst_id] = inst
    return inst


def list_microhubs_functions() -> list[str]:
    """Las 105 ids de instancia, en el orden de los tiers del manifest."""
    return list(_build_index().keys())


def instance_dim(inst_id: str) -> int:
    """Dimensión de búsqueda (m = nº de hubs) de la instancia."""
    return int(_prepare_instance(inst_id).m)


def supported_dims_for(inst_id: str) -> tuple[int]:
    return (instance_dim(inst_id),)


def get_microhubs_problem(name: str, dim: int | None = None, seed: int = 0):
    """Devuelve (objective, lower, upper, info) para una instancia MLPAP.

    name : id de instancia ('L_00'..'5XL_14').
    dim  : se valida contra m de la instancia (cada instancia tiene la suya). Si
           se pasa None se usa m directamente.
    seed : siembra el RNG de la binarización estocástica (reproducibilidad).

    La objective recibe un vector continuo x de longitud m, lo binariza con
    sigmoid estocástico y devuelve f~ (objetivo penalizado, a minimizar).
    """
    inst = _prepare_instance(name)
    m = int(inst.m)
    if dim is not None and dim != m:
        raise ValueError(
            f"La instancia microhubs {name} tiene dimensión fija m={m} "
            f"(nº de hubs); pedida dim={dim}."
        )

    # RNG dedicado a la binarización: determinista por (instancia, semilla).
    # Mezclamos el id de instancia en la semilla para que dos instancias con la
    # misma seed de corrida no compartan exactamente la secuencia de bits.
    # OJO: se usa crc32 y NO hash(), porque el hash de str en Python está
    # aleatorizado por proceso (PYTHONHASHSEED): con hash() cada worker generaba
    # una secuencia de binarización distinta para la misma (instancia, semilla),
    # de modo que las corridas no eran reproducibles y A vs A' ni siquiera
    # compartían la misma realización del ruido.
    bin_seed = (int(seed) * 1_000_003 + (zlib.crc32(name.encode()) % 1_000_003)) % (2**32)
    rng = np.random.default_rng(bin_seed)

    def objective(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        # Transfer sigmoide ESTOCÁSTICO: y_j = 1 con prob sigmoid(x_j).
        probs = _core.sigmoid(x)
        y = (rng.random(probs.shape) < probs).astype(np.int64)
        # eval_pop evalúa una población; aquí una sola fila.
        ft, _fr, _vt = _core.eval_pop(inst, y[None, :])
        return float(ft[0])

    info = {
        "family": "microhubs",
        "name": name,
        "dim": m,
        "n_clients": int(inst.n),
        "m_hubs": m,
        "Pmin": int(inst.Pmin),
        "Pmax": int(inst.Pmax),
        "lower": LOWER,
        "upper": UPPER,
        "f_global": None,   # óptimo desconocido a priori (como BBOB/FIR)
    }
    return objective, LOWER, UPPER, info
