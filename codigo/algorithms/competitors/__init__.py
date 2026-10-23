"""Competidores reimplementados desde sus papers, con interfaz y presupuesto comunes.

Acceso uniforme:
    from algorithms.competitors import get_competitor, COMPETITORS
    comp = get_competitor("ASGK")           # instancia
    result = comp.optimize(obj, dim, lo, hi, max_fes, seed)
"""

# --- Competidores originales (tesis) ---
from algorithms.competitors.jade import JADE
from algorithms.competitors.apgsk_imode import APGSK_IMODE
from algorithms.competitors.ea4eig import EA4eig
from algorithms.competitors.umoeaii import UMOEAII
from algorithms.competitors.asgk import ASGK
from algorithms.competitors.amalgam import AMALGAM

# --- Competidores adicionales (segundo lote) ---
from algorithms.competitors.gwo import GWO
from algorithms.competitors.igwo import IGWO
from algorithms.competitors.hho import HHO
from algorithms.competitors.lhho import LHHO
from algorithms.competitors.dbo import DBO
from algorithms.competitors.bwo import BWO
from algorithms.competitors.so import SO
from algorithms.competitors.bka import BKA
from algorithms.competitors.goose import GOOSE
from algorithms.competitors.lshade import LSHADE
from algorithms.competitors.lshade_epsin import LSHADE_EpSin
from algorithms.competitors.ebowithcmar import EBOwithCMAR
from algorithms.competitors.sacs import SACS
from algorithms.competitors.krmg import KRMG
from algorithms.competitors.fgin import FGIN

_REGISTRY = {
    # originales
    "ASGK": ASGK,
    "EA4eig": EA4eig,
    "APGSK-IMODE": APGSK_IMODE,
    "AMALGAM": AMALGAM,
    "UMOEAII": UMOEAII,
    "JADE": JADE,
    # segundo lote
    "BKA": BKA,
    "GOOSE": GOOSE,
    "GWO": GWO,
    "HHO": HHO,
    "DBO": DBO,
    "BWO": BWO,
    "SO": SO,
    "IGWO": IGWO,
    "LHHO": LHHO,
    "LSHADE": LSHADE,
    "LSHADE-EpSin": LSHADE_EpSin,
    "EBOwithCMAR": EBOwithCMAR,
    "SACS": SACS,
    "KRMG": KRMG,
    "FGIN": FGIN,
}

COMPETITORS = list(_REGISTRY.keys())

# Subconjunto del SEGUNDO lote (los 15 nuevos que se piden ejecutar ahora).
NEW_COMPETITORS = [
    "BKA", "GOOSE", "GWO", "HHO", "DBO", "BWO", "SO", "IGWO", "LHHO",
    "LSHADE", "LSHADE-EpSin", "EBOwithCMAR", "SACS", "KRMG", "FGIN",
]


def get_competitor(name: str):
    """Devuelve una instancia del competidor por nombre (case-insensitive)."""
    for key, cls in _REGISTRY.items():
        if key.lower() == name.lower():
            return cls()
    raise KeyError(f"Competidor desconocido: {name}. Disponibles: {COMPETITORS}")
