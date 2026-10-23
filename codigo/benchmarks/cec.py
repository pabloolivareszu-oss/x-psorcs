"""Suites CEC vía la librería opfunu (implementaciones oficiales).

Soporta las tres suites exigidas:
  - CEC2014: 30 funciones, dims 10/20/30/50/100.
  - CEC2017: 29 funciones (opfunu omite F2), dims 10/20/30/50/100.
  - CEC2022: 12 funciones, dims SOLO 10 y 20 (definición oficial).

Interfaz uniforme:
    get_cec_problem(name, dim, year) -> (objective, lower, upper, info)
con `objective(x)->float` (minimiza) y `lower/upper` escalares (dominios CEC son
hipercubos simétricos, p.ej. [-100, 100]).
"""

from __future__ import annotations

import numpy as np
import contextlib
import io
import os

import opfunu


class CECDimNotSupported(ValueError):
    """La función CEC concreta no admite la dimensión pedida (p.ej. F11 CEC2017 en 20D)."""


SUPPORTED_YEARS = (2014, 2017, 2022)
DEFAULT_YEAR = 2017

# Dimensiones oficiales soportadas por cada suite CEC.
SUPPORTED_DIMS = {
    2014: (10, 20, 30, 50, 100),
    2017: (10, 20, 30, 50, 100),
    2022: (10, 20),  # CEC2022 está definida solo en 10D y 20D
}

# Exclusiones por FUNCIÓN: en opfunu, ciertas funciones CEC están rotas en ciertas
# dims — (a) faltan matrices de rotación y construirlas hace sys.exit() (mata el
# proceso), o (b) devuelven NaN incluso en el óptimo. Se codifican explícitamente
# y se filtran ANTES de ejecutar para no contaminar los resultados con NaN.
#   Verificado empíricamente con opfunu 1.0.1.
EXCLUDED_FUNC_DIMS = {
    2017: {
        10: {"F17", "F29"},  # devuelven NaN en todo punto, incluido x_global
        20: {"F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19",
             "F28", "F29", "F30"},  # falta matriz de rotación -> sys.exit (F30 ni existe)
    },
    2014: {},
    2022: {},
}


def is_func_dim_supported(name: str, dim: int, year: int = DEFAULT_YEAR) -> bool:
    """True si esa función CEC concreta admite esa dimensión (sin construirla)."""
    if year not in SUPPORTED_DIMS:
        return False
    if dim not in SUPPORTED_DIMS[year]:
        return False
    canonical = name.upper() if str(name).upper().startswith("F") else f"F{name}"
    return canonical not in EXCLUDED_FUNC_DIMS.get(year, {}).get(dim, set())


@contextlib.contextmanager
def _silence_opfunu():
    """Silencia el printf de C de opfunu (mensajes 'M_*.txt not found')."""
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


def _func_class(name: str, year: int):
    """Resuelve 'F5' (o '5') a la clase opfunu, p.ej. F52017."""
    if year not in SUPPORTED_YEARS:
        raise KeyError(f"Año CEC no soportado: {year}. Disponibles: {SUPPORTED_YEARS}")
    module = getattr(opfunu.cec_based, f"cec{year}")
    num = "".join(ch for ch in str(name) if ch.isdigit())
    if not num:
        raise ValueError(f"Nombre CEC inválido: {name!r} (esperado 'F1'..).")
    cls_name = f"F{num}{year}"
    if not hasattr(module, cls_name):
        available = sorted(
            (a for a in dir(module) if a.startswith("F") and a.endswith(str(year))),
            key=lambda s: int(s[1:-4]),
        )
        raise KeyError(
            f"Función CEC{year} desconocida: {name} (clase {cls_name}). "
            f"Disponibles: {[f'F{a[1:-4]}' for a in available]}"
        )
    return getattr(module, cls_name)


def get_cec_problem(name: str, dim: int, year: int = DEFAULT_YEAR):
    """Devuelve (objective, lower, upper, info) para una función CEC.

    Lanza CECDimNotSupported si esa FUNCIÓN concreta no admite esa dimensión.
    En opfunu muchas funciones híbridas/compuestas de CEC2017 (F10-F30) solo
    tienen matrices de rotación para 10/30/50/100, no para 20; eso solo se sabe
    al construir/evaluar, no por la lista nominal de dims de la suite.
    """
    if dim not in SUPPORTED_DIMS[year]:
        raise CECDimNotSupported(
            f"CEC{year} no soporta dim={dim}. Dimensiones válidas: {SUPPORTED_DIMS[year]}."
        )
    if not is_func_dim_supported(name, dim, year):
        raise CECDimNotSupported(
            f"CEC{year} {name} no admite dim={dim} (sin matriz de rotación en opfunu)."
        )
    cls = _func_class(name, year)
    with _silence_opfunu():
        problem = cls(ndim=dim)

    lower = float(problem.lb[0])
    upper = float(problem.ub[0])

    def objective(x: np.ndarray) -> float:
        return float(problem.evaluate(np.asarray(x, dtype=float)))

    canonical = name.upper() if str(name).upper().startswith("F") else f"F{name}"
    info = {
        "family": f"CEC{year}",
        "year": year,
        "name": canonical,
        "dim": dim,
        "f_global": float(problem.f_global),
        "lower": lower,
        "upper": upper,
    }
    return objective, lower, upper, info


def list_cec_functions(year: int = DEFAULT_YEAR) -> list[str]:
    module = getattr(opfunu.cec_based, f"cec{year}")
    classes = [a for a in dir(module) if a.startswith("F") and a.endswith(str(year))]
    nums = sorted(int(a[1:-4]) for a in classes)
    return [f"F{n}" for n in nums]
