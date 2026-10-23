"""Benchmark rail — caso ferroviario (programación de tripulaciones como set covering).

Tres instancias de la OR-Library (rail507, rail516, rail582): m piezas de trabajo
(filas) y n jornadas factibles enumeradas (columnas, ~47k–63k). El problema es

    min sum_j c_j x_j   s.a.   sum_j a_rj x_j >= 1  para toda fila r,  x in {0,1}^n

Implementación según la guía de traspaso del proyecto CPB
(caso_ferroviario/guia_caso_ferroviario.tex), reimplementada sin su código
(qmb_grid.py no está disponible):

  - Lector por COLUMNAS (formato rail, no el de las scp*): m n / por columna:
    costo, k, k filas (base 1). Doble CSR: por columna y por fila.
  - Reparación = Chvátal goloso (columna de mínimo costo / filas nuevas) +
    eliminación de redundancia de la más cara a la más barata. Variable
    controlada: idéntica para todos los algoritmos.
  - Binarización V2-ELITR (mejor esquema de la Tabla T10 de la guía):
        u = 2v - 1,  T = |tanh(u)|,  bit_j = e_j si r_j <= T_j, si no 0,
    con e un miembro de la élite sorteado por ruleta proporcional a su calidad.

Adaptación declarada: en CPB la élite es el 30 % de la generación actual. Aquí
SIA y los competidores evalúan de a un vector, así que la función objetivo
mantiene un ARCHIVO DE ÉLITE con las ELITE_SIZE (=12, 30 % de 40) mejores
soluciones reparadas vistas en la corrida. Con el archivo vacío la reparación
parte de la solución vacía: es la inicialización constructiva de la guía.
Consecuencia: el objetivo depende de la historia de evaluaciones de la corrida.

POR VERIFICAR contra el manuscrito CPB (eq:rules): la regla exacta de ELITR
(qué ocurre cuando r > T) y la normalización de "calidad" de la ruleta.

Dominio [0, 1]^n. f_global = BKS, de modo que error_to_optimum = f - BKS y el
RPD = 100 (f - BKS) / BKS se calcula en el análisis (analysis/rail_report.py).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
from numba import njit

DATA_DIR = Path(__file__).resolve().parent / "rail_data"
INSTANCES = ("rail507", "rail516", "rail582")
LOWER, UPPER = 0.0, 1.0
ELITE_SIZE = 12

_CACHE: dict[str, dict] = {}


# --------------------------------------------------------------- lectura
def _csr_transpose(col_ptr: np.ndarray, col_rows: np.ndarray, m: int):
    """De la vista por columna (col_ptr, col_rows) a la vista por fila."""
    n = len(col_ptr) - 1
    counts = np.bincount(col_rows, minlength=m)
    row_ptr = np.zeros(m + 1, dtype=np.int64)
    np.cumsum(counts, out=row_ptr[1:])
    col_of_entry = np.repeat(np.arange(n, dtype=np.int32), np.diff(col_ptr))
    order = np.argsort(col_rows, kind="stable")
    row_cols = col_of_entry[order].astype(np.int32)
    return row_ptr, row_cols


def load_rail(path: Path) -> dict:
    """Lee una instancia rail (formato por columnas, índices base 1)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        t = fh.read().split()
    p = 0
    m = int(t[p]); p += 1
    n = int(t[p]); p += 1
    cost = np.empty(n, dtype=np.float64)
    col_ptr = np.zeros(n + 1, dtype=np.int64)
    blocks = []
    for j in range(n):
        cost[j] = float(t[p]); p += 1
        k = int(t[p]); p += 1
        blocks.append(np.array(t[p:p + k], dtype=np.int32) - 1)
        p += k
        col_ptr[j + 1] = col_ptr[j] + k
    if p != len(t):
        raise ValueError(f"{path.name}: sobran {len(t) - p} tokens; ¿lector de filas en vez de columnas?")
    col_rows = np.concatenate(blocks).astype(np.int32)
    row_ptr, row_cols = _csr_transpose(col_ptr, col_rows, m)
    inst = dict(m=m, n=n, cost=cost, col_ptr=col_ptr, col_rows=col_rows,
                row_ptr=row_ptr, row_cols=row_cols)
    validate(inst)
    return inst


def validate(inst: dict) -> None:
    """Verificación mínima de la guía (Sec. 3.2)."""
    m = inst["m"]
    if inst["col_rows"].min() < 0 or inst["col_rows"].max() >= m:
        raise ValueError("Índices de fila fuera de rango: ¿base 1 sin convertir?")
    if len(inst["col_rows"]) != len(inst["row_cols"]):
        raise ValueError("Las dos vistas CSR no suman lo mismo.")
    empty = np.flatnonzero(np.diff(inst["row_ptr"]) == 0)
    if len(empty):
        raise ValueError(f"{len(empty)} filas sin ninguna columna que las cubra: lector desalineado.")


# --------------------------------------------------------------- reparación
@njit(cache=True)
def repair_eval(bits, cost, col_ptr, col_rows, row_ptr, row_cols, m):
    """Chvátal goloso + eliminación de redundancia. Modifica `bits` en sitio.

    Devuelve el costo de la solución factible resultante.
    """
    n = cost.shape[0]
    cover = np.zeros(m, dtype=np.int32)
    for j in range(n):
        if bits[j]:
            for q in range(col_ptr[j], col_ptr[j + 1]):
                cover[col_rows[q]] += 1

    n_unc = 0
    for r in range(m):
        if cover[r] == 0:
            n_unc += 1

    if n_unc > 0:
        # newcov[j] = filas descubiertas que cubriría la columna j (si está inactiva).
        newcov = np.zeros(n, dtype=np.int32)
        for r in range(m):
            if cover[r] == 0:
                for q in range(row_ptr[r], row_ptr[r + 1]):
                    newcov[row_cols[q]] += 1
        while n_unc > 0:
            best_j = -1
            best_ratio = np.inf
            for j in range(n):
                if newcov[j] > 0 and not bits[j]:
                    ratio = cost[j] / newcov[j]
                    if ratio < best_ratio:
                        best_ratio = ratio
                        best_j = j
            bits[best_j] = 1
            for q in range(col_ptr[best_j], col_ptr[best_j + 1]):
                r = col_rows[q]
                if cover[r] == 0:
                    n_unc -= 1
                    for q2 in range(row_ptr[r], row_ptr[r + 1]):
                        newcov[row_cols[q2]] -= 1
                cover[r] += 1

    # Eliminación de redundancia: de la columna activa más cara a la más barata.
    active = np.nonzero(bits)[0]
    order = active[np.argsort(-cost[active], kind="mergesort")]
    for j in order:
        redundant = True
        for q in range(col_ptr[j], col_ptr[j + 1]):
            if cover[col_rows[q]] < 2:
                redundant = False
                break
        if redundant:
            bits[j] = 0
            for q in range(col_ptr[j], col_ptr[j + 1]):
                cover[col_rows[q]] -= 1

    total = 0.0
    for j in range(n):
        if bits[j]:
            total += cost[j]
    return total


def is_feasible(bits: np.ndarray, inst: dict) -> bool:
    cover = np.bincount(inst["col_rows"][np.repeat(bits.astype(bool), np.diff(inst["col_ptr"]))],
                        minlength=inst["m"])
    return bool((cover >= 1).all())


# --------------------------------------------------------------- binarización
def v2_transfer(v: np.ndarray) -> np.ndarray:
    """V2 sobre la señal simetrizada u = 2v - 1 (código 2 de la guía)."""
    return np.abs(np.tanh(2.0 * v - 1.0))


class EliteArchive:
    """Las `size` mejores soluciones reparadas distintas vistas en la corrida."""

    def __init__(self, size: int = ELITE_SIZE):
        self.size = size
        self.bits: list[np.ndarray] = []
        self.costs: list[float] = []

    def offer(self, bits: np.ndarray, cost: float) -> None:
        if len(self.costs) == self.size and cost >= self.costs[-1]:
            return
        for c, b in zip(self.costs, self.bits):
            if c == cost and np.array_equal(b, bits):
                return
        k = int(np.searchsorted(self.costs, cost, side="right"))
        self.costs.insert(k, cost)
        self.bits.insert(k, bits.copy())
        if len(self.costs) > self.size:
            self.costs.pop(); self.bits.pop()

    def sample(self, rng: np.random.Generator) -> np.ndarray | None:
        """Ruleta proporcional a la calidad normalizada (minimización).

        POR VERIFICAR: normalización exacta de CPB. Aquí q_i = (f_max - f_i) /
        (f_max - f_min), con un piso para que el peor miembro no tenga peso 0.
        """
        if not self.costs:
            return None
        c = np.asarray(self.costs)
        span = c.max() - c.min()
        q = np.ones_like(c) if span <= 0 else (c.max() - c) / span + 0.05
        return self.bits[int(rng.choice(len(c), p=q / q.sum()))]


# --------------------------------------------------------------- problema
def _load_bks() -> dict:
    return json.loads((DATA_DIR / "bks.json").read_text(encoding="utf-8"))["instances"]


def _instance(name: str) -> dict:
    if name not in _CACHE:
        for cand in (DATA_DIR / f"{name}.txt", DATA_DIR / f"{name}.gz", DATA_DIR / name):
            if cand.exists():
                _CACHE[name] = load_rail(cand)
                break
        else:
            raise FileNotFoundError(
                f"No está la instancia {name} en {DATA_DIR}. Ejecuta: python tools/fetch_rail.py")
        if _CACHE[name]["m"] != int(name.replace("rail", "")):
            raise ValueError(f"{name}: m={_CACHE[name]['m']} no coincide con el nombre.")
    return _CACHE[name]


def list_rail_functions() -> list[str]:
    return list(INSTANCES)


def instance_dim(name: str) -> int:
    """Dimensión de búsqueda = n columnas. Se lee de bks.json sin cargar la instancia."""
    return int(_load_bks()[name]["n"])


def get_rail_problem(name: str, dim: int | None = None, seed: int = 0):
    """(objective, lower, upper, info) para una instancia rail.

    seed siembra el RNG de la binarización (se pasa run_id, como en microhubs).
    Se siembra de forma explícita con enteros: nada de hash(), que Python
    aleatoriza por proceso y rompió la reproducibilidad de microhubs.
    """
    if name not in INSTANCES:
        raise KeyError(f"Instancia rail desconocida: {name!r}. Usa {INSTANCES}.")
    inst = _instance(name)
    n, m = inst["n"], inst["m"]
    if dim is not None and dim != n:
        raise ValueError(f"{name} tiene dimensión fija n={n}; pedida dim={dim}.")
    bks = _load_bks()[name]

    rng = np.random.default_rng([int(seed), m])
    elite = EliteArchive()
    args = (inst["cost"], inst["col_ptr"], inst["col_rows"], inst["row_ptr"], inst["row_cols"], m)

    def objective(x: np.ndarray) -> float:
        T = v2_transfer(np.clip(np.asarray(x, dtype=float), 0.0, 1.0))
        e = elite.sample(rng)
        take = rng.random(n) <= T
        bits = np.zeros(n, dtype=np.uint8) if e is None else np.where(take, e, 0).astype(np.uint8)
        f = float(repair_eval(bits, *args))
        elite.offer(bits, f)
        return f

    objective.elite = elite  # acceso para verificación (tools/rail_smoke.py)
    info = {
        "family": "rail", "name": name, "dim": n, "m_rows": m,
        "lower": LOWER, "upper": UPPER,
        "f_global": float(bks["bks"]) if bks.get("bks") is not None else None,
        "bks_verified": bool(bks.get("verified", False)),
    }
    return objective, LOWER, UPPER, info
