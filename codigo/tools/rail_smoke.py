"""Pruebas de humo del caso ferroviario (fase 1 de CASO_FERROVIARIO.md).

Uso (desde RecursosPaper/codigo/):
    python tools/rail_smoke.py            # solo mini-instancia sintética (segundos)
    python tools/rail_smoke.py --real     # además rail516, 1 semilla, 2.000 FEs

Verifica:
  1. Lector: una mini-instancia escrita en formato rail se lee igual en las dos
     vistas CSR y la validación detecta filas no cubiertas.
  2. Reparación: la solución devuelta cubre todas las filas, el costo es la suma
     de las columnas activas y no queda ninguna columna redundante.
  3. Presupuesto: SIA-base con pop 40 y presupuesto exacto consume lo declarado.
  4. Reproducibilidad: dos corridas con la misma semilla dan el mismo best.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks import rail  # noqa: E402


def _write_mini(path: Path, rng: np.random.Generator, m=30, n=200) -> None:
    """Instancia aleatoria factible en formato por columnas (base 1)."""
    lines = [f"{m} {n}"]
    for j in range(n):
        k = int(rng.integers(1, 6))
        rows = rng.choice(m, size=k, replace=False) + 1
        lines.append(f"{int(rng.integers(1, 4))} {k} " + " ".join(map(str, rows)))
    # garantiza que toda fila esté cubierta: la última columna cubre todo, cara
    lines[-1] = f"50 {m} " + " ".join(str(r) for r in range(1, m + 1))
    path.write_text("\n".join(lines))


def check_repair(inst: dict, bits: np.ndarray) -> float:
    args = (inst["cost"], inst["col_ptr"], inst["col_rows"], inst["row_ptr"], inst["row_cols"], inst["m"])
    f = rail.repair_eval(bits, *args)
    assert rail.is_feasible(bits, inst), "la reparación dejó filas descubiertas"
    assert abs(f - inst["cost"][bits.astype(bool)].sum()) < 1e-9, "costo != suma de columnas activas"
    for j in np.flatnonzero(bits):  # ninguna activa redundante
        b2 = bits.copy(); b2[j] = 0
        assert not rail.is_feasible(b2, inst), f"la columna {j} quedó redundante"
    return f


def smoke_mini() -> None:
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "mini.txt"
        _write_mini(p, rng)
        inst = rail.load_rail(p)
    assert inst["m"] == 30 and inst["n"] == 200
    assert np.array_equal(np.sort(inst["col_rows"]), np.sort(np.repeat(np.arange(30), np.diff(inst["row_ptr"]))))
    print("[1] lector: OK (dos vistas CSR consistentes)")

    for trial, dens in enumerate((0.0, 0.02, 0.5, 1.0)):
        bits = (rng.random(inst["n"]) < dens).astype(np.uint8)
        f = check_repair(inst, bits)
        print(f"[2] reparación desde densidad {dens:.2f}: OK, costo {f:.0f}, {int(bits.sum())} columnas")

    # Instancia con la fila 3 sin ninguna columna: el lector debe rechazarla.
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bad.txt"
        p.write_text("3 2\n1 1 1\n1 1 2\n")
        try:
            rail.load_rail(p)
            raise AssertionError("la validación no detectó la fila sin cobertura")
        except ValueError:
            print("[1b] validación detecta filas sin cobertura: OK")


def smoke_real(budget: int = 2000) -> None:
    from experiments.run_sia_online import run_single
    pop = 40
    iters = (budget - pop) // pop
    out = []
    with tempfile.TemporaryDirectory() as d:
        for rep in (1, 2):
            s = run_single("rail", "rail516", rail.instance_dim("rail516"), 1, pop, iters,
                           3, 5, 10, 200, 7, Path(d) / f"r{rep}", "base")
            out.append(s)
    assert out[0]["function_evaluations"] == budget, out[0]["function_evaluations"]
    assert out[0]["best_fitness"] == out[1]["best_fitness"], "misma semilla, distinto best"
    s = out[0]
    rpd = 100 * (s["best_fitness"] - s["f_global"]) / s["f_global"] if s["f_global"] else None
    print(f"[3] presupuesto: {s['function_evaluations']} FEs = {budget}: OK")
    print(f"[4] reproducibilidad: best {s['best_fitness']:.0f} en las dos corridas: OK")
    print(f"    RPD vs BKS (sin verificar) = {rpd:.2f}%  |  tiempo {s['elapsed_seconds']} s")
    if rpd is not None and rpd < 0:
        raise SystemExit("RPD negativo: el BKS está mal. Detente.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="también rail516 con 2.000 FEs")
    a = ap.parse_args()
    smoke_mini()
    if a.real:
        smoke_real()
    print("\nHumo OK.")
