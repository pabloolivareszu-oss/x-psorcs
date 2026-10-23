"""Sonda de memoria (herramienta de diagnóstico): ¿crece la memoria de un proceso de corrida en corrida?

Replica lo que hace una celda de run.py: un mismo proceso encadena varias corridas
del mismo método sobre la misma instancia. Tras cada corrida registra la memoria
residente (RSS) del proceso. Uso:

    sonda_memoria.py <metodo> <instancia> <corridas> <evaluaciones> <salida.json>
"""
import gc, json, sys, time
from pathlib import Path

sys.path.insert(0, "/Users/server/TesisMDJuan")
import psutil

metodo, instancia, corridas, fes, salida = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
proc = psutil.Process()
mb = lambda: proc.memory_info().rss / 2**20

import warnings; warnings.filterwarnings("ignore")
from benchmarks.registry import rail_dim_for
dim = rail_dim_for(instancia)
pop = 40
registro = {"metodo": metodo, "instancia": instancia, "evaluaciones": fes,
            "rss_mb_inicio": round(mb(), 1), "corridas": []}

base = Path(salida).parent / f"sonda_out_{metodo}_{instancia}"
for r in range(1, corridas + 1):
    t = time.perf_counter()
    seed = dim * 1000 + r
    if metodo.startswith("SIA"):
        from experiments.run_sia_online import run_single
        run_single(family="rail", benchmark=instancia, dim=dim, run_id=r, pop_size=pop,
                   max_iter=(fes - pop) // pop, horizon=3, update_interval=5, xai_interval=10,
                   window_size=200, seed=seed, out_dir=base / f"run_{r:02d}",
                   mode="online" if metodo == "SIA-online" else "base")
    else:
        from experiments.run_competitors import run_one
        run_one(metodo, "rail", instancia, dim, r, fes, seed, base / f"run_{r:02d}")
    antes_gc = mb(); gc.collect(); despues_gc = mb()
    registro["corridas"].append({"corrida": r, "segundos": round(time.perf_counter() - t, 1),
                                 "rss_mb": round(antes_gc, 1), "rss_mb_tras_gc": round(despues_gc, 1)})
    Path(salida).write_text(json.dumps(registro, indent=1))
print(json.dumps(registro))
