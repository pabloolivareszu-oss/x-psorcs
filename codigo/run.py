"""Orquestador único del experimento completo (paralelizado).

Corre TODO el flujo de la tesis:
  1. SIA-base   (A)   y  SIA-online (A')   sobre todas las familias.
  2. Competidores: los 20 registrados.
       originales (6): ASGK, EA4eig, APGSK-IMODE, AMALGAM, UMOEAII, JADE
       2º lote (15):   BKA, GOOSE, GWO, HHO, DBO, BWO, SO, IGWO, LHHO, LSHADE,
                       LSHADE-EpSin, EBOwithCMAR, SACS, KRMG, FGIN
  3. Análisis por familia (Wilcoxon, Friedman, figuras, tabla resumen).

Familias y dimensiones (se filtran automáticamente a las soportadas):
  - cec2014, cec2017 : dims 10, 20, 50
  - cec2022          : dims 10, 20         (no soporta 50)
  - bbob             : dims 10, 20         (no soporta 50)
  - classical        : dims 10, 20, 50
  - fir              : dim 31 (fija)       (diseño de filtros FIR, FIR1..FIR8)
  - microhubs        : dim = m por instancia (MLPAP, 105 instancias en 6 tiers)
  - rail             : dim = n columnas por instancia (rail507/516/582, set covering).
                       Protocolo propio (CPB): ver CASO_FERROVIARIO.md.

Paralelización: pool de procesos (default 10). La unidad de trabajo es una celda
(algoritmo, familia, función, dimensión); cada worker ejecuta sus N corridas.
El progreso se imprime por celda y se persiste en results/run_status.json, lo que
permite REANUDAR: al relanzar, las celdas ya completadas se saltan.

Uso:
    python run.py                      # experimento completo, 10 procesos
    python run.py --jobs 14            # más procesos
    python run.py --smoke             # prueba rápida (pocas funciones/corridas)
    python run.py --families cec2017  # solo una familia
    python run.py --no-analysis       # solo generar datos, sin análisis
    python run.py --resume            # reanuda saltando celdas completadas
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

# Silencia warnings ruidosos (requests/urllib3, RuntimeWarning de opfunu, etc.).
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from benchmarks.registry import list_functions, supported_dims  # noqa: E402
from paths import RESULTS  # noqa: E402  (raíz única; ver paths.py)

STATUS_FILE = RESULTS / "run_status.json"
LOG_FILE = RESULTS / "run.log"          # todo el output de consola
ERROR_FILE = RESULTS / "errors.log"     # solo los fallos, con su mensaje


class _Tee:
    """Duplica lo impreso a consola Y a un archivo de log (line-buffered)."""

    def __init__(self, stream, path: Path):
        self.stream = stream
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, data):
        self.stream.write(data)
        self.fh.write(data)

    def flush(self):
        self.stream.flush()
        self.fh.flush()


def log_error(key: str, error: str) -> None:
    """Añade un fallo al archivo de errores con timestamp."""
    ERROR_FILE.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(ERROR_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {key}  ->  {error}\n")

# Algoritmos competidores: TODOS los registrados (5 originales + 15 del 2º lote).
#   originales: ASGK, EA4eig, APGSK-IMODE, AMALGAM, UMOEAII, JADE
#   2º lote:    BKA, GOOSE, GWO, HHO, DBO, BWO, SO, IGWO, LHHO, LSHADE,
#               LSHADE-EpSin, EBOwithCMAR, SACS, KRMG, FGIN
from algorithms.competitors import COMPETITORS as _ALL_COMPETITORS
COMPETITORS = list(_ALL_COMPETITORS)
# Modos del SIA: A (base) y A' (online).
SIA_MODES = ["base", "online"]

# Familias y dimensiones objetivo (se filtran a las válidas por familia).
FAMILY_DIMS = {
    "cec2014": [10, 20, 50],
    "cec2017": [10, 20, 50],
    "cec2022": [10, 20, 50],   # 50 se filtra (cec2022 solo 10/20)
    "bbob":    [10, 20, 50],   # 50 se filtra (bbob no soporta 50)
    "classical": [10, 20, 50],
    "fir":     [31],           # FIR: dimensión fija D = orden(30) + 1 = 31
    # antenna y pv: la dim depende del CASO (16/32 y 5/7/9 respectivamente);
    # build_tasks la resuelve por función. Placeholder, no se usa.
    "antenna": [0],
    "pv":      [0],
    # microhubs: la dim NO es de la familia (cada instancia tiene su m = nº de
    # hubs, de 25 a 320). build_tasks resuelve la dim por instancia; este valor
    # es solo un placeholder y no se usa para esta familia.
    "microhubs": [0],
    # rail: igual que microhubs, la dim es n (columnas) de cada instancia.
    "rail": [0],
    # microhubs_cont: ídem, la dim es 2p coordenadas de la instancia (24 a 320).
    "microhubs_cont": [0],
}

# Presupuesto EXACTO en todas las familias: SIA gasta pop evaluaciones en la
# inicialización, así que corre (max_fes - pop) // pop iteraciones y consume
# exactamente max_fes, igual que los competidores (30 + 99*30 = 3000; en rail,
# 40 + 499*40 = 20000). La campaña original usaba pop + iters*pop = 3030 para SIA
# frente a 3000 de los competidores; desde la campaña v2 el presupuesto es idéntico.
EXACT_BUDGET_ALL_FAMILIES = True

# Presupuesto por defecto: pop x iter = 30 x 100 = 3000 FEs.
DEFAULT_POP = 30
DEFAULT_ITER = 100
DEFAULT_RUNS = 30


# --------------------------------------------------------------- memoria
class MemoryGuard(threading.Thread):
    """Vigila la memoria disponible del sistema mientras trabaja el pool.

    Si cae por debajo de la reserva, mata el proceso worker con más memoria
    residente antes de que el sistema empiece a paginar y se cuelgue (es lo que
    obligó a reiniciar la máquina en la campaña v2). Matar un worker rompe el pool:
    el bucle principal lo detecta, reduce el paralelismo y reanuda las celdas que no
    terminaron, que retoman desde su última corrida completa (_completed_run).
    """

    def __init__(self, reserve_gb: float, interval: float = 3.0, settle: float = 20.0,
                 report_every: float = 300.0):
        super().__init__(daemon=True)
        self.reserve = reserve_gb * 2**30
        self.interval = interval
        self.settle = settle          # espera tras matar, para que la memoria se libere
        self.report_every = report_every   # cada cuánto se registra el estado de memoria
        self.trips: list[dict] = []
        self._stop_evt = threading.Event()

    @staticmethod
    def snapshot(me, top: int = 5) -> dict:
        """Estado de memoria: disponible, lo que ocupa la campaña y los procesos
        mayores del sistema (agrupados por nombre). Deja constancia en el log para
        poder diagnosticar una saturación aunque la cause un proceso externo."""
        import collections
        import psutil
        vm = psutil.virtual_memory()
        ours = 0
        for p in [me] + me.children(recursive=True):
            try:
                ours += p.memory_info().rss
            except psutil.Error:
                pass
        by_name: collections.Counter = collections.Counter()
        for p in psutil.process_iter(["name", "memory_info"]):
            # Procesos de otros usuarios o del sistema: psutil devuelve None si no hay
            # permiso para leer su memoria. Se omiten.
            mi = p.info.get("memory_info")
            if mi is not None:
                by_name[p.info.get("name") or "?"] += mi.rss
        return {"disponible_gb": round(vm.available / 2**30, 1),
                "campana_gb": round(ours / 2**30, 1),
                "swap_gb": round(psutil.swap_memory().used / 2**30, 1),
                "mayores": [(n, round(r / 2**30, 1)) for n, r in by_name.most_common(top)]}

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        # La guardia no puede morir: si un hilo de protección cae en silencio, la
        # campaña sigue corriendo sin protección, que es el peor modo de fallo. Todo
        # error se registra y la vigilancia continúa.
        import psutil
        me = psutil.Process()
        state = {"last_report": 0.0}
        while not self._stop_evt.wait(self.interval):
            try:
                self._check(psutil, me, state)
            except Exception as e:  # noqa: BLE001
                print(f"{time.strftime('%H:%M:%S')} [guardia de memoria] error interno "
                      f"(la vigilancia continúa): {type(e).__name__}: {e}")
                log_error("guardia de memoria (error interno)", f"{type(e).__name__}: {e}")

    def _check(self, psutil, me, state: dict) -> None:
        """Una comprobación: registro periódico y, si hace falta, detener un worker."""
        avail = psutil.virtual_memory().available
        if time.monotonic() - state["last_report"] >= self.report_every:
            state["last_report"] = time.monotonic()
            s = self.snapshot(me)
            print(f"{time.strftime('%H:%M:%S')} [memoria] disponible {s['disponible_gb']} GB, "
                  f"campaña {s['campana_gb']} GB, swap {s['swap_gb']} GB; "
                  f"mayores: {', '.join(f'{n} {g}' for n, g in s['mayores'])}")
        if avail >= self.reserve:
            return
        workers = []
        for p in me.children(recursive=True):
            try:
                if "resource_tracker" in " ".join(p.cmdline()):
                    continue
                workers.append((p.memory_info().rss, p))
            except psutil.Error:
                pass
        if not workers:
            return
        rss, victim = max(workers, key=lambda w: w[0])
        try:
            victim.kill()
        except psutil.Error:
            return
        trip = {"hora": time.strftime("%Y-%m-%d %H:%M:%S"), "pid": victim.pid,
                "rss_gb": round(rss / 2**30, 2), "disponible_gb": round(avail / 2**30, 2),
                "estado_sistema": self.snapshot(me, top=10)}
        self.trips.append(trip)
        print(f"{time.strftime('%H:%M:%S')} [guardia de memoria] disponible "
              f"{trip['disponible_gb']} GB < reserva {self.reserve / 2**30:.0f} GB: "
              f"detenido el worker {trip['pid']} ({trip['rss_gb']} GB)")
        log_error("guardia de memoria", json.dumps(trip, ensure_ascii=False))
        self._stop_evt.wait(self.settle)


# --------------------------------------------------------------- worker
def _completed_run(out_dir: Path, seed: int, **expected) -> dict | None:
    """summary.json de una corrida YA completada con esta misma configuración, o None.

    Permite reanudar una celda por corrida: si el proceso se interrumpe (guardia de
    memoria, corte de luz, reinicio), al relanzar solo se repiten las corridas que
    faltan. Es seguro porque summary.json es lo último que escriben los runners, y
    porque cada corrida tiene su propia semilla: saltarse las ya hechas da el mismo
    resultado que no haberse interrumpido. Se exige que coincidan la semilla y los
    parámetros de presupuesto, para no reutilizar nunca una corrida de otra
    configuración (por ejemplo, la de un piloto).
    """
    p = out_dir / "summary.json"
    if not p.exists():
        return None
    try:
        s = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None   # escrito a medias: la corrida se repite
    if s.get("seed") != seed or any(s.get(k) != v for k, v in expected.items()):
        return None
    return s


def _run_cell(task: dict) -> dict:
    """Ejecuta TODAS las corridas de una celda (algoritmo, familia, func, dim).

    Se ejecuta en un proceso worker. Importa los runners aquí para que cada
    proceso tenga su propio estado (evita problemas de fork en Windows-spawn).
    """
    import warnings as _w
    _w.filterwarnings("ignore")
    from experiments.run_sia_online import run_single, RESULTS_ROOT as SIA_ROOT
    from experiments.run_competitors import run_one, RESULTS_ROOT as COMP_ROOT

    algo = task["algo"]            # 'SIA-base' | 'SIA-online' | nombre competidor
    family = task["family"]
    func = task["func"]
    dim = task["dim"]
    runs = task["runs"]
    pop = task["pop"]
    iters = task["iters"]
    max_fes = pop * iters

    t0 = time.perf_counter()
    bests = []
    reused = 0                     # corridas que ya estaban completas en disco
    try:
        for run_id in range(1, runs + 1):
            seed = dim * 1000 + run_id
            if algo in ("SIA-base", "SIA-online"):
                mode = "base" if algo == "SIA-base" else "online"
                out_dir = SIA_ROOT / algo / family / func / f"dim_{dim}" / f"run_{run_id:02d}"
                sia_iters = (max_fes - pop) // pop if EXACT_BUDGET_ALL_FAMILIES else iters
                s = _completed_run(out_dir, seed, pop_size=pop, max_iter=sia_iters, mode=mode)
                if s is None:
                    s = run_single(
                        family=family, benchmark=func, dim=dim, run_id=run_id,
                        pop_size=pop, max_iter=sia_iters, horizon=3, update_interval=5,
                        xai_interval=10, window_size=200, seed=seed,
                        out_dir=out_dir, mode=mode,
                    )
                else:
                    reused += 1
            else:
                out_dir = COMP_ROOT / algo / family / func / f"dim_{dim}" / f"run_{run_id:02d}"
                s = _completed_run(out_dir, seed, max_fes=max_fes)
                if s is None:
                    s = run_one(algo, family, func, dim, run_id, max_fes, seed, out_dir)
                else:
                    reused += 1
            bests.append(s["best_fitness"])
        ok = True
        err = None
    except Exception as e:  # noqa: BLE001 — reportamos el fallo, no abortamos el pool
        ok = False
        err = f"{type(e).__name__}: {e}"

    return {
        "key": task["key"], "algo": algo, "family": family, "func": func, "dim": dim,
        "ok": ok, "error": err,
        "best": min(bests) if bests else None,
        "n_done": len(bests),
        "n_reused": reused,
        "seconds": round(time.perf_counter() - t0, 1),
    }


# --------------------------------------------------------------- planning
def _valid_func_dim(family: str, func: str, dim: int) -> bool:
    """True si esa función concreta admite esa dimensión.

    Varias funciones híbridas/compuestas de CEC2017 (F10-F19, F28-F30) no tienen
    matriz de rotación para dim 20 en opfunu, y construirlas mata el proceso
    (sys.exit interno). Por eso se filtran por una lista conocida SIN construir.
    BBOB/classical no tienen este problema.
    """
    if not family.lower().startswith("cec"):
        return True
    from benchmarks.cec import is_func_dim_supported
    digits = "".join(ch for ch in family if ch.isdigit())
    year = int(digits) if digits else 2017
    return is_func_dim_supported(func, dim, year)


def build_tasks(families: list[str], algos: list[str], runs: int,
                pop: int, iters: int, funcs_override: dict | None) -> list[dict]:
    """Genera la lista de celdas (algoritmo, familia, función, dim) a ejecutar.

    Filtra de antemano las (función, dim) que no existen en la suite (p.ej. CEC2017
    híbridas en 20D), para que no aparezcan como fallos.
    """
    tasks = []
    n_skipped = 0
    for family in families:
        funcs = (funcs_override or {}).get(family) or list_functions(family)
        # microhubs: la dimensión NO es de la familia sino de cada instancia (m
        # hubs). Para cada instancia su dim es su propio m, no las dims genéricas.
        if family == "microhubs":
            from benchmarks.registry import microhubs_dim_for
            func_dims = {func: [microhubs_dim_for(func)] for func in funcs}
        elif family == "rail":
            from benchmarks.registry import rail_dim_for
            func_dims = {func: [rail_dim_for(func)] for func in funcs}
        elif family == "microhubs_cont":
            from benchmarks.registry import microhubs_cont_dim_for
            func_dims = {func: [microhubs_cont_dim_for(func)] for func in funcs}
        elif family == "antenna":
            from benchmarks.registry import antenna_dim_for
            func_dims = {func: [antenna_dim_for(func)] for func in funcs}
        elif family == "pv":
            from benchmarks.registry import pv_dim_for
            func_dims = {func: [pv_dim_for(func)] for func in funcs}
        else:
            dims = supported_dims(family, FAMILY_DIMS.get(family, [10, 20, 50]))
            func_dims = {func: dims for func in funcs}
        for func in funcs:
            for dim in func_dims[func]:
                if not _valid_func_dim(family, func, dim):
                    n_skipped += 1
                    continue
                for algo in algos:
                    key = f"{algo}|{family}|{func}|d{dim}"
                    tasks.append({
                        "key": key, "algo": algo, "family": family,
                        "func": func, "dim": dim, "runs": runs,
                        "pop": pop, "iters": iters,
                    })
    if n_skipped:
        print(f"[plan] {n_skipped} combinaciones (función, dim) no existen en la "
              f"suite y se omiten (p.ej. CEC2017 híbridas en 20D).")
    return tasks


def load_status() -> dict:
    if STATUS_FILE.exists():
        try:
            return json.loads(STATUS_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_status(status: dict) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(status, indent=2))


# --------------------------------------------------------------- analysis
def run_analysis_phase(families: list[str], funcs_override: dict | None,
                       with_competitors: list[str]) -> None:
    """Lanza run_analysis por familia (en el proceso principal)."""
    from experiments import run_analysis as ra

    for family in families:
        funcs = (funcs_override or {}).get(family) or list_functions(family)
        if family == "rail":
            # rail tiene su propio reporte (RPD, Wilcoxon por instancia sobre semillas,
            # Finner y Vargha-Delaney): el análisis genérico hace Friedman sobre celdas
            # y superficies 3D, que no aplican con 3 instancias de ~60k variables.
            from analysis import rail_report
            print(f"\n=== ANÁLISIS rail ({len(funcs)} instancias) ===")
            rail_report.main(funcs)
            continue
        if family == "microhubs":
            # Dims = el conjunto de m distintos entre las instancias pedidas
            # (uno por tier: 25, 40, 70, 120, 200, 320).
            from benchmarks.registry import microhubs_dim_for
            dims = sorted({microhubs_dim_for(f) for f in funcs})
        elif family == "microhubs_cont":
            from benchmarks.registry import microhubs_cont_dim_for
            dims = sorted({microhubs_cont_dim_for(f) for f in funcs})
        elif family == "antenna":
            from benchmarks.registry import antenna_dim_for
            dims = sorted({antenna_dim_for(f) for f in funcs})
        elif family == "pv":
            from benchmarks.registry import pv_dim_for
            dims = sorted({pv_dim_for(f) for f in funcs})
        else:
            dims = supported_dims(family, FAMILY_DIMS.get(family, [10, 20, 50]))
        argv = [
            "--family", family,
            "--benchmarks", *funcs,
            "--dims", *map(str, dims),
            "--with-competitors", *with_competitors,
        ]
        print(f"\n=== ANÁLISIS {family} ({len(funcs)} funcs, dims {dims}) ===")
        old = sys.argv
        try:
            sys.argv = ["run_analysis", *argv]
            ra.main()
        except SystemExit:
            pass
        except Exception as e:  # noqa: BLE001
            print(f"[análisis {family}] error: {type(e).__name__}: {e}")
        finally:
            sys.argv = old


# --------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description="Orquestador del experimento completo")
    p.add_argument("--jobs", type=int, default=10, help="Procesos paralelos (default 10).")
    p.add_argument("--families", nargs="+", default=list(FAMILY_DIMS),
                   help="Familias a correr (default: todas).")
    p.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    p.add_argument("--pop-size", type=int, default=DEFAULT_POP)
    p.add_argument("--max-iter", type=int, default=DEFAULT_ITER)
    p.add_argument("--resume", action="store_true",
                   help="Salta celdas ya completadas (según run_status.json).")
    p.add_argument("--no-analysis", action="store_true", help="Solo datos, sin análisis.")
    p.add_argument("--smoke", action="store_true",
                   help="Prueba rápida: 2 funcs/familia, 3 corridas, 30 iters.")
    p.add_argument("--mem-reserve-gb", type=float, default=8.0,
                   help="Memoria del sistema que la guardia mantiene libre (default 8 GB): "
                        "por debajo, detiene el worker más grande y reduce el paralelismo.")
    p.add_argument("--funcs", nargs="+", default=None,
                   help="Restringe las funciones/instancias (p.ej. --funcs rail516 para el piloto).")
    p.add_argument("--only", nargs="+", default=None,
                   help="Corre SOLO estos algoritmos (p.ej. GWO HHO LSHADE). "
                        "No toca los resultados de los demás.")
    p.add_argument("--new-competitors", action="store_true",
                   help="Atajo: corre SOLO los 15 competidores nuevos del segundo "
                        "lote (BKA, GOOSE, GWO, HHO, DBO, BWO, SO, IGWO, LHHO, "
                        "LSHADE, LSHADE-EpSin, EBOwithCMAR, SACS, KRMG, FGIN), "
                        "sin SIA ni borrar nada.")
    args = p.parse_args()

    # Tee: a partir de aquí, todo lo impreso va también a results/run.log.
    RESULTS.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(sys.__stdout__, LOG_FILE)
    print(f"\n{'='*60}\nINICIO {time.strftime('%Y-%m-%d %H:%M:%S')}  "
          f"(log: {LOG_FILE.name}, errores: {ERROR_FILE.name})\n{'='*60}")

    families = args.families
    runs, pop, iters = args.runs, args.pop_size, args.max_iter
    funcs_override = None
    if args.funcs:
        funcs_override = {f: list(args.funcs) for f in families}

    if args.smoke:
        runs, iters = 3, 30
        funcs_override = {f: list_functions(f)[:2] for f in families}
        print("[smoke] modo prueba: 2 funcs/familia, 3 corridas, 30 iters.")

    # Selección de algoritmos a ejecutar.
    if args.new_competitors:
        from algorithms.competitors import NEW_COMPETITORS
        algos = list(NEW_COMPETITORS)
        print(f"[modo] solo competidores nuevos ({len(algos)}): {algos}")
    elif args.only:
        algos = list(args.only)
        print(f"[modo] solo algoritmos: {algos}")
    else:
        algos = ["SIA-base", "SIA-online"] + COMPETITORS
    tasks = build_tasks(families, algos, runs, pop, iters, funcs_override)

    # El estado se carga SIEMPRE, para que cada invocación se acumule sobre las
    # anteriores en vez de sobrescribirlas (antes, sin --resume, arrancaba vacío y
    # al guardar borraba las celdas de invocaciones previas). --resume solo decide
    # si se saltan las celdas ya completadas.
    status = load_status()
    if args.resume:
        pending = [t for t in tasks if status.get(t["key"], {}).get("ok") is not True]
    else:
        pending = list(tasks)
    skipped = len(tasks) - len(pending)

    total_runs = len(pending) * runs
    print(f"Familias: {families}")
    print(f"Algoritmos: {algos}")
    print(f"Celdas totales: {len(tasks)}  |  pendientes: {len(pending)}  "
          f"|  saltadas (ya hechas): {skipped}")
    print(f"Corridas a ejecutar: {total_runs}  |  procesos: {args.jobs}")
    print(f"Presupuesto: pop={pop} x iter={iters} = {pop*iters} FEs por corrida\n")

    if not pending:
        print("Nada pendiente. (¿--resume con todo completo?)")
    else:
        t_start = time.perf_counter()
        cells_done = 0
        runs_done = 0                 # progreso por CORRIDA (1/total_runs)
        failures = []
        jobs = args.jobs
        remaining = list(pending)
        guard = MemoryGuard(args.mem_reserve_gb)
        guard.start()
        print(f"Guardia de memoria: reserva {args.mem_reserve_gb:g} GB; "
              f"un proceso nuevo por celda; reanudación por corrida.")
        try:
            while remaining:
                broken = False
                # max_tasks_per_child=1: cada celda en un proceso nuevo, así la memoria
                # que retiene un worker vuelve al sistema al terminar la celda en vez
                # de acumularse durante toda la campaña.
                with ProcessPoolExecutor(max_workers=jobs, max_tasks_per_child=1) as ex:
                    futs = {ex.submit(_run_cell, t): t for t in remaining}
                    for fut in as_completed(futs):
                        try:
                            res = fut.result()
                        except BrokenProcessPool:
                            broken = True     # la guardia mató un worker
                            break
                        remaining.remove(futs[fut])
                        status[res["key"]] = {
                            "ok": res["ok"], "error": res["error"],
                            "best": res["best"], "n_done": res["n_done"],
                            "n_reused": res["n_reused"], "seconds": res["seconds"],
                        }
                        save_status(status)  # persistir tras cada celda (reanudable)
                        cells_done += 1
                        # Cada celda aporta sus corridas al contador global por corrida.
                        runs_done += res["n_done"]
                        tag = "OK " if res["ok"] else "FALLO"
                        extra = "" if res["ok"] else f"  <-- {res['error']}"
                        if res["n_reused"]:
                            extra += f"  ({res['n_reused']} corridas ya completas, reanudada)"
                        if not res["ok"]:
                            failures.append(res["key"])
                            log_error(res["key"], res["error"])  # a results/errors.log
                        bests = f"best={res['best']:.3e}" if res["best"] is not None else "best=NA"
                        pct = 100 * runs_done / total_runs if total_runs else 100.0
                        eta = _eta(t_start, runs_done, total_runs)
                        print(f"{time.strftime('%H:%M:%S')} [{runs_done}/{total_runs} {pct:5.1f}%] "
                              f"(celda {cells_done}/{len(pending)}) {tag} {res['key']:<40} "
                              f"{bests}  {res['seconds']:>5}s  ETA {eta}{extra}")
                if not broken:
                    continue
                if jobs > 1:
                    jobs = max(1, jobs - max(1, jobs // 4))
                    print(f"{time.strftime('%H:%M:%S')} [guardia de memoria] pool detenido; "
                          f"se reanudan {len(remaining)} celdas con {jobs} procesos")
                else:
                    # Ni en solitario cabe: con un solo proceso las celdas corren en
                    # orden, así que la que estaba en curso es la primera pendiente.
                    t = remaining.pop(0)
                    err = "memoria insuficiente incluso con 1 proceso (guardia de memoria)"
                    status[t["key"]] = {"ok": False, "error": err, "best": None,
                                        "n_done": 0, "n_reused": 0, "seconds": 0.0}
                    save_status(status)
                    cells_done += 1
                    failures.append(t["key"])
                    log_error(t["key"], err)
                    print(f"{time.strftime('%H:%M:%S')} FALLO {t['key']}  <-- {err}")
        finally:
            guard.stop()

        elapsed = time.perf_counter() - t_start
        print(f"\nFIN {time.strftime('%Y-%m-%d %H:%M:%S')}  "
              f"Datos completos en {elapsed/60:.1f} min. "
              f"Corridas: {runs_done}/{total_runs}. "
              f"Celdas OK: {cells_done - len(failures)}/{cells_done}.")
        if failures:
            print(f"FALLOS ({len(failures)}): {failures[:10]}"
                  f"{' ...' if len(failures) > 10 else ''}")

    # Fase de análisis.
    if not args.no_analysis:
        run_analysis_phase(families, funcs_override, COMPETITORS)

    print("\n=== EXPERIMENTO COMPLETO ===")
    print(f"Resultados en: {RESULTS}")
    print(f"Estado/progreso: {STATUS_FILE}")


def _eta(t_start: float, done: int, total: int) -> str:
    if done == 0:
        return "??"
    rate = (time.perf_counter() - t_start) / done
    rem = rate * (total - done)
    m, s = divmod(int(rem), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


if __name__ == "__main__":
    main()
