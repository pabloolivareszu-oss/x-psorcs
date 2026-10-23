"""Escribe resultados/config.json: el manifiesto de la campaña v3.

Registra todo lo necesario para reproducir la campaña o auditarla después: el
commit exacto del código, la máquina, los hilos, el protocolo de cada familia, los
algoritmos (con los excluidos y su motivo) y los parámetros EFECTIVOS del
controlador adaptativo, leídos de su firma y no copiados a mano.

Lo invoca `campana/lanzar.sh config`. Puede re-ejecutarse: sobrescribe el
manifiesto y conserva la fecha de la primera escritura.
"""

from __future__ import annotations

import inspect
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paths import RESULTS  # noqa: E402


def _sh(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "desconocido"


def _sysctl(key: str) -> str:
    return _sh("sysctl", "-n", key)


def main() -> None:
    import numpy as np
    import scipy
    import sklearn

    from adaptation.adaptation_controller import AdaptationController
    from algorithms.competitors import COMPETITORS

    out = RESULTS / "config.json"
    previo = json.loads(out.read_text()) if out.exists() else {}

    firma = inspect.signature(AdaptationController.__init__)
    controlador = {n: p.default for n, p in firma.parameters.items() if n != "self"}

    excluidos_rail = {
        "LSHADE": "población inicial 18·D (~322 GB en rail516); >40 GB en 26 s",
        "LSHADE-EpSin": "población inicial 18·D (~322 GB en rail516); >40 GB en 25 s",
        "EBOwithCMAR": "población inicial 18·D (~322 GB en rail516); >40 GB en 25 s",
        "EA4eig": "matrices D×D y descomposiciones O(D³); pico 35,6 GB en rail516 y "
                  "no completó 1.000 evaluaciones en 10 min",
    }
    todos = ["SIA-base", "SIA-online"] + list(COMPETITORS)

    sucio = _sh("git", "status", "--porcelain", "--untracked-files=no")
    cfg = {
        "campana": "v3",
        "escrito_primera_vez": previo.get("escrito_primera_vez",
                                          time.strftime("%Y-%m-%d %H:%M:%S")),
        "actualizado": time.strftime("%Y-%m-%d %H:%M:%S"),
        "motivo": ("Repetición completa del experimento con el controlador adaptativo "
                   "corregido (estado absorbente), en una carpeta independiente y con una "
                   "única versión del código para todas las familias. Sustituye a la "
                   "campaña v2, cuyo caso ferroviario quedó incompleto al saturarse la "
                   "memoria de la máquina."),
        "codigo": {
            "commit": _sh("git", "rev-parse", "HEAD"),
            "rama": _sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "cambios_sin_commitear_en_archivos_trackeados": sucio != "",
        },
        "maquina": {
            "cpu": _sysctl("machdep.cpu.brand_string"),
            "nucleos_fisicos": _sysctl("hw.physicalcpu"),
            "nucleos_rendimiento": _sysctl("hw.perflevel0.physicalcpu"),
            "nucleos_eficiencia": _sysctl("hw.perflevel1.physicalcpu"),
            "ram_gb": round(int(_sysctl("hw.memsize") or 0) / 2**30),
            "sistema": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "ejecucion": {
            "procesos_paralelos": int(os.environ.get("JOBS", "20")),
            "proteccion_memoria": {
                "reserva_gb": float(os.environ.get("MEM_RESERVE", "8")),
                "un_proceso_por_celda": True,
                "reanudacion_por_corrida": True,
                "registro_de_memoria_cada_s": 300,
                "spotlight_excluido": (RESULTS / ".metadata_never_index").exists(),
            },
            "raiz_resultados": str(RESULTS),
            "hilos": {k: os.environ.get(k) for k in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")},
        },
        "protocolo": {
            "familias_originales": {
                "familias": ["classical", "cec2014", "cec2017", "cec2022", "bbob", "fir"],
                "algoritmos": todos,
                "corridas": 30, "poblacion": 30, "iteraciones": 100,
                "evaluaciones": 3000,
                "presupuesto_sia": "exacto: 30 de inicialización + 99 iteraciones × 30",
                "semillas": "dim·1000 + run_id, run_id = 1..30, iguales para todos",
            },
            "rail": {
                "instancias": ["rail507", "rail516", "rail582"],
                "algoritmos": [a for a in todos if a not in excluidos_rail],
                "excluidos": excluidos_rail,
                "corridas": 31, "poblacion": 40, "iteraciones": 500,
                "evaluaciones": 20000,
                "presupuesto_sia": "exacto: 40 de inicialización + 499 iteraciones × 40",
                "semillas": "n·1000 + run_id, run_id = 1..31, iguales para todos",
                "metrica": "RPD = 100 (f − BKS) / BKS",
                "protocolo_de": "CASO_FERROVIARIO.md (decisión del autor)",
            },
            "fuera_de_alcance": ["microhubs", "antenna", "pv", "microhubs_cont",
                                 "microhubs_tw"],
        },
        "sia_online": {
            "controlador": controlador,
            "horizonte": 3, "actualizacion_modelo_cada": 5, "xai_cada": 10,
            "ventana": 200,
        },
        "notas": [
            "SO consume menos evaluaciones que el presupuesto en todas las familias "
            "(2.745 de 3.000 en CEC 2022, 945 de 1.000 en rail): su número de iteraciones "
            "se fija como max_fes // N y algunas fases evalúan menos de N individuos. Es "
            "una propiedad de la implementación, no se modifica; solo puede "
            "desfavorecer a SO, nunca favorecerlo.",
            "GWO se vectorizó sobre las dimensiones sin cambiar su comportamiento: "
            "resultados idénticos bit a bit en las siete familias (ver README).",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    print(f"config.json escrito en {out}")
    print(f"  commit {cfg['codigo']['commit'][:10]} en '{cfg['codigo']['rama']}'"
          + ("  (OJO: hay cambios sin commitear)" if sucio else ""))


if __name__ == "__main__":
    main()
