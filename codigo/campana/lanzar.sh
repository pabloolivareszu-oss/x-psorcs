#!/bin/bash
# Lanzador de la campaña v3. Uso, desde RecursosPaper/codigo:
#
#   campana/lanzar.sh config         escribe ../resultados/config.json
#   campana/lanzar.sh fase1          familias originales a 3.000 evaluaciones
#   campana/lanzar.sh analisis1      análisis de las familias originales
#   campana/lanzar.sh rail-piloto    caso ferroviario: 1 corrida en rail516
#   campana/lanzar.sh rail-limpiar   borra las corridas del piloto
#   campana/lanzar.sh rail-campana   caso ferroviario: 31 corridas, 3 instancias
#   campana/lanzar.sh rail-analisis  reporte del caso ferroviario
#   campana/lanzar.sh auditoria      consolida ../resultados/auditoria.json
#
# Todas las fases escriben SOLO en ../resultados/ y pasan siempre --resume:
# si una fase se corta (o se reinicia la máquina), se relanza con el mismo comando
# y continúa donde quedó, a nivel de CORRIDA, no solo de celda.
#
# Protección de memoria (la campaña v2 saturó la máquina y obligó a reiniciarla):
#   - run.py corre cada celda en un proceso nuevo, que devuelve su memoria al
#     sistema al terminar;
#   - la guardia de memoria de run.py mantiene libres MEM_RESERVE GB: por debajo,
#     detiene el worker más grande, baja el paralelismo un 25 % y reanuda;
#   - cada 5 minutos el log registra la memoria disponible, la de la campaña y los
#     procesos mayores del sistema;
#   - la carpeta de resultados se excluye de Spotlight (~1 millón de archivos).

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
JOBS="${JOBS:-20}"
MEM_RESERVE="${MEM_RESERVE:-8}"
export JOBS MEM_RESERVE
RUN="$PY run.py --jobs $JOBS --resume --mem-reserve-gb $MEM_RESERVE"

# Raíz de resultados aislada (ver paths.py).
export TESIS_RESULTS_DIR="${TESIS_RESULTS_DIR:-../resultados}"

# Un hilo por proceso: con 20 procesos en paralelo, cada librería numérica abriría
# sus propios hilos y competirían por los núcleos. También hace reproducibles los
# tiempos de la auditoría.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1

FAMILIAS_ORIGINALES="classical cec2014 cec2017 cec2022 bbob fir"

# Caso ferroviario: 19 métodos. Excluidos por no escalar a n ≈ 50.000 (ver README):
#   LSHADE, LSHADE-EpSin, EBOwithCMAR  población inicial 18·D (~322 GB en rail516)
#   EA4eig                             matrices D×D; 35,6 GB y >10 min por 1.000 evaluaciones
#
# Los cuatro métodos lentos van primero (por corrida en rail516: IGWO 947 s,
# APGSK-IMODE 780 s, ASGK 697 s, BWO 304 s; el resto 57-121 s). run.py reparte las
# celdas en el orden de esta lista y cada celda corre sus 31 corridas en serie, así
# que poner las largas al principio evita que queden esperando al final de la cola
# y acorta el tiempo de reloj. No afecta a los resultados: la semilla es por corrida.
RAIL_METODOS="IGWO APGSK-IMODE ASGK BWO SIA-base SIA-online AMALGAM BKA DBO FGIN GOOSE \
GWO HHO JADE KRMG LHHO SACS SO UMOEAII"

fase="${1:-}"
case "$fase" in
  config)
    mkdir -p $TESIS_RESULTS_DIR
    # Spotlight no indexa una carpeta que contiene este archivo (mecanismo de macOS).
    touch $TESIS_RESULTS_DIR/.metadata_never_index
    "$PY" campana/escribir_config.py
    ;;
  fase1)
    $RUN --families $FAMILIAS_ORIGINALES --runs 30 --pop-size 30 --max-iter 100 --no-analysis
    ;;
  analisis1)
    # Solo el análisis, llamando directamente a run_analysis_phase en vez de a run.py:
    # así escribe en su propio log y no se intercala con run.log si a la vez corre
    # otra fase (la auditoría parsea run.log por fases).
    "$PY" -c "import run; run.run_analysis_phase('$FAMILIAS_ORIGINALES'.split(), None, run.COMPETITORS)" \
        > $TESIS_RESULTS_DIR/analisis1.log 2>&1
    ;;
  rail-piloto)
    $RUN --families rail --funcs rail516 --runs 1 --pop-size 40 --max-iter 500 \
        --no-analysis --only $RAIL_METODOS
    ;;
  rail-limpiar)
    # Imprescindible antes de la campaña: con --resume, las celdas del piloto (1 sola
    # corrida) se saltarían como si estuvieran completas.
    rm -rf $TESIS_RESULTS_DIR/SIA_online/*/rail $TESIS_RESULTS_DIR/competitors/*/rail
    "$PY" - <<'EOF'
import json, pathlib
import os
p = pathlib.Path(os.environ["TESIS_RESULTS_DIR"]) / "run_status.json"
if p.exists():
    st = json.loads(p.read_text())
    piloto = {k: v for k, v in st.items() if k.split("|")[1] == "rail"}
    for k in piloto:
        st.pop(k)
    p.write_text(json.dumps(st, indent=2))
    print(f"retiradas {len(piloto)} celdas del piloto de run_status.json")
EOF
    ;;
  rail-campana)
    $RUN --families rail --runs 31 --pop-size 40 --max-iter 500 \
        --no-analysis --only $RAIL_METODOS
    ;;
  rail-analisis)
    "$PY" -m analysis.rail_report
    ;;
  auditoria)
    "$PY" campana/auditar.py
    ;;
  *)
    sed -n '2,13p' "$0"
    exit 1
    ;;
esac
