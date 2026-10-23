"""Raíz única de resultados para todo el pipeline.

Todas las rutas de salida (corridas de SIA y competidores, análisis, log, estado)
derivan de RESULTS. Por defecto es `../resultados/` (RecursosPaper/resultados), la carpeta
con las corridas del paper; una campaña distinta se aísla fijando la variable de entorno:

    TESIS_RESULTS_DIR=../resultados_nueva python run.py ...

Una ruta relativa se interpreta respecto de esta carpeta (RecursosPaper/codigo). Los procesos
worker de run.py heredan la variable, así que basta con fijarla en el lanzador.

Existe para que dos campañas con versiones distintas del controlador no puedan
mezclar sus corridas en la misma carpeta.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = (ROOT / os.environ.get("TESIS_RESULTS_DIR", "../resultados")).resolve()
