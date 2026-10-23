# Campaña v3

Repetición completa del experimento con el **controlador adaptativo corregido**, todas
las familias ejecutadas con **una única versión del código**. Sustituye a la campaña v2,
cuyo caso ferroviario quedó a dos celdas de terminar cuando la memoria de la máquina se
saturó y hubo que reiniciarla. Los resultados de la v2 (`campana_v2/`) y de la campaña
original (`results/`) no se usan.

## Por qué existe

El controlador de SIA-online tenía un estado absorbente: la probabilidad de un operador
podía caer a cero y ninguna regla la devolvía. Al corregirlo (commit `577a878`: piso de
0,05 por operador, decaimiento de 0,05 hacia el prior y cooldown de 5 iteraciones)
SIA-online pasa a ser un algoritmo distinto, y hay que repetir el experimento.

La v3 añade sobre la v2 la **protección de memoria** descrita abajo. Para que todos los
resultados correspondan exactamente al mismo código, se re-ejecuta todo, también las
familias que la v2 sí había completado.

## Protocolo

| | familias originales | caso ferroviario |
|---|---|---|
| familias | classical, CEC 2014, CEC 2017, CEC 2022, BBOB, FIR | rail507, rail516, rail582 |
| algoritmos | 23 (SIA-base, SIA-online y 21 competidores) | 19 (4 excluidos, ver abajo) |
| corridas | 30 | 31 |
| población × iteraciones | 30 × 100 | 40 × 500 |
| evaluaciones | **3.000 exactas para todos** | **20.000 exactas para todos** |
| semillas | `dim·1000 + run_id`, iguales para todos | `n·1000 + run_id`, iguales para todos |
| métrica | error al óptimo (fitness crudo en BBOB y FIR) | RPD = 100 (f − BKS) / BKS |

**Fuera de alcance:** microhubs, y las familias antenna, pv, microhubs_cont y
microhubs_tw.

**Presupuesto exacto también para SIA**: corre `(max_fes − pop) // pop` iteraciones,
30 + 99 × 30 = 3.000 y en rail 40 + 499 × 40 = 20.000. En la campaña original consumía
3.030 frente a 3.000 de los competidores.

El caso ferroviario conserva el protocolo que fijó el autor en
[`CASO_FERROVIARIO.md`](../CASO_FERROVIARIO.md).

## Cómo se ejecuta

Desde `RecursosPaper/codigo`:

```bash
campana/lanzar.sh config          # carpeta, exclusión de Spotlight y manifiesto
campana/lanzar.sh fase1           # familias originales
campana/lanzar.sh analisis1       # su análisis
campana/lanzar.sh rail-piloto     # 1 corrida en rail516, para medir tiempos
campana/lanzar.sh rail-limpiar    # borrar el piloto antes de la campaña
campana/lanzar.sh rail-campana    # 31 corridas × 3 instancias
campana/lanzar.sh rail-analisis   # reporte del caso ferroviario
campana/lanzar.sh auditoria       # auditoria.json
```

Todas las fases escriben solo en `../resultados/`, usan 20 procesos con un hilo
cada uno y pasan siempre `--resume`. Si una fase se corta, **incluso por un reinicio**,
se relanza con el mismo comando y continúa desde la última corrida completa. El avance
se sigue con `tail -f ../resultados/run.log`.

## Protección de memoria

En la v2 la máquina se saturó tras ~13 h de campaña y hubo que reiniciarla. Se midió
después cuánta memoria usa cada método sobre la instancia más grande (rail507), en
cuatro corridas seguidas dentro del mismo proceso, como hace una celda real (ver
`diagnostico/`): todos quedan entre 0,3 y 0,8 GB y crecen unos pocos MB por corrida.
Los 19 juntos suman unos 8,5 GB. **Los procesos de la campaña no explican la
saturación**, y el registro del sistema no es accesible para confirmar qué la causó.
Por eso la protección cubre también causas externas:

1. **Un proceso nuevo por celda** (`max_tasks_per_child=1` en `run.py`). La memoria que
   retiene un worker vuelve al sistema al terminar cada celda, en vez de acumularse
   durante toda la campaña.
2. **Guardia de memoria** (`MemoryGuard` en `run.py`). Vigila cada 3 s la memoria
   disponible del sistema. Si baja de la reserva (8 GB por defecto, `MEM_RESERVE`),
   detiene el worker con más memoria antes de que el sistema empiece a paginar, reduce
   el paralelismo un 25 % y reanuda. Si ni con un solo proceso cabe, marca la celda
   como fallida en vez de colgar la máquina.
3. **Reanudación por corrida** (`_completed_run` en `run.py`). Al relanzar, cada celda
   salta las corridas que ya tienen un `summary.json` válido con la misma semilla y el
   mismo presupuesto. Una interrupción cuesta como mucho la corrida en curso, y el
   resultado es idéntico al de una celda sin interrumpir, porque cada corrida tiene su
   propia semilla.
4. **Registro de memoria cada 5 minutos** en `run.log`: memoria disponible, la que usa la
   campaña, el swap y los cinco procesos mayores del sistema. Cada actuación de la
   guardia guarda además una instantánea de los diez mayores. Si vuelve a ocurrir una
   saturación, quedará constancia de qué la produjo.
5. **Exclusión de Spotlight** (`results/.metadata_never_index`). La campaña escribe
   cerca de un millón de archivos pequeños; la indexación de volúmenes así es una causa
   conocida de picos de memoria en macOS.

## Qué queda registrado

| archivo | contenido |
|---|---|
| `results/config.json` | Commit exacto, máquina, hilos, protección de memoria, protocolo y parámetros efectivos del controlador |
| `results/run.log` | Cada celda con hora, progreso, segundos, mejor valor y ETA; inicio y fin de cada fase; registro de memoria; actuaciones de la guardia |
| `results/errors.log` | Cada fallo y cada actuación de la guardia, con fecha |
| `results/run_status.json` | Por celda: ok/error, corridas, corridas reutilizadas, mejor valor, segundos |
| `results/auditoria.json` | Consolidado: celdas, presupuesto por corrida, salud de SIA-online, tiempos, ETA, memoria, reanudaciones |
| `results/analysis/` | Análisis estadístico por familia y reporte del caso ferroviario |
| `diagnostico/` | Medición de memoria por método que motivó la protección |

## Cambios de código que usa esta versión

1. **Controlador corregido** (`adaptation/adaptation_controller.py`).
2. **Raíz de resultados única** (`paths.py`), configurable con `TESIS_RESULTS_DIR`.
3. **Estado acumulativo** (`run.py`): el estado ya no se sobrescribe entre invocaciones.
4. **Presupuesto exacto para SIA en todas las familias** (`run.py`).
5. **Hora de reloj en cada línea del log** (`run.py`).
6. **GWO vectorizado** (`algorithms/competitors/gwo.py`), con resultados idénticos bit a
   bit a la versión original en las siete familias y 33 veces más rápido en rail.
7. **Protección de memoria** (`run.py`), descrita arriba.

## Caso ferroviario: compatibilidad y métodos excluidos

El código de rail se probó sobre las instancias reales antes de la v2: descarga y
validación de las instancias, prueba de humo, mini-corrida de cada método y reporte de
análisis. La guía prohíbe re-sintonizar los algoritmos, así que un método que no escala
**no se modifica, se excluye**:

| método | causa | evidencia en rail516 |
|---|---|---|
| L-SHADE | población inicial 18·D: ~851.000 individuos, ~322 GB | >40 GB en 26 s |
| L-SHADE-EpSin | población inicial 18·D | >40 GB en 25 s |
| EBOwithCMAR | población inicial 18·D | >40 GB en 25 s |
| EA4eig | matrices D×D (~18 GB cada una) y descomposiciones O(D³) | pico 35,6 GB; no completó 1.000 evaluaciones en 10 min |

**SO no consume todo el presupuesto**: fija sus iteraciones como `max_fes // N` y algunas
fases evalúan menos de N individuos (2.745 de 3.000 en CEC 2022; 18.045 de 20.000 en
rail516). Es una propiedad de su implementación, presente también en la campaña
original. No se modifica y se declara; solo puede desfavorecer a SO.

## Pendiente del autor

- Verificar los tres BKS contra Caprara, Fischetti y Toth (1999) y marcar
  `"verified": true` en `benchmarks/rail_data/bks.json`.
- Confirmar la regla ELITR exacta contra el manuscrito CPB (`eq:rules`).
