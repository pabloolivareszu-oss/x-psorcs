# Caso ferroviario (rail): guía de ejecución

> **Para quién es.** Para el agente (Claude) o la persona que ejecute la campaña en la máquina con cómputo. Todo está **construido y probado en pequeño, pero no se ha ejecutado sobre las instancias reales**. Lee el documento completo antes de lanzar algo, sigue los pasos en orden y **detente en cada criterio de parada**.
>
> Referencia del caso: `caso_ferroviario/guia_caso_ferroviario.tex` (guía de traspaso del proyecto CPB).

---

## 1. Qué es y qué se decidió

- **Problema.** Set covering de programación de tripulaciones ferroviarias, con tres instancias de la OR-Library:

  | Instancia | Filas | Columnas | BKS |
  |---|---|---|---|
  | rail507 | 507 | ~63.009 | 174 |
  | rail516 | 516 | ~47.311 | 182 |
  | rail582 | 582 | ~55.515 | 211 |

  Los BKS son de Caprara et al. (1999) y **están sin verificar**. El objetivo es minimizar Σ c_j x_j de modo que cada fila quede cubierta al menos una vez.
- **Rol en la tesis.** Es un nuevo caso aplicado donde se compara SIA-base (A) contra SIA-online (A′) y contra los 21 competidores.
- **Decisiones tomadas por el autor (no reabrir):**
  - **Protocolo CPB, no el de la tesis:**
    - población 40;
    - **20.000 evaluaciones exactas para todos los métodos**;
    - **31 corridas**;
    - métrica RPD = 100 (f − BKS) / BKS.

    La razón es que 3.000 evaluaciones no significan nada sobre ~60.000 variables, y así los números se pueden leer junto a la Tabla T10 de CPB.
  - **Binarización V2-ELITR**, el mejor esquema de T10. **Reparación**: Chvátal más eliminación de redundancia, idéntica para todos los métodos.
  - Se implementó **desde la guía**, sin el código CPB (`qmb_grid.py`). Por eso la comparación con T10 es de referencia y no exacta.
  - **Controlador de SIA-online:** la versión **corregida** de `adaptation/adaptation_controller.py`, con piso de 0,05 por operador, decaimiento de 0,05 hacia el prior y cooldown de 5. **No cambiar a la versión anterior.** El detalle está en el artifact de auditoría: al corregir el estado absorbente, CEC 2022 pasó de p = 0,025 a p = 0,79.
  - Se corre con competidores.

## 2. Qué se agregó al código

| Archivo | Qué hace |
|---|---|
| `benchmarks/rail.py` | Lector por columnas (doble CSR), reparación en numba, V2-ELITR con archivo de élite y `get_rail_problem` |
| `benchmarks/rail_data/bks.json` | m, n y BKS con su procedencia (`verified: false`) |
| `benchmarks/rail_data/.gitignore` | Las instancias **no se suben** (términos de la OR-Library) |
| `benchmarks/registry.py` | Familia `rail`, con `rail_dim_for(nombre)` = n columnas |
| `run.py` | Familia `rail` con dimensión por instancia, **presupuesto exacto** para SIA (`EXACT_BUDGET_FAMILIES`), filtro `--funcs` y análisis con `rail_report` |
| `analysis/rail_report.py` | RPD, Wilcoxon pareado por instancia, Finner, Â de Vargha–Delaney y tabla LaTeX |
| `tools/fetch_rail.py` | Descarga y verifica las instancias |
| `tools/rail_smoke.py` | Pruebas de humo |

**Adaptaciones que hay que declarar en la memoria** (difieren de CPB):
1. **Élite.** En CPB es el 30% de la generación actual. Aquí, como SIA y los competidores evalúan de a un vector, la función objetivo mantiene un **archivo con las 12 mejores soluciones reparadas** de la corrida. Con el archivo vacío, la reparación parte de la solución vacía, que equivale a la inicialización constructiva. En consecuencia, el objetivo depende de la historia de evaluaciones.
2. **Semillas.** Son `n·1000 + run_id` con `run_id` de 1 a 31, iguales para todos los métodos, así que la comparación está pareada. No son literalmente las semillas 1 a 31 de CPB.
3. **Presupuesto de SIA.** 40 evaluaciones de inicialización más 499 iteraciones × 40 = 20.000.

**Por verificar antes de reportar:**
- La regla ELITR exacta (`eq:rules` del manuscrito CPB). Implementada: `bit = e_j` si `r ≤ T`, si no 0.
- La normalización de la ruleta de élite.
- **Los tres BKS**, contra la fuente primaria. Cuando estén verificados, marcar `"verified": true` en `bks.json`.

## 3. Pasos

Todos los comandos se ejecutan desde `RecursosPaper/codigo/`. **[EJEC]** marca un comando que ejecuta algo; **[MONITOR]** marca uno que solo observa.

### Paso 0: entorno  [EJEC]
```bash
pip install -r requirements.txt        # numpy, scipy, scikit-learn, numba, ...
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
```
Sin los hilos en 1, los tiempos no se reproducen y los procesos compiten entre sí.

### Paso 1: instancias  [EJEC]
```bash
python tools/fetch_rail.py
```
- **Resultado esperado:** para cada instancia, m = 507, 516 y 582, y n impreso. Si n difiere de `bks.json`, el script lo corrige y avisa.
- **Parar si** m no coincide o si alguna fila queda sin columna.
- **Después:** verificar los tres BKS contra Caprara, Fischetti y Toth (1999) y marcar `verified: true`.

### Paso 2: humo  [EJEC]
```bash
python tools/rail_smoke.py --real
```
Comprueba:
- el lector y la reparación sobre una mini-instancia;
- que rail516 con SIA-base y 2.000 evaluaciones consuma exactamente 2.000;
- que dos corridas con la misma semilla den el mismo best;
- que el RPD no sea negativo.

**Parar si** algo falla.

### Paso 3: piloto  [EJEC]
Se usa rail516 (la más pequeña), **1 corrida** y el presupuesto completo. Sirve para medir tiempos y memoria y para detectar qué métodos no escalan.
```bash
# 3a. los dos SIA y los competidores sin matrices n×n
python run.py --families rail --funcs rail516 --runs 1 --pop-size 40 --max-iter 500 \
  --jobs 8 --no-analysis \
  --only SIA-base SIA-online AMALGAM APGSK-IMODE ASGK BKA BWO DBO FGIN GOOSE GWO HHO \
         IGWO JADE KRMG LHHO LSHADE LSHADE-EpSin SACS SO UMOEAII
# 3b. EA4eig y EBOwithCMAR, SOLOS: construyen covarianzas D×D (~18–32 GB con D≈50–63k)
python run.py --families rail --funcs rail516 --runs 1 --pop-size 40 --max-iter 500 \
  --jobs 1 --no-analysis --only EA4eig
python run.py --families rail --funcs rail516 --runs 1 --pop-size 40 --max-iter 500 \
  --jobs 1 --no-analysis --only EBOwithCMAR
```
Vigilar la memoria durante 3b **[MONITOR]** (`htop` o Monitor de Actividad). Si un método se queda sin memoria o empieza a paginar, se mata el proceso y se declara como **"no escala a n ≈ 50.000"**. Queda excluido de la campaña, y eso se reporta y no se oculta. Los fallos quedan en `results/errors.log`.

**Qué revisar en el piloto** [MONITOR], en `results/run_status.json` y en los `summary.json`:
- `function_evaluations` = 20.000 para todos los métodos;
- **en SIA-online, `n_model_updates > 0` y `n_adaptations > 0`**. Si son 0, el modelo nunca se entrenó porque la etiqueta tuvo una sola clase, y A′ se comporta igual que A. Hay que **parar e informar**: sería el mismo síntoma de meseta que tuvo microhubs;
- el tiempo por corrida de cada método (`seconds`).

**Estimación de la campaña:** Σ(tiempo por corrida de cada método) × 31 × 3 / número de procesos. Si supera lo razonable, **consultar al autor** antes de lanzar; no bajar semillas ni presupuesto por cuenta propia.

### Paso 4: campaña  [EJEC]
Antes de lanzar hay que **borrar las corridas del piloto**. Si no, una relanzada con `--resume` saltaría las celdas del piloto, que tienen 1 corrida, como si estuvieran completas:
```bash
rm -rf results/SIA_online/*/rail results/competitors/*/rail
python run.py --families rail --runs 31 --pop-size 40 --max-iter 500 \
  --jobs <núcleos físicos − 2> --no-analysis \
  --only SIA-base SIA-online <competidores que pasaron el piloto>
```
- Si se corta, se relanza igual con `--resume`. La unidad de trabajo es una celda (método × instancia) con sus 31 corridas.
- El avance se sigue con `tail -f results/run.log` **[MONITOR]**.

### Paso 5: análisis  [EJEC]
```bash
python -m analysis.rail_report
```
Genera en `results/analysis/rail/`:
- `rail_summary.csv`: RPD medio, mediana, desviación estándar, mejor, peor y tiempo;
- `rail_tests.csv`: Wilcoxon pareado de A vs A′ y de A′ vs cada competidor, con mejor/igual/peor, p, p de Finner y Â;
- `rail_table.tex`: formato de la plantilla de la guía, con la fila de referencia de T10 (V2-ELITR);
- `rail_report.json`: incluye los **avisos**. Revisarlos todos.

## 4. Criterios de parada

Si se cumple cualquiera, **detenerse e informar al autor**:
- **RPD negativo** en cualquier corrida: el BKS está mal.
- Una celda con un número de corridas distinto de 31, o evaluaciones distintas de 20.000 (lo avisa `rail_report`).
- Dos corridas con la misma semilla que no dan el mismo resultado.
- SIA-online con 0 actualizaciones del modelo o 0 adaptaciones.
- Un fallo de un método que no sea por memoria. Revisar `errors.log`.

## 5. Qué NO hacer

- No re-sintonizar ningún algoritmo ni los parámetros de SIA-online: horizonte 3, actualización cada 5, XAI cada 10, ventana 200.
- No cambiar la reparación ni la binarización. Son variables controladas.
- No calibrar nada sobre rail507, rail516 ni rail582. Son las instancias de reporte. Si hiciera falta calibrar, se usan otras instancias rail de la OR-Library (por ejemplo rail2536).
- No mezclar versiones del controlador ni reutilizar corridas de otra versión.
- No bajar semillas ni presupuesto para ahorrar tiempo sin consultarlo.
- No subir las instancias al repositorio.

## 6. Qué entregar de vuelta

- `results/SIA_online/*/rail/`, `results/competitors/*/rail/`, `results/run_status.json` y `results/errors.log`.
- `results/analysis/rail/` completo.
- Una nota breve con:
  - la máquina y el número de procesos;
  - los tiempos del piloto;
  - los métodos excluidos y por qué;
  - si los BKS quedaron verificados y contra qué fuente;
  - cualquier aviso de `rail_report.json`.
