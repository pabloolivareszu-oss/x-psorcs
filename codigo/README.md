# SIA online explicable y adaptativo — guía del experimento

Pipeline de un **Swarm Intelligence Algorithm (PSO) online, explicable y adaptativo**:
el aprendizaje, la explicabilidad (XAI) y la adaptación ocurren **dentro** del ciclo
iterativo del algoritmo, no después. Se compara contra 20 algoritmos de
competición (CEC y metaheurísticas recientes) y se analiza con pruebas estadísticas
y figuras de tesis.

```
SIA(t) → traza → modelo online → XAI → adaptación → SIA(t+1)
```

---

## 1. Qué se va a hacer (flujo del experimento)

El experimento tiene **cuatro fases**. Cada una es un comando.

```
┌─ FASE 1 ────────────────────────────────────────────────┐
│ A  = SIA-base    (PSO puro, sin pipeline)                │  run_sia_online --mode base
│ A' = SIA-online  (PSO + modelo + XAI + adaptación)       │  run_sia_online --mode online
└──────────────────────────────────────────────────────────┘
┌─ FASE 2 ────────────────────────────────────────────────┐
│ Competidores: los 20 registrados (mismo presupuesto de   │  run_competitors
│ evaluaciones que el SIA). Ver lista completa más abajo.  │
└──────────────────────────────────────────────────────────┘
┌─ FASE 3 ────────────────────────────────────────────────┐
│ Análisis: Wilcoxon (A vs A'), Friedman + post-hoc Holm,  │  run_analysis
│ boxplots, convergencia, superficies 3D, 7 figuras-       │
│ plantilla, tabla resumen CSV/LaTeX.                      │
└──────────────────────────────────────────────────────────┘
```

**Idea clave de la comparación:**
- **A vs A'** mide *tu aporte*: cuánto mejora el PSO al añadirle el pipeline online.
- **A' vs competidores** te posiciona frente al estado del arte (todos con el mismo
  presupuesto de evaluaciones de función, FEs).

**Presupuesto (justo para todos):** `FEs = población × iteraciones = 30 × 100 = 3000`.
Se mide en evaluaciones de la función objetivo, no en iteraciones, para que la
comparación entre algoritmos con estructuras distintas sea honesta.

---

## 2. Requisitos

- Python 3.12
- Paquetes: `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `opfunu` (CEC),
  `coco-experiment` (BBOB).

Comprobación rápida:
```powershell
python -c "import numpy, scipy, sklearn, matplotlib, opfunu, cocoex; print('OK')"
```

---

## 3. Benchmarks disponibles

| Familia (`--family`) | Funciones | Dimensiones válidas |
|---|---|---|
| `cec2014` | F1..F30 (30) | 10, 20, 30, 50, 100 |
| `cec2017` | F1..F30 (29; opfunu omite F2) | 10, 20, 30, 50, 100 |
| `cec2022` | F1..F12 (12) | **solo 10, 20** |
| `bbob`    | f01..f24 (24) | 2, 3, 5, 10, 20, 40 (**no 50**) |
| `classical` | sphere, rosenbrock, rastrigin, ackley, griewank | cualquiera |
| `fir` | FIR1..FIR8 (8) | **solo 31** (orden 30 fijo) |
| `microhubs` | 105 instancias (L_01..5XL_15) | **m por instancia** (25/40/70/120/200/320) |

> Si pides una dimensión que la familia no soporta (p.ej. 50 en BBOB o CEC2022),
> el runner **la omite con un aviso** y sigue con las válidas. No falsea resultados.

**Corridas independientes:** en CEC/classical son semillas distintas del mismo
problema; en BBOB son **instancias** distintas (práctica estándar de COCO).

**FIR (diseño de filtros):** familia adicional de 8 casos de diseño de filtros FIR
de orden 30 (dimensión fija D=31, dominio `[-1, 1]`). El objetivo es
`F(h) = Wp·MSE_p + Ws·MSE_s` sobre la respuesta en frecuencia (FFT, `N_fft=2048`),
con `Wp=1` y `Ws=100` (casos Easy/Wide) o `Ws=120` (Hard/Narrow). Como en BBOB, el
óptimo es desconocido a priori, por lo que el análisis usa el **fitness crudo**
(`best_fitness`). Los 8 casos: FIR1/2 (paso-bajo), FIR3/4 (paso-alto), FIR5/6
(paso-banda), FIR7/8 (rechaza-banda). Definición en `benchmarks/fir.py`. Ejemplo:
```powershell
python run.py --families fir
# o por fases:
python -m experiments.run_sia_online --family fir --benchmarks FIR1 FIR2 FIR3 FIR4 FIR5 FIR6 FIR7 FIR8 --dims 31 --runs 30
```

**microhubs (MLPAP — Microhub Location and Pedestrian Assignment Problem):**
problema combinatorio binario de localización de microhubs y asignación peatonal
de clientes. Batería de **105 instancias** en 6 tiers por tamaño (L, XL, 2XL, 3XL,
4XL, 5XL; de 25 a 320 hubs). La variable de decisión es el vector binario `y` de
apertura de hubs (longitud `m` = nº de hubs de cada instancia, **fija por
instancia**); las asignaciones cliente→hub las resuelve un decoder greedy y el
objetivo penalizado es `f̃(z) = f_raw(z) + π·v(z)` (coste + violaciones). Como los
algoritmos del proyecto son continuos, la función objetivo **binariza** el vector
continuo `x ∈ [-4,4]^m` con un transfer **sigmoide estocástico** (`y_j=1` con prob.
`sigmoid(x_j)`), reproducible por semilla. Óptimo desconocido → análisis por
fitness crudo. Datos y core (numba) en `benchmarks/microhubs_data/`, wrapper en
`benchmarks/microhubs.py`. Si un competidor no puede ejecutar una instancia
(p.ej. por la dimensión `m`), el fallo se registra en `results/errors.log` con su
causa y el pipeline continúa con el resto. Ejemplo:
```powershell
python run.py --families microhubs
```

### 3.1 Competidores disponibles (20)

Todos reimplementados desde su paper, con interfaz y presupuesto comunes
(`algorithms/competitors/`). Registrados en `_REGISTRY` de
`algorithms/competitors/__init__.py`:

| Grupo | Algoritmos |
|---|---|
| Originales (6) | `ASGK`, `EA4eig`, `APGSK-IMODE`, `AMALGAM`, `UMOEAII`, `JADE` |
| Segundo lote (15) | `BKA`, `GOOSE`, `GWO`, `HHO`, `DBO`, `BWO`, `SO`, `IGWO`, `LHHO`, `LSHADE`, `LSHADE-EpSin`, `EBOwithCMAR`, `SACS`, `KRMG`, `FGIN` |

> `python run.py` (sin flags) corre **SIA-base + SIA-online + los 20 competidores**.
> `--only ALGO ...` restringe a un subconjunto; `--new-competitors` corre solo los 15
> del segundo lote (sin SIA).

---

## 4. Cómo ejecutarlo (paso a paso)

Ejecuta todo desde la raíz del proyecto (`TesisDefinitivaPostgrado/`).
Sustituye las funciones/dimensiones según lo que quieras reportar.

### Paso 0 — (opcional) prueba rápida de humo
Verifica que todo corre antes de lanzar el experimento largo:
```powershell
python -m experiments.run_sia_online --mode base   --family cec2017 --benchmarks F1 --dims 10 --runs 3 --max-iter 30
python -m experiments.run_sia_online --mode online --family cec2017 --benchmarks F1 --dims 10 --runs 3 --max-iter 30
python -m experiments.run_competitors --competitors ASGK --family cec2017 --benchmarks F1 --dims 10 --runs 3
python -m experiments.run_analysis --family cec2017 --benchmarks F1 --dims 10 --with-competitors ASGK
```
Si esto termina sin error, borra `results/` y lanza el experimento real.

### Paso 1 — SIA base (A) y SIA online (A')
```powershell
python -m experiments.run_sia_online --mode base   --family cec2017 --benchmarks F1 F3 F5 F7 F9 --dims 10 20 50 --runs 30
python -m experiments.run_sia_online --mode online --family cec2017 --benchmarks F1 F3 F5 F7 F9 --dims 10 20 50 --runs 30
```

### Paso 2 — Competidores
Ejemplo con 5 competidores; pasa los que quieras (ver sección 3.1) o usa `run.py`
para correr los 20 de una vez.
```powershell
python -m experiments.run_competitors --competitors ASGK EA4eig APGSK-IMODE AMALGAM UMOEAII --family cec2017 --benchmarks F1 F3 F5 F7 F9 --dims 10 20 50 --runs 30
```

### Paso 3 — Análisis, figuras y tabla
```powershell
python -m experiments.run_analysis --family cec2017 --benchmarks F1 F3 F5 F7 F9 --dims 10 20 50 --with-competitors ASGK EA4eig APGSK-IMODE AMALGAM UMOEAII
```

> **Importante:** usa las **mismas** `--benchmarks` y `--dims` en las tres fases.
> El análisis empareja por (familia, función, dimensión); si no coinciden, faltarán
> celdas en las tablas/figuras.

### Repetir por familia
Repite los pasos 1–3 cambiando `--family`. Recuerda que cada familia tiene sus
funciones y dimensiones válidas (sección 3). Ejemplos:
```powershell
# CEC2014 (10/20/50)
... --family cec2014 --benchmarks F1 F4 F9 F15 --dims 10 20 50 ...
# CEC2022 (solo 10/20)
... --family cec2022 --benchmarks F1 F3 F6 --dims 10 20 ...
# BBOB (10/20; corridas = instancias)
... --family bbob --benchmarks f01 f06 f15 f21 --dims 10 20 ...
```

---

## 5. Qué se genera (estructura de resultados)

```
results/
├── SIA_online/
│   ├── SIA-base/<familia>/<func>/dim_<d>/run_<r>/      ← A
│   │   ├── convergence.csv   traces.csv   predictions.csv
│   │   ├── explanations.csv  adaptations.csv  summary.json
│   └── SIA-online/<familia>/<func>/dim_<d>/run_<r>/    ← A' (idem)
├── competitors/<ALGO>/<familia>/<func>/dim_<d>/run_<r>/
│   ├── convergence.csv   summary.json
└── analysis/<familia>/
    ├── wilcoxon_A_vs_Aprime.json     ← A vs A' (Wilcoxon pareado)
    ├── friedman.json  posthoc_holm.json
    ├── summary_table.csv             ← tabla larga (1 fila por problema×algoritmo)
    ├── summary_table.tex             ← tabla LaTeX lista para la tesis
    ├── boxplot_dim_*.png
    ├── convergence_<func>_dim_*.png
    ├── surface_<func>.png
    └── template/                     ← las 7 figuras-plantilla
        ├── 1_heatmap.png  2_gap_lines.png  3_bubble.png  4_contour.png
        ├── 5_convergence_facets.png  6_critical_difference.png  7_radar.png
```

### La tabla resumen (`summary_table.csv` / `.tex`)
Por cada problema × algoritmo: **best, mean, worst, std, median, iqr,
success_rate, rank**, símbolo **Wilcoxon +/=/−** vs SIA-online, más filas globales
de **W/T/L** y **ranking medio de Friedman**. El `.tex` resalta el mejor en negrita.

### Las pruebas estadísticas
- **Wilcoxon** (A vs A'): test pareado; el aporte del pipeline.
- **Friedman** (≥3 algoritmos): ranking medio global; requiere `--with-competitors`.
- **Post-hoc Holm**: SIA-online vs cada rival, con corrección por comparaciones múltiples.

---

## 6. Parámetros configurables

Comunes a `run_sia_online` y `run_competitors`:

| Flag | Default | Significado |
|---|---|---|
| `--family` | cec2017 | familia de benchmarks |
| `--benchmarks` | F1 F3 F5 | funciones a evaluar |
| `--dims` | 10 20 50 | dimensiones (se filtran por familia) |
| `--runs` | 30 | corridas independientes |
| `--pop-size` | 30 | tamaño de población |
| `--max-iter` | 100 | iteraciones → FEs = pop × iter |
| `--base-seed` | 0 | semilla base (reproducibilidad) |

Solo `run_sia_online`: `--mode {base,online}`, `--horizon` (k de la etiqueta),
`--update-interval` (cada cuántas iters reentrena el modelo), `--xai-interval`,
`--window-size`.

Solo `run_competitors`: `--competitors` (subconjunto de los 20 registrados; ver
sección 3.1), `--max-fes` (sobrescribe el presupuesto).

Solo `run_analysis`: `--with-competitors`, `--metric {error_to_optimum,best_fitness}`,
`--tau` (umbral de éxito), `--no-surfaces`, `--no-template`, `--no-table`.

Ayuda completa de cualquiera: `python -m experiments.<runner> --help`.

---

## 7. Notas metodológicas (para la tesis)

- **Presupuesto 3000 FEs**: justo entre todos, pero **corto** frente al estándar CEC
  (10000×D). Los competidores rinden por debajo de su potencial publicado; la
  comparación sigue siendo válida porque todos comparten presupuesto.
- **Competidores reimplementados desde el paper** (no portados del código MATLAB
  oficial). Cada archivo en `algorithms/competitors/` documenta su referencia y
  supuestos. Para fidelidad exacta, validar contra resultados publicados.
- **Métrica de gap**: `error_to_optimum = best − f*` en CEC/classical; en BBOB
  (sin óptimo público) se usa el gap relativo al mejor empírico entre métodos.
- **"Tasa de éxito"** (figuras): fracción de corridas con error < `--tau`; sustituye
  el concepto de "factibilidad" del estudio-plantilla, que no aplica a optimización
  continua sin restricciones.

---

## 8. Estructura del código

```
algorithms/      sia_online.py (PSO base+online)  ·  competitors/ (los 20 competidores)
benchmarks/      classical.py  cec.py  bbob.py  registry.py
online_learning/ trace_recorder · feature_builder · label_builder · online_model
explainability/  xai_module.py (Permutation Importance)
adaptation/      adaptation_controller.py (reglas + XAI-guided)
analysis/        statistical_tests · plots · template_plots · summary_table · metrics
experiments/     run_sia_online.py · run_competitors.py · run_analysis.py
```
