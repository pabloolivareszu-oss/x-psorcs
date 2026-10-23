# x-psorcs
Online, explainable control for particle swarm optimization, with traceable adaptations, fixed-budget benchmarks, and a railway crew scheduling case study.


# SIA-online

**Online, explainable control for particle swarm optimization.**

SIA-online observes a particle swarm during optimization, predicts whether its current state will lead to improvement, and uses model summaries and explicit rules to adapt the search. Each adaptation is recorded with its triggering rule and signal.

This repository supports the manuscript *Explaining and adapting particle swarm search online for continuous benchmarks and railway crew scheduling*, by Mauricio De Juan and Rodrigo Olivares, Universidad de Valparaíso.

## Method

The study compares two variants of the same swarm optimizer:

- **SIA-base:** fixed exploration and exploitation rates, with fixed probabilities for standard, diversification, and intensification operators. It includes the same three operators as SIA-online; it is not an unmodified canonical PSO.
- **SIA-online:** the same optimizer with a controller that updates those rates and operator probabilities during the run.

The controller combines five operations:

1. **Observe:** record 15 features describing fitness, diversity, stagnation, recent improvement, and the current configuration.
2. **Learn:** fit a random forest on a sliding window of states labeled by improvement within a short future horizon.
3. **Explain:** compute permutation importance and combine it with the sign of feature–prediction correlations over the window.
4. **Adapt:** apply explicit rules to exploration, exploitation, and operator probabilities.
5. **Record:** log the rule, its triggering evidence, and the resulting change.

Permutation importance describes the fitted model's dependence on its inputs. The directional score is an association over the training window, not a causal effect or a local explanation of an individual prediction. The predictive model estimates future search improvement; it does not approximate the objective function.

A probability floor keeps every operator available. Decay toward the initial configuration limits the persistence of parameter changes, and a cooldown limits repeated diversification-rule updates. The controller makes no additional objective-function evaluations, although fitting and explanation add solving time.

### Controller settings

| Setting | Value |
|---|---:|
| Improvement horizon | 3 iterations |
| Improvement tolerance | `1e-8` |
| Training window | Up to 200 examples |
| Minimum training data | 30 examples in total, with both classes represented |
| Random forest | 100 trees |
| Refit interval | 5 iterations |
| Explanation interval | 10 iterations |
| Permutation repetitions | 5 |
| Features retained per explanation | 3 |
| Minimum operator probability | 0.05 |
| Rate decay | 0.02 |
| Operator-probability decay | 0.05 |
| Diversification-rule cooldown | 5 iterations |

The probability prior is `(0.70, 0.15, 0.15)` for the standard, diversification, and intensification operators. Both initial control rates are `0.5`.

## Experimental scope

| Family | Evaluated dimensions | Reported function–dimension cells | Runs per cell | Nominal evaluations per run |
|---|---|---:|---:|---:|
| Classical functions | 10, 20, 50 | 15 | 30 | 3,000 |
| CEC 2014 | 10, 20, 50 | 90 | 30 | 3,000 |
| CEC 2017 | 10, 20, 50, where supported | 73 | 30 | 3,000 |
| CEC 2022 | 10, 20 | 24 | 30 | 3,000 |
| Noiseless BBOB | 10, 20 | 48 | 30 | 3,000 |
| FIR filter design | 31 | 8 | 30 | 3,000 |
| Railway crew scheduling | 47,311–63,009 variables | 3 instances | 31 | 20,000 |

The continuous evaluation contains **258 cells and 23 methods**: the two SIA variants and 21 competitor implementations. The railway comparison contains **19 methods** on `rail507`, `rail516`, and `rail582`.

The campaign driver uses common seeds, calculated as `dimension * 1000 + run_id`. In BBOB, the implementation cycles through instances 1–15, so 30 runs use each instance twice with different optimizer seeds.

Use `codigo/run.py` or the campaign launcher to reproduce the evaluation-budget convention. These account for the initial population. Direct use of lower-level experiment runners has a different iteration convention. SO consumes fewer evaluations than the nominal budget in some phases; the reported comparison retains this behavior.

On the railway instances, L-SHADE, L-SHADE-EpSin, EBOwithCMAR, and EA4eig are excluded because their population or matrix requirements exceed the available resources. Continuous candidate solutions share the same binarization and repair procedure.

### Competitors

The registered identifiers are `ASGK`, `EA4eig`, `APGSK-IMODE`, `AMALGAM`, `UMOEAII`, `JADE`, `BKA`, `GOOSE`, `GWO`, `HHO`, `DBO`, `BWO`, `SO`, `IGWO`, `LHHO`, `LSHADE`, `LSHADE-EpSin`, `EBOwithCMAR`, `SACS`, `KRMG`, and `FGIN`.

These are implementations included in this project, based on published algorithm descriptions. They should not be treated as the original authors' reference software or as reproductions of competition rankings.

## Reported findings

- SIA-online improves on SIA-base at the family level on classical functions, CEC 2014, CEC 2017, and FIR, including correction across the six family-level tests.
- Across individual benchmark cells, unpaired tests favor SIA-online in **30 cells**, detect no difference in **226**, and favor SIA-base in **2**.
- On `rail582`, SIA-online obtains the lowest mean relative percentage deviation among the 19 evaluated methods: **6.25%**, compared with **6.71%** for SIA-base.
- All **240,863 recorded adaptations** are linked to their triggering rules and signals.

Adaptive differential evolution variants achieve better results on the continuous suites. Predictive discrimination is weak on short benchmark runs, and an ablation isolating the contribution of explanation-driven rules was not performed. The benchmark families were used during controller development; the railway instances were not used to design the controller.

These results describe the supplied experimental campaign under its stated budgets. They do not establish a general performance advantage for PSO or an isolated causal contribution from explainability.

## Repository contents

Paths below are relative to the root containing `codigo/` and `paper/`.

| Path | Contents |
|---|---|
| `codigo/run.py` | Campaign driver, budget accounting, and run resumption |
| `codigo/algorithms/` | Shared swarm optimizer and competitor implementations |
| `codigo/online_learning/` | State features, delayed labels, and predictive model |
| `codigo/explainability/` | Permutation importance and directional summaries |
| `codigo/adaptation/` | Adaptation rules, decay, cooldown, and probability floor |
| `codigo/benchmarks/` | Benchmark wrappers and application problems |
| `codigo/experiments/` | Lower-level runners and analysis entry points |
| `codigo/analysis/` | Statistical analysis and railway reporting |
| `codigo/campana/` | Campaign launcher, configuration, audit, and report generation |
| `codigo/tools/fetch_rail.py` | Railway instance download and validation |
| `paper/` | Manuscripts, compiled documents, numerical macros, and supplementary material |
| `paper/data/` | Aggregated results and data used in manuscript figures and tables |
| `paper/scripts/` | Manuscript statistics and artifact-generation scripts |
| `anexos/` | Additional methodological documents |
| `resultados/` | Generated run outputs; absent from the supplied source archive |

Other application wrappers, including microhub, antenna, and photovoltaic problems, are retained in the code but are outside the reported study. Specify the desired families explicitly when running `run.py`.

## Installation

Use a virtual environment from the repository root. The requirements file documents Python 3.12 and pins the dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r codigo/requirements.txt
```

The manuscript records Python 3.11.15 on an Apple M2 Ultra with 64 GB RAM. This differs from the Python version stated in the installation notes; the dependency pins alone do not constitute a complete environment lock.

Check the scientific dependencies:

```bash
python -c "import numpy, scipy, sklearn, matplotlib, opfunu, cocoex, numba; print('Dependencies available')"
```

The campaign launcher requires Bash. Some campaign metadata and memory-monitoring utilities use macOS commands. Review those utilities before attempting the full launcher on another operating system; cross-platform campaign execution has not been validated here.

## Quick run

From the repository root, enter `codigo/` and use a separate output directory:

```bash
cd codigo
export TESIS_RESULTS_DIR=../resultados_smoke

python run.py \
  --families classical \
  --funcs sphere \
  --only SIA-base SIA-online \
  --runs 2 \
  --pop-size 30 \
  --max-iter 100 \
  --jobs 1 \
  --no-analysis \
  --resume
```

This runs Sphere in the driver's classical dimensions, 10, 20, and 50, with 3,000 objective evaluations per run for each SIA variant. It checks basic execution and logging; it does not reproduce the full study.

View the available campaign options with:

```bash
python run.py --help
```

## Full campaign

Run these commands from `codigo/` in a configured environment. `JOBS=1` is a conservative starting point; increase it according to available memory. The original launcher defaults to 20 workers and an 8 GB memory reserve.

```bash
export TESIS_RESULTS_DIR=../resultados_reproduction
export JOBS=1
export MEM_RESERVE=8

bash campana/lanzar.sh config
bash campana/lanzar.sh fase1
bash campana/lanzar.sh analisis1

python tools/fetch_rail.py
bash campana/lanzar.sh rail-campana
bash campana/lanzar.sh rail-analisis

bash campana/lanzar.sh auditoria
```

The railway download is required before running the railway campaign. The source archive contains the best-known-value metadata, but not the three instance files.

The launcher requests run resumption. Use a new results directory when changing the method, dependency environment, or experimental configuration; the resume checks do not identify every possible implementation change.

`TESIS_RESULTS_DIR` is interpreted relative to `codigo/`. It defaults to `../resultados` when unset.

## Outputs and analysis

SIA run directories contain `convergence.csv`, `traces.csv`, `predictions.csv`, `explanations.csv`, `adaptations.csv`, and `summary.json`. Competitor runs contain convergence records and summaries. Campaign-level files record configuration, execution status, and audit information.

Statistical analysis includes Vargha–Delaney effect sizes, family-level Wilcoxon tests, Friedman rankings, and corrected pairwise comparisons. Within-cell Mann–Whitney tests and A12 estimates summarize unpaired outcome distributions despite the common seeds. Railway comparisons also use paired tests and paired bootstrap estimates.

The source archive includes aggregated manuscript data and the numerical supplement, **not the full raw `resultados/` directory**. Compiling the manuscript from its existing data does not require rerunning the experiments. Recomputing the complete analysis requires the raw records.

## Manuscript and numerical artifacts

For the original manuscript layout, run from `paper/`:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

pdflatex supplementary.tex
pdflatex supplementary.tex
```

A LaTeX installation must provide the packages requested by the selected source, including TikZ, PGFPlots, and algorithm2e. The original source also uses Font Awesome icons.

For the revised academic package, select `Paper_SIA_academic.tex` as the main document and keep its `cifras.tex`, `data/`, class, and style files alongside it. Its bibliography is embedded and its diagram uses vector icons, so it needs two pdfLaTeX passes and no BibTeX pass.

To regenerate manuscript statistics after obtaining the complete raw campaign, run these scripts from `paper/`:

```bash
python ../codigo/campana/hipotesis.py
python ../codigo/campana/informe/generar.py
python scripts/estadistica.py
python scripts/figuras_extra.py
python scripts/estadistica_protocolo.py
python scripts/macros.py
python scripts/suplementario.py
```

These report and manuscript-generation scripts read the repository's fixed `resultados/` directory; they do not inherit the campaign's `TESIS_RESULTS_DIR` override. The report generator supplies `resultados/informe/datos.json`, which `estadistica.py` requires. If the campaign used another output directory, adjust the fixed input paths in a separate working copy before running these scripts. They overwrite derived numerical artifacts, so retain the supplied manuscript data for comparison.

## Citation and availability

The associated manuscript is:

> Mauricio De Juan and Rodrigo Olivares. *Explaining and adapting particle swarm search online for continuous benchmarks and railway crew scheduling*.

The supplied manuscript does not yet specify a public repository or dataset DOI. Its final citation and archival identifiers should be added when available.

No `LICENSE` file is included in the supplied source package. A reuse license has not been declared in this README.
