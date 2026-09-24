# Graph-Guided Evolutionary Optimisation for Multi-Objective Operating Theatre Scheduling

This repository contains the implementation, experimental results,
analysis scripts, and final solution exports for the Computational
Intelligence Optimisation (CIO) project on multi-objective elective
Operating Theatre (OT) scheduling.

The final proposed method is **Graph-PMX NSGA-II**: NSGA-II using a
resource-conflict-guided variant of Partially Mapped Crossover (PMX)
together with standard swap mutation. A Graph-Guided Local Search (GLS)
component was also investigated during the factorial component study,
but it was not retained in the final configuration because the
experimental evidence did not show a consistent benefit.

## 1. Problem formulation

The problem uses an indirect permutation representation. Each chromosome
is a permutation of surgery indices. A deterministic greedy decoder
processes surgeries in permutation order and attempts to assign each
surgery to an eligible:

-   operating theatre;
-   planning day; and
-   surgical team.

Hard feasibility is handled by the decoder. It checks OT and team
eligibility, daily capacity, team consecutive-day limits, and downstream
ICU/ward capacity according to the instance policy. If a surgery cannot
be placed feasibly, it remains unscheduled rather than creating an
infeasible assignment.

The supplied evaluator uses three minimisation objectives:

1.  **f1 --- negative priority-weighted throughput**;
2.  **f2 --- resource-utilisation imbalance**; and
3.  **f3 --- waiting-time fairness across priority classes**.

## 2. Algorithms

The official-scale comparison contains four algorithms:

-   **NSGA-II** --- standard NSGA-II with PMX and swap mutation;
-   **MOEA/D** --- decomposition-based multi-objective evolutionary
    algorithm;
-   **PLS** --- Pareto Local Search;
-   **Graph-PMX NSGA-II** --- NSGA-II using the proposed
    resource-conflict-guided PMX operator.

The component study additionally investigated **Graph-Guided Local
Search (GLS)** in a 2 × 2 factorial design to separate the effects of
Graph-PMX and GLS.

## 3. Repository structure

``` text
cio_skeleton/
├── cio/
│   ├── algorithms/
│   ├── analysis/
│   ├── harness/
│   ├── operators/
│   └── problem/
├── configs/
│   ├── nsga2_baseline.yaml
│   ├── moead_baseline.yaml
│   └── pls_baseline.yaml
├── instances/
├── scripts/
│   ├── generate_instances.py
│   ├── generate_reference_fronts.py
│   ├── run.py
│   └── analyse_final_results.py
├── results/
│   ├── final_baselines/
│   ├── final_solutions/
│   ├── reproducibility_analysis/
│   ├── paper_figures/
│   └── paper_tables/
├── README.md
└── requirements.txt
```

The exact contents may include additional development or analysis files,
but the directories above contain the main reproducibility material.

## 4. Environment setup

The experiments were conducted with Python 3.10.

From the `cio_skeleton` directory, install the required packages using:

``` powershell
python -m pip install -r requirements.txt
```

The exact dependency versions used by the project are recorded in
`requirements.txt`.

## 5. Running an experiment

Experiment configurations are stored as YAML files under `configs/`.

For example, from the `cio_skeleton` directory, a standard NSGA-II
experiment can be executed using:

``` powershell
python scripts/run.py --config configs/nsga2_baseline.yaml --instance instances/ot_small.json --out results/example_nsga2_small/
```

The experiment harness records each stochastic run separately as a JSON
file and also writes a `summary.json` file to the selected output
directory.

**Important:** the official 30-run results used in the paper are already
stored under `results/final_baselines/`. Re-running an experiment is not
required to reproduce the reported statistical analysis and may
overwrite or create results different from the archived submission
evidence if a different environment or configuration is used.

## 6. Official experimental protocol

Four OT instances were evaluated:

-   Small;
-   Medium;
-   Large; and
-   Hidden.

The Hidden instance was evaluated after method development and was not
used for further tuning.

All reported stochastic comparisons use **30 independent runs**. Matched
seed lists were used between compared algorithms/configurations to
support paired statistical testing.

For the official-scale experiment:

  Algorithm                    Main budget
  ------------------- --------------------
  NSGA-II               10,000 evaluations
  Graph-PMX NSGA-II     10,000 evaluations
  PLS                   10,000 evaluations
  MOEA/D                10,500 evaluations

NSGA-II and Graph-PMX therefore form the primary budget-matched
comparison. MOEA/D uses 105 Das-Dennis reference directions for three
objectives and consequently performs 10,500 evaluations under the
supplied builder configuration.

## 7. Reproducing the final analysis

The final statistical analysis can be reconstructed directly from the
archived per-run result files. It is **not necessary to rerun the
optimisation algorithms**.

From the `cio_skeleton` directory, run:

``` powershell
python -m scripts.analyse_final_results
```

The script audits the final result folders, reconstructs the empirical
reference fronts and run-level quality indicators, and performs the
final statistical analysis.

A successful result audit should find:

``` text
Small
  NSGA-II              runs=30
  MOEA/D               runs=30
  PLS                  runs=30
  Graph-PMX NSGA-II    runs=30

Medium
  NSGA-II              runs=30
  MOEA/D               runs=30
  PLS                  runs=30
  Graph-PMX NSGA-II    runs=30

Large
  NSGA-II              runs=30
  MOEA/D               runs=30
  PLS                  runs=30
  Graph-PMX NSGA-II    runs=30

Hidden
  NSGA-II              runs=30
  MOEA/D               runs=30
  PLS                  runs=30
  Graph-PMX NSGA-II    runs=30
```

All compared algorithms should report aligned seeds.

There are:

``` text
4 instances × 4 algorithms × 30 runs = 480 run-level records
```

The reconstructed empirical reference-front sizes used in the final
analysis are:

  Instance     Reference-front size
  ---------- ----------------------
  Small                        3590
  Medium                        642
  Large                         701
  Hidden                        591

The final hypervolume reference points are:

``` text
Small   : [134.0,   195.0,   4.7244]
Medium  : [2900.0, 1080.0, 41.61675376]
Large   : [18339.0, 2227.25, 75.94667843]
Hidden  : [13937.0, 1806.0, 36.58299443]
```

The reference point is constructed as the component-wise maximum
objective value in the pooled comparison data plus one.

## 8. Performance indicators

The final analysis uses:

-   **Hypervolume (HV)** --- higher is better;
-   **IGD+** --- lower is better; and
-   **Spread** --- larger values indicate wider spread under the project
    indicator implementation.

The reproducibility outputs report mean, median, sample standard
deviation, and interquartile range.

## 9. Statistical analysis

For each instance and performance indicator, the four
algorithms/configurations are first compared using the **Friedman
test**.

Where the omnibus test is significant, matched-seed pairwise comparisons
use the **Wilcoxon signed-rank test** with **Holm correction** across
the six pairwise comparisons.

Practical effect magnitude is reported using paired **rank-biserial
correlation**, interpreted using:

``` text
|r| < 0.10          negligible
|r| < 0.30          small
|r| < 0.50          medium
|r| >= 0.50         large
```

Statistical significance is assessed at `alpha = 0.05`.

## 10. Key final Graph-PMX vs NSGA-II results

The primary confirmatory comparison is Graph-PMX NSGA-II versus standard
NSGA-II under the same 10,000-evaluation budget.

  ------------------------------------------------------------------------------
  Instance   Metric           Relative         Holm p Significant   Effect
                                change                              magnitude
  ---------- ---------- -------------- -------------- ------------- ------------
  Small      HV                 +0.00%       0.967674 No            negligible

  Small      IGD+              -55.45%       0.140283 No            medium

  Small      Spread             +0.83%       0.792159 No            negligible

  Medium     HV                 +4.02%       0.004735 Yes           large

  Medium     IGD+              -13.20%       0.087939 No            medium

  Medium     Spread             +5.95%       0.036435 Yes           medium

  Large      HV                 +4.32%       0.003728 Yes           large

  Large      IGD+              -10.43%       0.076721 No            medium

  Large      Spread             +2.43%       0.381798 No            small

  Hidden     HV                 +2.57%       0.104840 No            medium

  Hidden     IGD+              -22.70%       0.000872 Yes           large

  Hidden     Spread            +21.68%       0.000137 Yes           large
  ------------------------------------------------------------------------------

Negative IGD+ change is favourable because lower IGD+ is better.
Positive HV and spread changes are favourable under the indicators used
in this project.

## 11. Final solution exports

The final Graph-PMX fronts from all 30 runs were merged, duplicate
objective vectors were removed, and the global non-dominated solutions
were decoded into realised OT schedules.

The final exports are stored under:

``` text
results/final_solutions/ot_small/
results/final_solutions/ot_medium/
results/final_solutions/ot_large/
results/final_solutions/ot_hidden/
```

Each instance directory contains:

``` text
nondominated_solutions.json
nondominated_objectives.csv
nondominated_schedules.csv
```

The final export audit produced:

  -----------------------------------------------------------------------
  Instance            Non-dominated          Scheduled        Unscheduled
                          solutions          surgeries          surgeries
  -------------- ------------------ ------------------ ------------------
  Small                          66             18--21               1--4

  Medium                        288           105--114              6--15

  Large                         302           228--238             42--52

  Hidden                        274           152--158             42--48
  -----------------------------------------------------------------------

All exported permutations passed the permutation-validity check, and all
realised schedules passed the decoder-based feasibility audit.
Unscheduled requests are allowed by the formulation and are penalised
through the first objective rather than treated as hard-constraint
violations.

## 12. Reproducibility outputs

The reconstructed analysis outputs are written to:

``` text
results/reproducibility_analysis/
```

Important files include:

``` text
all_run_metrics.csv
final_descriptive_statistics.csv
friedman_tests.csv
posthoc_wilcoxon_holm.csv
```

Paper-oriented tables and figures are stored under:

``` text
results/paper_tables/
results/paper_figures/
```

## 13. Notebook

`CIO_Week2_NextSteps_FINAL.ipynb` documents the development and
experimental workflow.

The notebook is supplementary to the archived raw run files. The
authoritative evidence for the final reported experiments is the set of
seeded JSON run logs under `results/final_baselines/`, together with the
reproducibility analysis generated from those files.

The plotting cells recreate the paper figures from saved results and do
not rerun the optimisation algorithms.

## 14. Reproducibility notes

To preserve the exact evidence used in the paper:

1.  Do not overwrite the archived folders under
    `results/final_baselines/`.
2.  Use `python -m scripts.analyse_final_results` to reconstruct the
    final indicators and statistical tests from the saved logs.
3.  Use the provided configuration files and requirements when executing
    new experiments.
4.  Treat newly generated experimental folders as separate from the
    archived official results.
5.  The primary method comparison is the budget-matched Graph-PMX
    NSGA-II versus standard NSGA-II comparison.

## 15. Summary

The experimental workflow consists of:

``` text
OT instance
    ↓
permutation representation
    ↓
NSGA-II / MOEA/D / PLS / Graph-PMX NSGA-II
    ↓
greedy feasibility-by-construction decoder
    ↓
three-objective evaluation
    ↓
30 independent seeded runs
    ↓
HV / IGD+ / Spread
    ↓
Friedman test
    ↓
paired Wilcoxon + Holm correction
    ↓
rank-biserial effect size
    ↓
final non-dominated schedule exports
```

The final selected method is **Graph-PMX NSGA-II**, based on the
component study and subsequent official-scale comparison.
