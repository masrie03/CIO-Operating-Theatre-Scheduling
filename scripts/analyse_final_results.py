# ============================================================
# FINAL RESULTS ANALYSIS
# Reproduces final statistical tables from saved experiment logs
# ============================================================

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from cio.analysis.indicators import (
    hypervolume,
    igd_plus,
    spread,
    union_non_dominated,
)

from cio.analysis.statistics import friedman


# ============================================================
# PATHS
# ============================================================

RESULT_ROOT = Path("results/final_baselines")

OUTPUT_DIR = Path("results/reproducibility_analysis")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


INSTANCES = {
    "Small": "small",
    "Medium": "medium",
    "Large": "large",
    "Hidden": "hidden",
}


ALGORITHM_DIRS = {
    "NSGA-II": "nsga2",
    "MOEA/D": "moead",
    "PLS": "pls",
    "Graph-PMX NSGA-II": "graph_pmx",
}


# ============================================================
# LOAD SAVED RUNS
# ============================================================

def load_json_runs(folder):

    folder = Path(folder)

    files = sorted(
        folder.glob("run_*.json")
    )

    if not files:

        raise FileNotFoundError(
            f"No run files found in: {folder}"
        )

    runs = []

    for path in files:

        with open(path, "r") as f:
            record = json.load(f)

        F = np.asarray(
            record["F"],
            dtype=float,
        )

        if F.ndim == 1:
            F = F.reshape(1, -1)

        runs.append({
            "seed":
                int(record["seed"]),

            "F":
                F,

            "evaluations":
                int(
                    record.get(
                        "evaluations",
                        0,
                    )
                ),

            "runtime_seconds":
                float(
                    record.get(
                        "runtime_seconds",
                        np.nan,
                    )
                ),

            "file":
                str(path),
        })

    return runs


# ============================================================
# INITIAL AUDIT
# ============================================================

def audit_results():

    print("=" * 75)
    print("FINAL ANALYSIS — RESULT AUDIT")
    print("=" * 75)

    all_complete = True

    for instance_label, instance_key in INSTANCES.items():

        print(
            f"\n{instance_label}"
        )

        reference_seeds = None

        for algorithm, directory_key in ALGORITHM_DIRS.items():

            folder = (
                RESULT_ROOT
                / f"{directory_key}_{instance_key}"
            )

            runs = load_json_runs(
                folder
            )

            seeds = [
                run["seed"]
                for run in runs
            ]

            if reference_seeds is None:
                reference_seeds = seeds

            aligned = (
                seeds == reference_seeds
            )

            complete = (
                len(runs) == 30
            )

            all_complete &= (
                complete and aligned
            )

            print(
                f"  {algorithm:<20} "
                f"runs={len(runs):>2} | "
                f"seeds_aligned={aligned}"
            )

    print()
    print(
        "Audit passed:",
        all_complete
    )

    return all_complete

# ============================================================
# RECONSTRUCT INDICATORS FROM RAW RUN LOGS
# ============================================================

def analyse_instance(
    instance_label,
    instance_key,
):

    # --------------------------------------------------------
    # 1. Load all four algorithms
    # --------------------------------------------------------

    runs_by_algorithm = {}

    for algorithm, directory_key in ALGORITHM_DIRS.items():

        folder = (
            RESULT_ROOT
            / f"{directory_key}_{instance_key}"
        )

        runs_by_algorithm[algorithm] = (
            load_json_runs(folder)
        )


    # --------------------------------------------------------
    # 2. Build shared empirical reference front
    # --------------------------------------------------------

    all_fronts = [
        run["F"]
        for runs in runs_by_algorithm.values()
        for run in runs
    ]

    all_points = np.vstack(
        all_fronts
    )

    reference_front = union_non_dominated(
        *all_fronts
    )

    hv_reference_point = (
        np.max(
            all_points,
            axis=0,
        )
        + 1.0
    )


    # --------------------------------------------------------
    # 3. Calculate metrics for every run
    # --------------------------------------------------------

    rows = []

    for algorithm, runs in runs_by_algorithm.items():

        for run in runs:

            F = run["F"]

            rows.append({
                "instance":
                    instance_label,

                "algorithm":
                    algorithm,

                "seed":
                    run["seed"],

                "evaluations":
                    run["evaluations"],

                "runtime_seconds":
                    run["runtime_seconds"],

                "front_size":
                    len(F),

                "hv":
                    hypervolume(
                        F,
                        hv_reference_point,
                    ),

                "igd_plus":
                    igd_plus(
                        F,
                        reference_front,
                    ),

                "spread":
                    spread(F),
            })


    metrics_df = pd.DataFrame(
        rows
    )


    # --------------------------------------------------------
    # 4. Descriptive statistics
    # --------------------------------------------------------

    summary_rows = []

    for algorithm in ALGORITHM_DIRS:

        subset = metrics_df[
            metrics_df["algorithm"]
            == algorithm
        ]


        row = {
            "Instance":
                instance_label,

            "Algorithm":
                algorithm,
        }


        for metric in [
            "hv",
            "igd_plus",
            "spread",
        ]:

            values = subset[
                metric
            ].to_numpy()


            q1 = np.percentile(
                values,
                25,
            )

            q3 = np.percentile(
                values,
                75,
            )


            row[
                f"{metric}_mean"
            ] = np.mean(values)

            row[
                f"{metric}_median"
            ] = np.median(values)

            row[
                f"{metric}_std"
            ] = np.std(
                values,
                ddof=1,
            )

            row[
                f"{metric}_iqr"
            ] = q3 - q1


        row["runtime_mean"] = (
            subset[
                "runtime_seconds"
            ].mean()
        )

        row["evaluation_mean"] = (
            subset[
                "evaluations"
            ].mean()
        )


        summary_rows.append(
            row
        )


    summary_df = pd.DataFrame(
        summary_rows
    )


    return {
        "runs":
            runs_by_algorithm,

        "metrics":
            metrics_df,

        "summary":
            summary_df,

        "reference_front":
            reference_front,

        "hv_reference_point":
            hv_reference_point,
    }

# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

def holm_correction(p_values):

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    order = np.argsort(
        p_values
    )

    adjusted = np.empty(
        len(p_values),
        dtype=float,
    )

    running_max = 0.0
    m = len(p_values)

    for rank, idx in enumerate(order):

        corrected = (
            (m - rank)
            * p_values[idx]
        )

        corrected = min(
            corrected,
            1.0,
        )

        running_max = max(
            running_max,
            corrected,
        )

        adjusted[idx] = (
            running_max
        )

    return adjusted


def paired_rank_biserial(x, y):

    diff = (
        np.asarray(x)
        - np.asarray(y)
    )

    diff = diff[
        diff != 0
    ]

    if len(diff) == 0:
        return 0.0

    ranks = (
        pd.Series(
            np.abs(diff)
        )
        .rank(
            method="average"
        )
        .to_numpy()
    )

    positive = ranks[
        diff > 0
    ].sum()

    negative = ranks[
        diff < 0
    ].sum()

    total = (
        positive
        + negative
    )

    if total == 0:
        return 0.0

    return (
        positive
        - negative
    ) / total


def effect_magnitude(r):

    value = abs(r)

    if value < 0.10:
        return "negligible"

    if value < 0.30:
        return "small"

    if value < 0.50:
        return "medium"

    return "large"


def run_statistical_analysis(
    metrics_df
):

    algorithms = [
        "NSGA-II",
        "MOEA/D",
        "PLS",
        "Graph-PMX NSGA-II",
    ]

    metrics = [
        "hv",
        "igd_plus",
        "spread",
    ]


    friedman_rows = []
    posthoc_rows = []


    # --------------------------------------------------------
    # Friedman omnibus tests
    # --------------------------------------------------------

    for instance_name in (
        metrics_df["instance"]
        .unique()
    ):

        instance_df = metrics_df[
            metrics_df["instance"]
            == instance_name
        ]


        for metric in metrics:

            columns = []

            for algorithm in algorithms:

                values = (
                    instance_df[
                        instance_df[
                            "algorithm"
                        ]
                        == algorithm
                    ]
                    .sort_values(
                        "seed"
                    )[metric]
                    .to_numpy()
                )

                columns.append(
                    values
                )


            matrix = np.column_stack(
                columns
            )

            result = friedman(
                matrix
            )


            friedman_rows.append({
                "instance":
                    instance_name,

                "metric":
                    metric,

                "statistic":
                    result[
                        "statistic"
                    ],

                "p_value":
                    result[
                        "p_value"
                    ],

                "n_runs":
                    result[
                        "n_runs"
                    ],

                "n_configurations":
                    result[
                        "n_configurations"
                    ],

                "significant_0.05":
                    result[
                        "p_value"
                    ] < 0.05,
            })


            # ------------------------------------------------
            # Post-hoc only if Friedman significant
            # ------------------------------------------------

            if (
                result["p_value"]
                >= 0.05
            ):
                continue


            pair_results = []


            for config_a, config_b in combinations(
                algorithms,
                2,
            ):

                a = (
                    instance_df[
                        instance_df[
                            "algorithm"
                        ]
                        == config_a
                    ]
                    .sort_values(
                        "seed"
                    )[metric]
                    .to_numpy()
                )

                b = (
                    instance_df[
                        instance_df[
                            "algorithm"
                        ]
                        == config_b
                    ]
                    .sort_values(
                        "seed"
                    )[metric]
                    .to_numpy()
                )


                try:

                    stat, raw_p = wilcoxon(
                        a,
                        b,
                        alternative="two-sided",
                    )

                except ValueError:

                    stat = 0.0
                    raw_p = 1.0


                effect = (
                    paired_rank_biserial(
                        a,
                        b,
                    )
                )


                pair_results.append({
                    "instance":
                        instance_name,

                    "metric":
                        metric,

                    "config_a":
                        config_a,

                    "config_b":
                        config_b,

                    "mean_a":
                        np.mean(a),

                    "mean_b":
                        np.mean(b),

                    "wilcoxon_stat":
                        stat,

                    "raw_p":
                        raw_p,

                    "effect_size":
                        effect,

                    "effect_magnitude":
                        effect_magnitude(
                            effect
                        ),
                })


            corrected = holm_correction(
                [
                    row["raw_p"]
                    for row in pair_results
                ]
            )


            for row, corrected_p in zip(
                pair_results,
                corrected,
            ):

                row["holm_p"] = (
                    corrected_p
                )

                row[
                    "significant_0.05"
                ] = (
                    corrected_p
                    < 0.05
                )

                posthoc_rows.append(
                    row
                )


    return (
        pd.DataFrame(
            friedman_rows
        ),
        pd.DataFrame(
            posthoc_rows
        ),
    )

# ============================================================
# MAIN
# ============================================================
# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    passed = audit_results()

    if not passed:
        raise RuntimeError(
            "Result audit failed."
        )

    all_metrics = []
    all_summaries = []

    print()
    print("=" * 75)
    print("RECONSTRUCTING FINAL INDICATORS")
    print("=" * 75)

    for instance_label, instance_key in INSTANCES.items():

        print(
            f"\nAnalysing {instance_label}..."
        )

        analysis = analyse_instance(
            instance_label,
            instance_key,
        )

        all_metrics.append(
            analysis["metrics"]
        )

        all_summaries.append(
            analysis["summary"]
        )

        print(
            "Reference-front size:",
            len(
                analysis[
                    "reference_front"
                ]
            )
        )

        print(
            "HV reference point:",
            analysis[
                "hv_reference_point"
            ]
        )


    # --------------------------------------------------------
    # Combine all instances
    # --------------------------------------------------------

    final_metrics_df = pd.concat(
        all_metrics,
        ignore_index=True,
    )

    final_summary_df = pd.concat(
        all_summaries,
        ignore_index=True,
    )


    # --------------------------------------------------------
    # Statistical analysis
    # --------------------------------------------------------

    friedman_df, posthoc_df = (
        run_statistical_analysis(
            final_metrics_df
        )
    )


    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    metrics_path = (
        OUTPUT_DIR
        / "all_run_metrics.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "final_descriptive_statistics.csv"
    )

    friedman_path = (
        OUTPUT_DIR
        / "friedman_tests.csv"
    )

    posthoc_path = (
        OUTPUT_DIR
        / "posthoc_wilcoxon_holm.csv"
    )


    # --------------------------------------------------------
    # Save reconstructed outputs
    # --------------------------------------------------------

    final_metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    final_summary_df.to_csv(
        summary_path,
        index=False,
    )

    friedman_df.to_csv(
        friedman_path,
        index=False,
    )

    posthoc_df.to_csv(
        posthoc_path,
        index=False,
    )


    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("RECONSTRUCTION COMPLETE")
    print("=" * 75)

    print(
        "Run-level rows:",
        len(final_metrics_df)
    )

    print(
        "Expected:",
        4 * 4 * 30
    )

    print()
    print("=" * 75)
    print("FRIEDMAN TESTS")
    print("=" * 75)

    print(
        friedman_df.to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    print(metrics_path)
    print(summary_path)
    print(friedman_path)
    print(posthoc_path)
