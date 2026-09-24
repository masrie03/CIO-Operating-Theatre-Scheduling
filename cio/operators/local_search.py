"""Pareto Local Search (PLS).

A simple multi-objective local search that explores neighbours of a single
solution and keeps any non-dominated improvements. Used both as a stand-alone
baseline and as a local-search component in memetic configurations.

Students may design alternative local search operators (problem-tailored
neighbourhoods, adaptive step size, restart strategies) and pass them in.
"""
from __future__ import annotations

import numpy as np
from typing import Protocol

from ..problem import Instance, OTSchedulingProblem
from ..problem.evaluator import evaluate as _evaluate
from ..problem.decoder import GreedyDecoder


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Strict Pareto dominance: a dominates b iff a <= b component-wise
    AND a < b in at least one component."""
    return bool(np.all(a <= b) and np.any(a < b))


class BaseLocalSearch(Protocol):
    def __call__(self, problem: OTSchedulingProblem, perm: np.ndarray, budget: int,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Returns (improved_perm, improved_F) after up to `budget` evaluations."""
        ...


class SwapNeighbourhoodLocalSearch:
    """Explores swap-neighbours of the current solution.

    Stops on first improvement found in each pass (first-improvement strategy).
    Repeats passes until no improvement is found or budget is exhausted.
    """

    def __call__(self, problem, perm, budget, rng):
        decoder = GreedyDecoder()
        inst = problem.instance
        current = perm.copy()
        sched = decoder(inst, current)
        current_f = np.array(_evaluate(inst, sched))
        evals = 1
        improved = True
        while improved and evals < budget:
            improved = False
            n = len(current)
            pair_order = rng.permutation(n * (n - 1) // 2)
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            for k in pair_order:
                if evals >= budget:
                    break
                i, j = pairs[int(k)]
                cand = current.copy()
                cand[i], cand[j] = cand[j], cand[i]
                sched = decoder(inst, cand)
                cand_f = np.array(_evaluate(inst, sched))
                evals += 1
                if dominates(cand_f, current_f):
                    current = cand
                    current_f = cand_f
                    improved = True
                    break
        return current, current_f
    
class GraphGuidedLocalSearch:
    """
    Problem-tailored local search for OT scheduling.

    Uses resource-conflict information to focus neighbourhood search on
    surgeries that compete strongly for hospital resources.

    Conflict information includes:
    - overlapping eligible OTs
    - overlapping eligible surgical teams
    - shared required equipment
    - ICU demand
    - long ward stays
    """

    def __init__(self, neighbourhood_size: int = 4):
        self.neighbourhood_size = neighbourhood_size

    def __call__(
        self,
        problem: OTSchedulingProblem,
        perm: np.ndarray,
        budget: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:

        inst = problem.instance

        # --------------------------------------------------
        # 1. Evaluate starting solution
        # --------------------------------------------------

        start_perm = perm.copy()

        start_sched = problem.decoder(
            inst,
            start_perm
        )

        start_F = np.array(
            _evaluate(inst, start_sched)
        )

        if budget <= 0:
            return start_perm, start_F

        # --------------------------------------------------
        # 2. Create local Pareto archive
        # --------------------------------------------------

        archive_perms = [
            start_perm.copy()
        ]

        archive_Fs = [
            start_F.copy()
        ]

        evaluations = 1

        # --------------------------------------------------
        # 3. Build conflict scores
        # --------------------------------------------------

        conflict_scores = self._conflict_scores(
            inst
        )

        # --------------------------------------------------
        # 4. Explore graph-guided neighbourhood
        # --------------------------------------------------

        while evaluations < budget:

            # Randomly choose one non-dominated archive
            # solution as the base for neighbourhood search
            base_idx = int(
                rng.integers(len(archive_perms))
            )

            base_perm = archive_perms[
                base_idx
            ]

            # Select a highly conflicted surgery
            surgery_idx = self._select_conflicted_surgery(
                base_perm,
                conflict_scores,
                rng,
            )

            position = int(
                np.where(
                    base_perm == surgery_idx
                )[0][0]
            )

            # Limit search to nearby permutation positions
            left = max(
                0,
                position - self.neighbourhood_size
            )

            right = min(
                len(base_perm),
                position + self.neighbourhood_size + 1
            )

            candidate_positions = [
                p
                for p in range(left, right)
                if p != position
            ]

            if not candidate_positions:
                evaluations += 1
                continue

            swap_position = int(
                rng.choice(candidate_positions)
            )

            candidate = base_perm.copy()

            candidate[position], candidate[swap_position] = (
                candidate[swap_position],
                candidate[position],
            )

            # --------------------------------------------------
            # 5. Evaluate candidate
            # --------------------------------------------------

            sched = problem.decoder(
                inst,
                candidate
            )

            candidate_F = np.array(
                _evaluate(inst, sched)
            )

            evaluations += 1

            # --------------------------------------------------
            # 6. Reject candidate if dominated by archive
            # --------------------------------------------------

            dominated = False

            for archived_F in archive_Fs:

                if (
                    dominates(
                        archived_F,
                        candidate_F
                    )
                    or np.array_equal(
                        archived_F,
                        candidate_F
                    )
                ):
                    dominated = True
                    break

            if dominated:
                continue

            # --------------------------------------------------
            # 7. Remove archive solutions dominated
            #    by the new candidate
            # --------------------------------------------------

            keep_indices = [
                i
                for i, archived_F
                in enumerate(archive_Fs)
                if not dominates(
                    candidate_F,
                    archived_F
                )
            ]

            archive_perms = [
                archive_perms[i]
                for i in keep_indices
            ]

            archive_Fs = [
                archive_Fs[i]
                for i in keep_indices
            ]

            # Add new non-dominated candidate
            archive_perms.append(
                candidate.copy()
            )

            archive_Fs.append(
                candidate_F.copy()
            )

        # --------------------------------------------------
        # 8. Select one balanced solution from archive
        # --------------------------------------------------

        F_matrix = np.array(
            archive_Fs
        )

        mins = F_matrix.min(
            axis=0
        )

        maxs = F_matrix.max(
            axis=0
        )

        ranges = np.where(
            maxs - mins < 1e-12,
            1.0,
            maxs - mins
        )

        normalised = (
            F_matrix - mins
        ) / ranges

        balance_scores = normalised.sum(
            axis=1
        )

        best_idx = int(
            np.argmin(balance_scores)
        )

        return (
            archive_perms[best_idx],
            archive_Fs[best_idx],
        )

    def _conflict_scores(
        self,
        instance,
    ) -> dict[int, float]:

        """
        Calculate weighted Resource Conflict Graph degree.

        Larger scores represent surgeries that compete more
        strongly with other surgeries for hospital resources.
        """

        scores = {
            idx: 0.0
            for idx in range(
                instance.n_surgeries
            )
        }

        for i in range(
            instance.n_surgeries
        ):

            surgery_i = (
                instance.surgery_by_index(i)
            )

            ots_i = set(
                instance.eligible_ots_for_surgery[
                    surgery_i.id
                ]
            )

            teams_i = set(
                instance.eligible_teams_for_surgery[
                    surgery_i.id
                ]
            )

            for j in range(
                i + 1,
                instance.n_surgeries
            ):

                surgery_j = (
                    instance.surgery_by_index(j)
                )

                ots_j = set(
                    instance.eligible_ots_for_surgery[
                        surgery_j.id
                    ]
                )

                teams_j = set(
                    instance.eligible_teams_for_surgery[
                        surgery_j.id
                    ]
                )

                conflict = 0.0

                # Shared OT alternatives
                if ots_i & ots_j:
                    conflict += 1.0

                # Shared surgical teams
                if teams_i & teams_j:
                    conflict += 1.0

                # Shared equipment
                if (
                    surgery_i.required_equipment
                    & surgery_j.required_equipment
                ):
                    conflict += 2.0

                # Both require ICU
                if (
                    surgery_i.icu_days > 0
                    and surgery_j.icu_days > 0
                ):
                    conflict += 1.0

                # Both impose long ward stays
                if (
                    surgery_i.ward_days >= 4
                    and surgery_j.ward_days >= 4
                ):
                    conflict += 1.0

                if conflict > 0:

                    scores[i] += conflict
                    scores[j] += conflict

        return scores

    def _select_conflicted_surgery(
        self,
        permutation,
        conflict_scores,
        rng,
    ) -> int:

        """
        Select randomly among highly conflicted surgeries
        to retain stochastic diversity.
        """

        ranked = sorted(
            permutation.tolist(),
            key=lambda idx: conflict_scores[
                int(idx)
            ],
            reverse=True,
        )

        # Top 25% most conflicted surgeries
        top_k = max(
            1,
            len(ranked) // 4
        )

        return int(
            rng.choice(
                ranked[:top_k]
            )
        )