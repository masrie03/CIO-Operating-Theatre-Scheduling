"""Operators and named hooks.

This module exposes baseline implementations of the variation, local-search,
constraint-handling, and replacement operators that students will design and
substitute. Students should subclass or instantiate alternative versions of
these classes and pass them to the algorithm builders in `cio.algorithms`.

Hooks exposed:
    Initialiser        - generates initial permutations
    Crossover          - combines two parent permutations
    Mutation           - perturbs a single permutation
    LocalSearch        - improves a solution via neighbourhood moves
    ConstraintHandler  - currently a no-op (decoder enforces feasibility);
                         students may design alternative decoders or repair ops

Note: pymoo has its own Operator class hierarchy. The wrappers below convert
our hook classes into pymoo operators so students can pass our high-level
hooks and have them work with pymoo's NSGA-II / MOEA/D.

For students:
    To plug in your own crossover operator, write a class with a __call__
    method matching the signature in `BaseCrossover`. Then pass it via the
    `crossover=` argument to the algorithm builder.
"""
from __future__ import annotations

import numpy as np
from typing import Protocol

from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling


# ---------------------------------------------------------------------------
# Initialiser (a.k.a. sampling)
# ---------------------------------------------------------------------------

class BaseInitialiser(Protocol):
    def sample(self, n_var: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        ...


class RandomPermutationInitialiser:
    """Baseline initialiser: uniform random permutations."""

    def sample(self, n_var: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        out = np.empty((n_samples, n_var), dtype=int)
        for i in range(n_samples):
            out[i] = rng.permutation(n_var)
        return out


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

class BaseCrossover(Protocol):
    def __call__(self, parent_a: np.ndarray, parent_b: np.ndarray,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        ...


class PMXCrossover:
    """Partially-mapped crossover (PMX). Standard permutation crossover."""

    def __call__(self, parent_a, parent_b, rng):
        n = len(parent_a)
        if n < 2:
            return parent_a.copy(), parent_b.copy()
        cx_pts = sorted(rng.choice(n, size=2, replace=False))
        a, b = cx_pts
        child1 = self._pmx_one(parent_a, parent_b, a, b)
        child2 = self._pmx_one(parent_b, parent_a, a, b)
        return child1, child2

    @staticmethod
    def _pmx_one(p1, p2, a, b):
        n = len(p1)
        child = np.full(n, -1, dtype=int)
        child[a:b + 1] = p1[a:b + 1]
        mapping = {int(p1[i]): int(p2[i]) for i in range(a, b + 1)}
        for i in list(range(0, a)) + list(range(b + 1, n)):
            val = int(p2[i])
            while val in child[a:b + 1]:
                val = mapping.get(val, val)
                # Safety: prevent infinite loop
                if val == -1:
                    break
            child[i] = val
        # Fill any remaining -1 (shouldn't happen for valid PMX)
        missing = [v for v in range(n) if v not in child]
        for i in range(n):
            if child[i] == -1 and missing:
                child[i] = missing.pop(0)
        return child

class GraphGuidedPMXCrossover:
    """
    Resource-conflict-guided PMX crossover.

    Standard PMX chooses two crossover points completely at random.

    This version uses the Resource Conflict Graph to choose one
    conflict-heavy surgery as an anchor, then creates a bounded
    crossover region around that surgery.

    The PMX repair mechanism itself is unchanged, preserving
    valid permutation offspring.
    """

    def __init__(
        self,
        instance,
        max_window: int = 6,
    ):
        self.instance = instance
        self.max_window = max_window

        # Compute once because the instance does not change
        self.conflict_scores = self._build_conflict_scores()


    def __call__(
        self,
        parent_a,
        parent_b,
        rng,
    ):
        n = len(parent_a)

        if n < 2:
            return (
                parent_a.copy(),
                parent_b.copy(),
            )

        # ----------------------------------------------------
        # 1. Select graph-informed anchor surgery
        # ----------------------------------------------------

        surgery_indices = np.asarray(
            parent_a,
            dtype=int
        )

        scores = np.asarray(
            [
                self.conflict_scores[int(idx)]
                for idx in surgery_indices
            ],
            dtype=float
        )

        # Keep all surgeries selectable while favouring
        # conflict-heavy surgeries.
        weights = scores + 1.0
        probabilities = weights / weights.sum()

        anchor_surgery = int(
            rng.choice(
                surgery_indices,
                p=probabilities
            )
        )


        # ----------------------------------------------------
        # 2. Find anchor position in parent A
        # ----------------------------------------------------

        anchor_pos = int(
            np.where(
                parent_a == anchor_surgery
            )[0][0]
        )


        # ----------------------------------------------------
        # 3. Build a bounded crossover region
        # ----------------------------------------------------

        window = min(
            self.max_window,
            n
        )

        half_window = max(
            1,
            window // 2
        )

        left = max(
            0,
            anchor_pos - half_window
        )

        right = min(
            n - 1,
            anchor_pos + half_window
        )

        # Guarantee at least two positions
        if left == right:

            if right < n - 1:
                right += 1
            else:
                left -= 1


        # ----------------------------------------------------
        # 4. Standard PMX using graph-guided region
        # ----------------------------------------------------

        child1 = PMXCrossover._pmx_one(
            parent_a,
            parent_b,
            left,
            right
        )

        child2 = PMXCrossover._pmx_one(
            parent_b,
            parent_a,
            left,
            right
        )

        return child1, child2


    def _build_conflict_scores(self):
        """
        Compute weighted Resource Conflict Graph degree.
        """

        instance = self.instance

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

                # Shared operating theatres
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

                # Both impose substantial ward demand
                if (
                    surgery_i.ward_days >= 4
                    and surgery_j.ward_days >= 4
                ):
                    conflict += 1.0

                if conflict > 0:

                    scores[i] += conflict
                    scores[j] += conflict

        return scores


class OrderCrossover:
    """Order crossover (OX). Often competitive with PMX on scheduling."""

    def __call__(self, parent_a, parent_b, rng):
        n = len(parent_a)
        if n < 2:
            return parent_a.copy(), parent_b.copy()
        cx_pts = sorted(rng.choice(n, size=2, replace=False))
        a, b = cx_pts
        child1 = self._ox_one(parent_a, parent_b, a, b)
        child2 = self._ox_one(parent_b, parent_a, a, b)
        return child1, child2

    @staticmethod
    def _ox_one(p1, p2, a, b):
        n = len(p1)
        child = np.full(n, -1, dtype=int)
        child[a:b + 1] = p1[a:b + 1]
        used = set(int(v) for v in p1[a:b + 1])
        remaining = [int(v) for v in p2 if int(v) not in used]
        idx = 0
        for i in list(range(b + 1, n)) + list(range(0, a)):
            child[i] = remaining[idx]
            idx += 1
        return child


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

class BaseMutation(Protocol):
    def __call__(self, perm: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        ...


class SwapMutation:
    """Swap two randomly chosen positions."""

    def __init__(self, p_per_individual: float = 1.0):
        self.p = p_per_individual

    def __call__(self, perm, rng):
        if rng.random() > self.p:
            return perm.copy()
        n = len(perm)
        if n < 2:
            return perm.copy()
        i, j = rng.choice(n, size=2, replace=False)
        out = perm.copy()
        out[i], out[j] = out[j], out[i]
        return out

class GraphGuidedMutation:
    """
    Resource-conflict-guided mutation for OT scheduling.

    Instead of swapping two completely random surgeries, this operator:

    1. Identifies surgeries with high resource-conflict scores.
    2. Selects one highly conflicted surgery.
    3. Swaps it with another nearby surgery in the permutation.

    The intention is to make mutation more problem-aware while
    preserving the permutation representation.
    """

    def __init__(
        self,
        instance,
        p_per_individual: float = 1.0,
        neighbourhood_size: int = 4,
        top_fraction: float = 0.25,
    ):
        self.instance = instance
        self.p = p_per_individual
        self.neighbourhood_size = neighbourhood_size
        self.top_fraction = top_fraction

        # Build conflict scores once.
        # They do not need to be recalculated for every mutation.
        self.conflict_scores = self._build_conflict_scores()


    def __call__(self, perm, rng):

        # Same probability behaviour as lecturer's SwapMutation
        if rng.random() > self.p:
            return perm.copy()

        n = len(perm)

        if n < 2:
            return perm.copy()


        # ----------------------------------------------------
        # 1. Select surgery probabilistically using resource-conflict scores
        # ----------------------------------------------------
        surgery_indices = np.asarray(
            perm,
            dtype=int
        )

        scores = np.asarray(
            [self.conflict_scores[int(idx)] for idx in surgery_indices],
            dtype=float
        )

        # Small positive baseline ensures that every surgery
        # retains some probability of being selected.
        weights = scores + 1.0

        probabilities = (
            weights / weights.sum()
        )

        selected_surgery = int(
            rng.choice(
                surgery_indices,
                p=probabilities
            )
        )

        # Find its current permutation position
        position = int(
            np.where(
                perm == selected_surgery
            )[0][0]
        )


        # ----------------------------------------------------
        # 3. Choose nearby swap position
        # ----------------------------------------------------

        left = max(
            0,
            position - self.neighbourhood_size
        )

        right = min(
            n,
            position + self.neighbourhood_size + 1
        )

        possible_positions = [
            p
            for p in range(left, right)
            if p != position
        ]


        # Defensive fallback
        if not possible_positions:
            return perm.copy()


        swap_position = int(
            rng.choice(possible_positions)
        )


        # ----------------------------------------------------
        # 4. Perform mutation
        # ----------------------------------------------------

        out = perm.copy()

        out[position], out[swap_position] = (
            out[swap_position],
            out[position],
        )

        return out


    def _build_conflict_scores(self):
        """
        Build an approximate Resource Conflict Graph degree.

        Larger score = surgery competes more strongly with other
        surgeries for hospital resources.
        """

        instance = self.instance

        scores = {
            idx: 0.0
            for idx in range(instance.n_surgeries)
        }


        for i in range(instance.n_surgeries):

            surgery_i = instance.surgery_by_index(i)

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

                surgery_j = instance.surgery_by_index(j)

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


                # Shared operating theatres
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


                # Both place substantial demand on ward beds
                if (
                    surgery_i.ward_days >= 4
                    and surgery_j.ward_days >= 4
                ):
                    conflict += 1.0


                if conflict > 0:

                    scores[i] += conflict
                    scores[j] += conflict


        return scores
    
class InsertionMutation:
    """Pick a random element and re-insert it at a random position."""

    def __init__(self, p_per_individual: float = 1.0):
        self.p = p_per_individual

    def __call__(self, perm, rng):
        if rng.random() > self.p:
            return perm.copy()
        n = len(perm)
        if n < 2:
            return perm.copy()
        i, j = rng.choice(n, size=2, replace=False)
        out = perm.copy().tolist()
        val = out.pop(int(i))
        out.insert(int(j), val)
        return np.asarray(out, dtype=int)


class InversionMutation:
    """Reverse a random sub-sequence."""

    def __init__(self, p_per_individual: float = 1.0):
        self.p = p_per_individual

    def __call__(self, perm, rng):
        if rng.random() > self.p:
            return perm.copy()
        n = len(perm)
        if n < 2:
            return perm.copy()
        a, b = sorted(rng.choice(n, size=2, replace=False))
        out = perm.copy()
        out[a:b + 1] = out[a:b + 1][::-1]
        return out


# ---------------------------------------------------------------------------
# pymoo wrappers
# ---------------------------------------------------------------------------

class _SamplingWrapper(Sampling):
    def __init__(self, initialiser: BaseInitialiser, seed: int | None = None):
        super().__init__()
        self.initialiser = initialiser
        self.rng = np.random.default_rng(seed)

    def _do(self, problem, n_samples, **kwargs):
        return self.initialiser.sample(problem.n_var, n_samples, self.rng)


class _CrossoverWrapper(Crossover):
    def __init__(self, crossover_op: BaseCrossover, prob: float = 0.9, seed: int | None = None):
        super().__init__(n_parents=2, n_offsprings=2, prob=prob)
        self.crossover_op = crossover_op
        self.rng = np.random.default_rng(seed)

    def _do(self, problem, X, **kwargs):
        # X has shape (n_parents=2, n_matings, n_var)
        _, n_matings, n_var = X.shape
        out = np.empty((self.n_offsprings, n_matings, n_var), dtype=int)
        for k in range(n_matings):
            c1, c2 = self.crossover_op(X[0, k], X[1, k], self.rng)
            out[0, k] = c1
            out[1, k] = c2
        return out


class _MutationWrapper(Mutation):
    def __init__(self, mutation_op: BaseMutation, prob: float = 1.0, seed: int | None = None):
        super().__init__(prob=prob)
        self.mutation_op = mutation_op
        self.rng = np.random.default_rng(seed)

    def _do(self, problem, X, **kwargs):
        out = np.empty_like(X)
        for i in range(X.shape[0]):
            out[i] = self.mutation_op(X[i], self.rng)
        return out


def to_pymoo_sampling(initialiser: BaseInitialiser, seed: int | None = None) -> Sampling:
    return _SamplingWrapper(initialiser, seed=seed)


def to_pymoo_crossover(crossover_op: BaseCrossover, prob: float = 0.9, seed: int | None = None) -> Crossover:
    return _CrossoverWrapper(crossover_op, prob=prob, seed=seed)


def to_pymoo_mutation(mutation_op: BaseMutation, prob: float = 1.0, seed: int | None = None) -> Mutation:
    return _MutationWrapper(mutation_op, prob=prob, seed=seed)
