from __future__ import annotations

from typing import Protocol

import numpy as np

from .instance import Instance
from .schedule import Assignment, Schedule


class BaseDecoder(Protocol):
    def __call__(
        self,
        instance: Instance,
        permutation: np.ndarray,
    ) -> Schedule:
        ...


class SmartDecoder:
    """Greedily schedules surgeries using resource-aware placement."""

    def __call__(
        self,
        instance: Instance,
        permutation: np.ndarray,
    ) -> Schedule:
        sched = Schedule()

        for day in range(1, instance.horizon_days + 1):
            sched.icu_occupancy[day] = 0
            sched.ward_occupancy[day] = 0

        for idx in permutation:
            surgery = instance.surgery_by_index(int(idx))
            assignment = self._try_place(instance, surgery, sched)

            if assignment is None:
                sched.unscheduled_surgery_ids.append(surgery.id)
            else:
                self._apply_assignment(instance, assignment, surgery, sched)

        return sched

    def _try_place(
        self,
        instance: Instance,
        surgery,
        sched: Schedule,
    ) -> Assignment | None:
        eligible_ot_ids = instance.eligible_ots_for_surgery.get(surgery.id, [])
        eligible_team_ids = instance.eligible_teams_for_surgery.get(
            surgery.id, []
        )

        if not eligible_ot_ids or not eligible_team_ids:
            return None

        ots_by_id = {ot.id: ot for ot in instance.ots}
        teams_by_id = {team.id: team for team in instance.teams}

        latest_no_penalty_day = min(
            instance.horizon_days,
            surgery.earliest_day + surgery.max_wait_days,
        )

        day_order = list(
            range(surgery.earliest_day, latest_no_penalty_day + 1)
        )
        day_order.extend(
            range(latest_no_penalty_day + 1, instance.horizon_days + 1)
        )

        candidates: list[Assignment] = []

        for day in day_order:
            if not self._post_op_resources_ok(
                instance, surgery, day, sched
            ):
                continue

            for ot_id in eligible_ot_ids:
                ot = ots_by_id.get(ot_id)
                if ot is None:
                    continue

                ot_hours_used = sched.ot_hours_used.get((ot_id, day), 0.0)

                if (
                    ot_hours_used + surgery.duration_hr
                    > ot.daily_hours + 1e-9
                ):
                    continue

                for team_id in eligible_team_ids:
                    team = teams_by_id.get(team_id)
                    if team is None:
                        continue

                    team_hours_used = sched.team_hours_used.get(
                        (team_id, day), 0.0
                    )

                    if (
                        team_hours_used + surgery.duration_hr
                        > team.daily_hours_max + 1e-9
                    ):
                        continue

                    if not self._consecutive_days_ok(team, day, sched):
                        continue

                    candidates.append(
                        Assignment(
                            surgery_id=surgery.id,
                            ot_id=ot_id,
                            day=day,
                            team_id=team_id,
                            start_hour=ot_hours_used,
                            duration_hr=surgery.duration_hr,
                        )
                    )

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda assignment: self._resource_score(
                instance,
                assignment,
                surgery,
                sched,
            ),
        )

    def _resource_score(
        self,
        instance: Instance,
        assignment: Assignment,
        surgery,
        sched: Schedule,
    ) -> tuple:
        """
        Lower is better.
        1. Avoid lateness.
        2. Prefer earlier day.
        3. Preserve scarce OT/team capacity.
        4. Use OT capacity efficiently.
        """

        latest_no_penalty_day = min(
            instance.horizon_days,
            surgery.earliest_day + surgery.max_wait_days,
        )

        delay_penalty = max(
            0,
            assignment.day - latest_no_penalty_day,
        )

        # How much OT capacity remains after this placement?
        ot = next(
            ot for ot in instance.ots
            if ot.id == assignment.ot_id
        )

        current_ot_hours = sched.ot_hours_used.get(
            (assignment.ot_id, assignment.day),
            0.0,
        )

        remaining_ot_hours = (
            ot.daily_hours
            - current_ot_hours
            - assignment.duration_hr
        )

        # How much team capacity remains?
        team = next(
            team for team in instance.teams
            if team.id == assignment.team_id
        )

        current_team_hours = sched.team_hours_used.get(
            (assignment.team_id, assignment.day),
            0.0,
        )

        remaining_team_hours = (
            team.daily_hours_max
            - current_team_hours
            - assignment.duration_hr
        )

        # Prefer tight packing, but only after lateness/day.
        capacity_left = (
            remaining_ot_hours
            + remaining_team_hours
        )

        return (
            float(delay_penalty),
            assignment.day,
            float(capacity_left),
        )

    def _post_op_resources_ok(
        self,
        instance: Instance,
        surgery,
        scheduled_day: int,
        sched: Schedule,
    ) -> bool:
        for day in range(
            scheduled_day,
            scheduled_day + surgery.icu_days,
        ):
            if day > instance.horizon_days:
                break

            occupancy = sched.icu_occupancy.get(day, 0)
            if (
                not instance.overflow_allowed
                and occupancy + 1 > instance.icu_capacity
            ):
                return False

        ward_start = scheduled_day + surgery.icu_days

        for day in range(
            ward_start,
            ward_start + surgery.ward_days,
        ):
            if day > instance.horizon_days:
                break

            occupancy = sched.ward_occupancy.get(day, 0)
            if (
                not instance.overflow_allowed
                and occupancy + 1 > instance.ward_capacity
            ):
                return False

        return True

    def _consecutive_days_ok(
        self,
        team,
        day: int,
        sched: Schedule,
    ) -> bool:
        active_days = set(sched.team_active_days.get(team.id, []))

        # Scheduling another surgery on an already-active day does not add
        # another consecutive workday.
        if day in active_days:
            return True

        active_days.add(day)
        run_length = 1

        previous_day = day - 1
        while previous_day in active_days:
            run_length += 1
            previous_day -= 1

        next_day = day + 1
        while next_day in active_days:
            run_length += 1
            next_day += 1

        return run_length <= team.max_consecutive_days

    def _apply_assignment(
        self,
        instance: Instance,
        assignment: Assignment,
        surgery,
        sched: Schedule,
    ) -> None:
        sched.assignments.append(assignment)

        ot_key = (assignment.ot_id, assignment.day)
        sched.ot_hours_used[ot_key] = (
            sched.ot_hours_used.get(ot_key, 0.0)
            + assignment.duration_hr
        )

        team_key = (assignment.team_id, assignment.day)
        sched.team_hours_used[team_key] = (
            sched.team_hours_used.get(team_key, 0.0)
            + assignment.duration_hr
        )

        active_days = sched.team_active_days.setdefault(
            assignment.team_id, []
        )
        if assignment.day not in active_days:
            active_days.append(assignment.day)

        for day in range(
            assignment.day,
            assignment.day + surgery.icu_days,
        ):
            if day > instance.horizon_days:
                break

            sched.icu_occupancy[day] = (
                sched.icu_occupancy.get(day, 0) + 1
            )

        ward_start = assignment.day + surgery.icu_days

        for day in range(
            ward_start,
            ward_start + surgery.ward_days,
        ):
            if day > instance.horizon_days:
                break

            sched.ward_occupancy[day] = (
                sched.ward_occupancy.get(day, 0) + 1
            )