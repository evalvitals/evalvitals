"""Agent-trace data model: Step / Trajectory / FailureCase.trajectory."""

from __future__ import annotations

from evalvitals.core.case import (
    FailureCase,
    Inputs,
    Label,
    Step,
    StepRole,
    Trajectory,
)


def test_trajectory_from_records():
    records = [
        {"role": "user", "content": "book a flight"},
        {"role": "actor", "tool_call": {"name": "search", "args": {"q": "flights"}}},
        {"role": "tool", "observation": "no results"},
        {"role": "actor", "content": "sorry, none found"},
    ]
    traj = Trajectory.from_records(
        records, sample_id="s1", goal="book a flight", outcome=Label.FAIL
    )
    assert len(traj) == 4
    assert traj.steps[0].role is StepRole.USER
    assert traj.steps[1].tool_call["name"] == "search"
    assert traj.steps[1].idx == 1
    assert traj.outcome is Label.FAIL


def test_step_annotation_fields_default_none():
    s = Step(idx=0)
    assert s.is_first_error is None
    assert s.failure_mode is None
    # analyzers write these:
    s.is_first_error = True
    s.failure_mode = "FM-2.4"
    assert s.is_first_error and s.failure_mode == "FM-2.4"


def test_failurecase_carries_trajectory():
    traj = Trajectory.from_records([{"role": "actor", "content": "hi"}], sample_id="s2")
    case = FailureCase(inputs=Inputs(prompt="goal"), trajectory=traj, label=Label.FAIL)
    assert case.trajectory is traj
    assert len(case.trajectory) == 1


def test_unit_case_has_no_trajectory():
    case = FailureCase.from_prompt("just a prompt")
    assert case.trajectory is None
