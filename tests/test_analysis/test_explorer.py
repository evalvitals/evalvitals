from __future__ import annotations

from evalvitals.analysis import M2ExplorerAgent
from evalvitals.eval_agent.sandbox import ExperimentSandbox

_GOOD_CODE = """
import json
from pathlib import Path

rows = json.loads(Path("records.json").read_text())
fails = [r for r in rows if r.get("label") == "fail"]
passes = [r for r in rows if r.get("label") == "pass"]
payload = {
    "observations": [f"{len(fails)} fail rows and {len(passes)} pass rows"],
    "candidate_signals": [
        {
            "name": "flag",
            "rationale": "flag is concentrated in fail rows",
            "suggested_test": "signal_label_assoc",
        }
    ],
    "plots": [],
    "tables": {"counts": {"fail": len(fails), "pass": len(passes)}},
    "caveats": ["exploratory only"],
    "recommended_confirmatory_tests": ["Run StatsAnalysisAgent.analyze_records on flag"],
}
print("EXPLORATORY_RESULT_JSON=" + json.dumps(payload))
"""


class ScriptedJudge:
    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        if self._responses:
            return self._responses.pop(0)
        return _GOOD_CODE


def _rows() -> list[dict]:
    return [
        {"case_id": f"c{i}", "label": "fail" if i < 3 else "pass", "flag": int(i < 3)}
        for i in range(6)
    ]


def test_m2_explorer_runs_generated_local_analysis(tmp_path):
    agent = M2ExplorerAgent(
        judge=ScriptedJudge(f"```python\n{_GOOD_CODE}\n```"),
        sandbox=ExperimentSandbox(workdir=tmp_path, cleanup=False),
    )

    report = agent.explore_records(_rows(), question="Find failure patterns.")

    assert report.ok
    assert report.attempts == 1
    assert report.candidate_signal_names == ["flag"]
    assert report.tables["counts"] == {"fail": 3, "pass": 3}
    assert "exploratory only" in report.caveats
    assert (tmp_path / "records.json").exists()


def test_m2_explorer_uses_inspector_for_repair(tmp_path):
    bad_code = "raise RuntimeError('broken')"
    judge = ScriptedJudge(f"```python\n{bad_code}\n```")
    inspector = ScriptedJudge(f"```python\n{_GOOD_CODE}\n```")
    agent = M2ExplorerAgent(
        judge=judge,
        inspector=inspector,
        sandbox=ExperimentSandbox(workdir=tmp_path, cleanup=False),
        max_attempts=2,
    )

    report = agent.explore_records(_rows())

    assert report.ok
    assert report.attempts == 2
    assert len(judge.prompts) == 1
    assert len(inspector.prompts) == 1
    assert "Previous code" in inspector.prompts[0]


def test_m2_explorer_reports_missing_backend(tmp_path):
    agent = M2ExplorerAgent(sandbox=ExperimentSandbox(workdir=tmp_path, cleanup=False))

    report = agent.explore_records(_rows())

    assert not report.ok
    assert "no code-writing backend" in report.error
    assert report.data_profile["n_rows"] == 6
