"""Tests for M1 tier (b) ProbeGenerator (sandbox black-box probe generation).

A scripted "LLM" returns a real, runnable probe script, so the full
collect-outputs -> generate -> sandbox-run -> parse-into-Result path is
exercised deterministically (no network, no real coding agent).
"""

from __future__ import annotations

from evalvitals.core.capability import Capability
from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label
from evalvitals.eval_agent import ProbeAgent, ProbeGenerator
from tests.conftest import FakeModel

# A valid probe: flags cases whose output contains "sorry" (a refusal probe).
_GOOD_PROBE = '''
import json
data = json.load(open("m1_probe_input.json"))
per_case, n_ref = [], 0
for c in data["cases"]:
    refused = int("sorry" in c["output"].lower())
    n_ref += refused
    per_case.append({"sample_id": c["id"], "refused": refused})
print('PROBE_RESULT_JSON=' + json.dumps({
    "findings": {"refusal_rate": n_ref / max(len(data["cases"]), 1)},
    "per_case": per_case,
}))
'''

_NO_MARKER_PROBE = 'print("no contract line here")'


class ScriptedJudge:
    def __init__(self, code: str) -> None:
        self._code = code

    def generate(self, prompt: str, **kw) -> str:
        return f"```python\n{self._code}\n```"


class RefusingModel(FakeModel):
    """Outputs a refusal for prompts containing 'danger', else a normal answer."""

    def generate(self, inputs, **kwargs) -> str:
        prompt = str(getattr(inputs, "prompt", "")).lower()
        return "Sorry, I cannot help." if "danger" in prompt else "The answer is 3."


def _cases() -> CaseBatch:
    return CaseBatch([
        FailureCase(id="c0", inputs=Inputs(prompt="danger one"), label=Label.FAIL),
        FailureCase(id="c1", inputs=Inputs(prompt="danger two"), label=Label.FAIL),
        FailureCase(id="c2", inputs=Inputs(prompt="benign q"), label=Label.PASS),
        FailureCase(id="c3", inputs=Inputs(prompt="benign r"), label=Label.PASS),
    ])


# ── generator unit tests ────────────────────────────────────────────────────


def test_generate_runs_and_returns_result():
    gen = ProbeGenerator(judge=ScriptedJudge(_GOOD_PROBE))
    result, probe = gen.generate("detect refusals", RefusingModel(), _cases(), name="refusal")
    assert result is not None
    assert result.analyzer == "generated:refusal"
    assert result.findings["refusal_rate"] == 0.5  # 2 of 4 refused
    per_case = {e["sample_id"]: e["refused"] for e in result.findings["per_case"]}
    assert per_case == {"c0": 1, "c1": 1, "c2": 0, "c3": 0}
    assert probe is not None and probe.code


def test_run_cached_reuses_without_llm():
    gen = ProbeGenerator(judge=ScriptedJudge(_GOOD_PROBE))
    _, probe = gen.generate("first", RefusingModel(), _cases(), name="refusal")
    assert probe is not None
    again = gen.run_cached(probe, RefusingModel(), _cases())
    assert again is not None and again.findings["refusal_rate"] == 0.5


def test_missing_marker_returns_none():
    gen = ProbeGenerator(judge=ScriptedJudge(_NO_MARKER_PROBE))
    result, probe = gen.generate("bad", RefusingModel(), _cases(), name="bad")
    assert result is None and probe is None


def test_generator_unavailable_without_backend():
    gen = ProbeGenerator()
    assert gen.available is False
    result, probe = gen.generate("x", RefusingModel(), _cases())
    assert result is None and probe is None


# ── ProbeAgent integration ──────────────────────────────────────────────────


def test_probe_agent_generates_when_catalog_empty():
    # max_analyzers=0 → no catalog analyzers run → codegen must fire.
    gen = ProbeGenerator(judge=ScriptedJudge(_GOOD_PROBE))
    agent = ProbeAgent(max_analyzers=0, allow_codegen=True, probe_generator=gen)
    model = RefusingModel(capabilities={Capability.GENERATE})
    results = agent.probe(model, _cases())
    assert "generated:probe1" in results
    assert results["generated:probe1"].findings["refusal_rate"] == 0.5
    # Schema records the generated probe.
    assert "generated:probe1" in agent.last_schema.selected_analyzers


def test_probe_agent_reuses_generated_probe_across_calls():
    gen = ProbeGenerator(judge=ScriptedJudge(_GOOD_PROBE))
    agent = ProbeAgent(max_analyzers=0, allow_codegen=True, probe_generator=gen)
    model = RefusingModel(capabilities={Capability.GENERATE})
    agent.probe(model, _cases())
    assert len(agent._generated_probes) == 1
    # Second call reuses the cached probe (no second generation).
    agent.probe(model, _cases())
    assert len(agent._generated_probes) == 1


def test_probe_agent_no_codegen_when_disabled():
    gen = ProbeGenerator(judge=ScriptedJudge(_GOOD_PROBE))
    agent = ProbeAgent(max_analyzers=0, allow_codegen=False, probe_generator=gen)
    model = RefusingModel(capabilities={Capability.GENERATE})
    results = agent.probe(model, _cases())
    assert not any(k.startswith("generated:") for k in results)
