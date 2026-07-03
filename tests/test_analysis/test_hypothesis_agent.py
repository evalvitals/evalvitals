from __future__ import annotations

from evalvitals.analysis.hypothesis_agent import Hypothesis, HypothesisAgent, _parse_hypotheses


class ScriptedJudge:
    def __init__(self, response: str) -> None:
        self._response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        return self._response


_REPORT = {
    "question": "What predicts yield?",
    "takeaways": [{
        "title": "Higher temperature batches yield more (70.1% vs 88.4%).",
        "analysis": "Mean yield rises with temperature across bins.",
    }],
    "observations": ["30 batches, no missing values."],
    "candidate_signals": [{"name": "temperature", "display_name": "Temperature", "rationale": "r=0.86"}],
}

_GOOD_RESPONSE = """\
HYPOTHESIS: Higher temperature accelerates the reaction, raising yield.
BASIS: Higher temperature batches yield more (70.1% vs 88.4%).
TEST: Run a controlled temperature-ramp experiment holding pressure fixed.

HYPOTHESIS: Catalyst B underperforms due to a side reaction at high pressure.
BASIS: Temperature signal only
TEST: Compare catalyst B vs C yield holding temperature fixed."""


def test_parse_hypotheses_splits_multiple_entries():
    out = _parse_hypotheses(_GOOD_RESPONSE)
    assert len(out) == 2
    assert out[0].statement == "Higher temperature accelerates the reaction, raising yield."
    assert out[0].basis == "Higher temperature batches yield more (70.1% vs 88.4%)."
    assert out[0].test_design == "Run a controlled temperature-ramp experiment holding pressure fixed."
    assert out[1].statement.startswith("Catalyst B underperforms")


def test_parse_hypotheses_no_hypothesis_marker_yields_empty():
    assert _parse_hypotheses("NO_HYPOTHESIS") == []
    assert _parse_hypotheses("") == []


def test_parse_hypotheses_dedupes_restated_hypothesis_in_a_cli_trajectory():
    """CliAgentResult.raw_output for the CLI-agent backend is the full
    rendered tool-call trajectory, not just a final answer — an agent that
    narrates a plan before its final answer can restate the same hypothesis
    twice. This must not double-count it (observed on a real claude_code run)."""
    trajectory = (
        "[assistant] Let me think through this...\n"
        + _GOOD_RESPONSE
        + "\n\n[assistant] Here is my final answer.\n"
        + _GOOD_RESPONSE
    )

    out = _parse_hypotheses(trajectory)

    assert len(out) == 2
    statements = [h.statement for h in out]
    assert len(statements) == len(set(statements))
    assert out[0].statement == "Higher temperature accelerates the reaction, raising yield."


def test_propose_uses_judge_and_returns_parsed_hypotheses():
    judge = ScriptedJudge(_GOOD_RESPONSE)
    agent = HypothesisAgent(judge=judge)

    out = agent.propose(_REPORT)

    assert len(out) == 2
    assert all(isinstance(h, Hypothesis) for h in out)
    # the prompt actually carries the takeaway/observation/signal content
    assert "Higher temperature batches yield more" in judge.prompts[0]
    assert "temperature" in judge.prompts[0].lower()


def test_propose_returns_empty_without_a_configured_backend():
    agent = HypothesisAgent()  # no judge, no cli_config
    assert agent.available is False
    assert agent.propose(_REPORT) == []


def test_propose_returns_empty_when_report_has_nothing_to_reason_over():
    judge = ScriptedJudge(_GOOD_RESPONSE)
    agent = HypothesisAgent(judge=judge)

    out = agent.propose({"question": "q"})

    assert out == []
    assert judge.prompts == []  # never even called the backend


def test_propose_never_raises_on_backend_failure():
    class BrokenJudge:
        def generate(self, prompt: str) -> str:
            raise RuntimeError("boom")

    agent = HypothesisAgent(judge=BrokenJudge())
    assert agent.propose(_REPORT) == []
