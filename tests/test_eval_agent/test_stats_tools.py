"""Tests for the M2 statistical-tool layer (Plan A, 2026-06-05).

Covers:
  - build_stats_input: per-case signals / labels / scalars / strategy groups
  - each catalog tool on synthetic data (signal_label_assoc, bootstrap_diff,
    mcnemar_evalue, friedman_nemenyi, single_rate_evalue, rank_corr)
  - default_plan shape from the data
  - fdr_correct (e-BH across e-values)
  - StatsAnalysisAgent orchestration + backward compat (no data → threshold only)
"""

from __future__ import annotations

from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Label
from evalvitals.core.result import Result
from evalvitals.eval_agent import StatsAnalysisAgent, build_stats_input, fdr_correct
from evalvitals.eval_agent.stages.stats_tools import (
    StatsInput,
    default_plan,
    run_stats_tool,
)

# ── fixtures ────────────────────────────────────────────────────────────────


def _labeled_cases() -> CaseBatch:
    """4 FAIL + 4 PASS cases with stable ids c0..c7."""
    cases = []
    for i in range(4):
        cases.append(FailureCase(id=f"c{i}", inputs=Inputs(prompt=f"p{i}"), label=Label.FAIL))
    for i in range(4, 8):
        cases.append(FailureCase(id=f"c{i}", inputs=Inputs(prompt=f"p{i}"), label=Label.PASS))
    return CaseBatch(cases)


def _attention_result() -> Result:
    """Per-case binary signal perfectly correlated with FAIL, plus a scalar."""
    per_case = [{"sample_id": f"c{i}", "low_img_attn": 1 if i < 4 else 0} for i in range(8)]
    return Result(
        analyzer="attention",
        model="fake",
        findings={"image_token_attention_ratio": 0.03, "per_case": per_case},
    )


def _strategy_result(n_strategies: int) -> Result:
    """findings['by_strategy'] = {name: {case_id: success}} over 6 shared cases."""
    ids = [f"c{i}" for i in range(6)]
    by_strategy = {}
    for s in range(n_strategies):
        # strategy 0 best (mostly success), later strategies progressively worse
        by_strategy[f"strat{s}"] = {cid: int((i + s) % (s + 2) != 0) for i, cid in enumerate(ids)}
    return Result(analyzer="ab", model="fake", findings={"by_strategy": by_strategy})


# ── build_stats_input ───────────────────────────────────────────────────────


def test_build_stats_input_extracts_labels_signals_scalars():
    inp = build_stats_input({"attention": _attention_result()}, _labeled_cases())
    assert inp.labels["c0"] is True and inp.labels["c4"] is False
    assert len(inp.labels) == 8
    assert "attention.low_img_attn" in inp.per_case
    assert inp.per_case["attention.low_img_attn"]["c0"] == 1.0
    assert inp.scalars["attention.image_token_attention_ratio"] == 0.03
    assert inp.groups is None


def test_build_stats_input_no_data_is_empty_labels():
    inp = build_stats_input({"attention": _attention_result()}, None)
    assert inp.labels == {}
    assert "attention.low_img_attn" in inp.per_case  # signals still harvested


def test_build_stats_input_strategy_groups():
    inp = build_stats_input({"ab": _strategy_result(3)}, _labeled_cases())
    assert inp.groups is not None
    assert set(inp.groups) == {"strat0", "strat1", "strat2"}
    assert len(inp.groups["strat0"]) == 6


# ── individual tools ────────────────────────────────────────────────────────


def test_signal_label_assoc_detects_perfect_correlation():
    inp = build_stats_input({"attention": _attention_result()}, _labeled_cases())
    r = run_stats_tool("signal_label_assoc", inp, {"signal": "attention.low_img_attn"})
    assert r.ok
    assert r.effect == 1.0  # signal group fails 100%, control 0%
    assert r.details["fail_rate_signal"] == 1.0
    assert r.details["fail_rate_control"] == 0.0


def test_signal_label_assoc_sparse_binary_missing_is_control():
    # Sparse flag: only the 4 FAIL cases carry a per_case entry; the 4 PASS
    # cases are absent → must be treated as signal-absent control, not skipped.
    per_case = [{"sample_id": f"c{i}", "flag": 1} for i in range(4)]
    res = Result(analyzer="agent", model="fake", findings={"per_case": per_case})
    inp = build_stats_input({"agent": res}, _labeled_cases())
    r = run_stats_tool("signal_label_assoc", inp, {"signal": "agent.flag"})
    assert r.ok and r.effect == 1.0
    assert r.details["n_signal"] == 4 and r.details["n_control"] == 4


def test_signal_label_assoc_degenerate_split_is_skipped():
    # signal absent for everyone → one group empty → not ok
    res = Result(analyzer="attention", model="fake", findings={
        "per_case": [{"sample_id": f"c{i}", "low_img_attn": 0} for i in range(8)],
    })
    inp = build_stats_input({"attention": res}, _labeled_cases())
    r = run_stats_tool("signal_label_assoc", inp, {"signal": "attention.low_img_attn"})
    assert not r.ok and "empty" in (r.error or "")


def test_single_rate_evalue_runs():
    inp = build_stats_input({"attention": _attention_result()}, _labeled_cases())
    r = run_stats_tool("single_rate_evalue", inp, {"p0": 0.5})
    assert r.ok and r.e_value is not None
    assert r.details["fails"] == 4 and r.details["n"] == 8


def test_single_rate_evalue_default_p0_is_descriptive_only():
    """Without an explicit p0, the tool must NOT report a comparable effect or a
    reject — its rate−0.5 would otherwise pollute |effect| ranking and surface a
    meaningless "vs p0=0.50 → reject" verdict on an enriched batch (defect 6)."""
    inp = build_stats_input({"attention": _attention_result()}, _labeled_cases())
    r = run_stats_tool("single_rate_evalue", inp, {})  # no p0 -> unjustified default
    assert r.ok
    assert r.effect is None and r.reject is False
    assert r.details["p0_justified"] is False
    assert "descriptive only" in r.summary


def test_single_rate_evalue_explicit_p0_is_interpretable():
    """A justified p0 (the natural base rate) restores effect + reject."""
    inp = build_stats_input({"attention": _attention_result()}, _labeled_cases())
    r = run_stats_tool("single_rate_evalue", inp, {"p0": 0.05})
    assert r.ok and r.effect is not None
    assert r.details["p0_justified"] is True


def test_mcnemar_evalue_two_strategies():
    inp = build_stats_input({"ab": _strategy_result(2)}, _labeled_cases())
    r = run_stats_tool("mcnemar_evalue", inp, {})
    assert r.ok and r.e_value is not None
    assert r.effect is not None and r.ci is not None


def test_friedman_nemenyi_three_strategies():
    inp = build_stats_input({"ab": _strategy_result(3)}, _labeled_cases())
    r = run_stats_tool("friedman_nemenyi", inp, {})
    assert r.ok and r.p_value is not None
    assert "avg_ranks" in r.details and len(r.details["avg_ranks"]) == 3


def test_friedman_requires_three_groups():
    inp = build_stats_input({"ab": _strategy_result(2)}, _labeled_cases())
    r = run_stats_tool("friedman_nemenyi", inp, {})
    assert not r.ok and ">=3" in (r.error or "")


def test_rank_corr_on_continuous_signal():
    # continuous signal increasing with fail-ness
    per_case = [{"sample_id": f"c{i}", "entropy": float(8 - i)} for i in range(8)]
    res = Result(analyzer="token_entropy", model="fake", findings={"per_case": per_case})
    inp = build_stats_input({"token_entropy": res}, _labeled_cases())
    r = run_stats_tool("rank_corr", inp, {"signal": "token_entropy.entropy"})
    assert r.ok and r.effect is not None  # FAIL cases (c0..3) have highest entropy → positive tau
    assert r.effect > 0


# ── planner + FDR ───────────────────────────────────────────────────────────


def test_default_plan_picks_relevant_tools():
    inp = build_stats_input({"attention": _attention_result()}, _labeled_cases())
    plan = default_plan(inp)
    tools = {t for t, _, _ in plan}
    assert "signal_label_assoc" in tools
    assert "single_rate_evalue" in tools


def test_default_plan_empty_without_labels_or_groups():
    inp = StatsInput()  # nothing testable
    assert default_plan(inp) == []


def test_fdr_correct_with_evalues():
    inp = build_stats_input({"ab": _strategy_result(2)}, _labeled_cases())
    r = run_stats_tool("mcnemar_evalue", inp, {})
    out = fdr_correct([r])
    assert out["method"] == "e-BH"
    assert out["n_tested"] == 1


def test_llm_narrowing_never_drops_paired_tools():
    """Judge narrowing by tool name must retain mcnemar/friedman — they exist
    only when an intervention produced strategy groups and carry the causal
    verdicts (regression: narrowing silently dropped the paired contrasts)."""
    from evalvitals.eval_agent import StatsAnalysisAgent
    from evalvitals.eval_agent.stages.protocol import ExperimentProtocol

    class NarrowJudge:
        def generate(self, prompt, **kw):
            return '{"tools": ["signal_label_assoc"], "rationale": "x"}'

    # labels + per-case signal + 3 strategy groups → plan has assoc + friedman + mcnemar
    res = {
        "a": Result(analyzer="a", model="m", findings={
            "per_case": [{"sample_id": f"c{i}", "flag": i % 2} for i in range(6)],
            "by_strategy": {
                s: {f"c{i}": float((i + j) % 2) for i in range(6)}
                for j, s in enumerate(["baseline", "v1", "v2"])
            },
        }),
    }
    agent = StatsAnalysisAgent(judge=NarrowJudge())
    rep = agent.analyze(res, model_name="m", data=_labeled_cases(),
                        protocol=ExperimentProtocol(description="d"))
    tools = [p["tool"] for p in rep.stats_plan]
    assert "friedman_nemenyi" in tools
    assert "mcnemar_evalue" in tools


def test_fdr_correct_no_evalues():
    inp = build_stats_input({"attention": _attention_result()}, _labeled_cases())
    r = run_stats_tool("signal_label_assoc", inp, {"signal": "attention.low_img_attn"})
    out = fdr_correct([r])  # unpaired test → no e-value
    assert out["n_tested"] == 0 and out["rejected_tools"] == []


# ── agent orchestration + backward compat ───────────────────────────────────


def test_agent_runs_tool_layer_with_data():
    agent = StatsAnalysisAgent()  # no judge → deterministic plan
    rep = agent.analyze({"attention": _attention_result()}, model_name="fake", data=_labeled_cases())
    assert rep.stats_tool == "selected_tools"
    assert rep.stats_results
    assert any(r.tool == "signal_label_assoc" and r.ok for r in rep.stats_results)
    assert rep.stats_plan
    # evidence chain mentions the stats verdict
    assert any("stats:" in step for step in rep.evidence_chain)


def test_agent_backward_compat_without_data():
    agent = StatsAnalysisAgent()
    rep = agent.analyze({"attention": _attention_result()}, model_name="fake")
    assert rep.stats_tool == "threshold_rules"
    assert rep.stats_results == []
    # still a valid AnalysisReport-compatible object
    assert rep.model_name == "fake"


def test_agent_backward_compat_unlabeled_data():
    cases = CaseBatch([FailureCase(id="u0", inputs=Inputs(prompt="x"), label=Label.UNKNOWN)])
    agent = StatsAnalysisAgent()
    rep = agent.analyze({"attention": _attention_result()}, model_name="fake", data=cases)
    # UNKNOWN labels are dropped → no testable data → threshold only
    assert rep.stats_tool == "threshold_rules"
    assert rep.stats_results == []
