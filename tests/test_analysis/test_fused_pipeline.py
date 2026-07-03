"""Phase B2 — fused explore -> confirm pipeline."""

from __future__ import annotations

from evalvitals.analysis import CandidateSignal, ExploratoryAnalysisReport
from evalvitals.analysis.fused_pipeline import (
    FusedReport,
    _split_records,
    run_fused_analysis,
)


class _FakeExplorer:
    """Stand-in for ExploratoryAnalysisAgent: returns scripted candidates, records what it saw."""

    def __init__(self, candidates, *, observations=(), recommended=(), dashboard_storyboard=()):
        self._candidates = candidates
        self._observations = list(observations)
        self._recommended = list(recommended)
        self._dashboard_storyboard = list(dashboard_storyboard)
        self.seen_ids: list[str] | None = None

    def explore_records(self, rows, *, question=""):
        self.seen_ids = [r["case_id"] for r in rows]
        return ExploratoryAnalysisReport(
            ok=True,
            question=question,
            observations=self._observations,
            dashboard_storyboard=self._dashboard_storyboard,
            candidate_signals=list(self._candidates),
            recommended_confirmatory_tests=self._recommended,
        )


def _dataset(n_each: int = 15) -> list[dict]:
    """Small objects fail; large objects pass — a strong, separable signal."""
    fails = [
        {"case_id": f"f{i}", "label": "fail", "obj_size": 20, "attention": 0.1}
        for i in range(n_each)
    ]
    passes = [
        {"case_id": f"p{i}", "label": "pass", "obj_size": 80, "attention": 0.6}
        for i in range(n_each)
    ]
    return fails + passes


# ---------------------------------------------------------------------------
# Double-blind firewall: discover on EXPLORE, confirm on disjoint CONFIRM
# ---------------------------------------------------------------------------

def test_split_is_disjoint_stratified_and_deterministic():
    rows = _dataset()
    explore, confirm = _split_records(rows, frac=0.3, seed=0, label_col="label")
    assert confirm is not None
    e_ids = {r["case_id"] for r in explore}
    c_ids = {r["case_id"] for r in confirm}
    assert e_ids.isdisjoint(c_ids)
    assert len(e_ids) + len(c_ids) == len(rows)
    # both strata represented in the held-out split
    assert any(r["label"] == "fail" for r in confirm)
    assert any(r["label"] == "pass" for r in confirm)
    # deterministic
    explore2, confirm2 = _split_records(rows, frac=0.3, seed=0, label_col="label")
    assert [r["case_id"] for r in confirm] == [r["case_id"] for r in confirm2]


def test_explorer_only_sees_explore_rows_verdict_computed_on_confirm():
    rows = _dataset()
    explore, confirm = _split_records(rows, frac=0.3, seed=0, label_col="label")
    confirm_ids = {r["case_id"] for r in confirm}

    explorer = _FakeExplorer([
        CandidateSignal(
            name="explored.small_peripheral",
            rationale="small + low-attention objects fail",
            suggested_test="signal_label_assoc",
            recipe={"name": "explored.small_peripheral", "kind": "expr",
                    "expr": "(obj_size < 40) and (attention < 0.3)"},
        )
    ])
    rep = run_fused_analysis(rows, explorer=explorer, confirm_split=0.3, seed=0)

    # the explorer saw EXACTLY the explore rows (never confirm)
    assert set(explorer.seen_ids).isdisjoint(confirm_ids)
    assert set(explorer.seen_ids) == {r["case_id"] for r in explore}
    assert rep.split["mode"] == "held_out"


def test_fused_report_preserves_agent_dashboard_storyboard():
    rows = _dataset()
    explorer = _FakeExplorer(
        [CandidateSignal(name="obj_size")],
        dashboard_storyboard=[{
            "id": "analysis",
            "title": "Analysis",
            "stages": ["M2"],
            "summary": "Agent-generated analysis panel",
            "items": ["Method: compare size"],
            "artifact_refs": ["candidate_signals"],
        }],
    )
    rep = run_fused_analysis(rows, explorer=explorer, confirm_split=0.3, seed=0)
    assert rep.dashboard_storyboard[0]["summary"] == "Agent-generated analysis panel"
    assert rep.to_dict()["dashboard_storyboard"][0]["id"] == "analysis"


# ---------------------------------------------------------------------------
# Bridge: explorer recipe operationalized on CONFIRM and host-confirmed
# ---------------------------------------------------------------------------

def test_bridged_signal_is_confirmed_on_held_out_split():
    rows = _dataset()
    explorer = _FakeExplorer([
        CandidateSignal(
            name="explored.small_peripheral",
            recipe={"name": "explored.small_peripheral", "kind": "expr",
                    "expr": "(obj_size < 40) and (attention < 0.3)"},
        )
    ])
    rep = run_fused_analysis(rows, explorer=explorer, confirm_split=0.3, seed=0)

    sig = next(s for s in rep.candidate_signals if s.name == "explored.small_peripheral")
    assert sig.source == "explorer"
    assert sig.host_adjudicated is True
    assert sig.reject is True                 # strong, separable -> CI excludes 0
    assert sig.confirmed_on == "held_out"
    assert sig.p_value is not None
    assert sig.fdr_corrected is True
    assert sig.correction_method == "BH"
    assert rep.adjudication["families"]["bh"]["n_tested"] >= 1


def test_catalog_and_explorer_sources_are_tagged():
    rows = _dataset()
    explorer = _FakeExplorer([
        CandidateSignal(name="obj_size", rationale="explorer also points at obj_size"),
        CandidateSignal(
            name="explored.small_peripheral",
            recipe={"name": "explored.small_peripheral", "kind": "expr",
                    "expr": "obj_size < 40"},
        ),
    ])
    rep = run_fused_analysis(rows, explorer=explorer, confirm_split=0.3, seed=0)
    by_name = {s.name: s for s in rep.candidate_signals}

    assert by_name["obj_size"].source == "both"          # catalog column + explorer named it
    assert by_name["explored.small_peripheral"].source == "explorer"
    assert "attention" in by_name and by_name["attention"].source == "catalog"


# ---------------------------------------------------------------------------
# Graceful degradation: nothing silently dropped
# ---------------------------------------------------------------------------

def test_undermined_candidates_route_to_recommended_tests():
    rows = _dataset()
    explorer = _FakeExplorer(
        [
            CandidateSignal(name="vibes", rationale="hard to pin down"),  # no recipe
            CandidateSignal(  # malformed recipe -> cannot operationalize
                name="explored.broken",
                rationale="bad expr",
                recipe={"name": "explored.broken", "kind": "expr", "expr": "obj_size <"},
            ),
        ],
        recommended=["run a paired prompt-contrast experiment"],
    )
    rep = run_fused_analysis(rows, explorer=explorer, confirm_split=0.3, seed=0)
    joined = " | ".join(rep.recommended_confirmatory_tests)

    assert "run a paired prompt-contrast experiment" in joined  # explorer's own
    assert "vibes" in joined                                    # descriptive-only
    assert "explored.broken" in joined                          # recipe failed to compile
    # neither leaked into the confirmed candidate set
    names = {s.name for s in rep.candidate_signals}
    assert "vibes" not in names and "explored.broken" not in names


# ---------------------------------------------------------------------------
# Determinism + in-sample fallback
# ---------------------------------------------------------------------------

def test_run_is_deterministic_for_same_seed():
    rows = _dataset()
    cand = [CandidateSignal(
        name="explored.s",
        recipe={"name": "explored.s", "kind": "expr", "expr": "obj_size < 40"},
    )]
    a = run_fused_analysis(rows, explorer=_FakeExplorer(list(cand)), confirm_split=0.3, seed=0)
    b = run_fused_analysis(rows, explorer=_FakeExplorer(list(cand)), confirm_split=0.3, seed=0)
    assert a.to_dict()["candidate_signals"] == b.to_dict()["candidate_signals"]
    assert a.adjudication == b.adjudication


def test_small_batch_falls_back_to_in_sample_with_caveat():
    rows = [
        {"case_id": "a", "label": "fail", "obj_size": 20},
        {"case_id": "b", "label": "pass", "obj_size": 80},
    ]
    rep = run_fused_analysis(
        rows,
        explorer=_FakeExplorer([CandidateSignal(
            name="explored.s",
            recipe={"name": "explored.s", "kind": "expr", "expr": "obj_size < 40"},
        )]),
        confirm_split=0.3,
        seed=0,
    )
    assert rep.split["mode"] == "in_sample"
    assert any("IN-SAMPLE" in c for c in rep.caveats)
    assert isinstance(rep, FusedReport)
