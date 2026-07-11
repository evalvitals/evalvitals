"""Failure-mode clustering (analysis.failure_modes): the numpy fallback path
runs unconditionally (no sklearn/hdbscan dependency); sklearn-backed paths are
skipped when not installed.
"""

from __future__ import annotations

import sys

import pytest

from evalvitals.analysis.failure_modes import (
    FailureMode,
    FailureModeReport,
    _cluster_cosine_greedy,
    _hash_vectorize,
    cluster_failures,
)


def _two_mode_records(n_per_mode: int = 8) -> list[dict]:
    records = []
    for i in range(n_per_mode):
        records.append({
            "case_id": f"small_{i}", "label": "FAIL",
            "prompt": "small object missed detection too tiny to see",
        })
    for i in range(n_per_mode):
        records.append({
            "case_id": f"occl_{i}", "label": "FAIL",
            "prompt": "occluded object hidden behind another blocking item",
        })
    for i in range(5):
        records.append({
            "case_id": f"pass_{i}", "label": "PASS",
            "prompt": "correct detection clean and clear visible",
        })
    return records


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_zero_fail_cases_returns_empty_report():
    report = cluster_failures([{"case_id": "x", "label": "PASS", "prompt": "ok"}])
    assert report.n_fail_cases == 0
    assert report.clusters == []
    assert report.method == "none"


def test_too_few_fail_cases_returns_a_single_cluster():
    records = [
        {"case_id": "a", "label": "FAIL", "prompt": "a weird one-off bug"},
        {"case_id": "b", "label": "PASS", "prompt": "fine"},
    ]
    report = cluster_failures(records, min_cluster_size=3)
    assert report.method == "single_cluster"
    assert len(report.clusters) == 1
    assert report.clusters[0].case_ids == ["a"]
    assert report.n_fail_cases == 1


# ---------------------------------------------------------------------------
# Numpy fallback (always runs, no optional deps)
# ---------------------------------------------------------------------------

def test_hash_vectorize_is_deterministic_and_normalized():
    import numpy as np

    X = _hash_vectorize(["hello world", "hello world", "goodbye"], n_features=64)
    assert X.shape == (3, 64)
    assert (X[0] == X[1]).all()  # identical text -> identical vector
    assert not (X[0] == X[2]).all()
    norms = np.linalg.norm(X, axis=1)
    assert np.allclose(norms[norms > 0], 1.0)


def test_cosine_greedy_recovers_two_obviously_distinct_groups():
    X = _hash_vectorize([
        "small object missed detection too tiny to see",
        "small object missed detection too tiny to see",
        "occluded object hidden behind another blocking item",
        "occluded object hidden behind another blocking item",
    ], n_features=128)
    labels = _cluster_cosine_greedy(X, max_clusters=4, threshold=0.5)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_cluster_failures_forced_cosine_greedy_recovers_two_modes():
    report = cluster_failures(
        _two_mode_records(), method="cosine_greedy", min_cluster_size=3, max_clusters=4,
    )
    assert report.method == "cosine_greedy"
    assert report.n_fail_cases == 16
    sizes = sorted(c.size for c in report.clusters)
    assert sizes[-1] >= 6  # at least one large, coherent cluster recovered
    # every FAIL case lands in exactly one cluster or unclustered
    all_ids = {cid for c in report.clusters for cid in c.case_ids} | set(report.unclustered_ids)
    assert len(all_ids) == 16


def test_cluster_failures_falls_back_to_hashing_vectorizer_when_sklearn_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn.feature_extraction.text", None)
    report = cluster_failures(
        _two_mode_records(), method="cosine_greedy", min_cluster_size=3, max_clusters=4,
    )
    assert report.params["vectorizer"] == "hashing"
    assert report.n_fail_cases == 16


def test_cluster_failures_forced_hdbscan_raises_when_not_installed():
    with pytest.raises(ImportError):
        cluster_failures(_two_mode_records(), method="hdbscan")


def test_auto_method_falls_back_to_cosine_greedy_when_no_optional_deps(monkeypatch):
    monkeypatch.setitem(sys.modules, "hdbscan", None)
    monkeypatch.setitem(sys.modules, "sklearn.cluster", None)
    monkeypatch.setitem(sys.modules, "sklearn.feature_extraction.text", None)
    report = cluster_failures(_two_mode_records(), min_cluster_size=3, max_clusters=4)
    assert report.method == "cosine_greedy"
    assert report.params["vectorizer"] == "hashing"


# ---------------------------------------------------------------------------
# sklearn-backed paths (skipped when not installed)
# ---------------------------------------------------------------------------

def test_cluster_failures_with_sklearn_agglomerative_recovers_two_modes():
    pytest.importorskip("sklearn")
    report = cluster_failures(
        _two_mode_records(), method="agglomerative", min_cluster_size=3, max_clusters=2,
    )
    assert report.method == "agglomerative"
    assert len(report.clusters) == 2
    sizes = sorted(c.size for c in report.clusters)
    assert sizes == [8, 8]


def test_cluster_failures_with_hdbscan_or_skip():
    pytest.importorskip("hdbscan")
    report = cluster_failures(_two_mode_records(), method="hdbscan", min_cluster_size=3)
    assert report.method == "hdbscan"


# ---------------------------------------------------------------------------
# LLM naming tier
# ---------------------------------------------------------------------------

class _FakeJudge:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.calls = 0

    def generate(self, prompt, **kwargs) -> str:
        self.calls += 1
        return next(self._responses)


def test_llm_naming_renames_clusters_on_valid_response():
    judge = _FakeJudge([
        '[{"cluster_id": 0, "name": "small_object_miss", "description": "objects too small to detect"},'
        ' {"cluster_id": 1, "name": "occlusion_miss", "description": "objects hidden behind others"}]'
    ])
    report = cluster_failures(
        _two_mode_records(), method="cosine_greedy", min_cluster_size=3,
        max_clusters=2, judge=judge,
    )
    names = {c.name for c in report.clusters}
    assert "small_object_miss" in names or "occlusion_miss" in names
    assert report.named_by.startswith("llm:")
    assert judge.calls == 1


def test_llm_naming_repairs_once_then_falls_back_to_top_terms(monkeypatch):
    judge = _FakeJudge(["not json at all", "still not json"])
    report = cluster_failures(
        _two_mode_records(), method="cosine_greedy", min_cluster_size=3,
        max_clusters=2, judge=judge,
    )
    # Falls back to top_terms naming — never crashes, never leaves clusters unnamed.
    assert report.named_by == "top_terms"
    assert all(c.name for c in report.clusters)
    assert judge.calls == 2  # one repair attempt, then give up


# ---------------------------------------------------------------------------
# to_dict() contract (pyod-style stable-key check)
# ---------------------------------------------------------------------------

def test_failure_mode_to_dict_has_stable_keys():
    fm = FailureMode(name="n", description="d", case_ids=["a"], exemplars=[{"x": 1}], size=1, top_terms=["t"])
    assert set(fm.to_dict()) == {"name", "description", "case_ids", "exemplars", "size", "top_terms"}


def test_failure_mode_report_to_dict_has_stable_keys():
    report = FailureModeReport(clusters=[], n_fail_cases=0, unclustered_ids=[], method="none")
    assert set(report.to_dict()) == {
        "clusters", "n_fail_cases", "unclustered_ids", "method", "named_by", "params",
    }


def test_as_hypothesis_context_empty_when_no_clusters():
    assert FailureModeReport().as_hypothesis_context() == ""


def test_as_hypothesis_context_summarizes_clusters():
    report = FailureModeReport(
        clusters=[FailureMode(name="small_obj", description="too small", size=5)],
        method="cosine_greedy",
    )
    text = report.as_hypothesis_context()
    assert "small_obj" in text and "too small" in text and "n=5" in text
