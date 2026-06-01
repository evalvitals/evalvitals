"""Dataset loaders → CaseBatch + answer verifiers."""

from __future__ import annotations

import json

from evalvitals.core.case import CaseBatch
from evalvitals.datasets import (
    GUIOSDataset,
    PureQADataset,
    WebSearchQADataset,
    contains_answer,
    exact_match,
    normalize,
)


def test_pure_qa_sample_loads():
    cb = PureQADataset.sample().load()
    assert isinstance(cb, CaseBatch) and len(cb) == 4
    c = cb[0]
    assert c.inputs.prompt.startswith("What is the capital") and c.expected == "Paris"
    assert "pure_qa" in c.tags and c.metadata["difficulty"] == "easy"


def test_pure_qa_from_records_and_jsonl(tmp_path):
    recs = [{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}]
    cb = PureQADataset.from_records(recs).load()
    assert len(cb) == 2 and cb[1].expected == "a2"

    p = tmp_path / "qa.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs))
    cb2 = PureQADataset.from_jsonl(str(p)).load()
    assert len(cb2) == 2 and cb2[0].inputs.prompt == "q1"


def test_web_search_and_gui_os_tags():
    w = WebSearchQADataset.sample().load()
    assert "web_search" in w[0].tags and w[0].metadata["requires_retrieval"] is True
    g = GUIOSDataset.sample().load()
    assert "gui_os" in g[0].tags and g[0].expected == {"dark_mode": True}
    assert g[0].inputs.prompt.startswith("Open Settings")


def test_verifiers():
    assert exact_match("Paris", " paris ")
    assert contains_answer("The capital is Paris.", "Paris")
    assert not contains_answer("It is Lyon.", "Paris")
    assert normalize("  Hello   World ") == "hello world"
