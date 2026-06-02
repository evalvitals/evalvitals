"""Dataset loaders → CaseBatch + answer verifiers."""

from __future__ import annotations

import json

from evalvitals.core.case import CaseBatch
from evalvitals.datasets import (
    GUIOSDataset,
    LLMQADataset,
    PureQADataset,
    Spatial457Dataset,
    VLMQADataset,
    WebSearchQADataset,
    contains_answer,
    exact_match,
    normalize,
)


def test_llm_qa_sample_loads():
    cb = LLMQADataset.sample().load()
    assert isinstance(cb, CaseBatch) and len(cb) == 4
    c = cb[0]
    assert c.inputs.prompt.startswith("What is the capital") and c.expected == "Paris"
    assert "llm_qa" in c.tags and c.metadata["difficulty"] == "easy"


def test_pure_qa_is_llm_qa_alias():
    assert PureQADataset is LLMQADataset  # back-compat


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


def test_vlm_qa_image_goes_to_inputs_not_metadata():
    recs = [{"question": "what color?", "answer": "red", "image": "<PIL>", "scene": "s1"}]
    cb = VLMQADataset.from_records(recs, meta_keys=("scene",)).load()
    c = cb[0]
    assert "vlm_qa" in c.tags
    assert c.inputs.image == "<PIL>" and c.expected == "red"
    assert c.metadata == {"scene": "s1"}          # image NOT dumped into metadata
    assert "image" not in c.metadata


def test_spatial457_subset_validation():
    import pytest
    with pytest.raises(ValueError):
        Spatial457Dataset(subset="not_a_subset")
    assert "L5_6d_spatial" in Spatial457Dataset.SUBTYPES and len(Spatial457Dataset.SUBTYPES) == 7


def test_spatial457_record_mapping():
    cb = Spatial457Dataset.from_records([
        {"image": "<img>", "image_filename": "superCLEVR_new_000001.png",
         "question_index": 100001, "question": "Is the red object in front of the car?", "answer": "True"},
    ], subset="L5_6d_spatial").load()
    c = cb[0]
    assert c.inputs.prompt.startswith("Is the red object") and c.expected == "True"
    assert c.inputs.image == "<img>"
    assert {"vlm_qa", "spatial457", "L5_6d_spatial"}.issubset(c.tags)
    assert c.metadata["image_filename"] == "superCLEVR_new_000001.png"
    assert c.metadata["question_index"] == 100001 and c.metadata["subset"] == "L5_6d_spatial"


def test_spatial457_sample():
    cb = Spatial457Dataset.sample().load()
    assert len(cb) == 2 and "spatial457" in cb[0].tags
    assert cb[0].metadata["dataset"] == "spatial457"
