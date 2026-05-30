"""Tests for FailureCase, CaseBatch, and as_casebatch normalisation."""

from __future__ import annotations

import pytest

from evalvitals.core import CaseBatch, FailureCase, Inputs, Label, as_casebatch
from evalvitals.core.case import Provenance, Source


def test_failurecase_from_prompt():
    case = FailureCase.from_prompt("hello")
    assert case.inputs.prompt == "hello"
    assert case.label == Label.UNKNOWN
    assert len(case.id) == 12


def test_as_casebatch_from_str():
    batch = as_casebatch("a prompt")
    assert isinstance(batch, CaseBatch)
    assert len(batch) == 1
    assert batch[0].inputs.prompt == "a prompt"


def test_as_casebatch_from_failurecase():
    case = FailureCase.from_prompt("x")
    batch = as_casebatch(case)
    assert len(batch) == 1
    assert batch[0] is case


def test_as_casebatch_from_inputs():
    batch = as_casebatch(Inputs(prompt="y"))
    assert len(batch) == 1
    assert batch[0].inputs.prompt == "y"


def test_as_casebatch_from_list():
    batch = as_casebatch(["a", "b", "c"])
    assert len(batch) == 3
    assert [c.inputs.prompt for c in batch] == ["a", "b", "c"]


def test_as_casebatch_passthrough():
    original = CaseBatch.from_prompts(["a", "b"])
    assert as_casebatch(original) is original


def test_as_casebatch_rejects_unknown():
    with pytest.raises(TypeError):
        as_casebatch(object())


def test_casebatch_filter_by_label():
    batch = CaseBatch(
        [
            FailureCase.from_prompt("a", label=Label.FAIL),
            FailureCase.from_prompt("b", label=Label.PASS),
            FailureCase.from_prompt("c", label=Label.FAIL),
        ]
    )
    fails = batch.filter(label=Label.FAIL)
    assert len(fails) == 2


def test_casebatch_filter_by_tags():
    batch = CaseBatch(
        [
            FailureCase.from_prompt("a", tags={"hallucination"}),
            FailureCase.from_prompt("b", tags={"format"}),
        ]
    )
    assert len(batch.filter(tags={"hallucination"})) == 1


def test_failurecase_to_dict_roundtrip_shape():
    case = FailureCase.from_prompt(
        "q",
        expected="gold",
        label=Label.FAIL,
        tags={"x"},
        provenance=Provenance(source=Source.AGENT),
    )
    d = case.to_dict()
    assert d["inputs"]["prompt"] == "q"
    assert d["label"] == "fail"
    assert d["tags"] == ["x"]
    assert d["provenance"]["source"] == "agent"
