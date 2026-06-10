"""Dataset loaders → CaseBatch, plus simple answer verifiers.

Text QA: ``LLMQADataset``.  Image+text QA: ``VLMQADataset`` and the
``Spatial457Dataset`` / ``VQARADDataset`` benchmarks.  ``PureQADataset`` is a
back-compat alias of ``LLMQADataset``.
"""

from evalvitals.datasets.base import (
    Dataset,
    cases_from_records,
    contains_answer,
    exact_match,
    normalize,
    read_jsonl,
)
from evalvitals.datasets.gui_os import GUIOSDataset
from evalvitals.datasets.llm_qa import LLMQADataset, PureQADataset
from evalvitals.datasets.vlm_qa import Spatial457Dataset, VLMQADataset, VQARADDataset
from evalvitals.datasets.web_search_qa import WebSearchQADataset

__all__ = [
    "Dataset",
    "LLMQADataset",
    "VLMQADataset",
    "Spatial457Dataset",
    "VQARADDataset",
    "PureQADataset",
    "WebSearchQADataset",
    "GUIOSDataset",
    "cases_from_records",
    "read_jsonl",
    "exact_match",
    "contains_answer",
    "normalize",
]
