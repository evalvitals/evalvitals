"""Dataset loaders → CaseBatch, plus simple answer verifiers."""

from evalvitals.datasets.base import (
    Dataset,
    cases_from_records,
    contains_answer,
    exact_match,
    normalize,
    read_jsonl,
)
from evalvitals.datasets.gui_os import GUIOSDataset
from evalvitals.datasets.pure_qa import PureQADataset
from evalvitals.datasets.web_search_qa import WebSearchQADataset

__all__ = [
    "Dataset",
    "PureQADataset",
    "WebSearchQADataset",
    "GUIOSDataset",
    "cases_from_records",
    "read_jsonl",
    "exact_match",
    "contains_answer",
    "normalize",
]
