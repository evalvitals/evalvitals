"""VLM (image + text) QA loaders → CaseBatch.

``VLMQADataset`` is the generic image+text QA loader; ``Spatial457Dataset`` wraps
the **RyanWW/Spatial457** benchmark (6D spatial-reasoning VQA) via the HuggingFace
``datasets`` library (optional dep — ``pip install evalvitals[data]``).

The image goes into ``Inputs.image`` (kept OUT of metadata so heavy PIL objects
aren't duplicated); the gold answer is ``expected``; cases are tagged ``vlm_qa``.

Spatial457:
  Paper: "Spatial457: A Diagnostic Benchmark for 6D Spatial Reasoning of Large
         Multimodal Models" — Wang et al., CVPR 2025 — arXiv:2502.08636
  Data:  https://huggingface.co/datasets/RyanWW/Spatial457
  Code:  https://github.com/XingruiWang/Spatial457
"""

from __future__ import annotations

from typing import Iterable

from evalvitals.core.case import CaseBatch, FailureCase, Inputs, Provenance, Source
from evalvitals.datasets.base import Dataset, read_jsonl


def _vlm_cases(
    records: Iterable[dict],
    *,
    prompt_key: str,
    answer_key: str,
    image_key: str,
    tags: set[str],
    meta_keys: tuple[str, ...] = (),
    base_meta: dict | None = None,
) -> CaseBatch:
    """Build image+text cases — image -> Inputs.image (NOT metadata); selected keys -> metadata."""
    out = CaseBatch()
    for rec in records:
        meta = dict(base_meta or {})
        for k in meta_keys:
            if k in rec:
                meta[k] = rec[k]
        out.append(FailureCase(
            inputs=Inputs(prompt=str(rec.get(prompt_key, "")), image=rec.get(image_key)),
            expected=rec.get(answer_key),
            tags=set(tags),
            provenance=Provenance(source=Source.DATASET),
            metadata=meta,
        ))
    return out


_VLM_SAMPLE = [
    {"question": "What is shown in the image?", "answer": "a placeholder", "image": None, "note": "attach a real image"},
]


class VLMQADataset(Dataset):
    """Generic image+text question-answering loader."""

    def __init__(
        self,
        records: list[dict] | None = None,
        path: str | None = None,
        *,
        prompt_key: str = "question",
        answer_key: str = "answer",
        image_key: str = "image",
        meta_keys: tuple[str, ...] = (),
    ) -> None:
        self._records = records
        self._path = path
        self._prompt_key, self._answer_key, self._image_key = prompt_key, answer_key, image_key
        self._meta_keys = meta_keys

    @classmethod
    def from_records(cls, records: list[dict], **kw) -> "VLMQADataset":
        return cls(records=records, **kw)

    @classmethod
    def from_jsonl(cls, path: str, **kw) -> "VLMQADataset":
        return cls(path=path, **kw)

    @classmethod
    def sample(cls) -> "VLMQADataset":
        return cls(records=_VLM_SAMPLE, meta_keys=("note",))

    def load(self) -> CaseBatch:
        records = self._records if self._records is not None else (
            read_jsonl(self._path) if self._path else _VLM_SAMPLE
        )
        return _vlm_cases(records, prompt_key=self._prompt_key, answer_key=self._answer_key,
                          image_key=self._image_key, tags={"vlm_qa"}, meta_keys=self._meta_keys)


# Spatial457 — RyanWW/Spatial457 row schema: {image: PIL, image_filename, question, answer, program, question_index}
_SPATIAL457_SAMPLE = [
    {"image": None, "image_filename": "superCLEVR_new_000001.png", "question_index": 100001,
     "question": "Is the large red object in front of the yellow car?", "answer": "True"},
    {"image": None, "image_filename": "superCLEVR_new_000002.png", "question_index": 100002,
     "question": "How many objects are behind the blue cube?", "answer": "2"},
]


class Spatial457Dataset(Dataset):
    """RyanWW/Spatial457 — 6D spatial-reasoning VQA (CVPR'25), 7 cascading subtypes.

    ``subset`` selects a difficulty/question-type split.  ``load()`` pulls from
    HuggingFace (``pip install evalvitals[data]``); use ``from_records`` / ``sample``
    offline.  Maps each row to a ``vlm_qa`` FailureCase (image -> Inputs.image,
    answer -> expected; image_filename/question_index/subset -> metadata).
    """

    SUBTYPES = (
        "L1_single", "L2_objects", "L3_2d_spatial", "L4_occ", "L4_pose",
        "L5_6d_spatial", "L5_collision",
    )

    def __init__(
        self,
        subset: str = "L5_6d_spatial",
        split: str = "test",
        *,
        records: list[dict] | None = None,
        hf_repo: str = "RyanWW/Spatial457",
        max_samples: int | None = None,
    ) -> None:
        if subset not in self.SUBTYPES:
            raise ValueError(f"unknown Spatial457 subset {subset!r}; choose from {self.SUBTYPES}")
        self.subset = subset
        self.split = split
        self.hf_repo = hf_repo
        self.max_samples = max_samples
        self._records = records

    @classmethod
    def from_records(cls, records: list[dict], subset: str = "L5_6d_spatial") -> "Spatial457Dataset":
        return cls(subset=subset, records=records)

    @classmethod
    def sample(cls, subset: str = "L5_6d_spatial") -> "Spatial457Dataset":
        return cls(subset=subset, records=_SPATIAL457_SAMPLE)

    def _load_hf(self) -> Iterable[dict]:
        try:
            from datasets import load_dataset
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "Spatial457Dataset.load() needs the HuggingFace 'datasets' library — "
                "pip install evalvitals[data] (or use from_records()/sample() offline)."
            ) from e
        try:  # subtypes are exposed as dataset configs
            ds = load_dataset(self.hf_repo, self.subset, split=self.split)
        except Exception:  # fall back: single config, filter by a subtype column if present
            ds = load_dataset(self.hf_repo, split=self.split)
            for col in ("question_type", "subtype", "level"):
                if col in (getattr(ds, "column_names", None) or []):
                    ds = ds.filter(lambda r, c=col: r.get(c) == self.subset)
                    break
        if self.max_samples is not None:
            ds = ds.select(range(min(self.max_samples, len(ds))))
        return ds

    def load(self) -> CaseBatch:
        rows = self._records if self._records is not None else self._load_hf()
        return _vlm_cases(
            rows, prompt_key="question", answer_key="answer", image_key="image",
            tags={"vlm_qa", "spatial457", self.subset},
            meta_keys=("image_filename", "question_index", "program"),
            base_meta={"dataset": "spatial457", "subset": self.subset},
        )


# VQA-RAD — flaviagiammarino/vqa-rad row schema: {image: PIL, question, answer}
_VQA_RAD_SAMPLE = [
    {"image": None, "question": "what imaging modality was used?", "answer": "ct"},
    {"image": None, "question": "is there evidence of a pneumothorax?", "answer": "no"},
    {"image": None, "question": "is the heart enlarged?", "answer": "yes"},
    {"image": None, "question": "what plane is this image taken in?", "answer": "axial"},
]

# Question fragments that identify "easy" identification questions (modality /
# plane / organ) a general VLM reliably answers — the M5 control (PASS) group.
_VQA_RAD_EASY_FRAGMENTS = (
    "modality", "what plane", "which plane", "plane is", "what organ",
    "which organ", "organ system", "part of the body", "what type of imaging",
    "what kind of image", "what imaging", "mri or ct", "ct or mri",
)


def _categorize_vqa_rad(question: str, answer: str) -> str:
    """``"easy"`` (modality/plane/organ), ``"presence"`` (closed yes/no), or ``"other"``."""
    q = " ".join(str(question).lower().split())
    if any(frag in q for frag in _VQA_RAD_EASY_FRAGMENTS):
        return "easy"
    if str(answer).strip().lower() in {"yes", "no"}:
        return "presence"
    return "other"


def _easy_answer_rubric(gold: str) -> dict:
    """Token-level ``any_of`` rubric for open identification answers.

    VQA-RAD gold strings are messy ("xray - plain film", tab-separated organ
    lists) — whole-string matching would mislabel correct answers as FAIL and
    pollute the control group.  Accept any significant gold token instead, with
    a cheap plural tolerance (kidneys → kidney).
    """
    import re as _re

    tokens: list[str] = []
    for tok in _re.findall(r"[a-z0-9]+", str(gold).lower()):
        if len(tok) < 2 or tok in tokens:
            continue
        tokens.append(tok)
        if tok.endswith("s") and len(tok) > 3 and tok[:-1] not in tokens:
            tokens.append(tok[:-1])
    return {"any_of": tokens or [str(gold).strip().lower()]}


class VQARADDataset(Dataset):
    """VQA-RAD — radiology VQA (Lau et al., 2018), public domain (CC0).

    Builds a **diagnosis-ready** case mix for the VL failure-analysis loop:

    - ``n_easy`` identification questions (modality/plane/organ) the model
      reliably PASSES — M5's control group;
    - ``n_presence`` closed yes/no finding-presence questions, balanced between
      gold "yes" and gold "no" — where presence hallucination (yes-bias)
      concentrates the failures.

    Presence cases carry ``metadata["pope_label"]`` so the ``pope`` analyzer can
    score them and emit per-case ``false_positive`` / ``false_negative``
    mechanism signals, plus a strict ``{"all_of": [gold], "none_of": [other]}``
    rubric for the discovery scorer.  Easy cases get a token-level ``any_of``
    rubric (gold strings are messy free text).

    Data: https://huggingface.co/datasets/flaviagiammarino/vqa-rad
    Paper: "A dataset of clinically generated visual questions and answers
           about radiology images" — Lau et al., Scientific Data 2018.
    """

    def __init__(
        self,
        split: str = "train",
        *,
        records: list[dict] | None = None,
        hf_repo: str = "flaviagiammarino/vqa-rad",
        n_easy: int = 6,
        n_presence: int = 12,
        seed: int = 0,
    ) -> None:
        self.split = split
        self.hf_repo = hf_repo
        self.n_easy = n_easy
        self.n_presence = n_presence
        self.seed = seed
        self._records = records

    @classmethod
    def from_records(cls, records: list[dict], **kw) -> "VQARADDataset":
        return cls(records=records, **kw)

    @classmethod
    def sample(cls) -> "VQARADDataset":
        return cls(records=_VQA_RAD_SAMPLE, n_easy=2, n_presence=2)

    def _load_hf(self) -> Iterable[dict]:
        try:
            from datasets import load_dataset
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "VQARADDataset.load() needs the HuggingFace 'datasets' library — "
                "pip install evalvitals[data] (or use from_records()/sample() offline)."
            ) from e
        return load_dataset(self.hf_repo, split=self.split)

    def load(self) -> CaseBatch:
        import random

        rows = self._records if self._records is not None else self._load_hf()

        easy: list[dict] = []
        pres_yes: list[dict] = []
        pres_no: list[dict] = []
        for rec in rows:
            cat = _categorize_vqa_rad(rec.get("question", ""), rec.get("answer", ""))
            if cat == "easy":
                easy.append(rec)
            elif cat == "presence":
                (pres_yes if str(rec["answer"]).strip().lower() == "yes" else pres_no).append(rec)

        rng = random.Random(self.seed)
        rng.shuffle(easy)
        rng.shuffle(pres_yes)
        rng.shuffle(pres_no)

        half = self.n_presence // 2
        picked_pres = pres_yes[:half] + pres_no[: self.n_presence - half]
        picked_easy = easy[: self.n_easy]

        out = CaseBatch()
        for i, rec in enumerate(picked_easy):
            out.append(FailureCase(
                id=f"rad_easy_{i}",
                inputs=Inputs(
                    prompt=f"{str(rec['question']).strip()} Answer briefly.",
                    image=rec.get("image"),
                ),
                expected=_easy_answer_rubric(rec["answer"]),
                tags={"vlm_qa", "med_vqa", "vqa_rad", "easy"},
                provenance=Provenance(source=Source.DATASET),
                metadata={"dataset": "vqa_rad", "category": "easy",
                          "gold_answer": str(rec["answer"]).strip()},
            ))
        for i, rec in enumerate(picked_pres):
            gold = str(rec["answer"]).strip().lower()
            other = "no" if gold == "yes" else "yes"
            out.append(FailureCase(
                id=f"rad_pres_{i}",
                inputs=Inputs(
                    prompt=f"{str(rec['question']).strip()} Answer yes or no.",
                    image=rec.get("image"),
                ),
                expected={"all_of": [gold], "none_of": [other]},
                tags={"vlm_qa", "med_vqa", "vqa_rad", "presence"},
                provenance=Provenance(source=Source.DATASET),
                metadata={"dataset": "vqa_rad", "category": "presence",
                          "pope_label": gold, "gold_answer": gold},
            ))
        return out
