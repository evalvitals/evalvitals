"""L2 scaffold tool catalog + pipeline executor for the fix module.

An L2 candidate fix wraps the *unchanged* model in a small pipeline: image
preprocessing tools applied to the case image, an optional prompt template,
and optional multi-sample aggregation.  Tools are a registered catalog (the
judge selects and parameterises them — same select-from-catalog pattern as
M1's analyzers); pipeline specs are plain dicts so they serialise into run
logs and can be re-executed.

PIL is imported lazily — the core package does not depend on pillow; any
environment that loads images for a VLM already has it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from evalvitals.core.case import FailureCase
    from evalvitals.core.model import Model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image tool catalog
# ---------------------------------------------------------------------------

def _as_pil(image: Any):
    """Decode *image* (PIL.Image | path/URL) to a PIL image, or ``None``."""
    if image is None:
        return None
    from PIL import Image

    if isinstance(image, Image.Image):
        return image
    try:
        return Image.open(str(image))
    except Exception as exc:
        logger.debug("fix_tools: cannot decode image %r: %s", image, exc)
        return None


def zoom_center(img, factor: float = 1.5):
    """Crop the central 1/factor region and resize back to the original size."""
    factor = max(1.0, float(factor))
    w, h = img.size
    cw, ch = int(w / factor), int(h / factor)
    left, top = (w - cw) // 2, (h - ch) // 2
    from PIL import Image

    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.LANCZOS)


def enhance_contrast(img, factor: float = 1.5):
    from PIL import ImageEnhance

    return ImageEnhance.Contrast(img).enhance(float(factor))


def sharpen(img, factor: float = 2.0):
    from PIL import ImageEnhance

    return ImageEnhance.Sharpness(img).enhance(float(factor))


def equalize(img):
    """Histogram equalization (high dynamic-range scans, e.g. radiology)."""
    from PIL import ImageOps

    return ImageOps.equalize(img.convert("RGB"))


def upscale(img, factor: float = 2.0):
    factor = max(1.0, float(factor))
    from PIL import Image

    w, h = img.size
    return img.resize((int(w * factor), int(h * factor)), Image.LANCZOS)


def crop_region(img, box=(0.25, 0.25, 0.75, 0.75)):
    """Crop a normalized (left, top, right, bottom) box, resize to original size.

    Like :func:`zoom_center` but for an arbitrary region — the building block
    for attention-guided cropping (L3a), also usable by coded pipelines.
    """
    from PIL import Image

    w, h = img.size
    left, top, right, bottom = (float(v) for v in box)
    left, top = max(0.0, min(left, 0.95)), max(0.0, min(top, 0.95))
    right, bottom = min(1.0, max(right, left + 0.05)), min(1.0, max(bottom, top + 0.05))
    px = (int(left * w), int(top * h), max(int(right * w), int(left * w) + 1),
          max(int(bottom * h), int(top * h) + 1))
    return img.crop(px).resize((w, h), Image.LANCZOS)


#: name -> (function, parameter hint for the judge prompt, description)
IMAGE_TOOLS: "dict[str, tuple[Callable, str, str]]" = {
    "zoom_center": (zoom_center, "factor: float >= 1 (default 1.5)",
                    "crop the central 1/factor region, resize back — magnifies small findings"),
    "enhance_contrast": (enhance_contrast, "factor: float (default 1.5)",
                         "global contrast boost — low-contrast findings"),
    "sharpen": (sharpen, "factor: float (default 2.0)",
                "edge sharpening — blurred or subtle boundaries"),
    "equalize": (equalize, "(no params)",
                 "histogram equalization — compressed dynamic range"),
    "upscale": (upscale, "factor: float >= 1 (default 2.0)",
                "resize up before encoding — more vision tokens per region"),
    "crop_region": (crop_region, "box: [left, top, right, bottom] normalized 0..1",
                    "crop an arbitrary region and resize back — magnify a known area"),
}


def catalog_text() -> str:
    """Render the tool catalog for inclusion in a judge prompt."""
    return "\n".join(
        f"- {name}: {desc}  [params: {params}]"
        for name, (_, params, desc) in IMAGE_TOOLS.items()
    )


def apply_image_ops(image: Any, ops: "list[dict[str, Any]]") -> Any:
    """Apply ``[{"tool": name, "params": {...}}, ...]`` to *image*.

    Unknown tools and per-op failures are skipped (logged); returns the
    original object when nothing could be applied (e.g. no image).
    """
    img = _as_pil(image)
    if img is None:
        return image
    for op in ops:
        name = str(op.get("tool", ""))
        entry = IMAGE_TOOLS.get(name)
        if entry is None:
            logger.warning("fix_tools: unknown tool %r skipped", name)
            continue
        fn = entry[0]
        try:
            img = fn(img, **dict(op.get("params") or {}))
        except Exception as exc:
            logger.warning("fix_tools: tool %r failed (%s); skipped", name, exc)
    return img


# ---------------------------------------------------------------------------
# Pipeline spec + executor
# ---------------------------------------------------------------------------

@dataclass
class PipelineSpec:
    """A serialisable L2 scaffold around the unchanged model.

    Attributes:
        name:            Short identifier for logs.
        image_ops:       Tool applications, in order (see :data:`IMAGE_TOOLS`).
        prompt_template: Must contain ``{prompt}``; identity by default.
        n_samples:       Model calls per case (majority vote when > 1).
    """

    name: str
    image_ops: "list[dict[str, Any]]" = field(default_factory=list)
    prompt_template: str = "{prompt}"
    n_samples: int = 1

    @classmethod
    def from_dict(cls, d: "dict[str, Any]") -> "PipelineSpec | None":
        """Validate a judge-proposed spec dict; ``None`` when unusable."""
        name = str(d.get("name", "")).strip()
        template = str(d.get("prompt_template") or "{prompt}")
        if not name or "{prompt}" not in template:
            return None
        ops = [
            {"tool": str(op["tool"]), "params": dict(op.get("params") or {})}
            for op in d.get("image_ops") or []
            if isinstance(op, dict) and str(op.get("tool", "")) in IMAGE_TOOLS
        ]
        try:
            n_samples = max(1, int(d.get("n_samples", 1)))
        except (TypeError, ValueError):
            n_samples = 1
        return cls(name=name, image_ops=ops, prompt_template=template,
                   n_samples=min(n_samples, 5))

    def to_dict(self) -> "dict[str, Any]":
        return {"name": self.name, "image_ops": self.image_ops,
                "prompt_template": self.prompt_template, "n_samples": self.n_samples}


def run_pipeline(
    model: "Model",
    case: "FailureCase",
    spec: PipelineSpec,
    score_fn: "Callable[[FailureCase, str], Optional[bool]]",
) -> "Optional[bool]":
    """Execute *spec* on one case; majority vote over scored samples.

    Returns ``None`` when the case cannot be scored (no rubric / all calls
    failed) — mirroring prompt_contrast's unscored semantics.
    """
    from evalvitals.core.case import Inputs

    inp = getattr(case, "inputs", None)
    prompt = str(getattr(inp, "prompt", "")) if inp is not None else ""
    image = getattr(inp, "image", None) if inp is not None else None
    if spec.image_ops:
        image = apply_image_ops(image, spec.image_ops)
    new_inputs = Inputs(prompt=spec.prompt_template.format(prompt=prompt), image=image)

    votes: "list[bool]" = []
    for _ in range(spec.n_samples):
        try:
            output = str(model.generate(new_inputs))
        except Exception as exc:
            logger.debug("run_pipeline: generate failed on %s: %s", case.id, exc)
            continue
        score = score_fn(case, output)
        if score is not None:
            votes.append(bool(score))
    if not votes:
        return None
    return sum(votes) * 2 >= len(votes)  # majority, ties -> True
