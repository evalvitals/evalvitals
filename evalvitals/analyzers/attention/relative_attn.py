"""Relative attention to image tokens — "MLLMs Know Where to Look" (arXiv 2502.17422).

For a task-specific prompt, divides its image-patch attention weights by those from
a generic baseline prompt to reveal which patches are *uniquely* important for the
question being asked.  VLM-only; requires a white-box model whose ``forward()``
populates ``trace.extras["image_token_mask"]`` and optionally
``trace.extras["image_spatial_shape"]``.

References:
- Paper: "MLLMs Know Where to Look" — https://arxiv.org/abs/2502.17422
- Code:  https://github.com/saccharomycetes/mllms_know
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from evalvitals.core.analyzer import Analyzer
from evalvitals.core.capability import Capability
from evalvitals.core.case import Inputs
from evalvitals.core.registry import register_analyzer
from evalvitals.core.result import Result

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch
    from evalvitals.core.model import Model


@dataclass
class RelativeAttentionResult(Result):
    """Result of relative attention analysis.

    Artifacts:
        attn_map    — 1-D float32 array of relative weights over image-patch tokens.
        spatial_map — 2-D array (H, W) reshaped from attn_map; same as attn_map when
                      the grid shape cannot be inferred.
    """

    @property
    def attn_map(self) -> np.ndarray:
        return self.artifacts["attn_map"]

    @property
    def spatial_map(self) -> np.ndarray:
        return self.artifacts.get("spatial_map", self.artifacts["attn_map"])

    def plot(self, ax=None, figsize=(6, 6), cmap="hot"):
        """Heatmap of the relative attention spatial map (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Visualisation requires matplotlib: pip install evalvitals[viz]"
            ) from None

        mat = self.spatial_map
        if mat.ndim == 1:
            raise ValueError(
                "Spatial shape is unknown (image_spatial_shape missing from trace.extras). "
                "The 1-D attn_map is stored in self.artifacts['attn_map']."
            )
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(mat, cmap=cmap)
        ax.set_title("Relative Attention Map")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if created:
            plt.tight_layout()
            return fig
        return ax


@register_analyzer("relative_attention")
class RelativeAttentionAnalyzer(Analyzer):
    """Relative attention to image-patch tokens.

    Runs two forward passes through the same VLM — one with the task-specific
    prompt, one with a generic ``general_prompt`` — then computes the element-wise
    ratio of their last-token attention weights over the image-patch positions.
    Regions with a high ratio are uniquely relevant to the task question.

    Reference:
        "MLLMs Know Where to Look" (arXiv 2502.17422)
        https://arxiv.org/abs/2502.17422 · https://github.com/saccharomycetes/mllms_know

    Hyper-parameters:
        general_prompt: Baseline prompt (same image, generic question).
        layer:          Decoder layer to read attention from (-1 = last).
                        The paper uses layer 22 for Qwen2.5-VL-7B-Instruct.
        top_k:          Number of highest-scoring patches to include in findings.

    Example::

        from PIL import Image
        from evalvitals.analyzers.attention import RelativeAttentionAnalyzer
        from evalvitals.core.case import Inputs

        analyzer = RelativeAttentionAnalyzer(layer=22)
        case = Inputs(prompt="What color is the car?", image=Image.open("scene.jpg"))
        result = analyzer.run(vlm_model, case)
        result.plot()
    """

    name = "relative_attention"
    requires = frozenset({Capability.ATTENTION})
    applies_to_modalities = frozenset({"image"})

    def __init__(
        self,
        general_prompt: str = "Describe the image.",
        layer: int = -1,
        top_k: int = 5,
    ) -> None:
        super().__init__(general_prompt=general_prompt, layer=layer, top_k=top_k)

    def _run(self, model: "Model", cases: "CaseBatch") -> RelativeAttentionResult:
        import torch

        case = cases[0]

        # Forward pass 1: task-specific prompt
        specific_trace = model.forward(case.inputs, capture={Capability.ATTENTION})

        # Forward pass 2: generic baseline — same image, generic question
        general_inputs = Inputs(prompt=self.general_prompt, image=case.inputs.image)
        general_trace = model.forward(general_inputs, capture={Capability.ATTENTION})

        # Image-token masks — each trace reports its own positions because the
        # two prompts have different sequence lengths.
        specific_mask = specific_trace.extras.get("image_token_mask")
        general_mask = general_trace.extras.get("image_token_mask")

        if specific_mask is None or general_mask is None:
            raise ValueError(
                "trace.extras['image_token_mask'] is missing. "
                "RelativeAttentionAnalyzer requires a VLM model whose forward() "
                "populates image token positions (hf_local with is_vlm=True)."
            )

        n_img = int(specific_mask.sum())
        if n_img == 0:
            raise ValueError("image_token_mask contains no True entries — no image tokens found.")
        if int(general_mask.sum()) != n_img:
            raise ValueError(
                f"Image token count differs between specific ({n_img}) and "
                f"general ({int(general_mask.sum())}) forward passes. "
                "Both prompts must use the same image."
            )

        def _img_attn(trace, mask) -> torch.Tensor:
            """Head-averaged attention from the last query position to image patches."""
            attns = trace.require(Capability.ATTENTION)
            a = attns[self.layer].float()  # (heads, seq, seq)
            return a.mean(dim=0)[-1, mask]  # (n_img_tokens,)

        specific_attn = _img_attn(specific_trace, specific_mask)
        general_attn = _img_attn(general_trace, general_mask)

        # Relative attention: ratio highlighting task-relevant patches
        rel = specific_attn / (general_attn + 1e-8)
        rel_np = rel.cpu().float().numpy()

        # Spatial reshape using the grid shape stored by the VLM backend
        spatial_shape = specific_trace.extras.get("image_spatial_shape")
        if spatial_shape is not None:
            h, w = spatial_shape
            if h * w == n_img:
                spatial_map = rel_np.reshape(h, w)
            else:
                spatial_map = rel_np
        else:
            side = int(n_img ** 0.5)
            spatial_map = rel_np.reshape(side, side) if side * side == n_img else rel_np

        # Top-k patches for agent-readable findings
        topk_idx = np.argsort(rel_np)[::-1][: self.top_k]
        top_patches = [
            {"patch_idx": int(i), "relative_weight": round(float(rel_np[i]), 4)}
            for i in topk_idx
        ]

        result = RelativeAttentionResult(
            analyzer=self.name,
            model=repr(model),
            cases=cases,
            artifacts={"attn_map": rel_np, "spatial_map": spatial_map},
        )
        result.findings = {
            "n_image_tokens": n_img,
            "n_layers": len(specific_trace.require(Capability.ATTENTION)),
            "layer_used": self.layer,
            "map_shape": list(spatial_map.shape),
            "top_patches": top_patches,
        }
        return result