"""vLLM offline backend — in-process, high-throughput logprob/generation.

This is the *distinct* role of vLLM (NOT the same as ``api``): an in-process
engine you submit batches to, with continuous batching and ``prompt_logprobs`` —
the throughput sweet spot for perturbation sweeps (RISE / SHAP) and logprob
scoring on open weights.  It exposes ``GENERATE`` + ``LOGPROBS``/``LOGITS`` but
**never attention or hidden states** (paged/fused kernels never materialise the
L×L matrix) — that's a structural limit, so white-box analyzers route to
``hf_local`` instead.

(A ``vllm serve`` HTTP endpoint, by contrast, is reached via the ``api`` backend
with ``base_url`` pointed at it — no separate adapter needed there.)

Stage 2: the offline engine wiring.  The capability declaration is live so
capability matching and compose() negotiation already behave correctly.
"""

from __future__ import annotations

from evalvitals.core.capability import Capability
from evalvitals.models.backends.base import Backend, RuntimeConfig


class VLLMOfflineBackend(Backend):
    kind = "vllm_offline"
    capabilities = frozenset(
        {Capability.GENERATE, Capability.TOOL_CALLS, Capability.LOGPROBS, Capability.LOGITS}
    )

    def build(self, spec, runtime: RuntimeConfig):
        raise NotImplementedError(
            "vllm_offline backend is planned for Stage 2 (in-process `from vllm import LLM`, "
            "batched generation + prompt_logprobs). For serving today, use the `api` backend "
            "pointed at a `vllm serve` endpoint."
        )
