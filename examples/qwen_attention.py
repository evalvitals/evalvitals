"""Example: Qwen attention analysis.

Shows the unified path — build a model from a spec key, run an analyzer:
  1. Friendly       — evalvitals.load(key)
  2. Config-driven  — load_config() + run()
  3. Hybrid shim    — model.call_attention()
  4. Explicit engine — compose(spec, backend, want=...)

Run:
    python examples/qwen_attention.py
"""

from __future__ import annotations

PROMPT = "The Eiffel Tower is located in the city of"

# ======================================================================
# Option 1 — friendly load() + canonical analyzer
# ======================================================================

import evalvitals
from evalvitals.analyzers.attention.summary import AttentionAnalyzer

model = evalvitals.load("qwen2.5-7b-instruct")   # hf_local backend (full internals)
analyzer = AttentionAnalyzer(layer=-1, head="mean", top_k=10)
result = analyzer.run(model, PROMPT)

print(result.summary())
print()
print("findings (the agent-readable answer):")
for item in result.findings["top_attended_tokens"]:
    print(f"  {item['weight']:.4f}  {item['token']!r}")

# ======================================================================
# Option 2 — config-driven dispatch
# ======================================================================

from evalvitals import load_config, run

config = load_config("configs/qwen_attention.yaml")
result2 = run(config, PROMPT)
assert result2.num_layers == result.num_layers

# ======================================================================
# Option 3 — hybrid convenience shim (auto-derived from capabilities)
# ======================================================================

result3 = model.call_attention(PROMPT)
assert result3.num_layers == result.num_layers
print("\nOptions 1-3 produced the same result. ✓")

# ======================================================================
# Option 4 — explicit engine: pick the backend, negotiate capabilities
# ======================================================================

from evalvitals import Capability
from evalvitals.models import compose

model4 = compose("qwen2.5-7b-instruct", "hf_local", want={Capability.ATTENTION})
assert model4.supports({Capability.ATTENTION})

# ======================================================================
# Discovery — what can an agent run on this model?
# ======================================================================

from evalvitals import registry

print("\nAnalyses compatible with this model (capability-matched):")
print(" ", registry.analyzers.names_compatible_with(model))

# ======================================================================
# Optional: visualise (requires pip install evalvitals[viz])
# ======================================================================

try:
    fig = result.plot(layer=-1, head="mean")
    fig.savefig("attention_heatmap.png", dpi=150, bbox_inches="tight")
    print("\nSaved attention_heatmap.png")
except ImportError:
    print("\n(Install matplotlib for visualisation: pip install evalvitals[viz])")
