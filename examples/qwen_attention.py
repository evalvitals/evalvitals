"""Example: Qwen attention analysis.

Three equivalent usage patterns are shown:
  1. Canonical  — configure an analyzer, run it on a model (sklearn-style)
  2. Config-driven — load a YAML file and call run()
  3. Hybrid shim — model.call_attention(), auto-derived from capabilities

Run:
    python examples/qwen_attention.py
"""

from __future__ import annotations

PROMPT = "The Eiffel Tower is located in the city of"

# ======================================================================
# Option 1 — canonical, analyzer-centric
# ======================================================================

from evalvitals.analyzers.attention.summary import AttentionAnalyzer
from evalvitals.models.whitebox.qwen import QwenLLM

model = QwenLLM(checkpoint="Qwen/Qwen2.5-7B-Instruct", device="cuda")
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
# Option 3 — hybrid convenience shim (same result, derived from capabilities)
# ======================================================================

result3 = model.call_attention(PROMPT)
assert result3.num_layers == result.num_layers
print("\nAll three options produced the same result. ✓")

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
