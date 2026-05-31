"""Real-model smoke test: Qwen/Qwen3-4B via the hf_local backend (needs a GPU).

Exercises the full merged stack end-to-end on actual weights:
  1. compose(spec, "hf_local") + capability set (eager forced for attention)
  2. generate() -> text
  3. forward(capture={ATTENTION, HIDDEN_STATES, LOGITS}) -> Trace shapes
  4. AttentionAnalyzer + TokenEntropyAnalyzer -> findings (agent-readable)
  5. Agent(wraps=handle) with a real tool -> Trajectory (QwenToolCodec on REAL
     Qwen output + apply_chat_template(tools=...))

Run on a GPU node, e.g.:
    /home/jiaqiliu/scratch/evalvital/.venv/bin/python examples/smoke_qwen3_4b.py
"""

from __future__ import annotations

import torch

from evalvitals import Agent, Tool, compose
from evalvitals.analysis.whitebox.attention import AttentionAnalyzer
from evalvitals.analysis.whitebox.uncertainty import TokenEntropyAnalyzer
from evalvitals.core.capability import Capability
from evalvitals.models import RuntimeConfig


def main() -> None:
    assert torch.cuda.is_available(), "no CUDA device visible"
    print("device:", torch.cuda.get_device_name(0))

    rt = RuntimeConfig(device="cuda", dtype="bfloat16", max_new_tokens=128)
    model = compose("qwen3-4b", "hf_local", rt)
    print("\n[1] composed:", repr(model))
    print("    capabilities:", sorted(c.value for c in model.capabilities))
    assert Capability.ATTENTION in model.capabilities
    assert Capability.TOOL_CALLS in model.capabilities  # conditional cap granted (instruct template)

    # 2. generate ------------------------------------------------------
    out = model.generate("The capital of France is", max_new_tokens=16)
    print("\n[2] generate ->", repr(out.strip()))

    # 3. forward + capture --------------------------------------------
    trace = model.forward(
        "The Eiffel Tower is in",
        capture={Capability.ATTENTION, Capability.HIDDEN_STATES, Capability.LOGITS},
    )
    print("\n[3] Trace: provided =", sorted(c.value for c in trace.provided))
    print("    seq_len      :", trace.seq_len)
    print("    n attn layers:", len(trace.attentions), "| layer0 shape:", tuple(trace.attentions[0].shape))
    print("    n hidden     :", len(trace.hidden_states), "| logits shape:", tuple(trace.logits.shape))
    print("    attn_semantics:", trace.extras.get("attn_semantics"))

    # 4. analyzers -----------------------------------------------------
    attn = AttentionAnalyzer(top_k=5).run(model, "The Eiffel Tower is in")
    print("\n[4a] AttentionAnalyzer findings:", attn.findings)
    ent = TokenEntropyAnalyzer(top_k=5).run(model, "The Eiffel Tower is in")
    print("[4b] TokenEntropyAnalyzer findings:", ent.findings)

    # 5. agent with a real tool ---------------------------------------
    def multiply(a: float, b: float) -> float:
        return a * b

    tool = Tool(
        name="multiply",
        description="Multiply two numbers and return the product.",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
        fn=multiply,
    )
    agent = Agent(
        model,
        tools=[tool],
        max_turns=4,
        system="You are a calculator. Use the multiply tool for any multiplication.",
    )
    traj = agent.run("What is 21 multiplied by 2? Use the multiply tool.")
    print("\n[5] Agent trajectory:")
    for s in traj.steps:
        line = f"    [{s.idx}] {s.role.value}"
        if s.tool_call:
            line += f"  tool_call={s.tool_call['name']}({s.tool_call['args']})"
        if s.observation is not None:
            line += f"  obs={s.observation!r}"
        if s.content and s.role.value == "actor" and not s.tool_call:
            line += f"  text={s.content[:80]!r}"
        print(line)
    print("    final_answer:", repr(traj.final_answer))
    print("    metrics     :", traj.metrics)

    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
