"""ToolCallCodec — the ONLY backend/model-specific piece of agent tool-calling.

The agent loop is identical across backends; what differs is how tools are
encoded into the request and how a tool call is decoded out of the model's
reply.  Two implementations:

  * :class:`OpenAIToolCodec` — native structured tool_calls (API / ``vllm serve``).
  * :class:`QwenToolCodec` — Hermes-style ``<tool_call>{...}</tool_call>`` text
    parsing (Qwen2.5/Qwen3 and many template-based local models).

``encode`` is shared (OpenAI tool schema, which HF ``apply_chat_template`` also
accepts).  Message-threading helpers let each codec put the assistant turn and
the tool result back into history in the form its backend expects.  Torch-free.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Optional

from evalvitals.core.tool import ChatTurn, Tool, ToolCall


class ToolCallCodec(ABC):
    """Encode tool schemas into a request; decode a tool call out of a reply."""

    name: str = "codec"

    def encode(self, tools: list[Tool]) -> list[dict]:
        """OpenAI tool schema — accepted by both APIs and HF apply_chat_template."""
        return [t.to_openai_schema() for t in tools]

    @abstractmethod
    def decode(self, turn: ChatTurn) -> Optional[ToolCall]:
        """Return the tool call in *turn*, or ``None`` if it's a final answer."""

    # -- message threading (sensible defaults; OpenAI overrides) -------
    def assistant_message(self, turn: ChatTurn, call: Optional[ToolCall]) -> dict:
        return {"role": "assistant", "content": turn.text}

    def tool_message(self, call: ToolCall, result) -> dict:
        return {"role": "tool", "content": str(result)}


class OpenAIToolCodec(ToolCallCodec):
    """Native OpenAI-compatible tool calling (structured tool_calls)."""

    name = "openai"

    def decode(self, turn: ChatTurn) -> Optional[ToolCall]:
        if not turn.raw_tool_calls:
            return None
        tc = turn.raw_tool_calls[0]
        fn = tc.get("function", tc)
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {"_raw": args}
        return ToolCall(name=fn.get("name", ""), args=args or {}, id=tc.get("id"))

    def assistant_message(self, turn: ChatTurn, call: Optional[ToolCall]) -> dict:
        msg: dict = {"role": "assistant", "content": turn.text or None}
        if turn.raw_tool_calls:
            msg["tool_calls"] = turn.raw_tool_calls
        return msg

    def tool_message(self, call: ToolCall, result) -> dict:
        return {"role": "tool", "tool_call_id": call.id or "call_0", "content": str(result)}


class QwenToolCodec(ToolCallCodec):
    """Hermes-style ``<tool_call>{json}</tool_call>`` parsing (Qwen2.5 / Qwen3 / many locals).

    Note: other families differ (Llama 3.1 uses a different convention) — add a
    sibling codec and route it in :func:`codec_for` when you onboard them.
    """

    name = "qwen"
    _PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

    def decode(self, turn: ChatTurn) -> Optional[ToolCall]:
        m = self._PATTERN.search(turn.text or "")
        if not m:
            return None
        try:
            payload = json.loads(m.group(1))
        except json.JSONDecodeError:
            return None
        args = payload.get("arguments", payload.get("args", {})) or {}
        return ToolCall(name=payload.get("name", ""), args=args)


# family -> local codec (default Hermes/Qwen style covers most template models)
_LOCAL_CODECS = {
    "qwen2": QwenToolCodec,
    "qwen3": QwenToolCodec,
    "qwen3_moe": QwenToolCodec,
    "qwen3_vl": QwenToolCodec,
    "qwen3_vl_moe": QwenToolCodec,
}


def codec_for(handle) -> ToolCallCodec:
    """Pick a codec for *handle*: OpenAI for API, family-routed for local."""
    from evalvitals.models.backends.api import APIModel

    if isinstance(handle, APIModel):
        return OpenAIToolCodec()
    spec = getattr(handle, "spec", None)
    family = getattr(spec, "family", "")
    return _LOCAL_CODECS.get(family, QwenToolCodec)()
