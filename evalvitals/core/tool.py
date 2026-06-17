"""Tool / ToolCall / ChatTurn — the value types the agent loop speaks.

Backend-agnostic and torch-free.  A :class:`Tool` is defined ONCE (name +
JSON-schema params + a python callable) and works on every backend: its
``to_openai_schema()`` is accepted both by OpenAI-compatible APIs and by
transformers' ``apply_chat_template(tools=...)``.  A :class:`ChatTurn` is the
uniform transport from a model handle back to the codec/agent: ``text`` plus, for
native-tool-calling backends, the backend's raw structured ``tool_calls``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Tool:
    """A callable tool the agent may invoke.

    ``parameters`` is a JSON-schema object describing the args (the standard
    OpenAI ``function.parameters`` shape).
    """

    name: str
    description: str = ""
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}})
    fn: Optional[Callable[..., Any]] = None  # the actual implementation (executor calls it)

    def to_openai_schema(self) -> dict:
        """OpenAI tool schema — also accepted by HF ``apply_chat_template(tools=)``."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCall:
    """A normalised tool invocation decoded from a model's output."""

    name: str
    args: dict = field(default_factory=dict)
    id: Optional[str] = None

    def to_dict(self) -> dict:
        return {"name": self.name, "args": self.args, "id": self.id}


@dataclass
class ChatTurn:
    """One model turn in an agent loop: assistant text + optional native tool calls.

    ``raw_tool_calls`` holds the BACKEND-NATIVE structured calls (e.g. OpenAI's
    ``[{"id","function":{"name","arguments"}}]``) when the backend does native
    tool-calling; it is ``None`` for template-based backends, where the call is
    embedded in ``text`` and the codec parses it out.
    """

    text: str = ""
    raw_tool_calls: Optional[list] = None
    finish_reason: Optional[str] = None
