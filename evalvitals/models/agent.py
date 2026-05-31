"""Agent — a backend-AGNOSTIC tool-calling loop over any handle.

``Agent(wraps=handle)`` works identically on an API model and a local model: the
loop only needs ``GENERATE`` + ``TOOL_CALLS`` (verified up front), never model
internals.  The single backend/model-specific piece is the
:class:`~evalvitals.models.toolcodec.ToolCallCodec` (auto-selected).  Tool
execution goes through a pluggable :class:`ToolExecutor` — swap in the existing
``APIToolHandler`` for production (image handling, etc.).

White-box backends additionally let an analyzer capture ONE step's internals via
``handle.forward(...)``, but the trajectory production here is the same loop.
Torch-free.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from evalvitals.core.capability import Capability, CapabilityError
from evalvitals.core.case import (
    FailureCase,
    Inputs,
    Label,
    Step,
    StepRole,
    Trajectory,
)
from evalvitals.core.tool import Tool, ToolCall
from evalvitals.models.toolcodec import ToolCallCodec, codec_for


class ToolExecutor:
    """Run a decoded :class:`ToolCall` against the registered tools.

    The simple default; production swaps in the existing ``APIToolHandler``
    (which also handles tool-output images, retries, etc.).
    """

    def __init__(self, tools: Iterable[Tool]) -> None:
        self._by_name = {t.name: t for t in tools}

    def execute(self, call: ToolCall) -> Any:
        tool = self._by_name.get(call.name)
        if tool is None:
            return f"[error: unknown tool {call.name!r}]"
        if tool.fn is None:
            return f"[error: tool {call.name!r} has no implementation]"
        try:
            return tool.fn(**(call.args or {}))
        except Exception as exc:  # surface tool errors to the model, don't crash the loop
            return f"[tool error in {call.name!r}: {exc}]"


def _as_case(data: Any) -> FailureCase:
    if isinstance(data, FailureCase):
        return data
    if isinstance(data, Inputs):
        return FailureCase(inputs=data)
    if isinstance(data, str):
        return FailureCase.from_prompt(data)
    raise TypeError(f"Agent.run expects str | Inputs | FailureCase, got {type(data).__name__}")


class Agent:
    """A tool-calling agent composed over a model handle (any backend)."""

    requires = frozenset({Capability.GENERATE, Capability.TOOL_CALLS})

    def __init__(
        self,
        handle,
        tools: Iterable[Tool],
        *,
        codec: Optional[ToolCallCodec] = None,
        executor: Optional[ToolExecutor] = None,
        max_turns: int = 10,
        system: Optional[str] = None,
    ) -> None:
        missing = self.requires - set(getattr(handle, "capabilities", frozenset()))
        if missing:
            raise CapabilityError(analyzer="Agent", model=repr(handle), missing=missing)
        if not hasattr(handle, "chat"):
            raise TypeError(
                f"{type(handle).__name__} has no chat(); agent mode needs a tool-aware chat method."
            )
        self.handle = handle
        self.tools = list(tools)
        self.codec = codec or codec_for(handle)
        self.executor = executor or ToolExecutor(self.tools)
        self.max_turns = max_turns
        self.system = system
        # An Agent still exposes the underlying model's capabilities (pure-model
        # analysis remains available on self.handle).
        self.capabilities = handle.capabilities

    def run(self, data: Any) -> Trajectory:
        """Drive the tool loop to completion and return a :class:`Trajectory`."""
        case = _as_case(data)
        goal = case.inputs.prompt
        encoded = self.codec.encode(self.tools)

        messages: list[dict] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": goal})

        steps: list[Step] = [Step(idx=0, role=StepRole.USER, content=goal)]
        final_answer: Optional[str] = None
        terminated = "max_turns"

        turn = 0
        for turn in range(1, self.max_turns + 1):
            chat = self.handle.chat(messages, tools=encoded)
            call = self.codec.decode(chat)
            steps.append(
                Step(
                    idx=len(steps),
                    role=StepRole.ACTOR,
                    content=chat.text,
                    tool_call=call.to_dict() if call else None,
                    span={"turn": turn},
                )
            )
            messages.append(self.codec.assistant_message(chat, call))

            if call is None:
                final_answer = chat.text
                terminated = "final"
                break

            observation = self.executor.execute(call)
            steps.append(
                Step(
                    idx=len(steps),
                    role=StepRole.TOOL,
                    content=call.name,
                    observation=observation,
                    span={"turn": turn},
                )
            )
            messages.append(self.codec.tool_message(call, observation))

        return Trajectory(
            sample_id=case.id,
            goal=goal,
            steps=steps,
            final_answer=final_answer,
            ground_truth=case.expected,
            outcome=Label.UNKNOWN,  # correctness is a separate analyzer's job
            metrics={
                "n_steps": len(steps),
                "n_turns": turn,
                "n_tool_calls": sum(1 for s in steps if s.tool_call),
                "terminated": terminated,
            },
        )

    def __repr__(self) -> str:
        return f"Agent(handle={self.handle!r}, tools={[t.name for t in self.tools]}, codec={self.codec.name})"
