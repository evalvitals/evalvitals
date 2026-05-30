"""Model — the abstract contract every analyzable model implements.

The key design move: a model exposes **one** rich forward pass,
``forward(inputs, capture={...}) -> Trace``, where ``capture`` names the
internals to record (attention, hidden states, …).  Analyzers consume the
``Trace``, not the model's framework-specific guts.  This decouples "run the
model and grab internals" from "interpret the internals", so one model serves
every compatible analyzer and one analyzer works across every compatible model.

The hybrid convenience API (``model.call_attention(prompt)``) is provided here
by ``__getattr__``: it resolves ``attention`` in the analyzer registry, checks
the model supports it, and delegates to ``AttentionAnalyzer().run(self, ...)``.
No mixins, no per-model wiring — the shim is derived from capabilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from evalvitals.core.capability import Capability

if TYPE_CHECKING:
    import torch


@dataclass
class Trace:
    """Captured internals from a single forward pass.

    Only the fields named in the ``capture`` set are populated; the rest stay
    ``None``.  ``provided`` records what was actually captured.
    """

    tokens: list[str]
    token_ids: list[int]
    provided: set[Capability] = field(default_factory=set)
    attentions: "list[torch.Tensor] | None" = None      # per layer: (heads, seq, seq)
    hidden_states: "list[torch.Tensor] | None" = None    # per layer: (seq, dim)
    logits: "torch.Tensor | None" = None                 # (seq, vocab)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def seq_len(self) -> int:
        return len(self.tokens)

    def require(self, capability: Capability) -> Any:
        """Return the field for *capability*, raising if it wasn't captured."""
        attr = {
            Capability.ATTENTION: "attentions",
            Capability.HIDDEN_STATES: "hidden_states",
            Capability.LOGITS: "logits",
        }.get(capability)
        value = getattr(self, attr) if attr else None
        if value is None:
            raise ValueError(
                f"Trace does not contain {capability.value!r}. "
                f"Was it included in the forward(capture=...) set? "
                f"Captured: {sorted(c.value for c in self.provided)}"
            )
        return value


class Model(ABC):
    """Abstract base for every analyzable model.

    Subclasses must:
      1. set the ``capabilities`` class attribute,
      2. implement :meth:`generate` and :meth:`forward`.

    They get ``call_<analyzer>(data, **kwargs)`` for free via :meth:`__getattr__`.
    """

    #: Capabilities this model provides. Override in subclasses.
    capabilities: frozenset[Capability] = frozenset()

    @abstractmethod
    def generate(self, inputs: Any, **kwargs) -> str:
        """Produce a text response for *inputs* (a prompt or :class:`Inputs`)."""

    @abstractmethod
    def forward(self, inputs: Any, capture: set[Capability]) -> Trace:
        """Run one forward pass, capturing the requested internals into a :class:`Trace`."""

    # ------------------------------------------------------------------
    # Capability checks
    # ------------------------------------------------------------------

    def supports(self, required: frozenset[Capability] | set[Capability]) -> bool:
        """True iff this model provides every capability in *required*."""
        return set(required).issubset(self.capabilities)

    # ------------------------------------------------------------------
    # Hybrid convenience shim:  model.call_<analyzer>(data, **kwargs)
    # ------------------------------------------------------------------

    def __getattr__(self, attr: str) -> Callable[..., Any]:
        # __getattr__ only fires for attributes not found normally, so this
        # never shadows real methods/fields.
        if attr.startswith("call_"):
            analyzer_name = attr[len("call_"):]
            from evalvitals.core.registry import registry

            if registry.analyzers.has(analyzer_name):
                analyzer_cls = registry.analyzers.get(analyzer_name)

                def _shim(data: Any, **kwargs) -> Any:
                    return analyzer_cls(**kwargs).run(self, data)

                _shim.__name__ = attr
                _shim.__doc__ = (
                    f"Convenience shim → {analyzer_cls.__name__}(**kwargs).run(self, data). "
                    f"Auto-derived from capabilities + registry."
                )
                return _shim

        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {attr!r}"
        )
