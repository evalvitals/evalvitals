"""Gemini API model — black-box GENERATE via Google's Gen AI SDK.

Install the optional dependency::

    pip install evalvitals[gemini]

Then use as the judge in DiagnosisAgent::

    from evalvitals.models.blackbox.gemini import GeminiModel
    from evalvitals.eval_agent import DiagnosisAgent

    judge = GeminiModel(api_key="AIza...")   # or set GEMINI_API_KEY env var
    agent = DiagnosisAgent(judge=judge)

``GeminiModel`` is also the default judge when ``DiagnosisAgent`` is created
without an explicit judge and ``GEMINI_API_KEY`` is in the environment.
"""

from __future__ import annotations

import os
from typing import Any

from evalvitals.core.capability import Capability
from evalvitals.models.blackbox.base import BlackboxModel, Trace


class GeminiModel(BlackboxModel):
    """Black-box wrapper around the Google Gemini API.

    Args:
        model_id:  Gemini model name (default: ``"gemini-2.0-flash"``).
        api_key:   API key.  Falls back to the ``GEMINI_API_KEY`` environment
                   variable when ``None``.

    Requires ``pip install google-genai`` (``evalvitals[gemini]``).
    """

    capabilities = frozenset({Capability.GENERATE})
    modalities   = frozenset({"text", "image"})

    def __init__(
        self,
        model_id: str = "gemini-2.0-flash",
        api_key: str | None = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            api_key=api_key or os.getenv("GEMINI_API_KEY"),
        )

    def generate(self, inputs: Any, **kwargs) -> str:
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "GeminiModel requires the google-genai package. "
                "Install it with: pip install 'evalvitals[gemini]'"
            ) from exc

        if not self.api_key:
            raise ValueError(
                "No Gemini API key found. Pass api_key= or set GEMINI_API_KEY."
            )

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model_id,
            contents=str(inputs),
        )
        return response.text

    def forward(self, inputs: Any, capture: set[Capability], spec: Any = None) -> Trace:  # type: ignore[override]
        raise NotImplementedError(
            "GeminiModel is a black-box model; only generate() is available."
        )

    def __repr__(self) -> str:
        return f"GeminiModel(model_id={self.model_id!r})"
