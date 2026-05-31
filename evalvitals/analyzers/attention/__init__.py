"""Attention analyzers (require the ATTENTION capability → white-box / HF eager).

NOTE: these import torch at module load; on the light (pure-API) install the whole
subpackage is skipped by ``analyzers/__init__`` (you can't run them without torch).
"""

from evalvitals.analyzers.attention.relative_attn import RelativeAttentionAnalyzer
from evalvitals.analyzers.attention.rollout import AttentionRolloutAnalyzer, RolloutResult
from evalvitals.analyzers.attention.sink import AttentionSinkAnalyzer
from evalvitals.analyzers.attention.summary import AttentionAnalyzer, AttentionResult

__all__ = [
    "AttentionAnalyzer",
    "AttentionResult",
    "AttentionRolloutAnalyzer",
    "RolloutResult",
    "AttentionSinkAnalyzer",
    "RelativeAttentionAnalyzer",
]
