"""White-box (local) model entry points — per-version convenience factories.

Identity lives in ``evalvitals.specs``; these are thin wrappers over
``compose(spec, "hf_local")``.  Import a specific version factory, e.g.::

    from evalvitals.models.whitebox import qwen3_8b, qwen3_vl_8b_instruct
"""

from evalvitals.models.whitebox import qwen as _qwen
from evalvitals.models.whitebox import qwen_vl as _qwen_vl
from evalvitals.models.whitebox.qwen import *  # noqa: F401,F403  (QwenLLM + qwen text factories)
from evalvitals.models.whitebox.qwen_vl import *  # noqa: F401,F403  (QwenVL + qwen-vl factories)

__all__ = sorted(set(_qwen.__all__) | set(_qwen_vl.__all__))
