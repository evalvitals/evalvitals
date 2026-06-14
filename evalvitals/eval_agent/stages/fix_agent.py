"""Fix module — tiered, validated repair attempts after the diagnosis loop.

Design (intervention-space tiers, see :mod:`fix_tiers`): the allowed tier is
an **input** (default L2).  The agent proposes candidate fixes inside the
allowed tiers, compiles every candidate to the same shape — a per-case success
function, exactly :mod:`~evalvitals.eval_agent.ab_runner`'s *strategy*
contract — and validates each against the unmodified baseline with the paired
machinery from :mod:`evalvitals.stats` (McNemar + e-value, never a bare p).

There is **no automatic escalation**: when no candidate within the allowed
tier validates, the outcome carries a *recommendation* to raise the tier,
routed from the verified hypotheses' mechanisms (:func:`route_min_tier`).

Executors by tier:

* **L1** — prompt transforms (judge-proposed templates).
* **L2 declarative** — catalog-tool pipelines (:mod:`fix_tools`): cheap,
  deterministic, validated first.
* **L2 coded** — the coding agent (CLI agent first, judge fallback) writes a
  brand-new pipeline as Python: multiple model calls per case, branching on
  intermediate outputs — only the model itself is unchanged.  The code runs
  sandboxed with bridged model access (:mod:`fix_pipeline`); labels and
  rubrics never reach it, so it cannot cheat by echoing gold answers.
* **L3a** — internals read (:mod:`fix_internals`): attention-guided crop
  (capture host-side, crop at the attention peak, re-ask); coded pipelines
  additionally get a bridged ``model_attend()`` when the tier allows.
* **L3b** — internals write (:mod:`fix_internals`): pre-audited intervention
  primitives (v1: visual embedding boost via a forward hook) — the judge
  selects and parameterises; never free codegen against the model handle.
* **L4** — parameter space: **defined, executor TODO** — the judge writes a
  :class:`~.fix_internals.FinetuneSpec` recipe which is recorded (never
  executed) so an escalation decision has something concrete to act on.

A *fixed* verdict means: paired McNemar rejects with positive net effect —
the candidate repairs significantly more cases than it breaks.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from evalvitals.analyzers.perturbation.prompt_contrast import _default_score
from evalvitals.eval_agent.stages.fix_internals import (
    INTERNALS_PRIMITIVES,
    FinetuneSpec,
    primitives_catalog_text,
)
from evalvitals.eval_agent.stages.fix_pipeline import (
    CodedPipelineResult,
    run_coded_pipeline,
    score_outputs,
)
from evalvitals.eval_agent.stages.fix_tiers import FixTier, parse_tier, route_min_tier
from evalvitals.eval_agent.stages.fix_tools import (
    PipelineSpec,
    catalog_text,
    run_pipeline,
)
from evalvitals.eval_agent.stages.probe_generator import _extract_code
from evalvitals.stats import compare

if TYPE_CHECKING:
    from evalvitals.core.case import CaseBatch, FailureCase
    from evalvitals.core.model import Model
    from evalvitals.eval_agent.cli_agent import CliAgentConfig
    from evalvitals.eval_agent.hypothesis import Hypothesis

logger = logging.getLogger(__name__)

_MAX_JUDGE_CANDIDATES = 3
_EXAMPLE_PROMPTS = 3

_L1_PROMPT = """\
You are designing PROMPT-LEVEL fixes (tier L1: the input space only) for a \
vision-language model failure.

VERIFIED FAILURE HYPOTHESES:
{hypotheses}

EXAMPLE FAILING PROMPTS:
{examples}

Propose up to {k} prompt rewrite strategies that could repair these failures
WITHOUT changing the model or adding pipeline steps.  Each strategy is a
template applied to every case prompt; it MUST contain the literal placeholder
{{prompt}}.

Reply with ONLY a JSON array:
[{{"name": "<short_snake_case>", "prompt_template": "<template with {{prompt}}>"}}]"""

_L2_PROMPT = """\
You are designing SCAFFOLD-LEVEL fixes (tier L2: a pipeline around the \
unchanged model) for a vision-language model failure.

VERIFIED FAILURE HYPOTHESES:
{hypotheses}

EXAMPLE FAILING PROMPTS:
{examples}

AVAILABLE IMAGE TOOLS (applied to the case image before the model sees it):
{catalog}

Propose up to {k} pipelines.  Each may chain image tools, rewrite the prompt
(template MUST contain {{prompt}}), and sample the model n_samples times
(majority vote).  Reply with ONLY a JSON array:
[{{"name": "<short_snake_case>",
   "image_ops": [{{"tool": "<catalog name>", "params": {{}}}}],
   "prompt_template": "{{prompt}}", "n_samples": 1}}]"""

_L2_CODE_PROMPT = """\
You are writing a PYTHON PIPELINE (tier L2: a scaffold around the unchanged \
vision-language model) that repairs the failures described below.  Design any
pipeline you want — the only constraint is that the model itself is unchanged.

VERIFIED FAILURE HYPOTHESES:
{hypotheses}

EXAMPLE FAILING PROMPTS:
{examples}

EXECUTION CONTRACT:
- "{cases_file}" in the current directory: {{"cases": [{{"id": str, "prompt": str}}]}}
- A function  model_generate(case_id, prompt=None, image_ops=None) -> str  is
  ALREADY DEFINED in your namespace (do NOT import or redefine it).  It runs
  the ORIGINAL model on that case: optional prompt override, optional image
  transforms applied to the case's image first.  image_ops MUST be a list of
  {{"tool": "<name>", "params": {{...}}}} dicts using ONLY these tools
  (anything else is rejected with an error):
{catalog}{attend_hint}
- You may call the model SEVERAL times per case (budget ~6 calls/case) and
  branch on its outputs — e.g. ask where the finding could be, zoom there,
  re-ask; describe first, then decide; vote over variants.
- The LAST line of stdout MUST be exactly:
  {marker}{{"per_case": [{{"sample_id": "<case id>", "output": "<final answer text>"}}]}}
- Emit an entry for EVERY case.  The "output" is scored externally against the
  original question, so it must answer that question faithfully (e.g. contain
  a clear yes/no for yes/no questions).
- Standard library + numpy only.  No network, no file writes.  Keep it under
  ~80 lines.

Return ONLY the Python code{fences_hint}."""

_REPAIR_PROMPT_BODY = """\
Your previously written repair pipeline FAILED TO EXECUTE.

ERROR:
{error}

YOUR PREVIOUS CODE:
```python
{code}
```

Fix the code.  Follow the execution contract EXACTLY:
- the ONLY model access is the predefined model_generate(case_id, prompt=None, \
image_ops=None){attend_clause} — do not import or redefine it;
- image_ops must be a list of {{"tool": "<name>", "params": {{...}}}} dicts \
using ONLY these tools:
{catalog}
- read "{cases_file}", emit an entry for EVERY case, and end stdout with \
exactly:
  {marker}{{"per_case": [{{"sample_id": "<case id>", "output": "<final answer text>"}}]}}
- standard library + numpy only; no network, no file writes; under ~80 lines.
"""

_L3_PROMPT = """\
You are configuring WHITE-BOX intervention primitives (tier L3: the model's \
internals) against the failures below.  The primitives are pre-audited host \
code — you choose which to run and with what parameters.

VERIFIED FAILURE HYPOTHESES:
{hypotheses}

AVAILABLE PRIMITIVES:
{catalog}

Propose up to {k} configurations.  Reply with ONLY a JSON array:
[{{"primitive": "<name from the list>", "params": {{...}}}}]"""

_L4_PROMPT = """\
You are writing a PARAMETER-SPACE repair recipe (tier L4: fine-tuning) for \
the failures below.  The recipe is RECORDED for a human decision — it will \
not be executed automatically.

VERIFIED FAILURE HYPOTHESES:
{hypotheses}

Reply with ONLY a JSON object:
{{"dataset_recipe": "<how to build training data that generalises the failure
   mechanism — never just the observed failing cases>",
  "method": "lora|sft", "target": "vision_encoder|llm|projector|full",
  "eval_protocol": "<held-out repair effect + regression battery>",
  "rationale": "<why parameter-space change is the minimum effective tier>"}}"""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FixCandidate:
    """One proposed repair, compiled later to an ab_runner-style strategy.

    Attributes:
        tier:        Intervention space the candidate lives in.
        name:        Short identifier.
        kind:        ``"template"`` (L1) | ``"spec"`` (L2 declarative) |
                     ``"code"`` (L2 agent-written pipeline).
        payload:     Kind-specific — template: ``{"prompt_template": ...}``;
                     spec: a :class:`~.fix_tools.PipelineSpec` dict;
                     code: ``{"code": "<python source>"}``.
        source:      ``"judge"``, ``"cli:<provider>"`` or ``"default"``.
    """

    tier: FixTier
    name: str
    payload: "dict[str, Any]"
    kind: str = "spec"
    source: str = "judge"


@dataclass
class FixValidation:
    """Paired validation of one candidate against the unmodified baseline."""

    candidate: FixCandidate
    n_pairs: int = 0
    n_fixed: int = 0
    n_broken: int = 0
    fixed_cases: "list[str]" = field(default_factory=list)
    broken_cases: "list[str]" = field(default_factory=list)
    effect: "float | None" = None
    reject: bool = False
    fixed: bool = False
    summary: str = ""
    # Non-empty when the candidate never EXECUTED (sandbox crash, timeout,
    # bridge contract violation) — distinct from "executed and not effective".
    # Escalation must not treat these as evidence that the tier is exhausted.
    exec_error: str = ""


@dataclass
class FixOutcome:
    """Everything the fix module did, plus the escalation recommendation.

    ``recommendation`` is ``None`` when a candidate validated; otherwise
    ``{"recommend_tier": "L3a", "reason": ...}`` — the caller decides whether
    to re-run with a higher ``max_tier`` (never automatic).
    """

    max_tier: FixTier
    routed: "list[dict[str, str]]" = field(default_factory=list)
    attempted: "list[FixValidation]" = field(default_factory=list)
    best: "FixValidation | None" = None
    fixed: bool = False
    recommendation: "dict[str, Any] | None" = None

    def to_dict(self) -> "dict[str, Any]":
        return {
            "max_tier": self.max_tier.label,
            "routed": self.routed,
            "attempted": [
                {
                    "tier": v.candidate.tier.label,
                    "name": v.candidate.name,
                    "kind": v.candidate.kind,
                    "source": v.candidate.source,
                    "payload": v.candidate.payload,
                    "n_pairs": v.n_pairs,
                    "n_fixed": v.n_fixed,
                    "n_broken": v.n_broken,
                    "fixed_cases": v.fixed_cases,
                    "broken_cases": v.broken_cases,
                    "effect": v.effect,
                    "reject": v.reject,
                    "fixed": v.fixed,
                    "summary": v.summary,
                }
                for v in self.attempted
            ],
            "best": self.best.candidate.name if self.best else None,
            "fixed": self.fixed,
            "recommendation": self.recommendation,
        }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class FixAgent:
    """Propose and validate tiered fixes for the loop's verified hypotheses.

    Args:
        judge:            LLM proposing candidates (deterministic defaults
                          when ``None`` or unparseable).
        max_tier:         Highest allowed intervention tier (input, default L2).
        score_fn:         ``(case, output) -> bool | None``; defaults to the
                          rubric scorer shared with prompt_contrast.
        run_logger:       Optional RunLogger — records the outcome as a
                          ``fix`` event and coded pipelines as ``tool_codegen``.
        cli_config:       CLI coding-agent config; when set (provider != "llm")
                          it writes the L2 coded pipeline (judge fallback).
        allow_codegen:    Gate for the L2 coded-pipeline path (sandboxed,
                          bridged model access).  Declarative candidates do
                          not depend on this.
        sandbox:          Workdir provider for coded pipelines (fresh temp
                          dir when ``None``).
        exec_timeout_sec: Wall-clock limit for one coded-pipeline session
                          (includes the bridged model calls).
        max_validation_cases: When > 0 and the batch is larger, validate every
                          candidate on a label-stratified subset of this size
                          (all-FAIL-first; deterministic).  Every candidate
                          validation costs >= one model call per case, so an
                          unbounded batch makes coded pipelines time out.
    """

    def __init__(
        self,
        judge: "Model | None" = None,
        max_tier: "str | FixTier" = FixTier.L2_SCAFFOLD,
        score_fn: "Callable[[FailureCase, str], Optional[bool]] | None" = None,
        run_logger: "Any | None" = None,
        cli_config: "CliAgentConfig | None" = None,
        allow_codegen: bool = True,
        sandbox: "Any | None" = None,
        exec_timeout_sec: int = 600,
        max_validation_cases: int = 0,
    ) -> None:
        self._judge = judge
        self.max_tier = parse_tier(max_tier)
        self._score = score_fn or _default_score
        self.run_logger = run_logger
        self._cli_config = cli_config
        self._allow_codegen = allow_codegen
        self._sandbox = sandbox
        self._exec_timeout_sec = exec_timeout_sec
        self.max_validation_cases = max_validation_cases
        self._last_repair_prompt = ""
        self._last_usage: dict | None = None

    @property
    def codegen_available(self) -> bool:
        """True when the L2 coded-pipeline path has a code-writing backend."""
        return self._allow_codegen and (
            self._judge is not None
            or (self._cli_config is not None and self._cli_config.provider != "llm")
        )

    # -- public ---------------------------------------------------------

    def propose_and_validate(
        self,
        model: "Model",
        data: "CaseBatch",
        hypotheses: "list[Hypothesis]",
        prior_attempts: "list[FixValidation] | None" = None,
    ) -> FixOutcome:
        """Generate candidates within the allowed tiers, validate, recommend."""
        prior_text = self._format_prior(prior_attempts) if prior_attempts else ""
        prior_names: "frozenset[str]" = frozenset(
            v.candidate.name for v in (prior_attempts or [])
        )
        outcome = FixOutcome(max_tier=self.max_tier)
        routed_tiers: "list[FixTier]" = []
        for h in hypotheses:
            tier, why = route_min_tier(h)
            routed_tiers.append(tier)
            outcome.routed.append({
                "hypothesis": getattr(h, "statement", str(h))[:160],
                "min_tier": tier.label,
                "rationale": why,
            })

        data = self._validation_subset(data)
        baseline = self._baseline(model, data)
        if not any(v is not None for v in baseline.values()):
            logger.warning("FixAgent: no scorable case (no rubrics); nothing to validate")
            outcome.recommendation = self._recommend(routed_tiers, reason_prefix=(
                "no case carries a scoring rubric, so no fix can be validated"
            ))
            self._emit(outcome)
            return outcome

        for candidate in self._propose(hypotheses, data, model, prior_text, prior_names):
            validation = self._validate(candidate, model, data, baseline)
            outcome.attempted.append(validation)

        winners = [v for v in outcome.attempted if v.fixed]
        if winners:
            outcome.best = max(winners, key=lambda v: (v.effect or 0.0, -v.n_broken))
            outcome.fixed = True
        else:
            # Escalation needs the current tier to have been genuinely TRIED:
            # a candidate that never executed (sandbox crash/timeout/contract
            # violation) is an engineering failure, not evidence the tier is
            # exhausted.
            executed = [v for v in outcome.attempted if v.n_pairs > 0]
            never_ran = [v for v in outcome.attempted if v.n_pairs == 0]
            if never_ran and not executed:
                outcome.recommendation = {
                    "recommend_tier": self.max_tier.label,
                    "reason": (
                        "no candidate EXECUTED — escalating would be premature; "
                        f"fix candidate execution and retry within {self.max_tier.label}. "
                        "Failures: " + "; ".join(
                            f"{v.candidate.name}: {(v.exec_error or v.summary)[:120]}"
                            for v in never_ran[:3])
                    ),
                }
            else:
                outcome.recommendation = self._recommend(routed_tiers)
                if never_ran and outcome.recommendation is not None:
                    outcome.recommendation["reason"] += (
                        f" (caveat: {len(never_ran)} candidate(s) never executed: "
                        + ", ".join(v.candidate.name for v in never_ran[:3]) + ")")
        self._emit(outcome)
        return outcome

    def _validation_subset(self, data: "CaseBatch") -> "CaseBatch":
        """Label-stratified, deterministic subset for candidate validation."""
        cap = self.max_validation_cases
        if not cap or len(data) <= cap:
            return data
        import random as _random

        from evalvitals.core.case import CaseBatch, Label

        rng = _random.Random(0)
        fails = [c for c in data if c.label == Label.FAIL]
        passes = [c for c in data if c.label != Label.FAIL]
        rng.shuffle(fails)
        rng.shuffle(passes)
        n_fail = min(len(fails), max(cap // 2, cap - len(passes)))
        keep = fails[:n_fail] + passes[: cap - n_fail]
        logger.info("FixAgent: validating on %d/%d cases (%d fail, %d pass)",
                    len(keep), len(data), n_fail, len(keep) - n_fail)
        return CaseBatch(keep)

    # -- candidate generation --------------------------------------------

    def _propose(
        self,
        hypotheses: "list[Hypothesis]",
        data: "CaseBatch",
        model: "Model",
        prior_text: str = "",
        prior_names: "frozenset[str]" = frozenset(),
    ) -> "list[FixCandidate]":
        hyp_lines = "\n".join(
            f"- [{getattr(h, 'predicted_failure_mode', '')}] {getattr(h, 'statement', h)}"
            for h in hypotheses
        ) or "- (no verified hypotheses; failures are unexplained)"
        examples = "\n".join(
            f"- {str(getattr(c.inputs, 'prompt', ''))[:160]}"
            for c in list(data)
            if getattr(getattr(c, "label", None), "value", None) == "fail"
        )[: 1000] or "- (none)"

        candidates = self._l1_candidates(hyp_lines, examples, prior_text, prior_names)
        if self.max_tier >= FixTier.L2_SCAFFOLD:
            candidates += self._l2_candidates(hyp_lines, examples, prior_text, prior_names)
            if self.codegen_available:
                candidates += self._l2_coded_candidate(hyp_lines, examples, model, prior_text)
        if self.max_tier >= FixTier.L3A_INTERNALS_READ:
            candidates += self._l3_candidates(hyp_lines, model, prior_text, prior_names)
        if self.max_tier >= FixTier.L4_PARAMETERS:
            candidates += self._l4_candidates(hyp_lines)
        return candidates

    def _l1_candidates(
        self,
        hyp_lines: str,
        examples: str,
        prior_text: str = "",
        prior_names: "frozenset[str]" = frozenset(),
    ) -> "list[FixCandidate]":
        proposals = self._ask_judge(_L1_PROMPT.format(
            hypotheses=hyp_lines, examples=examples, k=_MAX_JUDGE_CANDIDATES) + prior_text)
        out: "list[FixCandidate]" = []
        for p in proposals:
            template = str(p.get("prompt_template", ""))
            name = str(p.get("name", "")).strip()
            if name and "{prompt}" in template:
                out.append(FixCandidate(
                    tier=FixTier.L1_PROMPT, name=name, kind="template",
                    payload={"prompt_template": template}))
        if not out and "attend_carefully" not in prior_names:
            out = [FixCandidate(
                tier=FixTier.L1_PROMPT, name="attend_carefully", kind="template",
                source="default",
                payload={"prompt_template": (
                    "Examine the image carefully, including small, subtle and "
                    "low-contrast regions, before answering. {prompt}")})]
        return out[:_MAX_JUDGE_CANDIDATES]

    def _l2_candidates(
        self,
        hyp_lines: str,
        examples: str,
        prior_text: str = "",
        prior_names: "frozenset[str]" = frozenset(),
    ) -> "list[FixCandidate]":
        proposals = self._ask_judge(_L2_PROMPT.format(
            hypotheses=hyp_lines, examples=examples, k=_MAX_JUDGE_CANDIDATES,
            catalog=catalog_text()) + prior_text)
        out: "list[FixCandidate]" = []
        for p in proposals:
            spec = PipelineSpec.from_dict(p) if isinstance(p, dict) else None
            if spec is not None:
                out.append(FixCandidate(
                    tier=FixTier.L2_SCAFFOLD, name=spec.name, payload=spec.to_dict()))
        if not out:
            defaults = [
                FixCandidate(
                    tier=FixTier.L2_SCAFFOLD, name="zoom_equalize", source="default",
                    payload=PipelineSpec(
                        name="zoom_equalize",
                        image_ops=[{"tool": "zoom_center", "params": {"factor": 1.6}},
                                   {"tool": "equalize", "params": {}}]).to_dict()),
                FixCandidate(
                    tier=FixTier.L2_SCAFFOLD, name="upscale_sharpen", source="default",
                    payload=PipelineSpec(
                        name="upscale_sharpen",
                        image_ops=[{"tool": "upscale", "params": {"factor": 2.0}},
                                   {"tool": "sharpen", "params": {"factor": 2.0}}]).to_dict()),
            ]
            out = [c for c in defaults if c.name not in prior_names]
        return out[:_MAX_JUDGE_CANDIDATES]

    def _l2_coded_candidate(
        self, hyp_lines: str, examples: str, model: "Model", prior_text: str = ""
    ) -> "list[FixCandidate]":
        """The coding agent writes a brand-new pipeline (CLI first, judge fallback)."""
        from evalvitals.core.capability import Capability
        from evalvitals.eval_agent.stages.fix_pipeline import (
            CASES_FILENAME,
            RESULT_MARKER,
        )

        enable_attend = (
            self.max_tier >= FixTier.L3A_INTERNALS_READ
            and Capability.ATTENTION in getattr(model, "capabilities", frozenset())
        )
        attend_hint = (
            "\n- A function  model_attend(case_id, prompt=None) -> "
            '{\"grid\": [[float,...],...], \"shape\": [H, W]}  is ALSO defined: '
            "the model's attention heatmap over image patches (read-only "
            "internals). Use it e.g. to find where the model looks, then "
            "crop_region there and re-ask."
        ) if enable_attend else ""
        code, source, prompt, raw = "", "", "", ""
        base = dict(hypotheses=hyp_lines, examples=examples, catalog=catalog_text(),
                    cases_file=CASES_FILENAME, marker=RESULT_MARKER,
                    attend_hint=attend_hint)
        if self._cli_config is not None and self._cli_config.provider != "llm":
            prompt = _L2_CODE_PROMPT.format(
                fences_hint=", written to a file named pipeline.py", **base) + prior_text
            code, raw = self._write_code_cli(prompt)
            source = f"cli:{self._cli_config.provider}"
        if not code.strip() and self._judge is not None:
            prompt = _L2_CODE_PROMPT.format(
                fences_hint=" inside a ```python code block", **base) + prior_text
            try:
                raw = str(self._judge.generate(prompt))
            except Exception as exc:
                logger.warning("FixAgent: code-writing judge call failed: %s", exc)
                raw = ""
            code = _extract_code(raw)
            # Syntax gate: a judge that answered in prose must not become a
            # "coded pipeline" candidate (the CLI path returns real files).
            if code.strip():
                import ast

                try:
                    ast.parse(code)
                except SyntaxError:
                    logger.warning("FixAgent: judge code failed to parse; dropped")
                    code = ""
            source = "judge"
        self._emit_codegen("coded_pipeline", prompt, source, code, raw,
                           ok=bool(code.strip()))
        if not code.strip():
            return []
        tier = FixTier.L3A_INTERNALS_READ if enable_attend else FixTier.L2_SCAFFOLD
        return [FixCandidate(tier=tier, name="coded_pipeline",
                             kind="code",
                             payload={"code": code, "enable_attend": enable_attend},
                             source=source)]

    def _write_code_cli(self, prompt: str) -> "tuple[str, str]":
        from pathlib import Path

        from evalvitals.eval_agent.cli_agent import create_cli_agent

        workdir = Path(self._workdir())
        agent = create_cli_agent(self._cli_config)  # type: ignore[arg-type]
        res = agent.run(prompt, workdir=workdir,
                        timeout_sec=self._cli_config.timeout_sec)  # type: ignore[union-attr]
        self._last_usage = res.usage
        if not res.ok:
            return "", res.raw_output
        py_files = {n: c for n, c in res.files.items() if n.endswith(".py")}
        if "pipeline.py" in py_files:
            return py_files["pipeline.py"], res.raw_output
        return (max(py_files.values(), key=len) if py_files else ""), res.raw_output

    def _workdir(self) -> str:
        if self._sandbox is None:
            from evalvitals.eval_agent.sandbox import ExperimentSandbox

            self._sandbox = ExperimentSandbox()
        return str(self._sandbox.workdir)

    @staticmethod
    def _format_prior(attempts: "list[FixValidation]") -> str:
        """Format failed prior attempts as a context block for judge prompts."""
        items = []
        for v in attempts:
            c = v.candidate
            if c.kind == "finetune_spec":
                continue
            effect = f"effect={v.effect:+.2f}" if v.effect is not None else "did not execute"
            broken = (f", broke {v.broken_cases[:3]}" if v.broken_cases else "")
            items.append(
                f"- [{c.tier.label}/{c.kind}] {c.name}: "
                f"{v.n_fixed} fixed / {v.n_broken} broken ({effect}{broken})"
            )
        if not items:
            return ""
        return (
            "\n\nPRIOR ATTEMPTS THAT DID NOT WORK — reason from these failures "
            "and design something FUNDAMENTALLY DIFFERENT (different mechanism, "
            "not just different parameters):\n" + "\n".join(items)
        )

    def _emit_codegen(
        self, name: str, prompt: str, source: str, code: str, raw: str, *, ok: bool
    ) -> None:
        if self.run_logger is None:
            return
        extra = ({"cli_usage": self._last_usage}
                 if source.startswith("cli:") and self._last_usage else None)
        try:
            self.run_logger.log_tool_codegen(
                module="fix_pipeline", name=name, need="L2 coded repair pipeline",
                source=source, ok=ok, code=code, prompt=prompt, raw_output=raw,
                error="" if ok else "no code produced", extra=extra,
            )
        except Exception as exc:  # logging must never break the fix step
            logger.debug("FixAgent: log_tool_codegen failed: %s", exc)

    def _l3_candidates(
        self,
        hyp_lines: str,
        model: "Model",
        prior_text: str = "",
        prior_names: "frozenset[str]" = frozenset(),
    ) -> "list[FixCandidate]":
        """Judge-parameterised configs of the pre-audited internals primitives."""
        catalog = primitives_catalog_text(model, self.max_tier)
        if not catalog:
            logger.info("FixAgent: no L3 primitive is available for %r", model)
            return []
        out: "list[FixCandidate]" = []
        for p in self._ask_judge(_L3_PROMPT.format(
                hypotheses=hyp_lines, catalog=catalog, k=_MAX_JUDGE_CANDIDATES) + prior_text):
            prim = INTERNALS_PRIMITIVES.get(str(p.get("primitive", "")))
            if prim is None or prim.tier > self.max_tier or not prim.available(model):
                continue
            out.append(FixCandidate(
                tier=prim.tier, name=prim.name, kind="primitive",
                payload={"primitive": prim.name, "params": dict(p.get("params") or {})}))
        if not out:
            defaults = {
                "attention_guided_crop": {"layer": -1, "crop_frac": 0.5},
                "visual_embedding_boost": {"gamma": 1.5},
            }
            for name, params in defaults.items():
                if name in prior_names:
                    continue
                prim = INTERNALS_PRIMITIVES[name]
                if prim.tier <= self.max_tier and prim.available(model):
                    out.append(FixCandidate(
                        tier=prim.tier, name=name, kind="primitive", source="default",
                        payload={"primitive": name, "params": params}))
        return out[:_MAX_JUDGE_CANDIDATES]

    def _l4_candidates(self, hyp_lines: str) -> "list[FixCandidate]":
        """L4 recipe — recorded for the escalation decision; executor is TODO."""
        spec: "FinetuneSpec | None" = None
        if self._judge is not None:
            raw = self._ask_judge_object(_L4_PROMPT.format(hypotheses=hyp_lines))
            if raw:
                spec = FinetuneSpec(
                    dataset_recipe=str(raw.get("dataset_recipe", "")),
                    method=str(raw.get("method", "lora")),
                    target=str(raw.get("target", "llm")),
                    eval_protocol=str(raw.get("eval_protocol", "")) or
                                  FinetuneSpec("").eval_protocol,
                    rationale=str(raw.get("rationale", "")),
                )
        if spec is None or not spec.dataset_recipe:
            spec = FinetuneSpec(
                dataset_recipe="TODO: synthesise training data generalising the "
                               "verified failure mechanism",
                rationale="default skeleton — no judge recipe available",
            )
        return [FixCandidate(tier=FixTier.L4_PARAMETERS, name="finetune_recipe",
                             kind="finetune_spec", payload=spec.to_dict(),
                             source="judge" if self._judge is not None else "default")]

    def _ask_judge_object(self, prompt: str) -> "dict[str, Any]":
        """Single-JSON-object variant of :meth:`_ask_judge`."""
        if self._judge is None:
            return {}
        try:
            raw = str(self._judge.generate(prompt))
        except Exception as exc:
            logger.warning("FixAgent: judge call failed: %s", exc)
            return {}
        match = re.search(r"\{.*\}", re.sub(r"<think>.*?</think>", "", raw,
                                              flags=re.DOTALL), flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _ask_judge(self, prompt: str) -> "list[dict[str, Any]]":
        if self._judge is None:
            return []
        try:
            raw = str(self._judge.generate(prompt))
        except Exception as exc:
            logger.warning("FixAgent: judge call failed: %s", exc)
            return []
        match = re.search(r"\[.*\]", re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL),
                          flags=re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("FixAgent: unparseable judge proposal; using defaults")
            return []
        return [p for p in parsed if isinstance(p, dict)] if isinstance(parsed, list) else []

    # -- strategy compilation + validation --------------------------------

    def _baseline(self, model: "Model", data: "CaseBatch") -> "dict[str, Optional[bool]]":
        scores: "dict[str, Optional[bool]]" = {}
        for case in data:
            try:
                output = str(model.generate(case.inputs))
            except Exception as exc:
                logger.debug("FixAgent: baseline generate failed on %s: %s", case.id, exc)
                scores[case.id] = None
                continue
            scores[case.id] = self._score(case, output)
        return scores

    def _strategy(self, candidate: FixCandidate) -> "Callable[[Model, FailureCase], Optional[bool]]":
        """Compile a candidate to a per-case success function (ab_runner shape)."""
        if candidate.kind == "template":
            template = candidate.payload["prompt_template"]

            def l1(model: "Model", case: "FailureCase") -> "Optional[bool]":
                from evalvitals.core.case import Inputs

                inp = case.inputs
                new_inputs = Inputs(
                    prompt=template.format(prompt=str(getattr(inp, "prompt", ""))),
                    image=getattr(inp, "image", None))
                try:
                    return self._score(case, str(model.generate(new_inputs)))
                except Exception:
                    return None
            return l1

        spec = PipelineSpec.from_dict(candidate.payload)
        if spec is None:  # already validated at proposal time; belt and braces
            return lambda model, case: None
        return lambda model, case: run_pipeline(model, case, spec, self._score)

    def _candidate_scores(
        self, candidate: FixCandidate, model: "Model", data: "CaseBatch"
    ) -> "dict[str, Optional[bool]]":
        """Per-case success of one candidate (batch path for coded pipelines)."""
        if candidate.kind == "primitive":
            prim = INTERNALS_PRIMITIVES[candidate.payload["primitive"]]
            return prim.run(model, data, self._score, candidate.payload.get("params"))
        if candidate.kind == "code":
            result = self._run_coded(candidate, model, data)
            return score_outputs(result, data, self._score)
        strategy = self._strategy(candidate)
        return {case.id: strategy(model, case) for case in data}

    def _run_coded(
        self, candidate: FixCandidate, model: "Model", data: "CaseBatch"
    ) -> "CodedPipelineResult":
        """Run a coded candidate; a failed run gets ONE coder repair round.

        The execution error (strict-bridge message, timeout, traceback tail) is
        fed back verbatim — the coder fixes its own contract violation instead
        of the candidate silently dying.  The final error, if any, is stashed
        in ``payload["exec_error"]`` for honest escalation accounting.
        """
        result = run_coded_pipeline(
            candidate.payload["code"], model, data,
            workdir=self._workdir(), timeout_sec=self._exec_timeout_sec,
            enable_attend=bool(candidate.payload.get("enable_attend")),
        )
        if not result.ok and self.codegen_available:
            logger.warning("FixAgent: coded pipeline failed (%s) — one repair round",
                           result.error)
            repaired, source, raw = self._repair_code(candidate, result.error)
            self._emit_codegen("coded_pipeline_repair", self._last_repair_prompt,
                               source, repaired, raw, ok=bool(repaired.strip()))
            if repaired.strip():
                candidate.payload["code"] = repaired
                result = run_coded_pipeline(
                    repaired, model, data,
                    workdir=self._workdir(), timeout_sec=self._exec_timeout_sec,
                    enable_attend=bool(candidate.payload.get("enable_attend")),
                )
        candidate.payload["exec_error"] = "" if result.ok else result.error
        if not result.ok:
            logger.warning("FixAgent: coded pipeline produced no result: %s",
                           result.error)
        return result

    def _repair_code(
        self, candidate: FixCandidate, error: str
    ) -> "tuple[str, str, str]":
        """Ask the coder to fix its failed pipeline; returns (code, source, raw)."""
        from evalvitals.eval_agent.stages.fix_pipeline import (
            CASES_FILENAME,
            RESULT_MARKER,
        )

        attend_clause = (
            " and model_attend(case_id, prompt=None)"
            if candidate.payload.get("enable_attend") else ""
        )
        base = _REPAIR_PROMPT_BODY.format(
            error=error[:600],
            code=str(candidate.payload.get("code", ""))[:4000],
            attend_clause=attend_clause,
            catalog=catalog_text(),
            cases_file=CASES_FILENAME,
            marker=RESULT_MARKER,
        )
        code, source, raw = "", "", ""
        if self._cli_config is not None and self._cli_config.provider != "llm":
            self._last_repair_prompt = base + "\nWrite the corrected code to a file named pipeline.py."
            code, raw = self._write_code_cli(self._last_repair_prompt)
            source = f"cli:{self._cli_config.provider}"
        if not code.strip() and self._judge is not None:
            self._last_repair_prompt = base + "\nReturn ONLY the corrected Python code inside a ```python code block."
            try:
                raw = str(self._judge.generate(self._last_repair_prompt))
            except Exception as exc:
                logger.warning("FixAgent: repair judge call failed: %s", exc)
                raw = ""
            code = _extract_code(raw)
            if code.strip():
                import ast

                try:
                    ast.parse(code)
                except SyntaxError:
                    code = ""
            source = "judge"
        return code, source, raw

    def _validate(
        self,
        candidate: FixCandidate,
        model: "Model",
        data: "CaseBatch",
        baseline: "dict[str, Optional[bool]]",
    ) -> FixValidation:
        v = FixValidation(candidate=candidate)
        if candidate.kind == "finetune_spec":
            v.summary = ("L4 executor TODO — fine-tune recipe recorded, "
                         "not executed (see candidate payload)")
            return v
        scores = self._candidate_scores(candidate, model, data)
        if isinstance(candidate.payload, dict):
            v.exec_error = str(candidate.payload.get("exec_error", "") or "")
        base_vec: "list[bool]" = []
        cand_vec: "list[bool]" = []
        for case in data:
            b = baseline.get(case.id)
            c = scores.get(case.id)
            if b is None or c is None:
                continue
            base_vec.append(b)
            cand_vec.append(c)
            if not b and c:
                v.n_fixed += 1
                v.fixed_cases.append(case.id)
            elif b and not c:
                v.n_broken += 1
                v.broken_cases.append(case.id)
        v.n_pairs = len(base_vec)
        if v.n_pairs == 0:
            v.summary = (f"never executed: {v.exec_error}" if v.exec_error
                         else "no scorable pair — candidate unvalidatable")
            return v
        try:
            stat = compare(base_vec, cand_vec, paired=True)
        except Exception as exc:
            v.summary = f"stats failed: {exc}"
            return v
        v.effect = stat.effect
        v.reject = bool(stat.reject)
        # Fixed = the paired test rejects with a net-positive effect: the
        # candidate repairs significantly more cases than it breaks.
        v.fixed = v.reject and (v.effect or 0.0) > 0
        v.summary = stat.summary()
        return v

    # -- recommendation + logging ------------------------------------------

    def _recommend(
        self, routed: "list[FixTier]", reason_prefix: str = ""
    ) -> "dict[str, Any] | None":
        above = sorted(t for t in routed if t > self.max_tier)
        if above:
            target = above[0]
            reason = (
                f"verified hypotheses route to {target.label} "
                f"({target.describe()}), beyond the allowed {self.max_tier.label}"
            )
        elif self.max_tier < FixTier.L4_PARAMETERS:
            target = FixTier(self.max_tier + 1)
            reason = (
                f"no candidate within {self.max_tier.label} validated; the next "
                f"intervention space is {target.describe()}"
            )
        else:
            return None  # already at L4 — nothing above to recommend
        if reason_prefix:
            reason = f"{reason_prefix}; {reason}"
        return {"recommend_tier": target.label, "reason": reason}

    def _emit(self, outcome: FixOutcome) -> None:
        if self.run_logger is None:
            return
        try:
            self.run_logger.log_fix(outcome)
        except Exception as exc:  # logging must never break the fix step
            logger.debug("FixAgent: log_fix failed: %s", exc)
