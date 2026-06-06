"""M1 — ProbeAgent: select suitable analyzers and execute them.

Combines tool selection (which analyzers to run for this model kind) with
execution (running each analyzer directly or inside a Docker container).

Direct mode (default, static selection)::

    agent = ProbeAgent(max_analyzers=4)
    results = agent.probe(model, data)   # {analyzer_name: Result}

LLM-guided mode — pass a judge model and a protocol::

    agent = ProbeAgent(judge=model, max_analyzers=4)
    results, schema = agent.probe_with_schema(model, data, protocol=protocol)
    print(schema.rationale)   # why these analyzers were chosen

When both *judge* and *protocol* are present the judge LLM reads the protocol
description and the catalog of available analyzers, then selects the most
relevant ones.  If the LLM call fails the agent falls back to
:class:`~evalvitals.eval_agent.probe.StrategyProbe` automatically.

Docker mode::

    agent = ProbeAgent(use_docker=True, docker_image="evalvitals:latest")
    results = agent.probe(model, data)

Docker mode is only engaged for **black-box-compatible** analyzers (those
whose requirements can be satisfied by a GENERATE-only API model).  Analyzers
that need white-box internals (ATTENTION, HIDDEN_STATES, …) always run
directly — they need the locally-loaded model and cannot be meaningfully
containerised without shipping the weights.

The Docker container receives a JSON payload via stdin::

    {"analyzer": "self_consistency", "params": {}, "cases": [...],
     "model_env": "GEMINI_API_KEY"}   # which env var carries the API key

and writes the analyzer findings as JSON to stdout.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import subprocess
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from evalvitals.core.capability import Capability
from evalvitals.core.experiment import Experiment, ExperimentRunner
from evalvitals.core.registry import registry
from evalvitals.eval_agent.stages.probe import ModelKind, StrategyProbe, get_analyzer_catalog

if TYPE_CHECKING:
    from evalvitals.core.analyzer import Analyzer
    from evalvitals.core.case import CaseBatch
    from evalvitals.core.model import Model
    from evalvitals.core.result import Result
    from evalvitals.eval_agent.stages.protocol import ExperimentProtocol, ProbingSchema

logger = logging.getLogger(__name__)

# Capabilities that a containerised (API-based) model can satisfy.
_BLACKBOX_CAPS = frozenset({Capability.GENERATE, Capability.LOGPROBS, Capability.TOOL_CALLS})

_SELECTION_PROMPT_TMPL = """\
You are selecting diagnostic analyzers for a model evaluation experiment.
Choose the analyzers that would surface the most useful evidence given the \
researcher's description.

EXPERIMENT DESCRIPTION:
{description}
{task_domain_section}
{success_criteria_section}
{failure_patterns_section}
MODEL TYPE: {model_kind}

AVAILABLE ANALYZERS ({n_available} compatible with this model):
{analyzer_list}
{prior_hypotheses_section}
Select up to {max_n} analyzers. Return ONLY a JSON object, no other text:
{{"analyzers": ["name1", "name2", ...], "rationale": "one sentence explaining the selection"}}
"""


def _is_blackbox_compatible(analyzer_cls: type) -> bool:
    """True when the analyzer's requires are all satisfiable by a black-box API model."""
    return analyzer_cls.requires <= _BLACKBOX_CAPS


class ProbeAgent:
    """M1: select analyzers and execute them (directly or via Docker).

    Args:
        probe:             Static selector — used when *judge* is absent or the
                           LLM call fails.  Defaults to ``StrategyProbe()``.
        judge:             Any :class:`~evalvitals.core.model.Model` with
                           ``Capability.GENERATE``.  When set **and** a
                           protocol is supplied to :meth:`probe`, the judge LLM
                           reads the protocol description and available analyzer
                           catalog to select analyzers dynamically.
        runner:            Executes direct (non-Docker) runs.  Defaults to a
                           fresh ``ExperimentRunner()``.
        max_analyzers:     Cap on analyzers per cycle.
        analyzer_overrides: Pre-instantiated analyzers for those requiring
                           mandatory constructor arguments.
        use_docker:        When ``True``, black-box-compatible analyzers are
                           run inside a Docker container instead of in-process.
        docker_image:      Docker image to use when *use_docker* is ``True``.
        docker_timeout:    Seconds before a Docker run is killed.
        model_env_var:     Name of the env var inside Docker that carries the
                           API key for the containerised model.
    """

    def __init__(
        self,
        probe: StrategyProbe | None = None,
        judge: "Model | None" = None,
        runner: ExperimentRunner | None = None,
        max_analyzers: int | None = None,
        analyzer_overrides: dict[str, "Analyzer"] | None = None,
        use_docker: bool = False,
        docker_image: str = "evalvitals:latest",
        docker_timeout: int = 300,
        model_env_var: str = "GEMINI_API_KEY",
    ) -> None:
        self.selector = probe or StrategyProbe()
        self.judge = judge
        self.runner = runner or ExperimentRunner()
        self.max_analyzers = max_analyzers
        self._overrides: dict[str, "Analyzer"] = analyzer_overrides or {}
        self.use_docker = use_docker
        self.docker_image = docker_image
        self.docker_timeout = docker_timeout
        self.model_env_var = model_env_var
        # Set by probe() / probe_with_schema() so callers can inspect why
        # each analyzer was chosen without changing the return type of probe().
        self.last_schema: "ProbingSchema | None" = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def probe(
        self,
        model: "Model",
        data: "CaseBatch",
        *,
        protocol: "ExperimentProtocol | None" = None,
        prior_hypotheses: list[Any] | None = None,
        hint_failure_modes: list[str] | None = None,
    ) -> dict[str, "Result"]:
        """Select analyzers and run each one, returning a ``{name: Result}`` dict.

        Analyzer selection strategy (in priority order):

        1. **LLM-guided** — when a *judge* model and a *protocol* are both
           present, the judge reads the protocol description plus an
           auto-generated catalog of available analyzers and picks the most
           relevant ones.  Any *prior_hypotheses* from M3 are included as
           additional context for cycle 2+ focused probing.
        2. **Static fallback** — when no judge or no protocol, uses
           :class:`~evalvitals.eval_agent.probe.StrategyProbe` with the
           optional *hint_failure_modes* list for priority-boosting (used by
           :class:`~evalvitals.eval_agent.loop.AutoDiagnoseLoop`).

        Analyzers are executed in parallel using a ``ThreadPoolExecutor`` so
        that independent analyzers do not wait on each other.

        Sets :attr:`last_schema` with the selection rationale so callers can
        inspect which analyzers ran and why without changing the return type.

        Args:
            model:               The model to probe.
            data:                Cases to run analyzers on.
            protocol:            Experiment protocol — enables LLM-guided
                                 selection when a judge is also configured.
            prior_hypotheses:    Hypotheses from M3 in prior cycles.  Passed to
                                 the judge as context for focused follow-up.
            hint_failure_modes:  Failure-mode tags for the static fallback path
                                 (e.g. from :class:`~evalvitals.eval_agent.loop.AutoDiagnoseLoop`).
        """
        rationale: str
        if self.judge is not None and protocol is not None:
            names, rationale = self._llm_select(protocol, model, prior_hypotheses)
        else:
            names = self.selector.select(
                model,
                max_analyzers=self.max_analyzers,
                hint_failure_modes=hint_failure_modes or None,
            )
            rationale = _static_rationale(names, hint_failure_modes)

        # Build (name, analyzer) pairs up front — _make_analyzer is cheap and
        # not thread-safe to call concurrently.
        tasks: list[tuple[str, "Analyzer"]] = []
        for name in names:
            analyzer = self._make_analyzer(name)
            if analyzer is not None:
                tasks.append((name, analyzer))

        if not tasks:
            self.last_schema = _build_schema([], rationale, protocol)
            return {}

        results: dict[str, "Result"] = {}

        def _run_one(name: str, analyzer: "Analyzer") -> tuple[str, "Result | None"]:
            cls = type(analyzer)
            if self.use_docker and _is_blackbox_compatible(cls):
                return name, self._run_in_docker(name, analyzer, data)
            return name, self._run_direct(analyzer, model, data)

        max_workers = min(len(tasks), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_run_one, name, analyzer): name
                for name, analyzer in tasks
            }
            for future in as_completed(futures):
                try:
                    name, result = future.result()
                    if result is not None:
                        results[name] = result
                except Exception as exc:
                    name = futures[future]
                    warnings.warn(
                        f"Analyzer '{name}' raised in parallel run: {exc}",
                        stacklevel=2,
                    )

        self.last_schema = _build_schema([t[0] for t in tasks], rationale, protocol)
        return results

    def probe_with_schema(
        self,
        model: "Model",
        data: "CaseBatch",
        *,
        protocol: "ExperimentProtocol | None" = None,
        prior_hypotheses: list[Any] | None = None,
        hint_failure_modes: list[str] | None = None,
    ) -> "tuple[dict[str, Result], ProbingSchema]":
        """Like :meth:`probe`, but also returns the :class:`~evalvitals.eval_agent.protocol.ProbingSchema`.

        The schema records which analyzers were selected and why — useful for
        M2/M5 to understand the M1 reasoning without re-running selection.

        Returns:
            ``(results_dict, schema)`` — the same dict as :meth:`probe` plus a
            :class:`~evalvitals.eval_agent.protocol.ProbingSchema`.
        """
        results = self.probe(
            model, data,
            protocol=protocol,
            prior_hypotheses=prior_hypotheses,
            hint_failure_modes=hint_failure_modes,
        )
        schema = self.last_schema
        if schema is None:
            from evalvitals.eval_agent.stages.protocol import ProbingSchema
            schema = ProbingSchema(selected_analyzers=list(results), protocol=protocol)
        return results, schema

    # ------------------------------------------------------------------
    # LLM-guided selection
    # ------------------------------------------------------------------

    def _llm_select(
        self,
        protocol: "ExperimentProtocol",
        model: "Model",
        prior_hypotheses: list[Any] | None,
    ) -> tuple[list[str], str]:
        """Ask the judge LLM to choose analyzers from the available catalog.

        Returns ``(selected_names, rationale_string)``.  Falls back to static
        selection on any error.
        """
        assert self.judge is not None  # caller guarantees this

        kind = self.selector.detect_kind(model)
        catalog = get_analyzer_catalog(model)
        if not catalog:
            return self._static_fallback(model), "no analyzers available"

        max_n = self.max_analyzers or len(catalog)

        analyzer_lines = "\n".join(
            f"  - {name}: {desc}" for name, desc in catalog.items()
        )

        task_domain_section = (
            f"\nTASK DOMAIN: {protocol.task_domain}\n" if protocol.task_domain else ""
        )
        success_criteria_section = (
            f"\nSUCCESS CRITERIA: {protocol.success_criteria}\n"
            if protocol.success_criteria
            else ""
        )
        failure_patterns_section = (
            f"\nADDITIONAL OBSERVATIONS: {protocol.failure_patterns}\n"
            if protocol.failure_patterns
            else ""
        )

        prior_section = ""
        if prior_hypotheses:
            lines = []
            for h in prior_hypotheses:
                stmt = getattr(h, "statement", str(h))
                status = ""
                s = getattr(h, "status", None)
                if s is not None:
                    status = f" [{getattr(s, 'value', str(s))}]"
                lines.append(f"  - {stmt}{status}")
            prior_section = (
                "\nPRIOR HYPOTHESES FROM THIS INVESTIGATION "
                "(select analyzers that help verify or refute these):\n"
                + "\n".join(lines)
                + "\n"
            )

        prompt = _SELECTION_PROMPT_TMPL.format(
            description=protocol.description,
            task_domain_section=task_domain_section,
            success_criteria_section=success_criteria_section,
            failure_patterns_section=failure_patterns_section,
            model_kind=kind.value,
            n_available=len(catalog),
            analyzer_list=analyzer_lines,
            prior_hypotheses_section=prior_section,
            max_n=max_n,
        )

        try:
            sig = inspect.signature(self.judge.generate)
            if "temperature" in sig.parameters:
                raw = self.judge.generate(prompt, temperature=0)
            else:
                raw = self.judge.generate(prompt)
        except Exception as exc:
            logger.warning("ProbeAgent LLM selection failed (%s) — using static fallback", exc)
            return self._static_fallback(model), "static fallback (LLM call failed)"

        return self._parse_llm_response(str(raw), set(catalog), max_n, model)

    def _parse_llm_response(
        self,
        raw: str,
        valid_names: set[str],
        max_n: int,
        model: "Model",
    ) -> tuple[list[str], str]:
        """Extract analyzer names and rationale from the LLM JSON response.

        Strips ``<think>…</think>`` blocks emitted by reasoning models (e.g.
        Qwen3) before searching for the JSON payload so that greedy matching
        does not swallow reasoning text as part of the JSON.
        """
        # Remove <think>…</think> reasoning blocks before JSON search
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Use a non-greedy search that stops at the first closing brace so we
        # don't accidentally span multiple JSON objects.
        match = re.search(r"\{[^{}]*\}", cleaned)
        if match:
            try:
                data = json.loads(match.group())
                names = [n for n in data.get("analyzers", []) if n in valid_names]
                rationale = str(data.get("rationale", "LLM-selected"))
                if names:
                    return names[:max_n], rationale
            except json.JSONDecodeError:
                pass

        # Soft parse: look for any known analyzer name mentioned in the response
        found = [n for n in sorted(valid_names) if n in cleaned]
        if found:
            logger.warning("ProbeAgent: LLM response not valid JSON — extracted names by text scan")
            return found[:max_n], "LLM-selected (text-extracted)"

        logger.warning("ProbeAgent: could not parse LLM response — using static fallback")
        return self._static_fallback(model), "static fallback (LLM parse failed)"

    def _static_fallback(self, model: "Model") -> list[str]:
        return self.selector.select(model, max_analyzers=self.max_analyzers)

    # ------------------------------------------------------------------
    # Execution strategies
    # ------------------------------------------------------------------

    def _run_direct(
        self,
        analyzer: "Analyzer",
        model: "Model",
        data: "CaseBatch",
    ) -> "Result | None":
        exp = Experiment(model=model, analyzer=analyzer, data=data)
        try:
            return self.runner.run(exp)
        except Exception as exc:
            warnings.warn(
                f"Analyzer '{analyzer.name}' raised during direct run: {exc}",
                stacklevel=3,
            )
            return None

    def _run_in_docker(
        self,
        name: str,
        analyzer: "Analyzer",
        data: "CaseBatch",
    ) -> "Result | None":
        """Run *analyzer* inside a Docker container, returning the Result or None."""
        from evalvitals.core.result import Result

        payload = json.dumps({
            "analyzer": name,
            "params": analyzer.get_params(),
            "cases": _serialize_cases(data),
            "model_env": self.model_env_var,
        })
        cmd = [
            "docker", "run", "--rm", "-i",
            "--env", self.model_env_var,
            self.docker_image,
            "python", "-m", "evalvitals.eval_agent._docker_runner",
        ]
        try:
            proc = subprocess.run(
                cmd,
                input=payload.encode(),
                capture_output=True,
                timeout=self.docker_timeout,
            )
        except FileNotFoundError:
            warnings.warn(
                "Docker is not available; falling back to direct execution "
                f"for analyzer '{name}'.",
                stacklevel=3,
            )
            return None
        except subprocess.TimeoutExpired:
            warnings.warn(f"Docker run for '{name}' timed out.", stacklevel=3)
            return None

        if proc.returncode != 0:
            warnings.warn(
                f"Docker run for '{name}' failed (exit {proc.returncode}): "
                f"{proc.stderr.decode()[:300]}",
                stacklevel=3,
            )
            return None

        try:
            raw = json.loads(proc.stdout)
            return Result(
                analyzer=name,
                model=f"docker:{self.docker_image}",
                findings=raw.get("findings", {}),
                metadata={"docker": True, "image": self.docker_image},
                cases=data,
            )
        except Exception as exc:
            warnings.warn(f"Could not parse Docker output for '{name}': {exc}", stacklevel=3)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_analyzer(self, name: str) -> "Analyzer | None":
        if name in self._overrides:
            return self._overrides[name]
        cls = registry.analyzers.get(name)
        try:
            return cls()
        except TypeError as exc:
            warnings.warn(
                f"Skipping analyzer '{name}': cannot instantiate with default args "
                f"({exc}). Pass an instance via analyzer_overrides.",
                stacklevel=3,
            )
            return None

    def detect_kind(self, model: "Model") -> ModelKind:
        """Delegate to the inner :class:`~evalvitals.eval_agent.probe.StrategyProbe`."""
        return self.selector.detect_kind(model)


def _static_rationale(names: list[str], hints: list[str] | None) -> str:
    parts = []
    if hints:
        parts.append(f"failure-mode hints: {', '.join(hints)}")
    return (
        f"Selected {len(names)} analyzer(s) ({', '.join(names)}) "
        + ("guided by " + " + ".join(parts) if parts else "by capability matching")
        + "."
    )


def _build_schema(
    selected: list[str],
    rationale: str,
    protocol: "ExperimentProtocol | None",
) -> "ProbingSchema":
    from evalvitals.eval_agent.stages.protocol import ProbingSchema
    return ProbingSchema(
        selected_analyzers=selected,
        rationale=rationale,
        protocol=protocol,
    )


def _serialize_cases(data: "CaseBatch") -> list[dict[str, Any]]:
    """Minimal JSON-serialisable representation of a CaseBatch for Docker."""
    out = []
    for case in data:
        inp = getattr(case, "inputs", None)
        out.append({
            "id": case.id,
            "prompt": str(inp.prompt) if inp else "",
            "label": getattr(case.label, "value", None) if hasattr(case, "label") else None,
            "metadata": getattr(case, "metadata", {}),
        })
    return out
