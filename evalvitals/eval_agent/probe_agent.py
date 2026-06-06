"""M1 — ProbeAgent: select suitable analyzers and execute them.

Combines tool selection (which analyzers to run for this model kind) with
execution (running each analyzer directly or inside a Docker container).

Direct mode (default)::

    agent = ProbeAgent(max_analyzers=4)
    results = agent.probe(model, data)   # {analyzer_name: Result}

Protocol-guided mode (new — :class:`~evalvitals.eval_agent.protocol.ExperimentProtocol`)::

    agent = ProbeAgent()
    results, schema = agent.probe_with_schema(model, data, protocol=protocol)
    print(schema.rationale)   # why these analyzers were selected

The protocol is converted to failure-mode hints via
:meth:`~evalvitals.eval_agent.protocol.ExperimentProtocol.probe_hints` and
forwarded to :class:`~evalvitals.eval_agent.probe.StrategyProbe` for
priority-boosting.  The returned :class:`~evalvitals.eval_agent.protocol.ProbingSchema`
records which analyzers ran and why (feeds M2/M5 for traceable reasoning).

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

import json
import subprocess
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from evalvitals.core.capability import Capability
from evalvitals.core.experiment import Experiment, ExperimentRunner
from evalvitals.core.registry import registry
from evalvitals.eval_agent.probe import ModelKind, StrategyProbe

if TYPE_CHECKING:
    from evalvitals.core.analyzer import Analyzer
    from evalvitals.core.case import CaseBatch
    from evalvitals.core.model import Model
    from evalvitals.core.result import Result
    from evalvitals.eval_agent.protocol import ExperimentProtocol, ProbingSchema

# Capabilities that a containerised (API-based) model can satisfy.
_BLACKBOX_CAPS = frozenset({Capability.GENERATE, Capability.LOGPROBS, Capability.TOOL_CALLS})


def _is_blackbox_compatible(analyzer_cls: type) -> bool:
    """True when the analyzer's requires are all satisfiable by a black-box API model."""
    return analyzer_cls.requires <= _BLACKBOX_CAPS


class ProbeAgent:
    """M1: select analyzers and execute them (directly or via Docker).

    Args:
        probe:             Selects and ranks analyzers.  Defaults to
                           ``StrategyProbe()``.
        runner:            Executes direct (non-Docker) runs.  Defaults to a
                           fresh ``ExperimentRunner()``.
        max_analyzers:     Cap on analyzers per cycle passed to
                           :meth:`StrategyProbe.select`.
        analyzer_overrides: Pre-instantiated analyzers for those requiring
                           mandatory constructor arguments.
        use_docker:        When ``True``, black-box-compatible analyzers are
                           run inside a Docker container instead of in-process.
        docker_image:      Docker image to use when *use_docker* is ``True``.
        docker_timeout:    Seconds before a Docker run is killed.
        model_env_var:     Name of the env var inside Docker that carries the
                           API key for the containerised model (e.g.
                           ``"GEMINI_API_KEY"`` or ``"OPENAI_API_KEY"``).
    """

    def __init__(
        self,
        probe: StrategyProbe | None = None,
        runner: ExperimentRunner | None = None,
        max_analyzers: int | None = None,
        analyzer_overrides: dict[str, "Analyzer"] | None = None,
        use_docker: bool = False,
        docker_image: str = "evalvitals:latest",
        docker_timeout: int = 300,
        model_env_var: str = "GEMINI_API_KEY",
    ) -> None:
        self.selector = probe or StrategyProbe()
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
        hint_failure_modes: list[str] | None = None,
        protocol: "ExperimentProtocol | None" = None,
    ) -> dict[str, "Result"]:
        """Select analyzers and run each one, returning a ``{name: Result}`` dict.

        Analyzers are executed in parallel using a ``ThreadPoolExecutor`` so
        that independent analyzers (e.g. self_consistency and attention both
        probing the same model) do not wait on each other.  Each analyzer
        still runs in its own thread; the GIL is released for I/O-heavy
        analyzers (API calls, file reads) giving near-linear speedup for
        black-box / API-based analyzers.

        Docker-dispatched analyzers are also parallelised — each container
        launch is an independent subprocess so the wall-clock time for N
        Docker analyzers is ≈ max(individual times), not their sum.

        Sets :attr:`last_schema` with the selection rationale so callers can
        inspect which analyzers ran and why without changing the return type.

        Args:
            model:              The model to probe.
            data:               Cases to run analyzers on.
            hint_failure_modes: Failure-mode tags from M3 — used to boost
                                analyzers that are relevant to outstanding
                                hypotheses (cycle 2+ focused probing).
            protocol:           Optional experiment protocol.  When supplied,
                                its :meth:`~evalvitals.eval_agent.protocol.ExperimentProtocol.probe_hints`
                                are merged into *hint_failure_modes* so the
                                selection is shaped by the user's prior.
        """
        # Merge protocol hints with any explicitly passed failure-mode hints.
        effective_hints = list(hint_failure_modes or [])
        if protocol is not None:
            for h in protocol.probe_hints():
                if h not in effective_hints:
                    effective_hints.append(h)

        names = self.selector.select(
            model,
            max_analyzers=self.max_analyzers,
            hint_failure_modes=effective_hints or None,
        )

        # Build (name, analyzer) pairs up front — _make_analyzer is cheap and
        # not thread-safe to call concurrently.
        tasks: list[tuple[str, "Analyzer"]] = []
        for name in names:
            analyzer = self._make_analyzer(name)
            if analyzer is not None:
                tasks.append((name, analyzer))

        if not tasks:
            self.last_schema = _build_schema([], effective_hints, protocol)
            return {}

        results: dict[str, "Result"] = {}

        def _run_one(name: str, analyzer: "Analyzer") -> tuple[str, "Result | None"]:
            cls = type(analyzer)
            if self.use_docker and _is_blackbox_compatible(cls):
                return name, self._run_in_docker(name, analyzer, data)
            return name, self._run_direct(analyzer, model, data)

        # Use min(len(tasks), 8) workers so we don't over-subscribe small runs
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

        self.last_schema = _build_schema([t[0] for t in tasks], effective_hints, protocol)
        return results

    def probe_with_schema(
        self,
        model: "Model",
        data: "CaseBatch",
        hint_failure_modes: list[str] | None = None,
        protocol: "ExperimentProtocol | None" = None,
    ) -> "tuple[dict[str, Result], ProbingSchema]":
        """Like :meth:`probe`, but also returns the :class:`~evalvitals.eval_agent.protocol.ProbingSchema`.

        The schema records which analyzers were selected and why — useful for
        M2/M5 to understand the M1 reasoning without re-running selection.

        Returns:
            ``(results_dict, schema)`` — the same dict as :meth:`probe` plus a
            :class:`~evalvitals.eval_agent.protocol.ProbingSchema`.
        """
        results = self.probe(model, data, hint_failure_modes=hint_failure_modes, protocol=protocol)
        schema = self.last_schema
        if schema is None:
            from evalvitals.eval_agent.protocol import ProbingSchema
            schema = ProbingSchema(selected_analyzers=list(results), protocol=protocol)
        return results, schema

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


def _build_schema(
    selected: list[str],
    hints: list[str],
    protocol: "ExperimentProtocol | None",
) -> "ProbingSchema":
    """Build a ProbingSchema from a completed probe() call."""
    from evalvitals.eval_agent.protocol import ProbingSchema

    rationale_parts: list[str] = []
    if hints:
        rationale_parts.append(f"failure-mode hints: {', '.join(hints)}")
    if protocol and protocol.task_domain:
        rationale_parts.append(f"task domain: {protocol.task_domain}")
    rationale = (
        f"Selected {len(selected)} analyzer(s) ({', '.join(selected)}) "
        + ("guided by " + " + ".join(rationale_parts) if rationale_parts else "by capability matching")
        + "."
    )
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
