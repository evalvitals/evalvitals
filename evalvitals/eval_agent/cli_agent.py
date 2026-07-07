"""CLI coding-agent backends for M4 ExperimentWriter.

Mirrors researchclaw/experiment/code_agent.py.

Supported providers:

    ``"llm"``         — single-pass LLM (default; handled by ExperimentWriter,
                        not by this module)
    ``"claude_code"``  — Claude Code CLI   (``claude -p``)
    ``"codex"``        — OpenAI Codex CLI  (``codex exec``)
    ``"opencode"``     — OpenCode CLI      (``opencode run``)
    ``"gemini_cli"``   — Gemini CLI        (``gemini -p``)
    ``"kimi_cli"``     — Kimi CLI          (``kimi chat``)
    ``"antigravity"``  — Antigravity CLI   (``agy -p``)

Usage::

    from evalvitals.eval_agent.cli_agent import CliAgentConfig, create_cli_agent

    cfg   = CliAgentConfig(provider="claude_code", model="sonnet", max_budget_usd=2.0)
    agent = create_cli_agent(cfg)
    result = agent.run(prompt, workdir=Path("runs/exp_01"), timeout_sec=300)
    if result.ok:
        code = result.files.get("experiment.py")
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

from evalvitals.eval_agent.cli_runtime import ProcessRun, SubprocessRunner, collect_py_files
from evalvitals.eval_agent.cli_skills import CodexSkillInstaller, SkillInstaller
from evalvitals.eval_agent.cli_transcript import RAW_OUTPUT_CAP, render_claude_stream
from evalvitals.eval_agent.cli_types import BINARY_DEFAULTS, CliAgentConfig, CliAgentResult

logger = logging.getLogger(__name__)

# Markers that identify an agy backend failure (quota, auth, …) in its log so
# AgyModel can surface *why* a response was empty instead of failing silently.
_AGY_ERROR_MARKERS = (
    "RESOURCE_EXHAUSTED", "code 429", "quota", "ineligible",
    "PERMISSION_DENIED", "UNAUTHENTICATED", "exhausted",
)


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _scan_agy_log(path: str) -> str:
    """Return the last agy-log error line (quota/auth/…), or ``""`` if none."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()[-300:]
    except OSError:
        return ""
    for line in reversed(lines):
        if any(m.lower() in line.lower() for m in _AGY_ERROR_MARKERS):
            msg = line.strip()
            idx = msg.rfind("] ")  # drop the glog "I0608 ...]" prefix
            return (msg[idx + 2:] if idx != -1 else msg)[:240]
    return ""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _CliAgentBase:
    """Subprocess runner shared by all CLI agent providers.

    Mirrors ``researchclaw.experiment.code_agent._CliAgentBase``.
    """

    _provider_name: str = "unknown"

    def __init__(
        self,
        binary_path: str,
        model: str = "",
        max_budget_usd: float = 5.0,
        timeout_sec: int = 600,
        extra_args: list[str] | None = None,
        skills: list[str] | None = None,
        allow_skills: bool = False,
    ) -> None:
        self._binary = binary_path
        self._model = model
        self._max_budget_usd = max_budget_usd
        self._timeout_sec = timeout_sec
        self._extra_args: list[str] = extra_args or []
        self._skills: list[str] = list(skills or [])
        # A vendored skill is useless unless the Skill tool is permitted.
        self._allow_skills: bool = bool(allow_skills or self._skills)
        self._runner = SubprocessRunner()
        self._skill_installer = self._make_skill_installer(self._skills)

    def _make_skill_installer(self, skills: list[str]) -> SkillInstaller:
        return SkillInstaller(skills)

    def _install_skills(self, workdir: Path) -> None:
        """Vendor each configured skill dir into ``<workdir>/.claude/skills/<name>/``
        so an ``--add-dir <workdir>`` agent auto-discovers it. Best-effort; a
        missing or unreadable skill is skipped, never fatal."""
        self._skill_installer.install(workdir)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        prompt: str,
        workdir: Path,
        timeout_sec: int | None = None,
    ) -> CliAgentResult:
        """Invoke the CLI agent with *prompt* and return collected files.

        Args:
            prompt:      Full prompt string forwarded to the CLI.
            workdir:     Directory the agent operates in (must already exist or
                         will be created).
            timeout_sec: Override the instance-level timeout.
        """
        timeout = timeout_sec if timeout_sec is not None else self._timeout_sec
        workdir.mkdir(parents=True, exist_ok=True)
        self._install_skills(workdir)
        cmd = self._build_cmd(prompt, workdir)
        logger.debug("%s: running %s", self._provider_name, cmd[0])

        run = self._run_subprocess(cmd, workdir, timeout)
        return self._build_result(workdir, run)

    # ------------------------------------------------------------------
    # Overridden by each provider
    # ------------------------------------------------------------------

    def _build_cmd(self, prompt: str, workdir: Path) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def _postprocess_output(self, stdout: str) -> tuple[str, dict | None]:
        """Render stdout for the coding log and extract usage if available.

        Base implementation keeps the historical behaviour: the first
        ``RAW_OUTPUT_CAP`` chars of stdout, no usage. Providers that emit a
        structured event stream (Claude Code's ``stream-json``) override this to
        render the full tool-call trajectory and pull out token/cost usage.
        """
        return stdout[:RAW_OUTPUT_CAP], None

    # ------------------------------------------------------------------
    # Core subprocess runner  (mirrors ARC's _run_subprocess)
    # ------------------------------------------------------------------

    def _run_subprocess(self, cmd: list[str], workdir: Path, timeout_sec: int) -> ProcessRun:
        """Run *cmd* in *workdir*; return ``(rc, stdout, stderr, elapsed, timed_out)``."""
        return self._runner.run(cmd, workdir, timeout_sec)

    def _build_result(
        self,
        workdir: Path,
        run: ProcessRun,
    ) -> CliAgentResult:
        files = collect_py_files(workdir)
        error: str | None = None
        if run.timed_out:
            error = f"[TIMEOUT] agent killed after {run.elapsed_sec:.0f}s"
        elif run.returncode != 0 and not files:
            error = f"Exited {run.returncode}: {run.stderr[:500]}"

        raw_output, usage = self._postprocess_output(run.stdout)
        logger.debug(
            "%s: rc=%d files=%s elapsed=%.1fs timed_out=%s",
            self._provider_name, run.returncode, list(files), run.elapsed_sec, run.timed_out,
        )
        return CliAgentResult(
            files=files,
            provider_name=self._provider_name,
            elapsed_sec=run.elapsed_sec,
            raw_output=raw_output,
            usage=usage,
            error=error,
        )


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

class ClaudeCodeAgent(_CliAgentBase):
    """Claude Code CLI backend (``claude -p``).

    Requires the ``claude`` binary installed and ``ANTHROPIC_API_KEY`` set.
    """

    _provider_name = "claude_code"

    def _build_cmd(self, prompt: str, workdir: Path) -> list[str]:
        # When skills are enabled, the Skill tool must be in the allowlist so the
        # agent can invoke a vendored/global skill (e.g. nature-figure) to style
        # the figures its own code writes under figures/.
        allowed = "Bash Edit Write Read" + (" Skill" if self._allow_skills else "")
        cmd = [
            self._binary, "-p", prompt,
            "--dangerously-skip-permissions",
            # stream-json emits one JSON event per step (assistant text, each
            # Bash/Edit/Write/Read tool call, tool results, final usage/cost) so
            # the coding *trajectory* is captured, not just the final text.  In
            # print mode it requires --verbose.  Rendered by _postprocess_output.
            "--output-format", "stream-json",
            "--verbose",
            "--allowed-tools", allowed,
            "--add-dir", str(workdir),
        ]
        if self._model:
            cmd += ["--model", self._model]
        if self._max_budget_usd:
            cmd += ["--max-budget-usd", str(self._max_budget_usd)]
        cmd.extend(self._extra_args)
        return cmd

    def _postprocess_output(self, stdout: str) -> tuple[str, dict | None]:
        return render_claude_stream(stdout)


class CodexAgent(_CliAgentBase):
    """OpenAI Codex CLI backend (``codex exec``).

    Requires ``codex`` installed and ``OPENAI_API_KEY`` set.
    """

    _provider_name = "codex"

    def _make_skill_installer(self, skills: list[str]) -> SkillInstaller:
        return CodexSkillInstaller(skills)

    def _build_cmd(self, prompt: str, workdir: Path) -> list[str]:
        cmd = [
            self._binary, "exec", prompt,
            "--sandbox", "workspace-write",
            "--json",
            "-C", str(workdir),
        ]
        if self._model:
            cmd += ["-m", self._model]
        cmd.extend(self._extra_args)
        return cmd


class OpenCodeAgent(_CliAgentBase):
    """OpenCode CLI backend (``opencode run``).

    Requires ``opencode`` installed.
    """

    _provider_name = "opencode"

    def _build_cmd(self, prompt: str, workdir: Path) -> list[str]:
        cmd = [
            self._binary, "run",
            "--message", prompt,
            "--cwd", str(workdir),
        ]
        if self._model:
            cmd += ["--model", self._model]
        cmd.extend(self._extra_args)
        return cmd


class GeminiCliAgent(_CliAgentBase):
    """Gemini CLI backend (``gemini -p``).

    Requires ``gemini`` installed and ``GEMINI_API_KEY`` set.
    """

    _provider_name = "gemini_cli"

    def _build_cmd(self, prompt: str, workdir: Path) -> list[str]:
        cmd = [
            self._binary, "-p", prompt,
            "--cwd", str(workdir),
        ]
        if self._model:
            cmd += ["--model", self._model]
        cmd.extend(self._extra_args)
        return cmd


class KimiCliAgent(_CliAgentBase):
    """Kimi CLI backend (``kimi chat``).

    Requires ``kimi`` installed and ``MOONSHOT_API_KEY`` set.
    """

    _provider_name = "kimi_cli"

    def _build_cmd(self, prompt: str, workdir: Path) -> list[str]:
        cmd = [
            self._binary, "chat",
            "--message", prompt,
            "--workdir", str(workdir),
        ]
        if self._model:
            cmd += ["--model", self._model]
        cmd.extend(self._extra_args)
        return cmd


class AntigravityAgent(_CliAgentBase):
    """Antigravity CLI backend (``agy -p``).

    Requires the ``agy`` binary installed.
    """

    _provider_name = "antigravity"

    def _build_cmd(self, prompt: str, workdir: Path) -> list[str]:
        cmd = [
            self._binary, "-p", prompt,
            "--dangerously-skip-permissions",
            "--add-dir", str(workdir),
        ]
        if self._model:
            cmd += ["--model", self._model]
        cmd.extend(self._extra_args)
        return cmd


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDER_CLASSES: dict[str, type[_CliAgentBase]] = {
    "claude_code": ClaudeCodeAgent,
    "codex":       CodexAgent,
    "opencode":    OpenCodeAgent,
    "gemini_cli":  GeminiCliAgent,
    "kimi_cli":    KimiCliAgent,
    "antigravity": AntigravityAgent,
}


class AgyModel:
    """agy CLI wrapped as a judge model — no API key required.

    Implements the minimal ``Model`` protocol (``generate() → str``) by
    running ``agy -p <prompt>`` in a subprocess and returning stdout.
    Any ``<think>…</think>`` reasoning blocks emitted by the underlying
    model are stripped so callers receive only the final answer.

    Usage::

        from evalvitals.eval_agent import AgyModel

        judge = AgyModel()
        diagnosis_agent = DiagnosisAgent(judge=judge)
        hypothesis_tester = HypothesisTester(judge=judge)

    Args:
        binary_path: Explicit path to the ``agy`` binary.  Auto-detected
                     via :func:`shutil.which` when empty.
        timeout_sec: Hard wall-clock limit per call (default 120 s).
        model:       Model identifier forwarded via ``--model`` (optional).
    """

    key = "agy"

    def __init__(
        self,
        binary_path: str = "",
        timeout_sec: int = 120,
        model: str = "",
    ) -> None:
        import os

        from evalvitals.core.capability import Capability

        binary = binary_path or shutil.which("agy") or ""
        if not binary or not os.path.isfile(binary) or not os.access(binary, os.X_OK):
            raise RuntimeError(
                "AgyModel: 'agy' binary not found or not executable. "
                "Set AGY_PATH=$(which agy) and re-run, or pass binary_path= explicitly."
            )
        self._binary = binary
        self._timeout_sec = timeout_sec
        self._model = model
        self.capabilities = frozenset({Capability.GENERATE})
        self.modalities = frozenset({"text"})

    def generate(
        self,
        inputs: object,
        *,
        images: "list[Path] | None" = None,
        **kwargs: object,
    ) -> str:
        """Run ``agy -p <inputs>`` and return the text response.

        agy logs backend errors (quota exhaustion, auth ineligibility, …) to its
        ``--log-file`` rather than stdout, and on such errors prints an *empty*
        response with exit code 0.  To avoid a silent empty string that callers
        misread as a parse failure, this captures the log and surfaces the real
        reason: a non-zero exit raises, and an empty response emits a warning
        naming the agy error before returning ``""`` for graceful fallback.

        Args:
            images: Optional list of image :class:`~pathlib.Path` objects to make
                    visible to the agent.  Each file is copied into a temporary
                    workspace directory passed via ``--add-dir``, and their names
                    are listed at the top of the prompt so the model knows to look
                    at them.
        """
        import pathlib

        fd, log_path = tempfile.mkstemp(prefix="agy_", suffix=".log")
        os.close(fd)
        img_dir: str | None = None
        try:
            # ── Build prompt and optional image workspace ─────────────────────
            prompt_text = str(inputs)
            if images:
                valid = [p for p in images if isinstance(p, pathlib.Path) and p.exists()]
                if valid:
                    img_dir = tempfile.mkdtemp(prefix="agy_imgs_")
                    for p in valid:
                        shutil.copy2(p, pathlib.Path(img_dir) / p.name)
                    names = ", ".join(p.name for p in valid)
                    prompt_text = f"Images available in workspace: {names}\n\n{prompt_text}"

            cmd = [
                self._binary, "-p", prompt_text,
                "--dangerously-skip-permissions", "--log-file", log_path,
            ]
            if img_dir:
                cmd += ["--add-dir", img_dir]
            if self._model:
                cmd += ["--model", self._model]

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=self._timeout_sec,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env={**os.environ},
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"AgyModel: agy timed out after {self._timeout_sec}s"
                ) from exc

            output = proc.stdout.strip()
            if proc.returncode != 0 and not output:
                reason = _scan_agy_log(log_path) or (proc.stderr or "").strip()[:240]
                raise RuntimeError(f"AgyModel: agy exited {proc.returncode}: {reason}")

            # Strip <think>…</think> reasoning blocks
            output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()
            if not output:
                reason = _scan_agy_log(log_path)
                warnings.warn(
                    "AgyModel: agy returned an empty response"
                    + (f" — {reason}" if reason else "")
                    + ". agy is likely rate-limited or quota-exhausted; the caller "
                    "will fall back to a non-LLM path.",
                    stacklevel=2,
                )
            return output
        finally:
            _safe_unlink(log_path)
            if img_dir:
                shutil.rmtree(img_dir, ignore_errors=True)

    def __repr__(self) -> str:
        return f"AgyModel(binary={self._binary!r})"


class ClaudeModel:
    """Claude Code CLI wrapped as a judge model — no API key required.

    Implements the minimal ``Model`` protocol (``generate() → str``) by running
    ``claude -p <prompt>`` non-interactively, reusing the user's existing
    Claude Code OAuth session (``~/.claude`` / ``~/.claude.json``).

    Container note: Claude Code refuses ``--dangerously-skip-permissions``
    when running as root unless ``IS_SANDBOX=1`` is set in the environment —
    the example docker-compose sets it.

    Usage::

        judge = ClaudeModel(model="claude-fable-5")
        diagnosis_agent = DiagnosisAgent(judge=judge)

    Args:
        binary_path: Explicit path to the ``claude`` binary.  Auto-detected
                     via :func:`shutil.which` when empty.
        timeout_sec: Hard wall-clock limit per call (default 240 s — judge
                     prompts carry statistical evidence + image attachments).
        model:       Model identifier forwarded via ``--model`` (e.g.
                     ``"claude-fable-5"``, ``"sonnet"``, ``"haiku"``).  Empty
                     uses the CLI session default.
        effort:      Effort level forwarded via ``--effort`` (e.g. ``"high"``).
                     Empty uses the CLI session default.
    """

    key = "claude"

    def __init__(
        self,
        binary_path: str = "",
        timeout_sec: int = 240,
        model: str = "",
        effort: str = "",
    ) -> None:
        from evalvitals.core.capability import Capability

        binary = binary_path or shutil.which("claude") or ""
        if not binary or not os.path.isfile(binary) or not os.access(binary, os.X_OK):
            raise RuntimeError(
                "ClaudeModel: 'claude' binary not found or not executable. "
                "Set CLAUDE_PATH=$(which claude) and re-run, or pass "
                "binary_path= explicitly."
            )
        self._binary = binary
        self._timeout_sec = timeout_sec
        self._model = model
        self._effort = effort
        self.capabilities = frozenset({Capability.GENERATE})
        self.modalities = frozenset({"text"})

    def generate(
        self,
        inputs: object,
        *,
        images: "list[Path] | None" = None,
        **kwargs: object,
    ) -> str:
        """Run ``claude -p <inputs>`` and return the text response.

        Args:
            images: Optional image paths made visible to the model: copied into
                    a temp workspace passed via ``--add-dir`` and listed at the
                    top of the prompt so the model knows to Read them.
        """
        import pathlib

        img_dir: str | None = None
        try:
            prompt_text = str(inputs)
            if images:
                valid = [p for p in images if isinstance(p, pathlib.Path) and p.exists()]
                if valid:
                    img_dir = tempfile.mkdtemp(prefix="claude_imgs_")
                    for p in valid:
                        shutil.copy2(p, pathlib.Path(img_dir) / p.name)
                    names = ", ".join(p.name for p in valid)
                    prompt_text = (
                        f"Images available in workspace: {names}\n\n{prompt_text}"
                    )

            cmd = [
                self._binary, "-p", prompt_text,
                "--dangerously-skip-permissions",
                "--output-format", "text",
            ]
            if img_dir:
                cmd += ["--add-dir", img_dir]
            if self._model:
                cmd += ["--model", self._model]
            if self._effort:
                cmd += ["--effort", self._effort]

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=self._timeout_sec,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env={**os.environ},
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"ClaudeModel: claude timed out after {self._timeout_sec}s"
                ) from exc

            output = (proc.stdout or "").strip()
            if proc.returncode != 0 and not output:
                reason = (proc.stderr or "").strip()[:240]
                raise RuntimeError(
                    f"ClaudeModel: claude exited {proc.returncode}: {reason}"
                )
            if not output:
                warnings.warn(
                    "ClaudeModel: claude returned an empty response — likely "
                    "rate-limited or out of quota; the caller will fall back "
                    "to a non-LLM path.",
                    stacklevel=2,
                )
            return output
        finally:
            if img_dir:
                shutil.rmtree(img_dir, ignore_errors=True)

    def __repr__(self) -> str:
        return f"ClaudeModel(binary={self._binary!r}, model={self._model!r}, effort={self._effort!r})"


def create_cli_agent(config: CliAgentConfig) -> _CliAgentBase:
    """Instantiate the appropriate CLI agent for *config*.

    Args:
        config: :class:`CliAgentConfig` with ``provider`` set to a non-``"llm"``
                value.

    Raises:
        ValueError:  When ``provider`` is ``"llm"`` or unknown.
        RuntimeError: When the CLI binary is not found on PATH and no explicit
                      ``binary_path`` was given.
    """
    provider = config.provider

    if provider == "llm":
        raise ValueError(
            "'llm' is not a CLI provider. Use ExperimentWriter directly "
            "(leave cli_agent=None or CliAgentConfig(provider='llm'))."
        )

    cls = _PROVIDER_CLASSES.get(provider)
    if cls is None:
        raise ValueError(
            f"Unknown CLI provider: {provider!r}. "
            f"Valid: {sorted(_PROVIDER_CLASSES)}"
        )

    binary = config.binary_path or shutil.which(BINARY_DEFAULTS[provider]) or ""
    if not binary:
        raise RuntimeError(
            f"CLI agent binary for {provider!r} not found in PATH. "
            f"Install '{BINARY_DEFAULTS[provider]}' or pass "
            f"CliAgentConfig(binary_path='/path/to/{BINARY_DEFAULTS[provider]}')."
        )

    return cls(
        binary_path=binary,
        model=config.model,
        max_budget_usd=config.max_budget_usd,
        timeout_sec=config.timeout_sec,
        extra_args=list(config.extra_args),
        skills=list(config.skills),
        allow_skills=config.allow_skills,
    )
