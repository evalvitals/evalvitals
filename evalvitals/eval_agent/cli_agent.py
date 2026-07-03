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

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

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
# Configuration
# ---------------------------------------------------------------------------

_VALID_PROVIDERS = frozenset(
    {"llm", "claude_code", "codex", "opencode", "gemini_cli", "kimi_cli", "antigravity"}
)

_BINARY_DEFAULTS: dict[str, str] = {
    "claude_code": "claude",
    "codex":       "codex",
    "opencode":    "opencode",
    "gemini_cli":  "gemini",
    "kimi_cli":    "kimi",
    "antigravity": "agy",
}


@dataclass(frozen=True)
class CliAgentConfig:
    """Configuration for a CLI-based coding-agent backend.

    Args:
        provider:        Which CLI agent to use.  ``"llm"`` (default) means no
                         CLI agent — ``ExperimentWriter`` falls back to its
                         single-pass LLM path.
        binary_path:     Explicit path to the CLI binary.  Auto-detected via
                         :func:`shutil.which` when empty.
        model:           Model-override flag forwarded to the binary (e.g.
                         ``"sonnet"`` for Claude Code, ``"gpt-4o"`` for Codex).
        max_budget_usd:  Spend cap forwarded to ``--max-budget-usd`` (Claude Code).
        timeout_sec:     Hard wall-clock limit for the agent subprocess.
        extra_args:      Additional flags appended verbatim to the CLI command.
        skills:          Paths to Agent-Skill directories (each containing a
                         ``SKILL.md``) to vendor into the sandbox per run, under
                         ``<workdir>/.claude/skills/<name>/``. Claude Code / agy
                         auto-discover skills inside an ``--add-dir`` directory.
                         A non-empty value implies ``allow_skills=True``.
        allow_skills:    Enable the ``Skill`` tool so the agent may invoke skills
                         (vendored here OR installed globally in ``~/.claude/skills``).
                         Off by default to preserve deterministic behavior.
    """

    provider: str = "llm"
    binary_path: str = ""
    model: str = ""
    max_budget_usd: float = 5.0
    timeout_sec: int = 600
    extra_args: tuple[str, ...] = ()
    skills: tuple[str, ...] = ()
    allow_skills: bool = False

    @property
    def skills_enabled(self) -> bool:
        return self.allow_skills or bool(self.skills)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class CliAgentResult:
    """Output of one CLI agent invocation.

    Attributes:
        files:         ``{filename: source_code}`` collected from the workdir
                       after the agent exits.
        provider_name: Short identifier of the provider that ran (e.g.
                       ``"claude_code"``).
        elapsed_sec:   Wall-clock seconds for the subprocess.
        raw_output:    The agent's narration for the coding log.  For providers
                       that emit a structured event stream (Claude Code's
                       ``stream-json``) this is the **rendered coding
                       trajectory** — assistant text, every Bash/Edit/Write/Read
                       tool call + result, and a token/cost footer.  For the
                       others it is the first ``_RAW_OUTPUT_CAP`` chars of stdout.
        usage:         Token/cost usage parsed from the stream (``cost_usd``,
                       ``num_turns``, in/out/cache tokens), or ``None`` when the
                       provider does not report it.
        error:         Non-``None`` when the agent timed out or exited non-zero
                       **and** produced no files.
    """

    files: dict[str, str]
    provider_name: str
    elapsed_sec: float
    raw_output: str = ""
    usage: dict | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.files)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _collect_py_files(workdir: Path) -> dict[str, str]:
    """Read all ``.py`` files from *workdir* (flat, non-recursive).

    Skips files whose names start with ``_codex_`` or ``_agent_`` (provider-
    internal temp files).  Mirrors the ARC pattern exactly.
    """
    files: dict[str, str] = {}
    for pyfile in sorted(workdir.glob("*.py")):
        if pyfile.name.startswith(("_codex_", "_agent_")):
            continue
        try:
            files[pyfile.name] = pyfile.read_text(encoding="utf-8")
        except OSError:
            pass
    return files


# ---------------------------------------------------------------------------
# Coding-trajectory rendering (Claude Code stream-json)
# ---------------------------------------------------------------------------

# Bounds so a single huge tool body (e.g. a Write of a 10k-line file, whose
# content is echoed in the tool_use input) can't blow the transcript up.  The
# file itself is still captured verbatim from the workdir; the transcript only
# needs the *action*, not a second copy of the payload.
_RAW_OUTPUT_CAP = 3000          # base providers: first N chars of stdout
_STREAM_TEXT_CAP = 4000         # per assistant-text / final-result block
_STREAM_TOOL_INPUT_CAP = 2000   # per tool_use input render
_STREAM_TOOL_RESULT_CAP = 2000  # per tool_result render
_STREAM_FALLBACK_CAP = 8000     # non-JSON stdout (degraded)


def _trunc(text: str, cap: int) -> str:
    text = text or ""
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n… [truncated {len(text) - cap} chars]"


def _blocks_text(content: object) -> str:
    """Flatten a message ``content`` (str, or list of content blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                out.append(str(b.get("text", "")))
            elif isinstance(b, str):
                out.append(b)
        return "\n".join(out)
    return ""


def _summarise_tool_use(name: str, inp: object) -> str:
    """One-line summary of a tool_use input (no huge file bodies inline)."""
    if not isinstance(inp, dict):
        return _trunc(str(inp), _STREAM_TOOL_INPUT_CAP)
    if name == "Bash":
        return _trunc("$ " + str(inp.get("command", "")), _STREAM_TOOL_INPUT_CAP)
    if name == "Write":
        return f"write {inp.get('file_path', '?')} ({len(str(inp.get('content', '')))} chars)"
    if name == "Edit":
        return f"edit {inp.get('file_path', '?')}"
    if name == "Read":
        span = ""
        if inp.get("offset") is not None or inp.get("limit") is not None:
            span = f" [offset={inp.get('offset')}, limit={inp.get('limit')}]"
        return f"read {inp.get('file_path', '?')}{span}"
    try:
        return _trunc(json.dumps(inp, default=str), _STREAM_TOOL_INPUT_CAP)
    except (TypeError, ValueError):
        return _trunc(str(inp), _STREAM_TOOL_INPUT_CAP)


def _render_claude_stream(stdout: str) -> tuple[str, dict | None]:
    """Render ``claude -p --output-format stream-json`` into a readable
    coding trajectory and pull out token/cost usage.

    Each stream line is one JSON event: ``system/init`` (session header),
    ``assistant`` (text + ``tool_use`` calls), ``user`` (``tool_result``), and a
    final ``result`` (usage + cost).  Returns ``(transcript, usage)``.  When the
    output is not the expected JSON stream (older CLI, error before the stream
    starts, …) it falls back to the raw stdout so output is never lost.
    """
    events: list[dict] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(ev, dict):
            events.append(ev)
    if not events:
        return _trunc(stdout, _STREAM_FALLBACK_CAP), None

    out: list[str] = []
    usage: dict | None = None
    tool_seq: dict[str, int] = {}   # tool_use_id -> step number
    step = 0

    for ev in events:
        etype = ev.get("type")
        if etype == "system" and ev.get("subtype") == "init":
            tools = ev.get("tools") or []
            out.append(
                f"=== session start | model={ev.get('model', '?')} | "
                f"tools={','.join(map(str, tools)) if tools else '-'} ==="
            )
        elif etype == "assistant":
            for b in (ev.get("message") or {}).get("content") or []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    txt = str(b.get("text", "")).strip()
                    if txt:
                        out.append(f"[assistant] {_trunc(txt, _STREAM_TEXT_CAP)}")
                elif b.get("type") == "tool_use":
                    step += 1
                    if b.get("id"):
                        tool_seq[b["id"]] = step
                    name = str(b.get("name", "tool"))
                    out.append(f"[#{step} {name}] {_summarise_tool_use(name, b.get('input'))}")
        elif etype == "user":
            for b in (ev.get("message") or {}).get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    n = tool_seq.get(b.get("tool_use_id"), "?")
                    err = " ERROR" if b.get("is_error") else ""
                    out.append(
                        f"[#{n} result{err}] "
                        f"{_trunc(_blocks_text(b.get('content')), _STREAM_TOOL_RESULT_CAP)}"
                    )
        elif etype == "result":
            u = ev.get("usage") or {}
            usage = {
                "cost_usd": ev.get("total_cost_usd"),
                "num_turns": ev.get("num_turns"),
                "duration_ms": ev.get("duration_ms"),
                "input_tokens": u.get("input_tokens"),
                "output_tokens": u.get("output_tokens"),
                "cache_read_input_tokens": u.get("cache_read_input_tokens"),
                "cache_creation_input_tokens": u.get("cache_creation_input_tokens"),
            }
            final = str(ev.get("result", "")).strip()
            if final:
                out.append(f"[final] {_trunc(final, _STREAM_TEXT_CAP)}")
            cost = usage["cost_usd"]
            dur = ev.get("duration_ms")
            out.append(
                "=== result: {sub} | turns={t} | cost={c} | "
                "tokens in={i} out={o} (cache_read={cr}) | wall={w} ===".format(
                    sub=ev.get("subtype", "?"),
                    t=usage["num_turns"],
                    c=(f"${cost:.4f}" if isinstance(cost, (int, float)) else "n/a"),
                    i=usage["input_tokens"], o=usage["output_tokens"],
                    cr=usage["cache_read_input_tokens"],
                    w=(f"{dur / 1000:.1f}s" if isinstance(dur, (int, float)) else "?"),
                )
            )

    return "\n".join(out), usage


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

    def _install_skills(self, workdir: Path) -> None:
        """Vendor each configured skill dir into ``<workdir>/.claude/skills/<name>/``
        so an ``--add-dir <workdir>`` agent auto-discovers it. Best-effort; a
        missing or unreadable skill is skipped, never fatal."""
        if not self._skills:
            return
        import shutil

        dest_root = workdir / ".claude" / "skills"
        for src in self._skills:
            src_path = Path(src)
            if not src_path.exists() or not src_path.is_dir():
                logger.warning("skill dir not found, skipping: %s", src)
                continue
            try:
                shutil.copytree(src_path, dest_root / src_path.name, dirs_exist_ok=True)
            except OSError as exc:
                logger.warning("could not vendor skill %s: %s", src, exc)

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

        rc, stdout, stderr, elapsed, timed_out = self._run_subprocess(
            cmd, workdir, timeout
        )
        return self._build_result(workdir, rc, stdout, stderr, elapsed, timed_out)

    # ------------------------------------------------------------------
    # Overridden by each provider
    # ------------------------------------------------------------------

    def _build_cmd(self, prompt: str, workdir: Path) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def _postprocess_output(self, stdout: str) -> tuple[str, dict | None]:
        """Render stdout for the coding log and extract usage if available.

        Base implementation keeps the historical behaviour — the first
        ``_RAW_OUTPUT_CAP`` chars of stdout, no usage.  Providers that emit a
        structured event stream (Claude Code's ``stream-json``) override this to
        render the full tool-call trajectory and pull out token/cost usage.
        """
        return stdout[:_RAW_OUTPUT_CAP], None

    # ------------------------------------------------------------------
    # Core subprocess runner  (mirrors ARC's _run_subprocess)
    # ------------------------------------------------------------------

    def _run_subprocess(
        self,
        cmd: list[str],
        workdir: Path,
        timeout_sec: int,
    ) -> tuple[int, str, str, float, bool]:
        """Run *cmd* in *workdir*; return ``(rc, stdout, stderr, elapsed, timed_out)``."""
        workdir.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        timed_out = False

        # The agent writes + runs scripts that `import evalvitals`; its bash
        # `python`/`python3` must be the SAME interpreter running this loop (the
        # project venv), not whatever base `python` is on PATH. The loop is
        # often started as `/path/.venv/bin/python run.py` WITHOUT activating
        # the venv, so prepend the running interpreter's bin dir to PATH.
        agent_env = {**os.environ}
        bindir = os.path.dirname(sys.executable)
        if bindir:
            agent_env["PATH"] = bindir + os.pathsep + agent_env.get("PATH", "")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=workdir,
            env=agent_env,
            start_new_session=True,  # own process group for clean kill
        )

        try:
            stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            # SIGTERM first; give it 5 s; then SIGKILL
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except OSError:
                pass
            try:
                stdout_bytes, stderr_bytes = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except OSError:
                    pass
                stdout_bytes, stderr_bytes = b"", b""
                try:
                    proc.communicate(timeout=5)
                except Exception:  # noqa: BLE001
                    pass

        elapsed = time.monotonic() - start
        rc = proc.returncode if proc.returncode is not None else -1
        return rc, _to_text(stdout_bytes), _to_text(stderr_bytes), elapsed, timed_out

    def _build_result(
        self,
        workdir: Path,
        rc: int,
        stdout: str,
        stderr: str,
        elapsed: float,
        timed_out: bool,
    ) -> CliAgentResult:
        files = _collect_py_files(workdir)
        error: str | None = None
        if timed_out:
            error = f"[TIMEOUT] agent killed after {elapsed:.0f}s"
        elif rc != 0 and not files:
            error = f"Exited {rc}: {stderr[:500]}"

        raw_output, usage = self._postprocess_output(stdout)
        logger.debug(
            "%s: rc=%d files=%s elapsed=%.1fs timed_out=%s",
            self._provider_name, rc, list(files), elapsed, timed_out,
        )
        return CliAgentResult(
            files=files,
            provider_name=self._provider_name,
            elapsed_sec=elapsed,
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
        return _render_claude_stream(stdout)


class CodexAgent(_CliAgentBase):
    """OpenAI Codex CLI backend (``codex exec``).

    Requires ``codex`` installed and ``OPENAI_API_KEY`` set.
    """

    _provider_name = "codex"

    def _install_skills(self, workdir: Path) -> None:
        """Vendor skill dirs, then surface them via the workdir's ``AGENTS.md``.

        Codex has no ``Skill`` tool and does not scan ``.claude/skills/``, but it
        does read ``AGENTS.md`` in its working directory — so the same vendored
        ``SKILL.md`` files are exposed as read-and-apply guides."""
        super()._install_skills(workdir)
        names = [Path(s).name for s in self._skills if Path(s).is_dir()]
        if not names:
            return
        section = "\n".join([
            "# Agent Skills (vendored)",
            "",
            "Before writing any figure/plot, read and APPLY these style guides:",
            "",
            *[f"- `.claude/skills/{n}/SKILL.md`" for n in names],
            "",
            "They govern chart-type choice and styling only — never change the "
            "data, the analysis, or the required output format.",
            "",
        ])
        agents_md = workdir / "AGENTS.md"
        try:
            existing = (
                agents_md.read_text(encoding="utf-8").rstrip() + "\n\n"
                if agents_md.exists() else ""
            )
            agents_md.write_text(existing + section, encoding="utf-8")
        except OSError as exc:
            logger.warning("could not write AGENTS.md for codex skills: %s", exc)

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

    binary = config.binary_path or shutil.which(_BINARY_DEFAULTS[provider]) or ""
    if not binary:
        raise RuntimeError(
            f"CLI agent binary for {provider!r} not found in PATH. "
            f"Install '{_BINARY_DEFAULTS[provider]}' or pass "
            f"CliAgentConfig(binary_path='/path/to/{_BINARY_DEFAULTS[provider]}')."
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
