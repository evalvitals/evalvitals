"""ClaudeModel — the Claude Code CLI wrapped as an M1–M5 judge.

Tested against a fake ``claude`` executable (a tiny shell script) so no real
CLI, auth, or network is involved.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from evalvitals.core.capability import Capability
from evalvitals.eval_agent import ClaudeModel


def _fake_claude(tmp_path: Path, body: str) -> str:
    """Write an executable fake claude binary and return its path."""
    p = tmp_path / "claude"
    p.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(p)


def test_missing_binary_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))  # nothing named claude on PATH
    with pytest.raises(RuntimeError, match="CLAUDE_PATH"):
        ClaudeModel()


def test_generate_returns_stdout(tmp_path):
    binary = _fake_claude(tmp_path, 'echo "HELLO FROM JUDGE"')
    judge = ClaudeModel(binary_path=binary)
    assert judge.generate("hi") == "HELLO FROM JUDGE"
    assert Capability.GENERATE in judge.capabilities


def test_model_flag_forwarded(tmp_path):
    # Echo back the argv so the test can assert the --model flag.
    binary = _fake_claude(tmp_path, 'echo "$@"')
    judge = ClaudeModel(binary_path=binary, model="claude-fable-5")
    out = judge.generate("question")
    assert "--model claude-fable-5" in out
    assert "--dangerously-skip-permissions" in out


def test_nonzero_exit_without_output_raises(tmp_path):
    binary = _fake_claude(tmp_path, 'echo "auth expired" >&2; exit 3')
    judge = ClaudeModel(binary_path=binary)
    with pytest.raises(RuntimeError, match="exited 3"):
        judge.generate("hi")


def test_empty_response_warns_and_returns_empty(tmp_path):
    binary = _fake_claude(tmp_path, "exit 0")
    judge = ClaudeModel(binary_path=binary)
    with pytest.warns(UserWarning, match="empty response"):
        assert judge.generate("hi") == ""


def test_images_listed_in_prompt_and_workspace_added(tmp_path):
    binary = _fake_claude(tmp_path, 'echo "$@"')
    img = tmp_path / "m2_effects.png"
    img.write_bytes(b"\x89PNG fake")
    judge = ClaudeModel(binary_path=binary)
    out = judge.generate("look at the figures", images=[img])
    assert "Images available in workspace: m2_effects.png" in out
    assert "--add-dir" in out


def test_timeout_raises(tmp_path):
    binary = _fake_claude(tmp_path, "sleep 5")
    judge = ClaudeModel(binary_path=binary, timeout_sec=1)
    with pytest.raises(RuntimeError, match="timed out"):
        judge.generate("hi")


def test_missing_image_paths_are_skipped(tmp_path):
    binary = _fake_claude(tmp_path, 'echo "$@"')
    judge = ClaudeModel(binary_path=binary)
    out = judge.generate("q", images=[Path("/nonexistent/x.png")])
    assert "Images available" not in out


def test_binary_path_must_be_executable(tmp_path):
    p = tmp_path / "claude"
    p.write_text("not executable", encoding="utf-8")
    os.chmod(p, 0o644)
    with pytest.raises(RuntimeError):
        ClaudeModel(binary_path=str(p))


def test_utf8_output_decodes_under_any_locale(tmp_path, monkeypatch):
    """Regression: Fable's answers contain UTF-8 punctuation (em dashes etc.);
    subprocess text decoding must not depend on the container locale
    ('ascii' codec can't decode byte 0xc3 killed M3 in the first run)."""
    binary = _fake_claude(tmp_path, "printf 'HYPOTHESIS: caf\\303\\251 \\342\\200\\224 fine\\n'")
    monkeypatch.setenv("LC_ALL", "C")
    monkeypatch.setenv("LANG", "C")
    judge = ClaudeModel(binary_path=binary)
    out = judge.generate("hi")
    assert "café — fine" in out
