from __future__ import annotations

import pytest

from evalvitals.analysis.cli import main as explore_main
from evalvitals.cli import main


def test_top_level_cli_help(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "EvalVitals command-line interface" in out
    # chat REPL is retired; the single-shot explore entry replaces it.
    assert "explore" in out
    assert "chat" not in out


def test_top_level_explore_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["explore", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "single-shot exploratory analysis" in out.lower() or "no interactive repl" in out.lower()


def test_top_level_dashboard_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["dashboard", "--help"])
    assert exc.value.code == 0
    assert "Streamlit dashboard" in capsys.readouterr().out


def test_explore_entry_help(capsys):
    with pytest.raises(SystemExit) as exc:
        explore_main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "evalvitals-explore" in out
    assert "--dashboard" in out


def test_chat_repl_is_retired():
    # The interactive chat shell and its CLI entry no longer exist.
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("evalvitals.analysis.chat")
    from evalvitals.analysis import cli

    assert not hasattr(cli, "chat_main")
