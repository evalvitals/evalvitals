from __future__ import annotations

import pytest

from evalvitals.analysis.cli import chat_main
from evalvitals.analysis.cli import main as m2_explore_main
from evalvitals.cli import main


def test_top_level_cli_help(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "EvalVitals command-line interface" in out
    assert "chat" in out


def test_top_level_chat_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["chat", "--help"])
    assert exc.value.code == 0
    assert "Start an interactive EvalVitals chat session" in capsys.readouterr().out


def test_top_level_dashboard_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["dashboard", "--help"])
    assert exc.value.code == 0
    assert "Streamlit dashboard" in capsys.readouterr().out


def test_m2_compat_help(capsys):
    with pytest.raises(SystemExit) as exc:
        chat_main(["--help"])
    assert exc.value.code == 0
    assert "evalvitals-m2-chat" in capsys.readouterr().out


def test_m2_explore_help(capsys):
    with pytest.raises(SystemExit) as exc:
        m2_explore_main(["--help"])
    assert exc.value.code == 0
    assert "evalvitals-m2-explore" in capsys.readouterr().out
