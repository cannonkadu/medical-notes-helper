"""Smoke tests for Medical Notes Helper CLI and entrypoints."""

from __future__ import annotations

from pathlib import Path

from app import __version__
from app.cli import main


def test_version_defined() -> None:
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_cli_hello_smoke(capsys, tmp_path: Path) -> None:
    storage_file = tmp_path / "notes.json"
    code = main(["hello", "--plain", "--storage", str(storage_file)])
    captured = capsys.readouterr()
    assert code == 0
    assert "Medical Notes Helper" in captured.out
    assert "anonymized administrative notes" in captured.out


def test_cli_suggest_smoke(capsys, tmp_path: Path) -> None:
    storage_file = tmp_path / "notes.json"
    code = main(
        [
            "suggest",
            "the patient arrived",
            "--top-k",
            "3",
            "--smoothing",
            "kneser_ney",
            "--plain",
            "--storage",
            str(storage_file),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "1." in captured.out
    assert "3." in captured.out


def test_cli_generate_and_stats_smoke(capsys, tmp_path: Path) -> None:
    storage_file = tmp_path / "notes.json"
    code = main(
        [
            "generate",
            "prior authorization",
            "--words",
            "6",
            "--plain",
            "--storage",
            str(storage_file),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "prior authorization" in captured.out.lower()

    code = main(["stats", "--plain", "--storage", str(storage_file)])
    captured = capsys.readouterr()
    assert code == 0
    assert "Vocab:" in captured.out


def test_cli_evaluate_smoke(capsys) -> None:
    code = main(["evaluate", "--plain"])
    captured = capsys.readouterr()
    assert code == 0
    assert "kneser_ney" in captured.out
    assert "laplace" in captured.out


def test_cli_feedback_and_add_phrase_smoke(capsys, tmp_path: Path) -> None:
    storage_file = tmp_path / "notes.json"
    code = main(
        [
            "feedback",
            "--context",
            "follow up",
            "--word",
            "appointment",
            "--accept",
            "--plain",
            "--storage",
            str(storage_file),
        ]
    )
    assert code == 0
    assert "Accepted" in capsys.readouterr().out

    code = main(
        [
            "add-phrase",
            "Telehealth intake packet was verified by clinic staff.",
            "--plain",
            "--storage",
            str(storage_file),
        ]
    )
    assert code == 0
    assert "Added custom phrase" in capsys.readouterr().out


def test_cli_interactive_draft_smoke(monkeypatch, capsys, tmp_path: Path) -> None:
    storage_file = tmp_path / "notes.json"
    inputs = iter(["the patient", "1", ":k 3", ":save Intake Note", ":quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    code = main(["draft", "--plain", "--storage", str(storage_file)])
    captured = capsys.readouterr()
    assert code == 0
    assert "Saved draft 'Intake Note'" in captured.out
    assert "Final draft:" in captured.out
