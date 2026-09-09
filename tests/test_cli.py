"""Test CLI parsing, commands, and argument handling."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest import mock

import pytest

from dentaltensor.cli import build_parser, cmd_info, cmd_predict, cmd_version, main


def test_cli_parser_help():
    """Verify CLI parser produces formatted help text."""
    parser = build_parser()
    help_text = parser.format_help()
    assert "DentalTensor Vision v1.0" in help_text
    assert "predict" in help_text
    assert "info" in help_text
    assert "version" in help_text
    assert "modal" not in help_text.lower()


def test_cli_version_command(capsys):
    """Verify version command output."""
    parser = build_parser()
    args = parser.parse_args(["version"])
    code = cmd_version(args)
    assert code == 0
    captured = capsys.readouterr()
    assert "dentaltensor-vision 1.0" in captured.out


def test_cli_info_command(capsys):
    """Verify info command output contains architecture, metrics, and thresholds."""
    parser = build_parser()
    args = parser.parse_args(["info"])
    code = cmd_info(args)
    assert code == 0
    captured = capsys.readouterr()
    assert "DentalTensor Vision v1.0" in captured.out
    assert "Nathan Asif" in captured.out
    assert "0.687" in captured.out  # mAP@50
    assert "0.680" in captured.out  # Precision
    assert "0.666" in captured.out  # Recall
    assert "0.35" in captured.out   # calculus threshold
    assert "0.55" in captured.out   # caries threshold
    assert "Local Pre-Trained Inference" in captured.out
    assert "modal" not in captured.out.lower()


def test_cli_predict_parser_arguments():
    """Verify predict argument options parsing including device and model."""
    parser = build_parser()
    args = parser.parse_args([
        "predict",
        "image.jpg",
        "--json",
        "--threshold", "0.65",
        "--output", "out.json",
        "--device", "cpu",
        "--model", "custom.pt",
    ])
    assert args.command == "predict"
    assert args.image == "image.jpg"
    assert args.json is True
    assert args.threshold == 0.65
    assert args.output == "out.json"
    assert args.device == "cpu"
    assert args.model == "custom.pt"


def test_cli_predict_missing_image(capsys):
    """Verify clean error code and message when image does not exist."""
    parser = build_parser()
    args = parser.parse_args(["predict", "non_existent_image_12345.jpg"])
    code = cmd_predict(args)
    assert code == 1
    captured = capsys.readouterr()
    assert "Error: Image file not found" in captured.err


def test_cli_main_entrypoint(monkeypatch, capsys):
    """Verify main entrypoint with --help."""
    monkeypatch.setattr(sys, "argv", ["dentaltensor", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "DentalTensor Vision v1.0" in captured.out
