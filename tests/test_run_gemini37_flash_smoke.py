"""Boundary tests for the provider-free Gemini 3.7 smoke CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_gemini37_flash_smoke as cli  # noqa: E402


def test_help_exposes_only_fixed_commands_and_no_unsafe_controls() -> None:
    help_text = cli._parser().format_help()
    for command in (
        "prepare",
        "preflight",
        "upload",
        "submit",
        "status",
        "freeze",
        "bind-annotations",
        "score",
        "show-status",
    ):
        assert command in help_text
    for forbidden in (
        "retry",
        "delete",
        "model-override",
        "panel",
        "credential",
        "api-key",
    ):
        assert forbidden not in help_text


def test_prepare_is_one_domain_transition_and_emits_aggregate_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    root = tmp_path / "run"
    monkeypatch.setattr(cli, "CANONICAL_SOURCE_ROOT", source)
    monkeypatch.setattr(cli, "PRIVATE_RUN_ROOT", root)
    calls: list[tuple[Path, Path]] = []

    def fake_prepare(*, source_root: Path, root: Path, created_at: str) -> dict[str, Any]:
        calls.append((source_root, root))
        return {
            "status": "inputs_frozen",
            "rows": [{"row_id": "private-row", "question": "private?"}],
            "source_protocol": {"path": "/private/source"},
            "estimated_cost_ceiling_usd": 0.0989,
        }

    monkeypatch.setattr(cli.smoke, "prepare_protocol", fake_prepare)
    assert cli.main(["prepare"]) == 0
    assert calls == [(source, root)]
    output_text = capsys.readouterr().out
    output = json.loads(output_text)
    assert output == {
        "status": "inputs_frozen",
        "counts": {"rows": 1},
        "estimated_cost_ceiling_usd": 0.0989,
    }
    assert "private-row" not in output_text
    assert "/private/source" not in output_text


@pytest.mark.parametrize(
    ("command", "function_name", "result"),
    [
        (
            "freeze",
            "freeze_predictions",
            {"status": "predictions_frozen", "decision": "quality_ready"},
        ),
        (
            "score",
            "score_postfreeze",
            {"status": "scored_postfreeze", "decision": "compatible_no_directional_claim"},
        ),
        (
            "show-status",
            "show_status",
            {"status": "inputs_frozen", "decision": "not_ready"},
        ),
        (
            "bind-annotations",
            "freeze_annotation_bindings",
            {"status": "annotations_frozen", "files": [{"sha256": "private"}]},
        ),
    ],
)
def test_provider_free_commands_construct_no_client(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    command: str,
    function_name: str,
    result: dict[str, Any],
) -> None:
    monkeypatch.setattr(cli, "CANONICAL_SOURCE_ROOT", tmp_path / "source")
    monkeypatch.setattr(cli, "PRIVATE_RUN_ROOT", tmp_path / "run")
    monkeypatch.setattr(cli, "CANONICAL_ANNOTATIONS_ROOT", tmp_path / "annotations")
    monkeypatch.setattr(
        cli, "client_factory", lambda **_: pytest.fail("client constructed"), raising=False
    )
    monkeypatch.setattr(cli.smoke, function_name, lambda *args, **kwargs: result)
    assert cli.main([command]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == result["status"]
    assert output.get("decision") == result.get("decision")


@pytest.mark.parametrize("command", ["preflight", "upload", "submit", "status"])
def test_live_commands_use_environment_key_and_one_transition(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    command: str,
) -> None:
    monkeypatch.setattr(cli, "PRIVATE_RUN_ROOT", tmp_path / "run")
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-secret")
    client_calls: list[str] = []

    def fake_client(**kwargs: Any) -> object:
        client_calls.append(kwargs["api_key"])
        return object()

    monkeypatch.setattr(cli, "client_factory", fake_client, raising=False)
    function_name = {
        "preflight": "preflight_models",
        "upload": "upload_rows",
        "submit": "submit_arm",
        "status": "record_status",
    }[command]

    def fake_transition(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "succeeded", "arm": kwargs.get("arm"), "rows": [{"question": "private"}]}

    monkeypatch.setattr(cli.smoke, function_name, fake_transition)
    argv = [command]
    if command in {"submit", "status"}:
        argv += ["--arm", "control_31pro"]
    assert cli.main(argv) == 0
    assert client_calls == ["synthetic-secret"]
    output_text = capsys.readouterr().out
    output = json.loads(output_text)
    assert output["status"] == "succeeded"
    assert "private" not in output_text


def test_invalid_extra_path_is_rejected_without_domain_call(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    called = False

    def fail_prepare(**_: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(cli.smoke, "prepare_protocol", fail_prepare)
    assert cli.main(["prepare", "--root", str(tmp_path / "arbitrary")]) != 0
    assert called is False
    assert "arbitrary" not in capsys.readouterr().err


def test_domain_error_is_generic_and_does_not_leak_private_details(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_prepare(**_: Any) -> dict[str, Any]:
        raise ValueError("row-000 What happens? /private/source Bearer secret")

    monkeypatch.setattr(cli.smoke, "prepare_protocol", fail_prepare)
    assert cli.main(["prepare"]) != 0
    output = capsys.readouterr()
    assert json.loads(output.out) == {"status": "error", "error": "command failed"}
    assert "row-000" not in output.out
    assert "What happens" not in output.out
    assert "/private/source" not in output.out
    assert "Bearer" not in output.out


def test_cli_rejects_private_status_strings_without_echoing_them(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli.smoke,
        "show_status",
        lambda _: {"status": "private-provider-uri", "decision": "private-question"},
    )
    assert cli.main(["show-status"]) != 0
    output = capsys.readouterr().out
    assert "private-provider-uri" not in output
    assert "private-question" not in output
    assert json.loads(output) == {"status": "error", "error": "command failed"}
