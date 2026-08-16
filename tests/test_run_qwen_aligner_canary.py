from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import scripts.run_qwen_aligner_canary as runner  # noqa: E402


def test_score_inputs_rejects_not_ready_before_loading_private_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner,
        "status_canary",
        lambda root: {"status": "predictions_frozen", "decision": "not_ready"},
    )

    def fail_load(path: Path):
        raise AssertionError("labels were opened")

    monkeypatch.setattr(runner, "_load_json", fail_load)

    with pytest.raises(ValueError, match="not ready|ready_for_quality_test"):
        runner._score_inputs(
            tmp_path,
            tmp_path / "annotations",
            tmp_path / "whisper.json",
        )


def test_score_inputs_calls_shared_guard_before_loading_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner,
        "status_canary",
        lambda root: {
            "status": "predictions_frozen",
            "decision": "ready_for_quality_test",
        },
    )
    guard_calls: list[Path] = []

    def reject_guard(root: Path) -> None:
        guard_calls.append(root)
        raise ValueError("shared quality readiness guard rejected")

    monkeypatch.setattr(runner, "require_quality_scoring_ready", reject_guard)

    def fail_load(path: Path):
        raise AssertionError("score inputs were opened")

    monkeypatch.setattr(runner, "_load_json", fail_load)

    with pytest.raises(ValueError, match="shared quality readiness guard rejected"):
        runner._score_inputs(
            tmp_path,
            tmp_path / "annotations",
            tmp_path / "whisper.json",
        )
    assert guard_calls == [tmp_path]
