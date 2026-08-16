from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import scripts.run_qwen_transcript_answer as cli  # noqa: E402


def test_cli_prepare_uses_only_fixed_private_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "fixed-root"
    monkeypatch.setattr(cli, "PRIVATE_ROOT", root)
    calls = []

    def fake_prepare(path: Path, *, prepared_at: str):
        calls.append((path, prepared_at))
        return {"status": "inputs_frozen"}

    monkeypatch.setattr(cli, "prepare_answer_test", fake_prepare)

    assert cli.main(["prepare"], now=lambda: "2026-08-16T12:00:00Z") == 0
    assert calls == [(root, "2026-08-16T12:00:00Z")]
    assert json.loads(capsys.readouterr().out) == {"status": "inputs_frozen"}


def test_cli_score_reads_only_fixed_judgment_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "fixed-root"
    root.mkdir()
    judgments = {"schema_version": 1, "reviewer": "owner", "rows": []}
    (root / "review-judgments-input.json").write_text(json.dumps(judgments))
    monkeypatch.setattr(cli, "PRIVATE_ROOT", root)
    calls = []

    def fake_score(path: Path, payload: dict, *, scored_at: str):
        calls.append((path, payload, scored_at))
        return {
            "status": "scored_postfreeze",
            "decision": "formulation_bridge_observed",
            "reviewed_answer_count": 24,
            "conditions": {
                "qwen_transcript": {"correct": 1},
                "whisper_transcript": {"correct": 1},
            },
        }

    monkeypatch.setattr(cli, "score_answer_review", fake_score)

    assert cli.main(["score"], now=lambda: "2026-08-16T12:02:00Z") == 0
    assert calls == [(root, judgments, "2026-08-16T12:02:00Z")]
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "status": "scored_postfreeze",
        "decision": "formulation_bridge_observed",
        "reviewed_answer_count": 24,
        "correct": {"qwen_transcript": 1, "whisper_transcript": 1},
    }


def test_cli_help_exposes_no_retry_delete_or_model_override(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match="0"):
        cli.main(["--help"])
    help_text = capsys.readouterr().out.casefold()
    assert "prepare" in help_text
    assert "review-packet" in help_text
    assert "retry" not in help_text
    assert "delete" not in help_text
    assert "--model" not in help_text
    assert "--root" not in help_text
