"""TDD coverage for the resumable VQA submission orchestrator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from trec_video_2026.vqa_submission import verify_prediction_freeze


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_vqa_submission.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_vqa_submission", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load run_vqa_submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_config(tmp_path: Path, *, row_count: int = 1) -> Path:
    config = tmp_path / "fixture-config.json"
    examples = [
        {
            "query_id": 7 + index,
            "video_id": f"fixture-video-{index + 1}",
            "category": "general",
            "question": "What happens in the clip?",
        }
        for index in range(row_count)
    ]
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "fixture-run",
                "model_id": "gemini-3.1-pro-preview",
                "fixture": True,
                "examples": examples,
            }
        ),
        encoding="utf-8",
    )
    return config


def _prepare_one_row(tmp_path: Path) -> tuple[object, Path]:
    module = _load_script()
    config = _fixture_config(tmp_path)
    run_dir = tmp_path / "run"
    assert module.main(
        ["prepare", "--config", str(config), "--output", str(run_dir)]
    ) == 0
    return module, run_dir


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_prepare_writes_manifest_and_refuses_overwrite(tmp_path: Path) -> None:
    module = _load_script()
    config = _fixture_config(tmp_path)
    output = tmp_path / "run"

    assert module.main(
        ["prepare", "--config", str(config), "--output", str(output)]
    ) == 0
    assert (output / "run-manifest.json").is_file()
    assert (output / "evidence" / "manifest.json").is_file()

    with pytest.raises(FileExistsError):
        module.main(
            ["prepare", "--config", str(config), "--output", str(output)]
        )


def test_run_resumes_only_hash_matching_completed_rows(tmp_path: Path) -> None:
    module, run_dir = _prepare_one_row(tmp_path)
    transport = module.FixtureTransport()

    first = module.run_prepared(run_dir, transport=transport, limit=1)
    second = module.run_prepared(run_dir, transport=transport)

    assert first.completed_rows == 1
    assert second.reused_rows == 1
    assert transport.generate_count > 0


def test_fixture_run_completes_ten_candidates_verifier_and_freeze(
    tmp_path: Path,
) -> None:
    module, run_dir = _prepare_one_row(tmp_path)

    assert module.main(
        ["run", "--prepared", str(run_dir), "--transport", "fixture"]
    ) == 0
    assert module.main(["freeze", "--prepared", str(run_dir)]) == 0

    assert len(_jsonl(run_dir / "candidates.jsonl")) == 10
    assert (run_dir / "failures.jsonl").read_text(encoding="utf-8") == ""
    freeze = json.loads((run_dir / "freeze.json").read_text(encoding="utf-8"))
    verify_prediction_freeze(run_dir / "ranked-predictions.json", freeze)


def test_fixture_run_writes_timeline_evidence_and_checkpoint_metadata(
    tmp_path: Path,
) -> None:
    module, run_dir = _prepare_one_row(tmp_path)

    summary = module.run_prepared(
        run_dir, transport=module.FixtureTransport(), limit=1
    )

    assert summary.completed_rows == 1
    timeline = _jsonl(run_dir / "timeline.jsonl")
    assert len(timeline) == 1
    assert timeline[0]["status"] == "completed"
    evidence = json.loads(
        (run_dir / "evidence" / "manifest.json").read_text(encoding="utf-8")
    )
    assert evidence["global_frame_count"] == 8
    assert evidence["materialized_frame_hashes"]
    assert len(_jsonl(run_dir / "candidates.jsonl")) == 10
    attempts = _jsonl(run_dir / "attempts.jsonl")
    assert attempts
    assert all(
        {
            "request_sha256",
            "response_sha256",
            "status",
            "elapsed_seconds",
            "usage",
            "uploaded_file_deletion_status",
            "failure_code",
        }.issubset(row)
        for row in attempts
    )
    terminal = json.loads(
        (run_dir / "terminal-status.json").read_text(encoding="utf-8")
    )
    assert terminal["status"] == "inferred"
    assert terminal["labels_loaded"] is False


def test_invalid_timeline_is_global_only_and_invalid_verifier_uses_precedence(
    tmp_path: Path,
) -> None:
    module, run_dir = _prepare_one_row(tmp_path)

    class InvalidTransport(module.FixtureTransport):
        def generate(self, request: dict[str, object]) -> dict[str, object]:
            if request["stage"] == "timeline":
                return {"events": [{"start_seconds": -1}]}
            if request["stage"] == "verifier":
                return {"invalid": True}
            return super().generate(request)

    module.run_prepared(run_dir, transport=InvalidTransport())
    evidence = json.loads(
        (run_dir / "evidence" / "manifest.json").read_text(encoding="utf-8")
    )
    assert evidence["timeline_fallback_reason"] == "invalid_timeline"
    assert evidence["local_frame_count"] == 0
    predictions = json.loads(
        (run_dir / "ranked-predictions.json").read_text(encoding="utf-8")
    )
    assert predictions["predictions"][0]["ranking_source"] == "deterministic-fallback"
    assert tuple(predictions["predictions"][0]["ranked_pass_ids"]) == tuple(
        module.FALLBACK_PASS_PRECEDENCE
    )


def test_candidate_shortfall_binds_configured_baseline_ranked_list(
    tmp_path: Path,
) -> None:
    module = _load_script()
    config = _fixture_config(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["baseline_ranked_list"] = ["configured baseline one", "configured baseline two"]
    config.write_text(json.dumps(payload), encoding="utf-8")
    run_dir = tmp_path / "run"
    assert module.main(
        ["prepare", "--config", str(config), "--output", str(run_dir)]
    ) == 0

    class DuplicateTransport(module.FixtureTransport):
        def generate(self, request: dict[str, object]) -> dict[str, object]:
            if request["stage"] in {"answer", "backfill"}:
                return {"answer": "the same response", "confidence": 0.5}
            return super().generate(request)

    module.run_prepared(run_dir, transport=DuplicateTransport())
    predictions = json.loads(
        (run_dir / "ranked-predictions.json").read_text(encoding="utf-8")
    )
    assert predictions["ranking_source"] == "configured-baseline-ranked-list"
    assert predictions["baseline_ranked_list"] == [
        "configured baseline one",
        "configured baseline two",
    ]
    terminal = json.loads(
        (run_dir / "terminal-status.json").read_text(encoding="utf-8")
    )
    assert terminal["candidate_shortfall"] == 9


def test_status_is_artifact_only_and_preflight_auth_error_is_safe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    module, run_dir = _prepare_one_row(tmp_path)
    module.run_prepared(run_dir, transport=module.FixtureTransport())
    assert module.main(["status", "--prepared", str(run_dir)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["inferred_rows"] == 1
    assert status["labels_loaded"] is False

    class NoCredentials(module.ProviderAuthenticationError):
        pass

    def fail_preflight() -> object:
        raise NoCredentials(module.AUTHENTICATION_ERROR_MESSAGE)

    monkeypatch.setattr(module, "preflight_google_transport", fail_preflight)
    assert module.main(["preflight-live"]) == 2
    assert capsys.readouterr().err.strip() == module.AUTHENTICATION_ERROR_MESSAGE


def test_preflight_success_emits_only_safe_fields(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "preflight_google_transport", lambda: object())
    monkeypatch.setattr(module, "_google_sdk_version", lambda: "2.12.1")

    assert module.main(["preflight-live"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "model_id": "gemini-3.1-pro-preview",
        "sdk_version": "2.12.1",
        "authentication_configured": True,
    }
