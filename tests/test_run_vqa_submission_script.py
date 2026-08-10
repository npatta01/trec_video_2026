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
                "fixture_mode": True,
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


def test_prepare_requires_explicit_fixture_mode(tmp_path: Path) -> None:
    module = _load_script()
    config = _fixture_config(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload.pop("fixture_mode")
    config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fixture_mode"):
        module.main(["prepare", "--config", str(config), "--output", str(tmp_path / "run")])


def test_two_row_fixture_keeps_private_evidence_and_one_record_per_row(
    tmp_path: Path,
) -> None:
    module = _load_script()
    config = _fixture_config(tmp_path, row_count=2)
    run_dir = tmp_path / "run"
    assert module.main(
        ["prepare", "--config", str(config), "--output", str(run_dir)]
    ) == 0
    assert module.run_prepared(run_dir, transport=module.FixtureTransport()).completed_rows == 2

    assert len(_jsonl(run_dir / "timeline.jsonl")) == 2
    assert len(_jsonl(run_dir / "candidates.jsonl")) == 20
    for query_id in (7, 8):
        row_manifest = run_dir / "evidence" / "rows" / str(query_id) / "manifest.json"
        assert row_manifest.is_file()
        assert json.loads(row_manifest.read_text(encoding="utf-8"))["query_id"] == query_id


def test_crash_writes_immediate_attempt_checkpoint_and_resumes_without_rebilling(
    tmp_path: Path,
) -> None:
    module, run_dir = _prepare_one_row(tmp_path)

    class CrashTransport(module.FixtureTransport):
        def generate(self, request: dict[str, object]) -> dict[str, object]:
            if self.generate_count == 3:
                raise KeyboardInterrupt("simulated crash")
            return super().generate(request)

    with pytest.raises(KeyboardInterrupt):
        module.run_prepared(run_dir, transport=CrashTransport())
    checkpoint_count = len(_jsonl(run_dir / "attempts.jsonl"))
    assert checkpoint_count >= 3
    with pytest.raises(ValueError, match="in_flight_unknown"):
        module.run_prepared(run_dir, transport=module.FixtureTransport())


def test_partial_run_cannot_freeze_and_manifest_tamper_rejects_status(
    tmp_path: Path,
) -> None:
    module = _load_script()
    config = _fixture_config(tmp_path, row_count=2)
    run_dir = tmp_path / "run"
    assert module.main(
        ["prepare", "--config", str(config), "--output", str(run_dir)]
    ) == 0
    module.run_prepared(run_dir, transport=module.FixtureTransport(), limit=1)
    with pytest.raises(ValueError, match="complete|ordered"):
        module.freeze(run_dir)

    manifest_path = run_dir / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_id"] = "tampered-model"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest|model"):
        module.status(run_dir)


def test_malformed_response_is_failed_and_retryable(tmp_path: Path) -> None:
    module, run_dir = _prepare_one_row(tmp_path)

    class MalformedOnce(module.FixtureTransport):
        def __init__(self) -> None:
            super().__init__()
            self.malformed = True

        def generate(self, request: dict[str, object]) -> dict[str, object]:
            response = super().generate(request)
            if self.malformed and request["stage"] == "answer":
                self.malformed = False
                return {}
            return response

    first = MalformedOnce()
    module.run_prepared(run_dir, transport=first)
    attempts = _jsonl(run_dir / "attempts.jsonl")
    assert any(row["status"] == "failed" for row in attempts)
    second = module.FixtureTransport()
    summary = module.run_prepared(run_dir, transport=second)
    assert summary.completed_rows == 1
    assert second.generate_count > 0


def test_shortfall_uses_ten_answer_baseline_candidates(tmp_path: Path) -> None:
    module = _load_script()
    config = _fixture_config(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["baseline_ranked_list"] = [f"baseline response {index}" for index in range(10)]
    config.write_text(json.dumps(payload), encoding="utf-8")
    run_dir = tmp_path / "run"
    assert module.main(
        ["prepare", "--config", str(config), "--output", str(run_dir)]
    ) == 0

    class DuplicateTransport(module.FixtureTransport):
        def generate(self, request: dict[str, object]) -> dict[str, object]:
            if request["stage"] in {"answer", "backfill"}:
                return {"answer": "duplicate response", "confidence": 0.5}
            return super().generate(request)

    module.run_prepared(run_dir, transport=DuplicateTransport())
    predictions = json.loads(
        (run_dir / "ranked-predictions.json").read_text(encoding="utf-8")
    )
    assert predictions["candidate_shortfall"] == 0
    assert len(predictions["predictions"][0]["candidates"]) == 10
    assert len({item["answer"] for item in predictions["predictions"][0]["candidates"]}) == 10


def test_nonfixture_prepare_resolves_media_and_records_source_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    dataset_root = tmp_path / "canonical-data"
    dataset_root.mkdir()
    media = dataset_root / "video.mp4"
    media.write_bytes(b"isolated synthetic media")
    config = tmp_path / "live-config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "live-run",
                "model_id": "gemini-3.1-pro-preview",
                "fixture_mode": False,
                "examples": [
                    {
                        "query_id": 1,
                        "video_id": "video-1",
                        "category": "general",
                        "question": "What happens?",
                        "media": "video.mp4",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "CANONICAL_DATA_ROOT", dataset_root)
    monkeypatch.setattr(
        module,
        "probe_media",
        lambda path: {
            "duration_seconds": 2.0,
            "has_video": True,
            "has_audio": True,
            "stream_types": ["video", "audio"],
        },
    )
    monkeypatch.setattr(
        module,
        "source_fingerprint",
        lambda path: {"sha256": "a" * 64, "size_bytes": path.stat().st_size},
    )
    run_dir = tmp_path / "run"
    assert module.main(
        [
            "prepare",
            "--config",
            str(config),
            "--dataset-root",
            str(dataset_root),
            "--output",
            str(run_dir),
        ]
    ) == 0
    manifest = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest["examples"][0]["source"]["sha256"] == "a" * 64
    assert manifest["examples"][0]["media"]["duration_seconds"] == 2.0


def _live_fixture_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load_script()
    dataset_root = tmp_path / "canonical-data"
    dataset_root.mkdir()
    media = dataset_root / "video.mp4"
    media.write_bytes(b"synthetic live media")
    config = tmp_path / "live-config.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "live-run",
                "model_id": "gemini-3.1-pro-preview",
                "fixture_mode": False,
                "examples": [
                    {
                        "query_id": 1,
                        "video_id": "video-1",
                        "question": "What happens?",
                        "media": "video.mp4",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "CANONICAL_DATA_ROOT", dataset_root)
    monkeypatch.setattr(
        module,
        "probe_media",
        lambda path: {
            "duration_seconds": 2.0,
            "has_video": True,
            "has_audio": True,
            "stream_types": ["video", "audio"],
        },
    )
    monkeypatch.setattr(
        module,
        "source_fingerprint",
        lambda path: {
            "sha256": module._hash_path(path),
            "size_bytes": path.stat().st_size,
        },
    )
    run_dir = tmp_path / "run"
    module.prepare(config, run_dir, dataset_root=dataset_root)
    return module, run_dir, media


def test_nonfixture_never_uses_fixture_evidence_and_timeline_has_source_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, run_dir, media = _live_fixture_run(tmp_path, monkeypatch)
    captured: list[dict[str, object]] = []

    def fail_fixture(*args: object, **kwargs: object) -> object:
        raise AssertionError("non-fixture run used synthetic fixture evidence")

    source_sha = module._hash_path(media)
    monkeypatch.setattr(
        module,
        "_decode_live_frame_inventory",
        lambda path, duration: [
            {
                "stream_index": 0,
                "frame_index": index,
                "actual_pts_seconds": (index + 0.5) * duration / 40,
                "time_base": "1/1000",
                "source_media_sha256": source_sha,
            }
            for index in range(40)
        ],
    )

    def fake_extract(path: Path, frame: object, destination: Path) -> str:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"frame-{destination.name}".encode())
        return module._hash_path(destination)

    class Capture(module.FixtureTransport):
        def generate(self, request: dict[str, object]) -> dict[str, object]:
            captured.append(request)
            return super().generate(request)

    monkeypatch.setattr(module, "_fixture_evidence", fail_fixture)
    monkeypatch.setattr(module, "_extract_live_frame", fake_extract)
    module.run_prepared(run_dir, transport=Capture())
    timeline = next(row for row in captured if row["stage"] == "timeline")
    assert timeline["media_path"] == str(media)
    assert timeline["audio_path"] == str(media)


def test_conservative_synthesis_receives_first_nine_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, run_dir = _prepare_one_row(tmp_path)
    original = module.build_answer_requests
    calls: list[tuple[str, ...]] = []

    def capture(row: object, timeline: object, plan: object, *, candidates=()):
        calls.append(tuple(candidate.answer for candidate in candidates))
        return original(row, timeline, plan, candidates=candidates)

    monkeypatch.setattr(module, "build_answer_requests", capture)
    module.run_prepared(run_dir, transport=module.FixtureTransport())
    assert len(calls) == 2
    assert calls[0] == ()
    assert len(calls[1]) == 9


def test_resume_does_not_duplicate_completed_timeline_record(tmp_path: Path) -> None:
    module, run_dir = _prepare_one_row(tmp_path)
    module.run_prepared(run_dir, transport=module.FixtureTransport())
    module.run_prepared(run_dir, transport=module.FixtureTransport())
    assert len(_jsonl(run_dir / "timeline.jsonl")) == 1


def test_started_checkpoint_without_result_fails_closed(tmp_path: Path) -> None:
    module, run_dir = _prepare_one_row(tmp_path)
    prepared = module.PreparedRun.load(run_dir)
    request = module._timeline_request(prepared, prepared.examples[0])
    module._append_jsonl(
        run_dir / "attempts.jsonl",
        {
            "query_id": 7,
            "video_id": "fixture-video-1",
            "stage": "timeline",
            "pass_id": None,
            "request": request,
            "request_sha256": module.canonical_request_sha256(request),
            "status": "started",
            "response": None,
        },
    )
    with pytest.raises(ValueError, match="in_flight_unknown"):
        module.run_prepared(run_dir, transport=module.FixtureTransport())


def test_malformed_timeline_is_failed_not_cached_completed(tmp_path: Path) -> None:
    module, run_dir = _prepare_one_row(tmp_path)

    class BadTimeline(module.FixtureTransport):
        def generate(self, request: dict[str, object]) -> dict[str, object]:
            if request["stage"] == "timeline":
                return {}
            return super().generate(request)

    module.run_prepared(run_dir, transport=BadTimeline())
    timeline_attempt = [
        row for row in _jsonl(run_dir / "attempts.jsonl") if row["stage"] == "timeline"
    ][-1]
    assert timeline_attempt["status"] == "failed"


def test_missing_pass_is_filled_by_unique_backfill(tmp_path: Path) -> None:
    module, run_dir = _prepare_one_row(tmp_path)

    class MissingOne(module.FixtureTransport):
        def generate(self, request: dict[str, object]) -> dict[str, object]:
            if request["stage"] == "answer" and request["pass_id"] == module.PASS_IDS[-1]:
                return {}
            if request["stage"] == "backfill":
                return {"answer": "independent backfill", "confidence": 0.4}
            return super().generate(request)

    module.run_prepared(run_dir, transport=MissingOne())
    rows = _jsonl(run_dir / "candidates.jsonl")
    assert [row["pass_id"] for row in rows] == list(module.PASS_IDS)
    assert len({row["answer"] for row in rows}) == 10


def test_baseline_materializes_after_frontier_failure(tmp_path: Path) -> None:
    module = _load_script()
    config = _fixture_config(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["baseline_ranked_list"] = [f"baseline {index}" for index in range(10)]
    config.write_text(json.dumps(payload), encoding="utf-8")
    run_dir = tmp_path / "run"
    module.prepare(config, run_dir)

    class AllBad(module.FixtureTransport):
        def generate(self, request: dict[str, object]) -> dict[str, object]:
            if request["stage"] in {"answer", "backfill"}:
                return {}
            return super().generate(request)

    module.run_prepared(run_dir, transport=AllBad())
    prediction = json.loads((run_dir / "ranked-predictions.json").read_text())
    assert prediction["candidate_shortfall"] == 0
    assert len(prediction["predictions"][0]["candidates"]) == 10


def test_freeze_rejects_terminal_labels_loaded(tmp_path: Path) -> None:
    module, run_dir = _prepare_one_row(tmp_path)
    module.run_prepared(run_dir, transport=module.FixtureTransport())
    terminal_path = run_dir / "terminal-status.json"
    terminal = json.loads(terminal_path.read_text())
    terminal["labels_loaded"] = True
    terminal_path.write_text(json.dumps(terminal))
    with pytest.raises(ValueError, match="labels_loaded"):
        module.freeze(run_dir)


def test_legitimate_plain_text_is_not_label_audit_leakage(tmp_path: Path) -> None:
    module, run_dir = _prepare_one_row(tmp_path)

    class ScoreAnswer(module.FixtureTransport):
        def generate(self, request: dict[str, object]) -> dict[str, object]:
            if request["stage"] == "answer":
                return {"answer": f"The score is {request['pass_id']}", "confidence": 0.5}
            return super().generate(request)

    module.run_prepared(run_dir, transport=ScoreAnswer())
    module.freeze(run_dir)


def test_status_does_not_refingerprint_removed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, run_dir, media = _live_fixture_run(tmp_path, monkeypatch)
    media.unlink()
    result = module.status(run_dir)
    assert result["status"] == "prepared"
