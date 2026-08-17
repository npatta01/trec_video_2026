from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import errors as genai_errors


import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_vqa_deadline_gemini_full as runner  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def panel_fixture(tmp_path: Path) -> dict[str, Any]:
    media_root = tmp_path / "testing"
    videos = media_root / "videos"
    original = media_root / "videos_original"
    videos.mkdir(parents=True)
    original.mkdir(parents=True)

    first = videos / "clip-a.webm"
    first.write_bytes(b"native-webm-a")
    second = original / "clip-b.mp4"
    second.write_bytes(b"native-mp4-b")

    rows = [
        {
            "availability": "available",
            "frame_paths": ["ignored/frame.jpg"],
            "frame_timestamps": [0.0, 2.0],
            "query_id": 1,
            "question": "What happens first?",
            "source_sha256": _sha256(first),
            "video_id": "clip-a.webm",
        },
        {
            "availability": "available",
            "frame_paths": ["ignored/frame.jpg"],
            "frame_timestamps": [0.0, 4.0],
            "query_id": 2,
            "question": "What happens next?",
            "source_sha256": _sha256(first),
            "video_id": "clip-a.webm",
        },
        {
            "availability": "available",
            "frame_paths": ["ignored/frame.jpg"],
            "frame_timestamps": [0.0, 3.0],
            "query_id": 3,
            "question": "What is shown?",
            "source_sha256": _sha256(second),
            "video_id": "clip-b",
        },
        {
            "availability": "unavailable",
            "frame_paths": [],
            "frame_timestamps": [],
            "query_id": 4,
            "question": "What is missing?",
            "source_sha256": None,
            "video_id": "missing-video",
        },
    ]
    requests = tmp_path / "requests.json"
    requests.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    return {"requests": requests, "media_root": media_root, "rows": rows}


class FakeFiles:
    def __init__(self, *, fail_stems: set[str] | None = None) -> None:
        self.upload_calls: list[tuple[Path, Any]] = []
        self.fail_stems = fail_stems or set()

    def upload(self, *, file: Path, config: Any) -> Any:
        self.upload_calls.append((file, config))
        if file.stem in self.fail_stems:
            raise RuntimeError("simulated upload failure")
        return SimpleNamespace(
            name=f"files/{file.stem}",
            uri=f"https://provider.invalid/files/{file.stem}",
            mime_type=config.mime_type,
            state="ACTIVE",
            size_bytes=file.stat().st_size,
        )


class FakeModels:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        index = len(self.calls)
        return {
            "response_id": f"response-{index}",
            "model_version": "gemini-3.7-flash-test",
            "text": f"direct answer {index}",
            "finish_reason": "STOP",
            "usage_metadata": {
                "prompt_token_count": 100,
                "candidates_token_count": 12,
                "thoughts_token_count": 88,
                "total_token_count": 200,
            },
        }


class FakeClient:
    def __init__(self, *, fail_upload_stems: set[str] | None = None) -> None:
        self.files = FakeFiles(fail_stems=fail_upload_stems)
        self.models = FakeModels()


def _server_error(code: int) -> genai_errors.ServerError:
    return genai_errors.ServerError(code, {"message": f"transient {code}"})


class Upload503ThenSuccessFiles(FakeFiles):
    def __init__(self) -> None:
        super().__init__()
        self.attempts_by_name: dict[str, int] = {}

    def upload(self, *, file: Path, config: Any) -> Any:
        self.attempts_by_name[file.name] = self.attempts_by_name.get(file.name, 0) + 1
        self.upload_calls.append((file, config))
        if file.name == "clip-a.webm" and self.attempts_by_name[file.name] == 1:
            raise _server_error(503)
        return SimpleNamespace(
            name=f"files/{file.stem}",
            uri=f"https://provider.invalid/files/{file.stem}",
            mime_type=config.mime_type,
            state="ACTIVE",
            size_bytes=file.stat().st_size,
        )


class Upload503ThenSuccessClient(FakeClient):
    def __init__(self) -> None:
        self.files = Upload503ThenSuccessFiles()
        self.models = FakeModels()


class Get503ThenSuccessFiles(FakeFiles):
    def __init__(self) -> None:
        super().__init__()
        self.get_calls: list[str] = []
        self.get_attempts_by_name: dict[str, int] = {}

    def upload(self, *, file: Path, config: Any) -> Any:
        self.upload_calls.append((file, config))
        return SimpleNamespace(
            name=f"files/{file.stem}",
            uri=f"https://provider.invalid/files/{file.stem}",
            mime_type=config.mime_type,
            state="PROCESSING",
            size_bytes=file.stat().st_size,
        )

    def get(self, *, name: str) -> Any:
        self.get_calls.append(name)
        self.get_attempts_by_name[name] = self.get_attempts_by_name.get(name, 0) + 1
        if name == "files/clip-a" and self.get_attempts_by_name[name] == 1:
            raise _server_error(503)
        upload_path, upload_config = next(
            (path, config)
            for path, config in self.upload_calls
            if f"files/{path.stem}" == name
        )
        return SimpleNamespace(
            name=name,
            uri=f"https://provider.invalid/{name}",
            mime_type=upload_config.mime_type,
            state="ACTIVE",
            size_bytes=Path(upload_path).stat().st_size,
        )


class Get503ThenSuccessClient(FakeClient):
    def __init__(self) -> None:
        self.files = Get503ThenSuccessFiles()
        self.models = FakeModels()


class Upload503TwiceFiles(FakeFiles):
    def __init__(self) -> None:
        super().__init__()
        self.attempts_by_name: dict[str, int] = {}

    def upload(self, *, file: Path, config: Any) -> Any:
        self.attempts_by_name[file.name] = self.attempts_by_name.get(file.name, 0) + 1
        self.upload_calls.append((file, config))
        if file.name == "clip-a.webm":
            raise _server_error(503)
        return SimpleNamespace(
            name=f"files/{file.stem}",
            uri=f"https://provider.invalid/files/{file.stem}",
            mime_type=config.mime_type,
            state="ACTIVE",
            size_bytes=file.stat().st_size,
        )


class Upload503TwiceClient(FakeClient):
    def __init__(self) -> None:
        self.files = Upload503TwiceFiles()
        self.models = FakeModels()


class UploadErrorFiles(FakeFiles):
    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error
        self.attempts_by_name: dict[str, int] = {}

    def upload(self, *, file: Path, config: Any) -> Any:
        self.attempts_by_name[file.name] = self.attempts_by_name.get(file.name, 0) + 1
        self.upload_calls.append((file, config))
        if file.name == "clip-a.webm":
            raise self.error
        return SimpleNamespace(
            name=f"files/{file.stem}",
            uri=f"https://provider.invalid/files/{file.stem}",
            mime_type=config.mime_type,
            state="ACTIVE",
            size_bytes=file.stat().st_size,
        )


class UploadErrorClient(FakeClient):
    def __init__(self, error: BaseException) -> None:
        self.files = UploadErrorFiles(error)
        self.models = FakeModels()


def test_upload_503_retries_exact_upload_once_and_freezes_retry_provenance(
    panel_fixture: dict[str, Any], tmp_path: Path
) -> None:
    client = Upload503ThenSuccessClient()
    sleeps: list[float] = []

    result = runner.run_full(
        requests_path=panel_fixture["requests"],
        media_root=panel_fixture["media_root"],
        output_root=tmp_path / "run",
        expected_count=4,
        client=client,
        sleep=sleeps.append,
    )

    assert result["status"] == "predictions_frozen"
    assert client.files.attempts_by_name == {"clip-a.webm": 2, "clip-b.mp4": 1}
    assert sleeps == [10]
    terminals = [
        json.loads(path.read_text())
        for path in (tmp_path / "run" / "uploads").glob("*.json")
        if not path.name.endswith(".attempt.json")
    ]
    terminal = next(item for item in terminals if item["retry_count"] == 1)
    assert terminal["provider_retry_policy"] == {
        "max_retries": 1,
        "code": 503,
        "backoff_seconds": 10,
        "workers": runner.UPLOAD_WORKERS,
    }
    assert terminal["retry_count"] == 1
    assert terminal["retry_phase"] == "upload"
    assert terminal["retry_first_error"]["type"] == "ServerError"
    assert terminal["possible_orphan_duplicate"] is True
    attempt = json.loads(
        (tmp_path / "run" / "uploads" / "clip-a.webm.attempt.json").read_text()
    )
    assert attempt["provider_retry_policy"] == terminal["provider_retry_policy"]


def test_get_503_retries_exact_get_once_without_orphan_duplicate(
    panel_fixture: dict[str, Any], tmp_path: Path
) -> None:
    client = Get503ThenSuccessClient()
    sleeps: list[float] = []

    result = runner.run_full(
        requests_path=panel_fixture["requests"],
        media_root=panel_fixture["media_root"],
        output_root=tmp_path / "run",
        expected_count=4,
        client=client,
        sleep=sleeps.append,
    )

    assert result["status"] == "predictions_frozen"
    assert len(client.files.upload_calls) == 2
    assert len(client.files.get_calls) == 3
    assert sleeps.count(10) == 1
    terminals = [
        json.loads(path.read_text())
        for path in (tmp_path / "run" / "uploads").glob("*.json")
        if not path.name.endswith(".attempt.json")
    ]
    terminal = next(item for item in terminals if item["retry_count"] == 1)
    assert terminal["retry_count"] == 1
    assert terminal["retry_phase"] == "get"
    assert terminal["possible_orphan_duplicate"] is False
    assert terminal["file_name"].startswith("files/clip-")


def test_two_upload_503s_freeze_terminal_after_one_retry_without_third_call(
    panel_fixture: dict[str, Any], tmp_path: Path
) -> None:
    client = Upload503TwiceClient()
    sleeps: list[float] = []

    result = runner.run_full(
        requests_path=panel_fixture["requests"],
        media_root=panel_fixture["media_root"],
        output_root=tmp_path / "run",
        expected_count=4,
        client=client,
        sleep=sleeps.append,
    )

    assert result["status"] == "predictions_frozen"
    assert result["upload_failed_count"] == 2
    assert client.files.attempts_by_name["clip-a.webm"] == 2
    assert sleeps == [10]
    terminal = json.loads(
        (tmp_path / "run" / "uploads" / "clip-a.webm.json").read_text()
    )
    assert terminal["status"] == "upload_failed"
    assert terminal["retry_count"] == 1
    assert terminal["possible_orphan_duplicate"] is True


@pytest.mark.parametrize(
    "error", [_server_error(500), ValueError("bad input")], ids=["server-500", "value"]
)
def test_non_503_upload_errors_are_terminal_without_retry(
    panel_fixture: dict[str, Any], tmp_path: Path, error: BaseException
) -> None:
    client = UploadErrorClient(error)
    sleeps: list[float] = []

    result = runner.run_full(
        requests_path=panel_fixture["requests"],
        media_root=panel_fixture["media_root"],
        output_root=tmp_path / "run",
        expected_count=4,
        client=client,
        sleep=sleeps.append,
    )

    assert result["upload_failed_count"] == 2
    assert client.files.attempts_by_name["clip-a.webm"] == 1
    assert sleeps == []
    terminal = json.loads(
        (tmp_path / "run" / "uploads" / "clip-a.webm.json").read_text()
    )
    assert terminal["retry_count"] == 0
    assert "retry_phase" not in terminal


def test_resume_preserves_existing_upload_failure_and_retries_only_missing_video(
    panel_fixture: dict[str, Any], tmp_path: Path
) -> None:
    output_root = tmp_path / "run"
    first_client = UploadErrorClient(ValueError("bad input"))
    runner.run_full(
        requests_path=panel_fixture["requests"],
        media_root=panel_fixture["media_root"],
        output_root=output_root,
        expected_count=4,
        client=first_client,
    )
    upload_dir = output_root / "uploads"
    preserved = {
        name: (upload_dir / name).read_bytes()
        for name in ("clip-a.webm.json", "clip-a.webm.attempt.json")
    }
    (upload_dir / "clip-b.json").unlink()
    (upload_dir / "clip-b.attempt.json").unlink()

    second_client = FakeClient()
    resumed = runner.run_full(
        requests_path=panel_fixture["requests"],
        media_root=panel_fixture["media_root"],
        output_root=output_root,
        expected_count=4,
        client=second_client,
    )

    assert resumed["status"] == "predictions_frozen"
    assert [path.name for path, _ in second_client.files.upload_calls] == [
        "clip-b.mp4"
    ]
    for name, content in preserved.items():
        assert (upload_dir / name).read_bytes() == content


def test_input_contract_resolves_media_and_marks_missing_without_calling_provider(
    panel_fixture: dict[str, Any],
) -> None:
    rows = runner.load_input_rows(
        panel_fixture["requests"],
        media_root=panel_fixture["media_root"],
        expected_count=4,
    )

    assert [row["status"] for row in rows] == ["ready", "ready", "ready", "unavailable"]
    assert rows[0]["media_path"] == rows[1]["media_path"]
    assert rows[2]["media_mime_type"] == "video/mp4"
    assert rows[3]["reason"] == "input_unavailable"


def test_mkv_delivery_plan_uses_supported_mime_or_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "clip.mkv"
    source.write_bytes(b"matroska")

    class Probe:
        stdout = json.dumps(
            {
                "streams": [
                    {"codec_type": "video", "codec_name": "vp9"},
                    {"codec_type": "audio", "codec_name": "opus"},
                ]
            }
        )

    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: Probe())
    plan = runner.probe_delivery_plan(source)

    assert plan is not None
    assert plan["delivery_mode"] == "copy_webm"
    assert plan["delivery_mime_type"] == "video/webm"
    assert plan["delivery_mime_type"] != "video/x-matroska"


def test_dry_run_is_default_safe_and_counts_unique_uploads(
    panel_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_client(**_: Any) -> Any:
        raise AssertionError("dry-run must not construct a provider client")

    monkeypatch.setattr(runner, "client_factory", fail_client)
    result = runner.run_full(
        requests_path=panel_fixture["requests"],
        media_root=panel_fixture["media_root"],
        expected_count=4,
        dry_run=True,
    )

    assert result["status"] == "dry_run"
    assert result["row_count"] == 4
    assert result["native_request_count"] == 3
    assert result["upload_count"] == 2
    assert result["unavailable_count"] == 1


def test_request_uses_promoted_lowres_four_fps_high_thinking_profile() -> None:
    request = runner.build_request(
        row={"query_id": 1, "video_id": "clip-a", "question": "What happens?"},
        upload={
            "file_uri": "https://provider.invalid/files/clip-a",
            "mime_type": "video/webm",
        },
    )

    video = request.contents[0].parts[0]
    assert request.model == runner.MODEL
    assert video.video_metadata.fps == 4.0
    assert (
        video.media_resolution.level
        == runner.types.PartMediaResolutionLevel.MEDIA_RESOLUTION_LOW
    )
    assert request.config.max_output_tokens == 2_048
    assert (
        request.config.thinking_config.thinking_level == runner.types.ThinkingLevel.HIGH
    )
    assert request.config.thinking_config.include_thoughts is False


def test_prohibited_content_without_candidate_count_is_classified_and_billed_exactly() -> None:
    # This is the provider payload observed on live query 87.  Gemini omits
    # candidatesTokenCount when safety blocks generation, but totalTokenCount
    # still includes the hidden thinking tokens that are billable as output.
    raw = {
        "candidates": [
            {"content": {}, "finishReason": "PROHIBITED_CONTENT", "index": 0}
        ],
        "modelVersion": "gemini-3.7-flash",
        "responseId": "safety-blocked-response",
        "usageMetadata": {
            "promptTokenCount": 21_462,
            "thoughtsTokenCount": 604,
            "totalTokenCount": 22_066,
        },
    }

    parsed = runner.parse_response(
        raw=raw,
        row={"query_id": 87, "video_id": "Calloused_Hands-5.webm"},
    )

    assert parsed["status"] == "failed"
    assert parsed["answer"] == ""
    assert parsed["failure_reason"] == "prohibited_content"
    assert parsed["finish_reason"] == "PROHIBITED_CONTENT"
    assert parsed["usage"]["output_tokens"] == 0
    assert parsed["usage"]["metadata"]["output_tokens_derived"] is True
    assert (
        parsed["usage"]["metadata"]["output_tokens_derivation"]
        == "total_tokens_minus_prompt_tokens"
    )
    assert runner._usage_cost(parsed["usage"]) == (21_462, 604, 0.0183615)


def test_nested_max_tokens_is_failed_even_when_text_is_nonempty() -> None:
    parsed = runner.parse_response(
        raw={
            "candidates": [
                {
                    "content": {"parts": [{"text": "truncated answer"}]},
                    "finishReason": "MAX_TOKENS",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 8,
                "totalTokenCount": 120,
            },
        },
        row={"query_id": 116, "video_id": "row-116.webm"},
    )

    assert parsed["status"] == "failed"
    assert parsed["answer"] == "truncated answer"
    assert parsed["finish_reason"] == "MAX_TOKENS"
    assert parsed["failure_reason"] == "max_tokens"
    assert parsed["usage"]["output_tokens"] == 8


def test_live_run_uploads_each_video_once_freezes_raw_before_parse_and_resumes(
    panel_fixture: dict[str, Any], tmp_path: Path
) -> None:
    client = FakeClient()
    output_root = tmp_path / "run"
    parse_seen: list[bool] = []

    def parse_response(*, raw: Any, row: Any) -> dict[str, Any]:
        parse_seen.append((output_root / "raw-freeze.json").is_file())
        return runner.parse_response(raw=raw, row=row)

    first = runner.run_full(
        requests_path=panel_fixture["requests"],
        media_root=panel_fixture["media_root"],
        output_root=output_root,
        expected_count=4,
        client=client,
        parse_response=parse_response,
        created_at="2026-08-16T20:00:00Z",
    )

    assert first["status"] == "predictions_frozen"
    assert first["native_request_count"] == 3
    assert first["completed_count"] == 3
    assert first["unavailable_count"] == 1
    assert first["prompt_tokens"] == 300
    assert first["output_tokens"] == 300
    assert len(client.files.upload_calls) == 2
    assert len(client.models.calls) == 3
    assert parse_seen == [True, True, True]

    freeze = json.loads((output_root / "predictions-freeze.json").read_text())
    assert freeze["status"] == "predictions_frozen"
    assert freeze["raw_freeze_sha256"]
    assert {row["status"] for row in freeze["rows"]} == {"succeeded", "unavailable"}
    assert all(
        {
            "query_id",
            "video_id",
            "status",
            "answer",
            "raw_response_sha256",
            "elapsed_seconds",
        }
        <= row.keys()
        for row in freeze["rows"]
    )
    unavailable = next(row for row in freeze["rows"] if row["status"] == "unavailable")
    assert unavailable["answer"] == ""
    assert unavailable["raw_response_sha256"] is None

    resumed_client = FakeClient()
    resumed = runner.run_full(
        requests_path=panel_fixture["requests"],
        media_root=panel_fixture["media_root"],
        output_root=output_root,
        expected_count=4,
        client=resumed_client,
    )
    assert resumed["status"] == "predictions_frozen"
    assert resumed_client.files.upload_calls == []
    assert resumed_client.models.calls == []


def test_upload_failure_is_terminal_fallback_and_does_not_abort_panel(
    panel_fixture: dict[str, Any], tmp_path: Path
) -> None:
    client = FakeClient(fail_upload_stems={"clip-a"})
    output_root = tmp_path / "run"

    first = runner.run_full(
        requests_path=panel_fixture["requests"],
        media_root=panel_fixture["media_root"],
        output_root=output_root,
        expected_count=4,
        client=client,
    )

    assert first["status"] == "predictions_frozen"
    assert first["completed_count"] == 1
    assert first["upload_failed_count"] == 2
    assert first["prompt_tokens"] == 100
    assert first["output_tokens"] == 100
    assert first["observed_cost_usd"] == pytest.approx(0.00045)
    freeze = json.loads((output_root / "predictions-freeze.json").read_text())
    assert [row["status"] for row in freeze["rows"]] == [
        "upload_failed",
        "upload_failed",
        "succeeded",
        "unavailable",
    ]

    resumed_client = FakeClient()
    runner.run_full(
        requests_path=panel_fixture["requests"],
        media_root=panel_fixture["media_root"],
        output_root=output_root,
        expected_count=4,
        client=resumed_client,
    )
    assert resumed_client.files.upload_calls == []
    assert resumed_client.models.calls == []


def _parallel_panel_fixture(tmp_path: Path, *, count: int = 6) -> dict[str, Any]:
    media_root = tmp_path / "testing"
    videos = media_root / "videos"
    videos.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for index in range(count):
        video_id = f"clip-{index}.webm"
        path = videos / video_id
        path.write_bytes(f"native-webm-{index}".encode())
        rows.append(
            {
                "availability": "available",
                "frame_paths": [],
                "frame_timestamps": [0.0, 2.0],
                "query_id": index + 1,
                "question": f"What is shown in clip {index}?",
                "source_sha256": _sha256(path),
                "video_id": video_id,
            }
        )
    requests = tmp_path / "requests.json"
    requests.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    return {"requests": requests, "media_root": media_root, "rows": rows}


class BlockingFiles(FakeFiles):
    def __init__(self, *, count: int) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._first_wave = threading.Barrier(runner.UPLOAD_WORKERS, timeout=5)
        self._count = count
        self._active = 0
        self.max_active = 0
        self.started: list[str] = []

    def upload(self, *, file: Path, config: Any) -> Any:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            self.started.append(file.name)
            first_wave = len(self.started) <= runner.UPLOAD_WORKERS
        try:
            if first_wave:
                self._first_wave.wait()
            else:
                time.sleep(0.01)
            return SimpleNamespace(
                name=f"files/{file.stem}",
                uri=f"https://provider.invalid/files/{file.stem}",
                mime_type=config.mime_type,
                state="ACTIVE",
                size_bytes=file.stat().st_size,
            )
        finally:
            with self._lock:
                self._active -= 1


class BlockingClient(FakeClient):
    def __init__(self, *, count: int) -> None:
        self.files = BlockingFiles(count=count)
        self.models = FakeModels()


def test_uploads_use_a_bounded_rolling_pool_and_terminal_artifacts(
    tmp_path: Path,
) -> None:
    panel = _parallel_panel_fixture(tmp_path)
    client = BlockingClient(count=6)

    result = runner.run_full(
        requests_path=panel["requests"],
        media_root=panel["media_root"],
        output_root=tmp_path / "run",
        expected_count=6,
        client=client,
    )

    assert result["status"] == "predictions_frozen"
    assert result["completed_count"] == 6
    assert client.files.max_active == runner.UPLOAD_WORKERS == 2
    assert len(client.files.upload_calls) == 0
    assert len(client.files.started) == 6
    uploads = [
        path
        for path in (tmp_path / "run" / "uploads").glob("*.json")
        if not path.name.endswith(".attempt.json")
    ]
    assert len(uploads) == 6
    assert len(list((tmp_path / "run" / "uploads").glob("*.attempt.json"))) == 6


def test_upload_interrupt_drains_started_workers_without_reserving_queued_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _parallel_panel_fixture(tmp_path, count=6)
    client = BlockingClient(count=6)
    output_root = tmp_path / "run"
    original_wait = runner.wait
    interrupted = False

    def interrupting_wait(*args: Any, **kwargs: Any) -> Any:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt()
        return original_wait(*args, **kwargs)

    monkeypatch.setattr(runner, "wait", interrupting_wait)

    with pytest.raises(KeyboardInterrupt):
        runner.run_full(
            requests_path=panel["requests"],
            media_root=panel["media_root"],
            output_root=output_root,
            expected_count=6,
            client=client,
        )

    upload_dir = output_root / "uploads"
    terminals = {
        path.name.removesuffix(".json")
        for path in upload_dir.glob("*.json")
        if not path.name.endswith(".attempt.json")
    }
    attempts = {
        path.name.removesuffix(".attempt.json")
        for path in upload_dir.glob("*.attempt.json")
    }
    assert len(client.files.started) == runner.UPLOAD_WORKERS
    assert len(attempts) == runner.UPLOAD_WORKERS
    assert terminals == attempts
    assert not (upload_dir / "clip-4.webm.attempt.json").exists()
    assert not (upload_dir / "clip-5.webm.attempt.json").exists()


def test_estimate_cap_is_checked_before_live_provider_access(
    panel_fixture: dict[str, Any], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="estimated cost"):
        runner.run_full(
            requests_path=panel_fixture["requests"],
            media_root=panel_fixture["media_root"],
            output_root=tmp_path / "run",
            expected_count=4,
            client=FakeClient(),
            max_estimated_cost_usd=0.000001,
        )


def test_observed_cost_guard_blocks_calls_before_the_cap_is_spent(
    panel_fixture: dict[str, Any], tmp_path: Path
) -> None:
    client = FakeClient()
    output_root = tmp_path / "run"

    result = runner.run_full(
        requests_path=panel_fixture["requests"],
        media_root=panel_fixture["media_root"],
        output_root=output_root,
        expected_count=4,
        client=client,
        max_observed_cost_usd=0.001,
    )

    assert result["status"] == "predictions_frozen"
    assert result["completed_count"] == 0
    assert client.models.calls == []
    freeze = json.loads((output_root / "predictions-freeze.json").read_text())
    assert [row["status"] for row in freeze["rows"]] == [
        "cost_cap_blocked",
        "cost_cap_blocked",
        "cost_cap_blocked",
        "unavailable",
    ]
