from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_vqa_mc_gemini_direct_full as runner  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def panel(tmp_path: Path) -> dict[str, object]:
    media_root = tmp_path / "testing"
    videos = media_root / "videos"
    videos.mkdir(parents=True)
    first = videos / "clip-a.webm"
    second = videos / "clip-b.webm"
    first.write_bytes(b"first-video")
    second.write_bytes(b"second-video")
    request_rows = [
        {
            "availability": "available",
            "frame_timestamps": [0.0, 2.0, 4.0],
            "query_id": 1,
            "question": "What happens?",
            "source_sha256": _sha(first),
            "video_id": "clip-a.webm",
        },
        {
            "availability": "available",
            "frame_timestamps": [0.0, 3.0],
            "query_id": 2,
            "question": "Who arrives?",
            "source_sha256": _sha(second),
            "video_id": "clip-b.webm",
        },
        {
            "availability": "unavailable",
            "frame_timestamps": [],
            "query_id": 3,
            "question": "What is missing?",
            "source_sha256": None,
            "video_id": "missing.webm",
        },
    ]
    requests = tmp_path / "requests.json"
    requests.write_text(json.dumps(request_rows), encoding="utf-8")
    task2_rows = [
        {
            "Q_ID": 1,
            "Video_ID": "clip-a.webm",
            "Question": "What happens?",
            "Options": [{"option": value} for value in ["It starts", "It stops", "It fades"]],
        },
        {
            "Q_ID": 2,
            "Video_ID": "clip-b.webm",
            "Question": "Who arrives?",
            "Options": [{"option": value} for value in ["A man", "A woman", "Nobody", "A dog"]],
        },
        {
            "Q_ID": 3,
            "Video_ID": "missing.webm",
            "Question": "What is missing?",
            "Options": [{"option": value} for value in ["Sound", "Video"]],
        },
    ]
    v1 = [
        {
            "Q_ID": row["Q_ID"],
            "Video_ID": row["Video_ID"],
            "Answers": [
                {"Rank": index, "Answer": option["option"]}
                for index, option in enumerate(row["Options"], start=1)
            ],
        }
        for row in task2_rows
    ]
    return {
        "media_root": media_root,
        "requests": requests,
        "task2": json.dumps(task2_rows).encode(),
        "task2_rows": task2_rows,
        "v1": v1,
    }


def test_load_panel_aligns_released_options_with_label_free_media(panel: dict[str, object]) -> None:
    rows = runner.load_panel(
        task2=panel["task2"],
        requests_path=panel["requests"],
        media_root=panel["media_root"],
        expected_count=3,
    )

    assert [row["query_id"] for row in rows] == [1, 2, 3]
    assert rows[0]["options"] == ["It starts", "It stops", "It fades"]
    assert rows[0]["status"] == "ready"
    assert rows[2]["status"] == "unavailable"


def test_build_request_is_blind_to_v1_and_ranks_every_released_option() -> None:
    row = {
        "query_id": 7,
        "video_id": "clip.webm",
        "question": "What happens?",
        "options": ["Old prior answer marker", "Second", "Third"],
        "input_sha256": "a" * 64,
    }
    request = runner.build_request(
        row=row,
        upload={"file_uri": "https://files.invalid/clip", "mime_type": "video/webm"},
    )
    dumped = request.model_dump(mode="json", by_alias=True, exclude_none=True)
    serialized = json.dumps(dumped)

    assert "prior_video_answer" not in serialized
    assert "current_top" not in serialized
    assert "audit" not in serialized.casefold()
    assert "Old prior answer marker" in serialized
    assert dumped["config"]["responseJsonSchema"]["properties"]["ranking"]["maxItems"] == 3
    assert dumped["metadata"]["query_id"] == "7"


@pytest.mark.parametrize(
    ("payload", "count"),
    [
        ({"ranking": [2, 1, 3]}, 3),
        ('{"ranking":[4,2,3,1]}', 4),
    ],
)
def test_parse_ranking_accepts_only_complete_option_permutations(payload: object, count: int) -> None:
    assert runner.parse_ranking(payload, option_count=count) == list(
        json.loads(payload)["ranking"] if isinstance(payload, str) else payload["ranking"]
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"ranking": [1, 1, 2]},
        {"ranking": [1, 2]},
        {"ranking": [0, 1, 2]},
        {"ranking": [1, 2, 4]},
        {"ranking": [True, 2, 3]},
    ],
)
def test_parse_ranking_rejects_incomplete_or_invalid_permutations(payload: object) -> None:
    with pytest.raises(ValueError):
        runner.parse_ranking(payload, option_count=3)


def test_one_full_prediction_freeze_assembles_both_v2_and_v3(panel: dict[str, object]) -> None:
    rows = runner.load_panel(
        task2=panel["task2"],
        requests_path=panel["requests"],
        media_root=panel["media_root"],
        expected_count=3,
    )
    predictions = {
        1: {"status": "passed", "ranking": [2, 3, 1]},
        2: {"status": "passed", "ranking": [4, 1, 2, 3]},
        3: {"status": "unavailable", "ranking": None},
    }

    result = runner.assemble_submissions(
        rows=rows,
        predictions=predictions,
        v1_submission=panel["v1"],
        weak_query_ids={1},
    )

    assert [answer["Answer"] for answer in result["v2"][0]["Answers"]] == [
        "It stops",
        "It fades",
        "It starts",
    ]
    assert result["v2"][1] == panel["v1"][1]
    assert [answer["Answer"] for answer in result["v3"][1]["Answers"]] == [
        "A dog",
        "A man",
        "A woman",
        "Nobody",
    ]
    assert result["v3"][2] == panel["v1"][2]
    assert result["v2_direct_count"] == 1
    assert result["v3_direct_count"] == 2
    assert result["v3_fallback_count"] == 1


def test_cost_gate_uses_batch_rates_and_prior_matched_usage() -> None:
    estimate = runner.estimate_cost(
        ready_rows=2433,
        ready_video_seconds=85407.454,
        historical_prompt_tokens=24_712_790,
        historical_output_tokens=1_078_734,
    )

    assert estimate["projected_cost_usd"] == pytest.approx(11.2899225)
    assert estimate["batch_input_usd_per_million"] == 0.375
    assert estimate["batch_output_usd_per_million"] == 1.875
    with pytest.raises(ValueError, match="cost cap"):
        runner.assert_cost_gate(estimate, max_cost_usd=5.0)
    runner.assert_cost_gate(estimate, max_cost_usd=12.0)


def test_canary_limit_preserves_panel_v1_alignment_and_intersects_weak_set(
    panel: dict[str, object]
) -> None:
    rows = runner.load_panel(
        task2=panel["task2"],
        requests_path=panel["requests"],
        media_root=panel["media_root"],
        expected_count=3,
    )
    selected = runner.limit_inputs(
        rows=rows,
        v1_submission=panel["v1"],
        weak_query_ids={1, 3},
        query_limit=2,
    )

    assert [row["query_id"] for row in selected["rows"]] == [1, 2]
    assert [row["Q_ID"] for row in selected["v1_submission"]] == [1, 2]
    assert selected["weak_query_ids"] == {1}


class _Files:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def upload(self, *, file: Path, config: object) -> object:
        self.calls.append(Path(file))
        return SimpleNamespace(
            name=f"files/{Path(file).stem}",
            uri=f"https://files.invalid/{Path(file).stem}",
            mime_type=config.mime_type,
            state="ACTIVE",
            size_bytes=Path(file).stat().st_size,
        )


class _Batches:
    def __init__(self) -> None:
        self.create_calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.create_calls.append(kwargs)
        return SimpleNamespace(
            name="batches/direct-options-canary",
            display_name=kwargs["config"].display_name,
            model=kwargs["model"],
            state="JOB_STATE_PENDING",
        )

    def get(self, *, name: str) -> object:
        assert name == "batches/direct-options-canary"
        responses = [
            {
                "response": {
                    "responseId": "response-1",
                    "modelVersion": "gemini-3.7-flash-test",
                    "candidates": [
                        {
                            "content": {"parts": [{"text": '{"ranking":[2,3,1]}'}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 100,
                        "candidatesTokenCount": 10,
                        "thoughtsTokenCount": 40,
                        "totalTokenCount": 150,
                    },
                },
                "metadata": {"query_id": "1"},
            },
            {
                "response": {
                    "responseId": "response-2",
                    "modelVersion": "gemini-3.7-flash-test",
                    "candidates": [
                        {
                            "content": {"parts": [{"text": '{"ranking":[4,1,2,3]}'}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 200,
                        "candidatesTokenCount": 10,
                        "thoughtsTokenCount": 90,
                        "totalTokenCount": 300,
                    },
                },
                "metadata": {"query_id": "2"},
            },
        ]
        return SimpleNamespace(
            name=name,
            display_name="trecvid-2026-vqa-mc-direct-options-v2-v3",
            model=runner.MODEL,
            state="JOB_STATE_SUCCEEDED",
            dest=SimpleNamespace(
                inlined_responses=list(reversed(responses))
            ),
        )


class _Client:
    def __init__(self) -> None:
        self.files = _Files()
        self.batches = _Batches()


def test_prepare_upload_and_submit_freeze_one_nonduplicated_batch(
    panel: dict[str, object], tmp_path: Path
) -> None:
    rows = runner.load_panel(
        task2=panel["task2"],
        requests_path=panel["requests"],
        media_root=panel["media_root"],
        expected_count=3,
    )
    root = tmp_path / "run"
    prepared = runner.prepare_run(
        rows=rows,
        root=root,
        v1_submission=panel["v1"],
        weak_query_ids={1},
        task2_sha256=hashlib.sha256(panel["task2"]).hexdigest(),
        v1_sha256="1" * 64,
        weak_queue_sha256="2" * 64,
        max_cost_usd=12.0,
        created_at="2026-08-22T12:00:00Z",
    )
    assert prepared["status"] == "prepared"
    assert prepared["ready_count"] == 2
    assert prepared["unavailable_count"] == 1

    client = _Client()
    uploaded = runner.upload_run(root=root, client=client, sleep=lambda _: None)
    assert uploaded == {"status": "uploaded", "uploaded_count": 2, "failed_count": 0}
    assert len(client.files.calls) == 2

    submitted = runner.submit_batch(
        root=root,
        client=client,
        submitted_at="2026-08-22T12:05:00Z",
    )
    assert submitted["status"] == "submitted"
    assert submitted["request_count"] == 2
    assert len(client.batches.create_calls) == 1
    src = client.batches.create_calls[0]["src"]
    serialized = json.dumps(
        [
            request.model_dump(mode="json", by_alias=True, exclude_none=True)
            for request in src.inlined_requests
        ]
    )
    assert "prior_video_answer" not in serialized
    assert "current_top" not in serialized

    status = runner.record_status(
        root=root,
        client=client,
        checked_at="2026-08-22T12:10:00Z",
    )
    assert status == {"status": "succeeded", "response_count": 2}

    frozen = runner.freeze_run(root=root, frozen_at="2026-08-22T12:11:00Z")
    assert frozen["status"] == "complete"
    assert frozen["passed_count"] == 2
    assert frozen["fallback_count"] == 1
    assert frozen["v2_direct_count"] == 1
    assert frozen["v3_direct_count"] == 2
    assert frozen["observed_cost_usd"] == pytest.approx(0.00039375)
    assert json.loads((root / "submission-v2.json").read_text())[0]["Answers"][0][
        "Answer"
    ] == "It stops"
    assert json.loads((root / "submission-v3.json").read_text())[1]["Answers"][0][
        "Answer"
    ] == "A dog"

    artifact_names = (
        "predictions-freeze.json",
        "submission-v2.json",
        "submission-v3.json",
        "summary.json",
    )
    before_resume = {name: (root / name).read_bytes() for name in artifact_names}
    resumed = runner.freeze_run(root=root, frozen_at="2026-08-22T12:12:00Z")
    assert resumed == frozen
    assert {name: (root / name).read_bytes() for name in artifact_names} == before_resume

    with pytest.raises(ValueError, match="already exists"):
        runner.submit_batch(
            root=root,
            client=client,
            submitted_at="2026-08-22T12:06:00Z",
        )
    assert len(client.batches.create_calls) == 1


def test_freeze_falls_back_for_provider_error_without_response_id(
    panel: dict[str, object], tmp_path: Path
) -> None:
    class ProviderErrorBatches(_Batches):
        def get(self, *, name: str) -> object:
            job = super().get(name=name)
            job.dest.inlined_responses[0] = {
                "error": {"code": 13, "message": "Internal error encountered."},
                "metadata": {"query_id": "2"},
            }
            return job

    rows = runner.load_panel(
        task2=panel["task2"],
        requests_path=panel["requests"],
        media_root=panel["media_root"],
        expected_count=3,
    )
    root = tmp_path / "provider-error"
    runner.prepare_run(
        rows=rows,
        root=root,
        v1_submission=panel["v1"],
        weak_query_ids={1, 2},
        task2_sha256=hashlib.sha256(panel["task2"]).hexdigest(),
        v1_sha256="1" * 64,
        weak_queue_sha256="2" * 64,
        max_cost_usd=12.0,
        created_at="2026-08-26T12:00:00Z",
    )
    client = _Client()
    client.batches = ProviderErrorBatches()
    runner.upload_run(root=root, client=client, sleep=lambda _: None)
    runner.submit_batch(root=root, client=client, submitted_at="2026-08-26T12:05:00Z")
    runner.record_status(root=root, client=client, checked_at="2026-08-26T12:10:00Z")

    frozen = runner.freeze_run(root=root, frozen_at="2026-08-26T12:11:00Z")

    assert frozen["passed_count"] == 1
    assert frozen["fallback_count"] == 2
    predictions = json.loads((root / "predictions-freeze.json").read_text())
    assert predictions["usage_complete"] is False
    provider_error = predictions["rows"][1]
    assert provider_error["status"] == "invalid"
    assert provider_error["response_id"] is None
    assert provider_error["error"] == {
        "code": 13,
        "message": "Internal error encountered.",
    }
    v3 = json.loads((root / "submission-v3.json").read_text())
    assert v3[1] == panel["v1"][1]


def test_freeze_rejects_empty_error_mapping_without_response_id(
    panel: dict[str, object], tmp_path: Path
) -> None:
    class EmptyErrorBatches(_Batches):
        def get(self, *, name: str) -> object:
            job = super().get(name=name)
            job.dest.inlined_responses[0] = {
                "error": {},
                "metadata": {"query_id": "2"},
            }
            return job

    rows = runner.load_panel(
        task2=panel["task2"],
        requests_path=panel["requests"],
        media_root=panel["media_root"],
        expected_count=3,
    )
    root = tmp_path / "empty-provider-error"
    runner.prepare_run(
        rows=rows,
        root=root,
        v1_submission=panel["v1"],
        weak_query_ids={1, 2},
        task2_sha256=hashlib.sha256(panel["task2"]).hexdigest(),
        v1_sha256="1" * 64,
        weak_queue_sha256="2" * 64,
        max_cost_usd=12.0,
        created_at="2026-08-26T12:00:00Z",
    )
    client = _Client()
    client.batches = EmptyErrorBatches()
    runner.upload_run(root=root, client=client, sleep=lambda _: None)
    runner.submit_batch(root=root, client=client, submitted_at="2026-08-26T12:05:00Z")
    runner.record_status(root=root, client=client, checked_at="2026-08-26T12:10:00Z")

    with pytest.raises(ValueError, match="response identity is missing"):
        runner.freeze_run(root=root, frozen_at="2026-08-26T12:11:00Z")


def test_prepare_run_records_a_distinct_repair_experiment_id(
    panel: dict[str, object], tmp_path: Path
) -> None:
    rows = runner.load_panel(
        task2=panel["task2"],
        requests_path=panel["requests"],
        media_root=panel["media_root"],
        expected_count=3,
    )

    runner.prepare_run(
        rows=rows,
        root=tmp_path / "repair",
        v1_submission=panel["v1"],
        weak_query_ids={1},
        task2_sha256=hashlib.sha256(panel["task2"]).hexdigest(),
        v1_sha256="1" * 64,
        weak_queue_sha256="2" * 64,
        max_cost_usd=12.0,
        created_at="2026-08-23T12:00:00Z",
        experiment_id="vqa-mc-gemini-direct-v2b-v3b-repair9-20260823",
    )

    protocol = json.loads((tmp_path / "repair" / "protocol.json").read_text())
    assert protocol["experiment_id"] == "vqa-mc-gemini-direct-v2b-v3b-repair9-20260823"


def test_prepare_cli_accepts_explicit_revised_release_sources(tmp_path: Path) -> None:
    task2 = tmp_path / "task2-v2.json"
    v1 = tmp_path / "v1-v2.json"
    weak_queue = tmp_path / "weak-v2.jsonl"

    args = runner._parser().parse_args(
        [
            "prepare",
            "--task2",
            str(task2),
            "--v1",
            str(v1),
            "--weak-queue",
            str(weak_queue),
            "--expected-weak-count",
            "817",
            "--experiment-id",
            "vqa-mc-task2-v2-refresh-20260826",
            "--output",
            str(tmp_path / "run"),
            "--max-cost-usd",
            "15",
        ]
    )

    assert args.task2 == task2
    assert args.v1 == v1
    assert args.weak_queue == weak_queue
    assert args.expected_weak_count == 817
    assert args.experiment_id == "vqa-mc-task2-v2-refresh-20260826"


def test_prepare_cli_freezes_explicit_revised_release_sources(
    panel: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task2 = tmp_path / "task2-v2.json"
    task2.write_bytes(panel["task2"])
    v1 = tmp_path / "v1-v2.json"
    v1.write_text(json.dumps(panel["v1"]), encoding="utf-8")
    weak_queue = tmp_path / "weak-v2.jsonl"
    weak_queue.write_text('{"Q_ID": 1}\n', encoding="utf-8")
    monkeypatch.setattr(runner, "REQUESTS_PATH", panel["requests"])
    monkeypatch.setattr(runner, "MEDIA_ROOT", panel["media_root"])

    result = runner._prepare_cli(
        output=tmp_path / "run",
        query_limit=None,
        max_cost_usd=12.0,
        task2_path=task2,
        v1_path=v1,
        weak_queue_path=weak_queue,
        expected_weak_count=1,
        expected_query_count=3,
        experiment_id="vqa-mc-task2-v2-refresh-20260826",
    )

    assert result["status"] == "prepared"
    protocol = json.loads((tmp_path / "run" / "protocol.json").read_text())
    assert protocol["experiment_id"] == "vqa-mc-task2-v2-refresh-20260826"
    assert protocol["source"]["task2_sha256"] == _sha(task2)
    assert protocol["source"]["v1_sha256"] == _sha(v1)
    assert protocol["source"]["weak_queue_sha256"] == _sha(weak_queue)
    assert protocol["weak_query_ids"] == [1]
