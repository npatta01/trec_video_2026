from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from trec_video_2026.vqa_experiments.vqa_mc_adaptive_agent import InspectionPlan
from trec_video_2026.vqa_experiments.vqa_mc_poc20 import sha256_file


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "vqa_mc_adaptive_runner",
    ROOT / "scripts/run_vqa_mc_gemini_adaptive_agent_poc20.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _row() -> dict[str, Any]:
    return {
        "query_id": "video-a",
        "video_id": "video-a",
        "stratum": "counting",
        "question": "How many times does the person jump?",
        "duration_seconds": 6.0,
        "options": [
            {"option_id": option_id, "text": text}
            for option_id, text in zip(
                "ABCD", ("once", "twice", "three times", "four times"), strict=True
            )
        ],
    }


def _upload() -> dict[str, str]:
    return {
        "file_uri": "https://provider.invalid/video-a",
        "mime_type": "video/webm",
    }


def _plan() -> InspectionPlan:
    return InspectionPlan(
        start_seconds=1.25,
        end_seconds=4.75,
        fps=8.0,
        resolution="high",
        evidence_gap="rapid_or_repeated_action",
        evidence_question="Count each complete jump.",
        source="model",
    )


def _request_payload(tmp_path: Path, *, include_label: bool = False) -> Path:
    row = _row()
    if include_label:
        row["correct_option_id"] = "B"
    payload = {
        "schema_version": 1,
        "experiment_id": "test-adaptive-agent",
        "rows": [row],
        "variants": [
            {
                "variant_id": "video-a:canonical",
                "video_id": "video-a",
                "variant_kind": "canonical",
                "display_order": list("ABCD"),
            }
        ],
    }
    path = tmp_path / "requests.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _historical_root(tmp_path: Path) -> Path:
    root = tmp_path / "historical"
    delivery = root / "deliveries/video-a/delivery-media/video-a.webm"
    delivery.parent.mkdir(parents=True)
    delivery.write_bytes(b"fake video")
    return root


def test_stage1_is_full_video_4fps_low() -> None:
    request = runner.build_stage1_request(row=_row(), upload=_upload())
    video = request.contents[0].parts[0]

    assert video.video_metadata.fps == 4.0
    assert video.video_metadata.start_offset is None
    assert video.video_metadata.end_offset is None
    assert video.media_resolution.level.name.endswith("LOW")
    assert request.config.response_json_schema["required"] == [
        "ranking",
        "inspection",
    ]


def test_stage2_uses_validated_interval_fps_and_resolution() -> None:
    request = runner.build_stage2_request(
        row=_row(),
        upload=_upload(),
        plan=_plan(),
        stage1_ranking=list("BACD"),
    )
    video = request.contents[0].parts[0]

    assert video.video_metadata.start_offset == "1.250s"
    assert video.video_metadata.end_offset == "4.750s"
    assert video.video_metadata.fps == 8.0
    assert video.media_resolution.level.name.endswith("HIGH")
    assert request.config.response_json_schema["required"] == ["ranking", "evidence"]


def test_worst_case_estimate_counts_two_calls_and_enforces_cap(
    tmp_path: Path,
) -> None:
    requests_path = _request_payload(tmp_path)
    payload = runner.load_requests(
        requests_path, expected_rows=None, expected_variants=None
    )
    estimate = runner.estimate_worst_case_cost(payload)

    assert estimate["row_count"] == 1
    assert estimate["request_count"] == 2
    assert estimate["video_seconds_per_stage"] == 6.0
    assert 0 < estimate["estimated_cost_usd"] < 0.75

    with pytest.raises(ValueError, match="cap"):
        runner.preflight(
            requests_path=requests_path,
            historical_root=_historical_root(tmp_path),
            max_cost_usd=0.000001,
            expected_rows=None,
            expected_variants=None,
        )


def test_runner_rejects_label_bearing_requests(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="label"):
        runner.load_requests(
            _request_payload(tmp_path, include_label=True),
            expected_rows=None,
            expected_variants=None,
        )


class _Models:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.requests: list[Any] = []

    def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        self.requests.append({"model": model, "contents": contents, "config": config})
        return self.responses.pop(0)


class _Client:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.models = _Models(responses)


def _provider_response(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": json.dumps(value),
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 10,
            "totalTokenCount": 110,
        },
    }


def test_run_freezes_both_stages_and_reuses_one_upload(tmp_path: Path) -> None:
    requests_path = _request_payload(tmp_path)
    historical_root = _historical_root(tmp_path)
    client = _Client(
        [
            _provider_response(
                {
                    "ranking": list("BACD"),
                    "inspection": {
                        "start_seconds": 1.25,
                        "end_seconds": 4.75,
                        "fps": 8.0,
                        "resolution": "high",
                        "evidence_gap": "rapid_or_repeated_action",
                        "evidence_question": "Count each complete jump.",
                    },
                }
            ),
            _provider_response(
                {
                    "ranking": list("ABCD"),
                    "evidence": [
                        {
                            "timestamp_seconds": 3.0,
                            "observation": "Only one complete jump is visible.",
                        }
                    ],
                }
            ),
        ]
    )
    uploads: list[Path] = []
    output = tmp_path / "output"

    result = runner.run(
        requests_path=requests_path,
        historical_root=historical_root,
        output_root=output,
        client=client,
        upload_file=lambda _client, path: uploads.append(path) or _upload(),
        expected_rows=None,
        expected_variants=None,
    )

    assert result["stage1_valid_count"] == 1
    assert result["stage2_valid_count"] == 1
    assert result["final_valid_count"] == 1
    assert len(uploads) == 1
    assert len(client.models.requests) == 2
    stage2_video = client.models.requests[1]["contents"][0].parts[0]
    assert stage2_video.video_metadata.start_offset == "1.250s"
    assert (output / "stage1/raw-freeze.json").is_file()
    assert (output / "inspection-requests-freeze.json").is_file()
    assert (output / "stage2/raw-freeze.json").is_file()
    final = json.loads((output / "final-predictions-freeze.json").read_text())
    assert final["stage2_raw_freeze_sha256"] == sha256_file(
        output / "stage2/raw-freeze.json"
    )
    assert final["rows"][0]["ranking"] == list("ABCD")
    assert final["rows"][0]["final_source"] == "stage2"


def test_invalid_stage1_plan_runs_recorded_fallback_inspection(tmp_path: Path) -> None:
    requests_path = _request_payload(tmp_path)
    historical_root = _historical_root(tmp_path)
    client = _Client(
        [
            _provider_response(
                {
                    "ranking": list("BACD"),
                    "inspection": {
                        "start_seconds": 5.0,
                        "end_seconds": 4.0,
                        "fps": 8.0,
                        "resolution": "high",
                        "evidence_gap": "rapid_or_repeated_action",
                        "evidence_question": "Count each complete jump.",
                    },
                }
            ),
            _provider_response(
                {
                    "ranking": list("ABCD"),
                    "evidence": [
                        {"timestamp_seconds": 3.0, "observation": "One jump."}
                    ],
                }
            ),
        ]
    )
    output = tmp_path / "fallback-output"

    result = runner.run(
        requests_path=requests_path,
        historical_root=historical_root,
        output_root=output,
        client=client,
        upload_file=lambda _client, _path: _upload(),
        expected_rows=None,
        expected_variants=None,
    )

    assert result["inspection_fallback_count"] == 1
    inspections = json.loads((output / "inspection-requests-freeze.json").read_text())
    assert inspections["rows"][0]["inspection"]["source"] == "fallback"
    video = client.models.requests[1]["contents"][0].parts[0]
    assert video.video_metadata.start_offset == "0.000s"
    assert video.video_metadata.end_offset == "6.000s"
    assert video.video_metadata.fps == 4.0
