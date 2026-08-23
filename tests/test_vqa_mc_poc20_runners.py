from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from trec_video_2026.vqa_experiments.vqa_mc_poc20 import VIDEO_IDS


ROOT = Path(__file__).parents[1]


def _load_script(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gemini = _load_script("vqa_mc_gemini", "run_vqa_mc_gemini_poc20.py")
qwen = _load_script("vqa_mc_qwen", "run_vqa_mc_qwen36_poc20.py")


@pytest.fixture
def request_fixture(tmp_path: Path) -> tuple[Path, Path]:
    media = tmp_path / "video-a.mkv"
    media.write_bytes(b"small fake video")
    row = {
        "query_id": "video-a",
        "video_id": "video-a",
        "stratum": "temporal",
        "question": "What happens first?",
        "options": [
            {"option_id": "A", "text": "alpha"},
            {"option_id": "B", "text": "beta"},
            {"option_id": "C", "text": "gamma"},
            {"option_id": "D", "text": "delta"},
        ],
        "duration_seconds": 10.0,
        "source_path": media.name,
        "source_sha256": qwen.sha256_file(media),
        "transcript": "[00:00:01.000] spoken evidence",
        "transcript_sha256": qwen.sha256_bytes(
            b"[00:00:01.000] spoken evidence"
        ),
    }
    payload = {
        "schema_version": 1,
        "experiment_id": "test-mc",
        "conditions": ["video-plus-transcript", "native-video"],
        "rows": [row],
        "variants": [
            {
                "variant_id": "video-a:canonical",
                "video_id": "video-a",
                "variant_kind": "canonical",
                "display_order": list("ABCD"),
            },
            {
                "variant_id": "video-a:reversed",
                "video_id": "video-a",
                "variant_kind": "reversed",
                "display_order": list("DCBA"),
            },
        ],
    }
    path = tmp_path / "requests.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, media


def test_gemini_request_uses_native_av_strict_schema_and_display_order() -> None:
    row = {
        "query_id": "video-a",
        "video_id": "video-a",
        "question": "What happens first?",
        "options": [
            {"option_id": letter, "text": text}
            for letter, text in zip("ABCD", ("alpha", "beta", "gamma", "delta"), strict=True)
        ],
    }
    variant = {"variant_id": "video-a:reversed", "display_order": list("DCBA")}

    request = gemini.build_request(
        row=row,
        variant=variant,
        upload={"file_uri": "https://provider.invalid/video", "mime_type": "video/webm"},
    )

    prompt = request.contents[0].parts[1].text
    assert request.model == "gemini-3.7-flash"
    assert request.contents[0].parts[0].file_data.file_uri.endswith("/video")
    assert prompt.index("D. delta") < prompt.index("A. alpha")
    assert request.config.response_mime_type == "application/json"
    assert request.config.response_json_schema["required"] == ["ranking"]
    assert request.config.max_output_tokens == 2048


def test_gemini_can_select_only_canonical_variants_for_bounded_recovery(
    request_fixture: tuple[Path, Path]
) -> None:
    """Catches the recovery run accidentally repeating order diagnostics."""
    requests_path, _ = request_fixture
    payload = gemini.load_requests(
        requests_path, expected_rows=None, expected_variants=None
    )

    selected = gemini.select_variants(payload, variant_kind="canonical")
    estimate = gemini.estimate_cost(payload, variant_kind="canonical")

    assert [row["variant_id"] for row in selected] == ["video-a:canonical"]
    assert estimate["request_count"] == 1
    assert estimate["video_seconds"] == 10.0


def test_gemini_cost_counts_variants_and_enforces_authorized_cap(
    request_fixture: tuple[Path, Path], tmp_path: Path
) -> None:
    requests_path, _ = request_fixture
    payload = gemini.load_requests(requests_path, expected_rows=None, expected_variants=None)

    estimate = gemini.estimate_cost(payload)

    assert estimate["request_count"] == 2
    assert estimate["video_seconds"] == 20.0
    assert estimate["estimated_cost_usd"] < 0.40
    with pytest.raises(ValueError, match="cap"):
        gemini.preflight(
            requests_path=requests_path,
            historical_root=tmp_path,
            max_cost_usd=0.000001,
            expected_rows=None,
            expected_variants=None,
        )


def test_both_runners_reject_a_label_bearing_request(
    request_fixture: tuple[Path, Path]
) -> None:
    path, _ = request_fixture
    payload = json.loads(path.read_text())
    payload["rows"][0]["correct_option_id"] = "A"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="label"):
        gemini.load_requests(path, expected_rows=None, expected_variants=None)
    with pytest.raises(ValueError, match="label"):
        qwen.load_requests(
            path,
            media_root=path.parent,
            expected_rows=None,
            expected_variants=None,
        )


def test_qwen_builds_matched_dual_view_payloads_with_strict_json(
    request_fixture: tuple[Path, Path]
) -> None:
    path, _ = request_fixture
    payload = qwen.load_requests(
        path,
        media_root=path.parent,
        expected_rows=None,
        expected_variants=None,
    )
    specs = qwen.build_request_specs(payload)

    assert len(specs) == 4
    assert [spec["condition_id"] for spec in specs] == [
        "video-plus-transcript",
        "native-video",
        "video-plus-transcript",
        "native-video",
    ]
    data_uri = qwen.video_data_uri(Path(payload["rows"][0]["video_path"]))
    bodies = [qwen.build_payload(spec, data_uri=data_uri) for spec in specs]
    assert "spoken evidence" in bodies[0]["messages"][0]["content"][-1]["text"]
    assert "spoken evidence" not in bodies[1]["messages"][0]["content"][-1]["text"]
    assert bodies[0]["response_format"]["type"] == "json_schema"
    assert bodies[0]["response_format"]["json_schema"]["strict"] is True
    assert bodies[2]["messages"][0]["content"][-1]["text"].index("D. delta") < bodies[2]["messages"][0]["content"][-1]["text"].index("A. alpha")


def test_qwen_run_freezes_raw_before_parsed_rankings(
    request_fixture: tuple[Path, Path], tmp_path: Path
) -> None:
    requests_path, _ = request_fixture
    calls: list[dict[str, Any]] = []

    def post(_url: str, payload: dict[str, Any]) -> bytes:
        calls.append(payload)
        return json.dumps(
            {
                "id": f"cmpl-{len(calls)}",
                "choices": [
                    {
                        "message": {"content": '{"ranking":["A","B","C","D"]}'},
                        "finish_reason": "stop",
                    }
                ],
            }
        ).encode()

    output = tmp_path / "output"
    result = qwen.run(
        requests_path=requests_path,
        media_root=requests_path.parent,
        output_root=output,
        server_url="http://127.0.0.1:8000/v1/chat/completions",
        post=post,
        expected_rows=None,
        expected_variants=None,
    )

    assert result["request_count"] == 4
    assert result["valid_count"] == 4
    assert len(calls) == 4
    assert (output / "raw-freeze.json").is_file()
    predictions = json.loads((output / "predictions-freeze.json").read_text())
    assert predictions["raw_freeze_sha256"] == qwen.sha256_file(output / "raw-freeze.json")
    assert all(row["ranking"] == list("ABCD") for row in predictions["rows"])


def test_qwen_runtime_identity_is_pinned() -> None:
    assert qwen.MODEL_ID == "Qwen/Qwen3.6-27B"
    assert qwen.MODEL_REVISION == "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
    assert "@sha256:" in qwen.RUNTIME_IMAGE


def test_qwen_dstack_config_is_immutable_durable_and_within_budget() -> None:
    """Catches identity drift, non-durable output, or a POC cost-cap violation."""
    path = ROOT / "dstack/qwen36-27b-vqa-mc-poc20.dstack.yml"
    text = path.read_text(encoding="utf-8")

    assert "name: qwen36-27b-vqa-mc-poc20-20260821-a2" in text
    assert qwen.MODEL_ID in text
    assert qwen.MODEL_REVISION in text
    assert qwen.RUNTIME_IMAGE in text
    assert "max_duration: 25m" in text
    assert "max_price: 3.35" in text
    assert "H100" in text
    assert "spot_policy: on-demand" in text
    assert "backends: [runpod, vastai]" in text
    assert "--max-model-len 32768" in text
    assert "--media-io-kwargs '{\"video\": {\"backend\": \"pyav\"}}'" in text
    assert (
        "hf://buckets/Npatta01/trec_video_2026/experiments/"
        "qwen36-27b-vqa-mc-poc20-20260821-a2" in text
    )
    assert "vqa2025-mc-poc20-20260821-a1/media" in text
    assert 'hf buckets sync "$OUTPUT_DIR" "$HF_REMOTE_URI"' in text
    assert '--exclude "vllm-server.log"' in text
    assert "--delete" not in text
    assert "HF_TOKEN=${{ secrets.hf_token }}" in text
    assert "../src:/dstack/run/src" in text
    assert "PYTHONPATH=/dstack/run/src" in text
    for video_id in VIDEO_IDS:
        assert f'--include "{video_id}.mkv"' in text
