from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/run_vqa_deadline_qwen36_canary.py"
SPEC = importlib.util.spec_from_file_location("deadline_qwen36_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _inputs(tmp_path: Path) -> Path:
    rows = []
    for index, video_id in enumerate(("video-a", "video-b", "video-c", "video-d", "video-e")):
        video = tmp_path / f"{video_id}.webm"
        video.write_bytes(f"video-{video_id}".encode())
        rows.append(
            {
                "video_id": video_id,
                "question": f"What happens in video {index}?",
                "video_path": str(video),
                "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                "transcript": f"[00:00:00.000] spoken words for {video_id}",
                "transcript_sha256": hashlib.sha256(
                    f"[00:00:00.000] spoken words for {video_id}".encode()
                ).hexdigest(),
            }
        )
    path = tmp_path / "requests.json"
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return path


def _response(answer: str) -> dict[str, object]:
    return {
        "id": "cmpl-test",
        "choices": [{"message": {"content": answer}, "finish_reason": "stop"}],
    }


def test_build_requests_are_matched_and_keep_exact_prompt_and_transcript(tmp_path: Path) -> None:
    rows = runner.load_input_rows(_inputs(tmp_path))

    requests = runner.build_requests(rows)

    assert len(requests) == 10
    assert [request["condition_id"] for request in requests] == [
        value
        for _ in range(5)
        for value in ("video-plus-transcript", "native-video")
    ]
    for index in range(0, 10, 2):
        with_transcript, native_only = requests[index : index + 2]
        assert with_transcript["video_id"] == native_only["video_id"]
        assert with_transcript["video_sha256"] == native_only["video_sha256"]
        assert with_transcript["prompt"] == runner.DIRECT_ANSWER_PROMPT
        assert native_only["prompt"] == runner.DIRECT_ANSWER_PROMPT
        assert with_transcript["transcript_sha256"]
        assert native_only["transcript_sha256"] is None
        assert with_transcript["payload"]["messages"][0]["content"][-1]["type"] == "text"
        assert native_only["payload"]["messages"][0]["content"][-1]["type"] == "text"
        assert runner.DIRECT_ANSWER_PROMPT in with_transcript["payload"]["messages"][0]["content"][-1]["text"]


def test_run_freezes_all_raw_responses_before_parsing_answers(tmp_path: Path) -> None:
    input_path = _inputs(tmp_path)
    output = tmp_path / "output"
    calls: list[dict[str, object]] = []

    def post(_url: str, payload: dict[str, object]) -> bytes:
        calls.append(payload)
        return json.dumps(_response(f"answer-{len(calls)}"), separators=(",", ":")).encode()

    result = runner.run_canary(
        requests_path=input_path,
        output_root=output,
        server_url="http://127.0.0.1:8000/v1/chat/completions",
        post=post,
    )

    assert result["request_count"] == 10
    assert result["raw_response_count"] == 10
    assert len(calls) == 10
    assert sorted(path.name for path in (output / "raw-responses").iterdir()) == [
        f"{index:02d}.json" for index in range(10)
    ]
    raw_freeze = json.loads((output / "raw-freeze.json").read_text())
    assert raw_freeze["status"] == "raw_responses_frozen"
    assert raw_freeze["raw_response_count"] == 10
    freeze = json.loads((output / "predictions-freeze.json").read_text())
    assert freeze["status"] == "predictions_frozen"
    assert freeze["raw_freeze_sha256"] == hashlib.sha256(
        (output / "raw-freeze.json").read_bytes()
    ).hexdigest()
    assert len(freeze["rows"]) == 10
    predictions = json.loads((output / "predictions.json").read_text())
    assert len(predictions) == 10
    assert [row["answer"] for row in predictions] == [f"answer-{i}" for i in range(1, 11)]
    assert all(row["status"] == "passed" for row in predictions)
    assert not any("reference" in json.dumps(row).lower() for row in predictions)


def test_load_input_rows_rejects_labels_and_requires_five_unique_videos(tmp_path: Path) -> None:
    path = _inputs(tmp_path)
    payload = json.loads(path.read_text())
    payload["rows"][0]["expected_answer"] = "must not be here"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="label|reference|answer"):
        runner.load_input_rows(path)


def test_dstack_config_pins_runtime_budget_and_foreground_sync() -> None:
    config = Path(__file__).parents[1] / "dstack/qwen36-27b-deadline-canary.dstack.yml"
    text = config.read_text(encoding="utf-8")

    assert "Qwen/Qwen3.6-27B" in text
    assert "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9" in text
    assert "docker.io/vllm/vllm-openai:v0.20.0@sha256:77797441eae630c2e79eefa03957b3d61a278670f2a9928d64ce102e7a0790cc" in text
    assert "max_duration: 2h" in text
    assert "max_price: 4.00" in text
    assert "H100" in text
    assert "--max-model-len 32768" in text
    assert "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/" in text
    assert "hf buckets sync" in text
    assert re.search(r"hf buckets sync[^\n&]+&", text) is None
    assert "--delete" not in text
    assert "name: qwen36-27b-deadline-canary-20260816-a5" in text
    assert "EXPERIMENT_RUN_NAME=qwen36-27b-deadline-canary-20260816-a5" in text
    assert "--media-io-kwargs '{\"video\": {\"backend\": \"pyav\"}}'" in text
    assert 'pip install --no-cache-dir "av==16.0.1"' in text
    assert 'import av; print(av.__version__)' in text
    assert text.index('pip install --no-cache-dir "av==16.0.1"') < text.index(
        "import av; print(av.__version__)"
    )
    assert "regions: [US-KS-2, US-CA-2]" in text
    assert "HF_CLI_VERSION=1.27.0" in text
    assert 'pip install --no-cache-dir "huggingface_hub==$HF_CLI_VERSION"' in text
    assert text.index('pip install --no-cache-dir "huggingface_hub==$HF_CLI_VERSION"') < text.index(
        'hf buckets sync "$MEDIA_URI"'
    )
    assert "MEDIA_URI=hf://buckets/" in text
    assert text.count('hf buckets sync "$MEDIA_URI"') == 1
    for video_id in (
        "BDKkVYmWqdM",
        "Bvv6lMUTlEw",
        "DC4FE3hI67U",
        "G5AgzHKJINU",
        "G6zu9D0j8hk",
    ):
        assert f'--include "{video_id}.mkv"' in text
        assert f"../data/vqa/2025/training/videos_original/{video_id}.mkv:" not in text


def test_checksums_exclude_mutable_server_log(tmp_path: Path) -> None:
    (tmp_path / "predictions-freeze.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "vllm-server.log").write_text("still growing\n", encoding="utf-8")

    runner.write_checksums(tmp_path)

    checksum_text = (tmp_path / "SHA256SUMS").read_text(encoding="utf-8")
    assert "predictions-freeze.json" in checksum_text
    assert "vllm-server.log" not in checksum_text
