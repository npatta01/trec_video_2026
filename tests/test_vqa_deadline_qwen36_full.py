from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_vqa_deadline_qwen36_full as runner  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def panel_fixture(tmp_path: Path) -> dict[str, Any]:
    media = tmp_path / "media"
    media.mkdir()
    rows: list[dict[str, Any]] = []
    whisper_rows: list[dict[str, Any]] = []
    for query_id in range(1, 9):
        video = media / f"clip-{query_id}.webm"
        video.write_bytes(f"video-{query_id}".encode())
        available = query_id != 8
        rows.append(
            {
                "query_id": query_id,
                "video_id": f"clip-{query_id}.webm",
                "question": f"What happens in clip {query_id}?",
                "availability": "available" if available else "unavailable",
                "source_sha256": _sha256(video) if available else None,
                "media_file": f"media/{video.name}" if available else None,
            }
        )
        if query_id != 7:
            whisper_rows.append(
                {
                    "query_id": query_id,
                    "video_id": video.name,
                    "quality": {"status": "accepted"},
                    "transcript": f"[00:00:00.000] words for clip {query_id}",
                }
            )
    requests = tmp_path / "requests.json"
    requests.write_text(json.dumps(rows) + "\n", encoding="utf-8")
    whisper = tmp_path / "requests.accepted-whisper.json"
    whisper.write_text(json.dumps(whisper_rows) + "\n", encoding="utf-8")
    return {"requests": requests, "whisper": whisper, "media_root": tmp_path, "rows": rows}


class FakeProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def post(self, _url: str, payload: dict[str, Any]) -> bytes:
        self.calls.append(payload)
        index = len(self.calls)
        return json.dumps(
            {
                "id": f"response-{index}",
                "choices": [
                    {"message": {"content": f"answer {index}"}, "finish_reason": "stop"}
                ],
            },
            separators=(",", ":"),
        ).encode()


def test_build_requests_preserves_two_matched_conditions_and_exact_prompt(
    panel_fixture: dict[str, Any],
) -> None:
    rows = runner.load_input_rows(
        panel_fixture["requests"],
        whisper_path=panel_fixture["whisper"],
        media_root=panel_fixture["media_root"],
        expected_count=8,
        verify_contract=False,
    )

    requests = runner.build_requests(rows)

    assert len(requests) == 16
    assert [request["condition_id"] for request in requests] == [
        condition
        for _ in range(8)
        for condition in (runner.TRANSCRIPT_CONDITION_ID, runner.NATIVE_CONDITION_ID)
    ]
    for index in range(0, 16, 2):
        transcript, native = requests[index : index + 2]
        assert transcript["query_id"] == native["query_id"]
        assert transcript["video_sha256"] == native["video_sha256"]
        assert transcript["prompt"] == runner.DIRECT_ANSWER_PROMPT
        assert native["prompt"] == runner.DIRECT_ANSWER_PROMPT
        assert native["transcript"] is None

    missing_transcript = next(
        request for request in requests if request["query_id"] == 7
        and request["condition_id"] == runner.TRANSCRIPT_CONDITION_ID
    )
    unavailable_media = next(request for request in requests if request["query_id"] == 8)
    assert missing_transcript["status"] == "unavailable"
    assert unavailable_media["status"] == "unavailable"


def test_partition_is_exactly_four_query_shards(panel_fixture: dict[str, Any]) -> None:
    rows = runner.load_input_rows(
        panel_fixture["requests"],
        whisper_path=panel_fixture["whisper"],
        media_root=panel_fixture["media_root"],
        expected_count=8,
        verify_contract=False,
    )

    shards = runner.partition_rows(rows, shard_count=4)

    assert len(shards) == 4
    assert sum(len(shard) for shard in shards) == 8
    assert sorted(row["query_id"] for shard in shards for row in shard) == list(range(1, 9))
    assert all(len(shard) == 2 for shard in shards)


def test_shard_freezes_raw_before_parse_and_resumes_without_duplicate_calls(
    panel_fixture: dict[str, Any], tmp_path: Path
) -> None:
    provider = FakeProvider()
    output = tmp_path / "shard-0"
    parse_seen: list[bool] = []

    def parse_response(*, raw: bytes, request: dict[str, Any]) -> dict[str, Any]:
        parse_seen.append((output / "raw-freeze.json").is_file())
        return runner.parse_response(raw=raw, request=request)

    result = runner.run_shard(
        requests_path=panel_fixture["requests"],
        whisper_path=panel_fixture["whisper"],
        media_root=panel_fixture["media_root"],
        output_root=output,
        shard_index=0,
        shard_count=4,
        expected_count=8,
        post=provider.post,
        parse_response=parse_response,
        created_at="2026-08-16T20:00:00Z",
        verify_contract=False,
    )

    assert result["status"] == "predictions_frozen"
    assert result["shard_index"] == 0
    assert result["row_count"] == 4
    assert result["request_count"] == 4
    assert len(provider.calls) == 4
    assert parse_seen == [True, True, True, True]
    freeze = json.loads((output / "predictions-freeze.json").read_text())
    assert freeze["status"] == "predictions_frozen"
    assert len(freeze["rows"]) == 4
    assert all(
        {"query_id", "video_id", "condition_id", "status", "answer", "raw_response_sha256"}
        <= row.keys()
        for row in freeze["rows"]
    )

    resumed_provider = FakeProvider()
    resumed = runner.run_shard(
        requests_path=panel_fixture["requests"],
        whisper_path=panel_fixture["whisper"],
        media_root=panel_fixture["media_root"],
        output_root=output,
        shard_index=0,
        shard_count=4,
        expected_count=8,
        post=resumed_provider.post,
        verify_contract=False,
    )
    assert resumed["status"] == "predictions_frozen"
    assert resumed_provider.calls == []


def test_failed_attempt_is_frozen_and_resume_does_not_retry(
    panel_fixture: dict[str, Any], tmp_path: Path
) -> None:
    calls = 0

    def failing_provider(_url: str, _payload: dict[str, Any]) -> bytes:
        nonlocal calls
        calls += 1
        raise TimeoutError("provider timeout")

    output = tmp_path / "failed-shard"
    runner.run_shard(
        requests_path=panel_fixture["requests"],
        whisper_path=panel_fixture["whisper"],
        media_root=panel_fixture["media_root"],
        output_root=output,
        shard_index=0,
        shard_count=4,
        expected_count=8,
        post=failing_provider,
        verify_contract=False,
    )

    assert calls == 4
    manifest = json.loads((output / "raw-response-manifest.json").read_text())
    assert all(record["status"] == "failed" for record in manifest["responses"])
    assert all(record["raw_response_sha256"] for record in manifest["responses"])

    def should_not_retry(_url: str, _payload: dict[str, Any]) -> bytes:
        raise AssertionError("terminal failed call was retried")

    resumed = runner.run_shard(
        requests_path=panel_fixture["requests"],
        whisper_path=panel_fixture["whisper"],
        media_root=panel_fixture["media_root"],
        output_root=output,
        shard_index=0,
        shard_count=4,
        expected_count=8,
        post=should_not_retry,
        verify_contract=False,
    )
    assert resumed["status"] == "predictions_frozen"
    predictions = json.loads((output / "predictions-freeze.json").read_text())
    assert all(row["status"] == "failed" for row in predictions["rows"])
    assert all(row["raw_response_sha256"] for row in predictions["rows"])


def test_aggregate_requires_four_shards_and_freezes_4886_contract(
    panel_fixture: dict[str, Any], tmp_path: Path
) -> None:
    shard_roots: list[Path] = []
    for shard_index in range(4):
        provider = FakeProvider()
        output = tmp_path / f"shard-{shard_index}"
        runner.run_shard(
            requests_path=panel_fixture["requests"],
            whisper_path=panel_fixture["whisper"],
            media_root=panel_fixture["media_root"],
            output_root=output,
            shard_index=shard_index,
            shard_count=4,
            expected_count=8,
            post=provider.post,
            verify_contract=False,
        )
        shard_roots.append(output)

    aggregate = runner.aggregate_shards(
        shard_roots,
        output_root=tmp_path / "aggregate",
        expected_query_count=8,
        shard_count=4,
    )

    assert aggregate["status"] == "predictions_frozen"
    assert aggregate["row_count"] == 16
    freeze = json.loads((tmp_path / "aggregate/predictions-freeze.json").read_text())
    assert len(freeze["rows"]) == 16
    assert len({(row["query_id"], row["condition_id"]) for row in freeze["rows"]}) == 16


def test_config_pins_runtime_and_syncs_each_shard() -> None:
    root = Path(__file__).parents[1]
    config_names = [
        "dstack/qwen36-27b-deadline-full.dstack.yml",
        "dstack/qwen36-27b-deadline-full-shard-1.dstack.yml",
        "dstack/qwen36-27b-deadline-full-shard-2.dstack.yml",
        "dstack/qwen36-27b-deadline-full-shard-3.dstack.yml",
    ]
    for shard_index, config_name in enumerate(config_names):
        text = (root / config_name).read_text(encoding="utf-8")
        assert f"qwen36-27b-deadline-full-20260816-a2-s{shard_index}" in text
        assert "Qwen/Qwen3.6-27B" in text
        assert "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9" in text
        assert "docker.io/vllm/vllm-openai:v0.20.0@sha256:77797441eae630c2e79eefa03957b3d61a278670f2a9928d64ce102e7a0790cc" in text
        assert (
            "HF_INPUT_URI=hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/"
            "experiments/deadline-qwen36-full-inputs-20260816-a2/input"
        ) in text
        assert (
            "HF_WHISPER_URI=hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/"
            "experiments/deadline-qwen36-whisper-full-20260816-a2/output"
        ) in text
        assert (
            'hf buckets cp "$HF_WHISPER_URI/requests.accepted-whisper.json" '
            '"$INPUT_DIR/requests.accepted-whisper.json"'
        ) in text
        assert (
            'hf buckets cp "$HF_WHISPER_URI/summary.json" '
            '"$INPUT_DIR/whisper-summary.json"'
        ) in text
        assert "openai/whisper-large-v3" in text
        assert "06f233fe06e710322aca913c1bc4249a0d71fce1" in text
        assert 'summary["media_count"] == 1946' in text
        assert 'summary["target_count"] == 1946' in text
        assert 'summary["completed_count"] == 1946' in text
        assert '[[ -s "$INPUT_DIR/requests.accepted-whisper.json" ]]' in text
        restore = (
            'hf buckets sync "$HF_REMOTE_URI/shards/$shard_index" '
            '"$OUTPUT_DIR/shard-$shard_index"'
        )
        persist = (
            'hf buckets sync "$OUTPUT_DIR/shard-$shard_index" '
            '"$HF_REMOTE_URI/shards/$shard_index"'
        )
        assert text.count(restore) == 1
        assert text.count(persist) >= 1
        assert 'hf buckets sync "$OUTPUT_DIR" ' not in text
        assert "/artifacts/deadline-qwen36-full-inputs-20260816-a2" not in text
        assert f"SHARD_INDEX={shard_index}" in text
        assert "max_duration: 2h" in text
        assert "max_price: 4.00" in text
        assert "H100" in text
        assert "--max-model-len 32768" in text
        assert '--media-io-kwargs \'{"video": {"backend": "pyav"}}\'' in text
        assert 'pip install --no-cache-dir "av==16.0.1"' in text
        assert "shard_index=\"$SHARD_INDEX\"" in text
        assert text.count("trap finalize EXIT") == 1
        assert text.count("trap - EXIT") == 1
        assert text.count("hf buckets sync") >= 2
        assert "--delete" not in text


def test_smoke_config_is_distinct_limited_and_whisper_gated() -> None:
    root = Path(__file__).parents[1]
    text = (root / "dstack/qwen36-27b-deadline-full-smoke.dstack.yml").read_text(encoding="utf-8")

    assert "name: qwen36-full-smoke-20260816-a2-s0" in text
    assert (
        "HF_REMOTE_URI=hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/"
        "experiments/qwen36-full-smoke-20260816-a2-s0"
    ) in text
    assert (
        "HF_INPUT_URI=hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/"
        "experiments/deadline-qwen36-full-inputs-20260816-a2/input"
    ) in text
    assert 'HF_WHISPER_URI=hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/' in text
    assert "SHARD_COUNT=4" in text
    assert "SHARD_INDEX=0" in text
    assert "--limit 5" in text
    assert "max_duration: 1h" in text
    assert "max_price: 4.00" in text
    assert 'summary["media_count"] == 1946' in text
    assert 'summary["target_count"] == 1946' in text
    assert 'summary["completed_count"] == 1946' in text
    assert 'summary.get("model", {})' in text
    assert 'openai/whisper-large-v3' in text
    assert '06f233fe06e710322aca913c1bc4249a0d71fce1' in text
    restore = (
        'hf buckets sync "$HF_REMOTE_URI/shards/$shard_index" '
        '"$OUTPUT_DIR/shard-$shard_index"'
    )
    persist = (
        'hf buckets sync "$OUTPUT_DIR/shard-$shard_index" '
        '"$HF_REMOTE_URI/shards/$shard_index"'
    )
    assert text.count(restore) == 1
    assert text.count(persist) >= 1
    assert text.count("trap finalize EXIT") == 1
    assert text.count("trap - EXIT") == 1
    assert "qwen36-27b-deadline-full-20260816-a2" not in text.replace(
        "qwen36-full-smoke-20260816-a2-s0", ""
    )
    assert "--delete" not in text
