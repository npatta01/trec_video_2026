"""Contract tests for the TimeLens2 compatibility selector adapter."""

import hashlib
import json
import math
from pathlib import Path

import pytest

from trec_video_2026.vqa_experiments import timelens2_selector
from trec_video_2026.vqa_experiments.timelens2_selector import (
    CompatibilityEvidenceStore,
    allocate_hamilton,
    build_hybrid_timestamps,
    build_messages,
    build_normalized_interval_result,
    canonical_sha256,
    freeze_compatibility_results,
    normalize_intervals,
    parse_raw_interval_result,
    prepare_compatibility_requests,
    run_with_compatibility_evidence,
    validate_compatibility_requests,
)

MODEL_REVISION = "90dd30b3eeff3520a30c830eff2209f4ad788ca9"
RUNTIME_IMAGE = (
    "docker.io/pytorch/pytorch@sha256:"
    "c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
)


def _metadata() -> dict[str, object]:
    return {
        "request_sha256": "a" * 64,
        "model_id": "MCG-NJU/TimeLens2-2B",
        "model_revision": MODEL_REVISION,
        "runtime_image": RUNTIME_IMAGE,
        "transformers_version": "4.57.6",
        "latency_seconds": 1.25,
        "peak_memory_bytes": 1234,
        "device": "NVIDIA A100-SXM4-80GB",
        "precision": "bfloat16",
    }


def test_raw_result_preserves_model_order_and_overlaps() -> None:
    raw_output = "[[8, 12], [1.5, 4], [3, 6]]"

    result = parse_raw_interval_result(
        request_id="row-1",
        video_id="video-1",
        duration_seconds=15.0,
        raw_output=raw_output,
        metadata=_metadata(),
    )

    assert result["raw_output"] == raw_output
    assert result["intervals"] == [[8.0, 12.0], [1.5, 4.0], [3.0, 6.0]]
    assert result["duration_seconds"] == 15.0
    assert result["request_sha256"] == "a" * 64


@pytest.mark.parametrize(
    ("raw_output", "duration", "message"),
    [
        ("[[1, NaN]]", 10.0, "finite"),
        ("[[4, 3]]", 10.0, "reversed"),
        ("[[-0.1, 3]]", 10.0, "outside"),
        ("[[1, 10.1]]", 10.0, "outside"),
        ("[[true, 3]]", 10.0, "number"),
        ('{"start": 1, "end": 2}', 10.0, "array"),
    ],
)
def test_raw_result_rejects_invalid_intervals(
    raw_output: str, duration: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_raw_interval_result(
            request_id="row-1",
            video_id="video-1",
            duration_seconds=duration,
            raw_output=raw_output,
            metadata=_metadata(),
        )


def test_normalization_is_separate_and_merges_only_sorted_overlaps() -> None:
    raw_intervals = [[8.0, 12.0], [1.5, 4.0], [3.0, 6.0], [7.0, 7.0]]

    normalized = normalize_intervals(raw_intervals)

    assert raw_intervals == [[8.0, 12.0], [1.5, 4.0], [3.0, 6.0], [7.0, 7.0]]
    assert normalized == [[1.5, 6.0], [7.0, 7.0], [8.0, 12.0]]


def test_normalized_result_links_to_raw_output_without_copying_it() -> None:
    raw = parse_raw_interval_result(
        request_id="row-1",
        video_id="video-1",
        duration_seconds=15.0,
        raw_output="[[8, 12], [1.5, 4], [3, 6]]",
        metadata=_metadata(),
    )

    normalized = build_normalized_interval_result(raw)

    assert normalized["schema_version"] == "timelens2-normalized-interval-result-v1"
    assert normalized["intervals"] == [[1.5, 6.0], [8.0, 12.0]]
    assert normalized["raw_output_sha256"] == (
        "51326c9dc63f107b03b828f6ccd8b2bbe90fd3dda102b32939058df4688c9610"
    )
    assert "raw_output" not in normalized


def test_hamilton_allocator_uses_largest_remainders_with_stable_ties() -> None:
    assert allocate_hamilton([5.0, 3.0, 2.0], seats=6) == [3, 2, 1]
    assert allocate_hamilton([1.0, 1.0, 1.0, 1.0], seats=2) == [1, 1, 0, 0]


def test_hamilton_rejects_positive_seats_for_zero_total_duration() -> None:
    with pytest.raises(ValueError, match="positive total weight"):
        allocate_hamilton([0.0, 0.0], seats=3)

    assert allocate_hamilton([0.0, 0.0], seats=0) == [0, 0]


def test_hybrid_allocator_rejects_only_zero_width_intervals() -> None:
    with pytest.raises(ValueError, match="positive total weight"):
        build_hybrid_timestamps(
            duration_seconds=12.0,
            normalized_intervals=[[2.0, 2.0], [7.0, 7.0]],
        )


def test_hybrid_allocator_returns_six_global_and_six_local_midpoints() -> None:
    allocation = build_hybrid_timestamps(
        duration_seconds=12.0,
        normalized_intervals=[[0.0, 6.0], [9.0, 12.0]],
    )

    assert allocation == {
        "global": [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
        "local": [0.75, 2.25, 3.75, 5.25, 9.75, 11.25],
        "local_counts": [4, 2],
        "all": [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 0.75, 2.25, 3.75, 5.25, 9.75, 11.25],
    }


def test_prepare_requests_writes_only_label_free_three_video_panel(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "data"
    rows = []
    for index in range(3):
        video_id = f"video-{index}"
        media = Path("vqa/videos") / f"{video_id}.mkv"
        annotation = Path("vqa/annotations") / f"{video_id}.json"
        media_path = dataset_root / media
        annotation_path = dataset_root / annotation
        media_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        media_path.write_bytes(f"media-{index}".encode())
        annotation_path.write_text(
            json.dumps(
                {
                    "question": f"question {index}",
                    "correct_answer": f"secret {index}",
                    "incorrect_answers": ["also secret"],
                }
            )
        )
        rows.append(
            {
                "video_id": video_id,
                "category": "temporal",
                "media": str(media),
                "annotation": str(annotation),
            }
        )
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"schema_version": 1, "examples": rows}))
    output = tmp_path / "private" / "compatibility-requests.json"
    expected_rows = tuple(
        (
            f"compatibility-{index + 1:02d}",
            f"video-{index}",
            f"video-{index}.mkv",
            hashlib.sha256(f"media-{index}".encode()).hexdigest(),
            hashlib.sha256(f"question {index}".encode()).hexdigest(),
        )
        for index in range(3)
    )

    prepared = prepare_compatibility_requests(
        gate_path=gate,
        dataset_root=dataset_root,
        output_path=output,
        container_media_root=Path("/dstack/input/media"),
        expected_gate_sha256=hashlib.sha256(gate.read_bytes()).hexdigest(),
        expected_rows=expected_rows,
    )

    assert prepared["schema_version"] == "timelens2-compatibility-requests-v1"
    assert len(prepared["requests"]) == 3
    assert [row["query"] for row in prepared["requests"]] == [
        "question 0",
        "question 1",
        "question 2",
    ]
    assert all(len(row["request_sha256"]) == 64 for row in prepared["requests"])
    encoded = output.read_text()
    assert "secret" not in encoded
    assert "correct_answer" not in encoded
    assert "incorrect_answers" not in encoded
    assert not any(math.isnan(value) for value in [0.0])


def _frozen_request_manifest(tmp_path: Path) -> Path:
    rows = (
        (
            "compatibility-01",
            "BDKkVYmWqdM",
            "BDKkVYmWqdM.mkv",
            "7ae4937b33b0caedbe30f6b2790c06affea52f5fb5f9666bb052424cda49a323",
            "5dfe17948025a8e9407aa9535bc2d308ff9e36bbb298a6fdf7182159946912fa",
        ),
        (
            "compatibility-02",
            "O60MtMuSpf8",
            "O60MtMuSpf8.mkv",
            "46606eb3b9ec901014a4ab2d0f3a8a562356a7af52527b1f16f3df7b88caa2bf",
            "b70521f9413ecc991631425483fd381e8e244a3db77c2d101325b7537a5383c2",
        ),
        (
            "compatibility-03",
            "Fk_j6re_Uiw",
            "Fk_j6re_Uiw.mkv",
            "aa2e987f61a2962bab5db68940b62e25376078b53c4748cffdfd2cd2c67a5c4b",
            "806eb114abcca191c0fd0fa6e10991071be7c65d5379bbfad9c522cba4bcb578",
        ),
    )
    gate = json.loads(Path("configs/vqa/three_video_gate.json").read_text())
    questions = {
        example["video_id"]: json.loads(
            (Path("data") / example["annotation"]).read_text()
        )["question"]
        for example in gate["examples"]
    }
    requests = []
    for request_id, video_id, filename, media_sha256, query_sha256 in rows:
        query = questions[video_id]
        assert hashlib.sha256(query.encode()).hexdigest() == query_sha256
        identity = {
            "request_id": request_id,
            "video_id": video_id,
            "query": query,
            "media_path": str(
                Path("data/vqa/2025/training/videos_original", filename).resolve()
            ),
            "media_sha256": media_sha256,
        }
        requests.append({**identity, "request_sha256": canonical_sha256(identity)})
    path = tmp_path / "requests.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "timelens2-compatibility-requests-v1",
                "source_gate_sha256": (
                    "1f00bc94868c29cd58f6b27a73add337ba67c35db0806d99ef2301646858e05f"
                ),
                "requests": requests,
            }
        )
    )
    return path


def test_runtime_requests_bind_to_exact_frozen_gate_and_media(tmp_path: Path) -> None:
    path = _frozen_request_manifest(tmp_path)

    validated = validate_compatibility_requests(path)

    assert [(row["request_id"], row["video_id"]) for row in validated] == [
        ("compatibility-01", "BDKkVYmWqdM"),
        ("compatibility-02", "O60MtMuSpf8"),
        ("compatibility-03", "Fk_j6re_Uiw"),
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "source_gate",
        "duplicate_request",
        "wrong_video",
        "wrong_query",
        "wrong_media_hash",
    ],
)
def test_runtime_requests_reject_panel_substitution(
    tmp_path: Path, mutation: str
) -> None:
    path = _frozen_request_manifest(tmp_path)
    payload = json.loads(path.read_text())
    if mutation == "source_gate":
        payload["source_gate_sha256"] = "0" * 64
    elif mutation == "duplicate_request":
        payload["requests"][1]["request_id"] = payload["requests"][0]["request_id"]
    elif mutation == "wrong_video":
        payload["requests"][1]["video_id"] = "substituted-video"
    elif mutation == "wrong_query":
        payload["requests"][1]["query"] = "substituted query"
    else:
        payload["requests"][1]["media_sha256"] = "f" * 64
    if mutation != "source_gate":
        for request in payload["requests"]:
            identity = {
                key: value for key, value in request.items() if key != "request_sha256"
            }
            request["request_sha256"] = canonical_sha256(identity)
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="frozen compatibility gate"):
        validate_compatibility_requests(path)


def test_messages_follow_official_video_grounding_prompt_contract() -> None:
    messages = build_messages(Path("/dstack/input/media/video.mkv"), "Find the event")

    assert messages == [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "file:///dstack/input/media/video.mkv",
                    "fps": 2.0,
                    "min_pixels": 32 * 32,
                    "max_pixels": 480 * 480,
                    "total_pixels": 128000 * 32 * 32,
                },
                {
                    "type": "text",
                    "text": (
                        'Given the query: "Find the event", return ALL time spans '
                        "(in seconds) where the query is relevant.\n"
                        "Output format MUST be a JSON array of [start, end] pairs.\n"
                    ),
                },
            ],
        }
    ]


def test_freeze_requires_exact_three_complete_raw_and_normalized_results(
    tmp_path: Path,
) -> None:
    raw_results = []
    normalized_results = []
    for index in range(3):
        raw = parse_raw_interval_result(
            request_id=f"row-{index}",
            video_id=f"video-{index}",
            duration_seconds=15.0,
            raw_output="[[1, 2]]",
            metadata={**_metadata(), "request_sha256": f"{index + 1:064x}"},
        )
        raw_results.append(raw)
        normalized_results.append(build_normalized_interval_result(raw))

    freeze_path = tmp_path / "selector-freeze.json"
    freeze = freeze_compatibility_results(
        raw_results=raw_results,
        normalized_results=normalized_results,
        output_path=freeze_path,
    )

    assert freeze["status"] == "frozen"
    assert freeze["completion"] == {"accepted": 3, "required": 3}
    assert len(freeze["raw_results_sha256"]) == 64
    assert len(freeze["normalized_results_sha256"]) == 64
    assert freeze_path.read_text() == json.dumps(freeze, sort_keys=True) + "\n"

    with pytest.raises(ValueError, match="exactly three"):
        freeze_compatibility_results(
            raw_results=raw_results[:2],
            normalized_results=normalized_results[:2],
            output_path=tmp_path / "incomplete.json",
        )


def test_evidence_wrapper_preserves_setup_failure_denominator(
    tmp_path: Path,
) -> None:
    def fail_during_setup(store: CompatibilityEvidenceStore) -> bool:
        store.checkpoint("model-load")
        raise RuntimeError("setup exploded")

    returncode = run_with_compatibility_evidence(tmp_path, fail_during_setup)

    assert returncode == 1
    summary = json.loads((tmp_path / "compatibility-summary.json").read_text())
    phase = json.loads((tmp_path / "phase-status.json").read_text())
    attempts_freeze = json.loads((tmp_path / "attempts-freeze.json").read_text())
    assert summary["completion"] == {"accepted": 0, "attempted": 0, "required": 3}
    assert summary["failure"] == {
        "phase": "model-load",
        "error_type": "RuntimeError",
        "error_message": "setup exploded",
    }
    assert phase["status"] == "failed"
    assert phase["phase"] == "model-load"
    assert attempts_freeze["completion"] == summary["completion"]
    assert len(attempts_freeze["attempts_sha256"]) == 64
    assert (tmp_path / "raw-attempts.jsonl").read_text() == ""
    assert (tmp_path / "raw-results.jsonl").read_text() == ""
    assert (tmp_path / "normalized-results.jsonl").read_text() == ""


def test_evidence_wrapper_persists_partial_attempt_and_inflight_oom(
    tmp_path: Path,
) -> None:
    raw = parse_raw_interval_result(
        request_id="compatibility-01",
        video_id="BDKkVYmWqdM",
        duration_seconds=15.0,
        raw_output="[[1, 2]]",
        metadata=_metadata(),
    )
    normalized = build_normalized_interval_result(raw)

    def fail_mid_panel(store: CompatibilityEvidenceStore) -> bool:
        store.begin_request(
            {
                "request_id": "compatibility-01",
                "video_id": "BDKkVYmWqdM",
                "request_sha256": "a" * 64,
                "query": "must not leak to automatic failures",
                "media_path": "/private/media.mkv",
            },
            phase="panel:compatibility-01",
        )
        store.record_attempt(
            {"status": "accepted", **raw},
            raw_result=raw,
            normalized_result=normalized,
        )
        persisted = (tmp_path / "raw-attempts.jsonl").read_text().splitlines()
        assert len(persisted) == 1
        assert json.loads(persisted[0])["status"] == "accepted"

        store.begin_request(
            {
                "request_id": "compatibility-02",
                "video_id": "O60MtMuSpf8",
                "request_sha256": "b" * 64,
                "query": "must not leak to automatic failures",
                "media_path": "/private/media.mkv",
            },
            phase="panel:compatibility-02",
        )
        raise RuntimeError("CUDA out of memory")

    returncode = run_with_compatibility_evidence(tmp_path, fail_mid_panel)

    assert returncode == 1
    attempts = [
        json.loads(line)
        for line in (tmp_path / "raw-attempts.jsonl").read_text().splitlines()
    ]
    summary = json.loads((tmp_path / "compatibility-summary.json").read_text())
    assert [attempt["status"] for attempt in attempts] == ["accepted", "failed"]
    assert attempts[1] == {
        "status": "failed",
        "request_id": "compatibility-02",
        "video_id": "O60MtMuSpf8",
        "request_sha256": "b" * 64,
        "phase": "panel:compatibility-02",
        "error_type": "RuntimeError",
        "error_message": "CUDA out of memory",
    }
    assert summary["completion"] == {"accepted": 1, "attempted": 2, "required": 3}
    assert summary["failure"]["phase"] == "panel:compatibility-02"
    assert len((tmp_path / "raw-results.jsonl").read_text().splitlines()) == 1
    assert len((tmp_path / "normalized-results.jsonl").read_text().splitlines()) == 1


def test_compatibility_runtime_uses_failure_persisting_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_wrapper(
        output_dir: Path,
        workload: object,
    ) -> int:
        captured["output_dir"] = output_dir
        captured["workload"] = workload
        return 17

    monkeypatch.setattr(
        timelens2_selector,
        "run_with_compatibility_evidence",
        fake_wrapper,
    )

    returncode = timelens2_selector._run_compatibility(
        model_dir=tmp_path / "model",
        requests_path=tmp_path / "requests.json",
        output_dir=tmp_path / "output",
    )

    assert returncode == 17
    assert captured["output_dir"] == tmp_path / "output"
    assert callable(captured["workload"])
