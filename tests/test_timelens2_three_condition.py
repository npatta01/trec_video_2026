"""Contracts for the label-isolated TimeLens2 frozen-30 comparison."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from trec_video_2026.vqa_experiments.timelens2_three_condition import (
    CONDITIONS,
    ThreeConditionEvidenceStore,
    build_synthetic_canary_assets,
    build_blind_grading_packet,
    build_three_condition_recipes,
    extract_three_condition_frames,
    freeze_three_condition_inference,
    make_ffmpeg_frame_decoder,
    prepare_frozen30_source,
    score_blind_judgments,
    validate_runtime_requests,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(tmp_path: Path, *, count: int = 30, leak: bool = False) -> Path:
    examples = []
    for index in range(count):
        video_id = f"video-{index:02d}"
        media = tmp_path / "media" / f"{video_id}.mkv"
        media.parent.mkdir(exist_ok=True)
        media.write_bytes(f"media-{index}".encode())
        row = {
            "category": "temporal",
            "media": f"media/{video_id}.mkv",
            "question": f"What happens in clip {index}?",
            "video_id": video_id,
        }
        if leak and index == 0:
            row["annotation"] = "private-answer.json"
        examples.append(row)
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_version": "frozen-test",
                "gate_id": "gate",
                "gate_ids": [],
                "gate_marker": "marker",
                "gate_selection_sha256": "0" * 64,
                "examples": examples,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return source


def test_prepare_frozen30_source_binds_exact_label_free_media(tmp_path: Path) -> None:
    source = _source(tmp_path)
    output = tmp_path / "prepared.json"

    prepared = prepare_frozen30_source(
        source_path=source,
        dataset_root=tmp_path,
        output_path=output,
        container_media_root=Path("/dstack/input/data"),
        expected_source_sha256=_sha256(source),
        duration_provider=lambda path: 10.0 + int(path.stem.rsplit("-", 1)[1]),
    )

    assert prepared["schema_version"] == "timelens2-three-condition-requests-v1"
    assert len(prepared["requests"]) == 30
    assert len({row["video_id"] for row in prepared["requests"]}) == 30
    assert prepared["requests"][0]["media_sha256"] == _sha256(
        tmp_path / "media/video-00.mkv"
    )
    assert prepared["requests"][0]["media_path"] == "/dstack/input/data/video-00.mkv"
    encoded = json.dumps(prepared).lower()
    assert "annotation" not in encoded
    assert "answer" not in encoded


@pytest.mark.parametrize("count", [29, 31])
def test_prepare_frozen30_source_rejects_wrong_denominator(
    tmp_path: Path, count: int
) -> None:
    source = _source(tmp_path, count=count)

    with pytest.raises(ValueError, match="exactly 30"):
        prepare_frozen30_source(
            source_path=source,
            dataset_root=tmp_path,
            output_path=tmp_path / "prepared.json",
            container_media_root=Path("/dstack/input/data"),
            expected_source_sha256=_sha256(source),
            duration_provider=lambda _: 12.0,
        )


def test_prepare_frozen30_source_rejects_label_path(tmp_path: Path) -> None:
    source = _source(tmp_path, leak=True)

    with pytest.raises(ValueError, match="label leakage"):
        prepare_frozen30_source(
            source_path=source,
            dataset_root=tmp_path,
            output_path=tmp_path / "prepared.json",
            container_media_root=Path("/dstack/input/data"),
            expected_source_sha256=_sha256(source),
            duration_provider=lambda _: 12.0,
        )


def test_three_condition_recipes_share_uniform_and_local_slots() -> None:
    recipes = build_three_condition_recipes(
        duration_seconds=12.0,
        normalized_intervals=[[3.0, 6.0]],
    )

    assert set(recipes) == set(CONDITIONS)
    assert [len(recipes[name]["slots"]) for name in CONDITIONS] == [12, 12, 18]
    uniform_ids = {slot["slot_id"] for slot in recipes["uniform-12"]["slots"]}
    hybrid_local_ids = {
        slot["slot_id"]
        for slot in recipes["hybrid-12"]["slots"]
        if slot["role"] == "local"
    }
    combined_ids = {slot["slot_id"] for slot in recipes["combined-18"]["slots"]}
    assert uniform_ids <= combined_ids
    assert hybrid_local_ids <= combined_ids
    assert len(uniform_ids | hybrid_local_ids) == 18
    assert all(
        3.0 < slot["requested_timestamp_seconds"] < 6.0
        for slot in recipes["hybrid-12"]["slots"]
        if slot["role"] == "local"
    )


def test_extractor_reuses_shared_slots_across_conditions(tmp_path: Path) -> None:
    media = tmp_path / "video.mkv"
    media.write_bytes(b"video")
    recipes = build_three_condition_recipes(
        duration_seconds=12.0,
        normalized_intervals=[[3.0, 6.0]],
    )
    calls: list[tuple[str, float]] = []

    def decoder(_media: Path, timestamp: float, output: Path) -> None:
        calls.append((output.stem, timestamp))
        output.write_bytes(f"frame-{timestamp:.6f}".encode())

    manifest = extract_three_condition_frames(
        video_id="video",
        media_path=media,
        recipes=recipes,
        output_dir=tmp_path / "frames",
        frame_decoder=decoder,
        frame_timestamps_provider=lambda _: [],
    )

    assert [len(manifest["conditions"][name]) for name in CONDITIONS] == [12, 12, 18]
    assert len({row["sha256"] for row in manifest["conditions"]["combined-18"]}) == 18
    assert len(calls) == 24  # 12 uniform + 6 hybrid-global + 6 shared local
    shared_uniform = manifest["conditions"]["uniform-12"][0]
    assert shared_uniform == manifest["conditions"]["combined-18"][0]


def test_extractor_moves_only_colliding_local_slot_inside_interval(
    tmp_path: Path,
) -> None:
    media = tmp_path / "video.mkv"
    media.write_bytes(b"video")
    recipes = build_three_condition_recipes(
        duration_seconds=12.0,
        normalized_intervals=[[3.0, 6.0]],
    )
    colliding_local = next(
        slot
        for slot in recipes["hybrid-12"]["slots"]
        if slot["role"] == "local"
    )
    colliding_global = min(
        recipes["uniform-12"]["slots"],
        key=lambda slot: abs(
            slot["requested_timestamp_seconds"]
            - colliding_local["requested_timestamp_seconds"]
        ),
    )
    collision_times = {
        round(colliding_local["requested_timestamp_seconds"], 6),
        round(colliding_global["requested_timestamp_seconds"], 6),
    }

    def decoder(_media: Path, timestamp: float, output: Path) -> None:
        payload = (
            b"same-source-frame"
            if round(timestamp, 6) in collision_times
            else f"frame-{timestamp:.6f}".encode()
        )
        output.write_bytes(payload)

    manifest = extract_three_condition_frames(
        video_id="video",
        media_path=media,
        recipes=recipes,
        output_dir=tmp_path / "frames",
        frame_decoder=decoder,
        frame_timestamps_provider=lambda _: [3.233, 3.267, 3.300],
    )

    all_rows = {
        row["slot_id"]: row
        for condition in manifest["conditions"].values()
        for row in condition
    }
    assert all_rows[colliding_global["slot_id"]]["resolution"] == "requested"
    assert all_rows[colliding_local["slot_id"]]["resolution"] == (
        "nearest-unused-source-pts"
    )
    assert 3.0 < all_rows[colliding_local["slot_id"]]["timestamp_seconds"] < 6.0
    assert len({row["sha256"] for row in manifest["conditions"]["combined-18"]}) == 18


def _complete_freeze_fixture(tmp_path: Path) -> dict[str, object]:
    requests = []
    raw = []
    normalized = []
    recipes = []
    frames = []
    answers = []
    for index in range(30):
        video_id = f"video-{index:02d}"
        request_sha = f"{index + 1:064x}"
        requests.append({"video_id": video_id, "request_sha256": request_sha})
        raw.append({"video_id": video_id, "request_sha256": request_sha})
        normalized.append({"video_id": video_id, "request_sha256": request_sha})
        condition_recipes = {
            name: {"slots": [{"slot_id": f"{name}-{slot}"} for slot in range(count)]}
            for name, count in zip(CONDITIONS, (12, 12, 18), strict=True)
        }
        condition_frames = {
            name: [
                {"slot_id": f"{name}-{slot}", "sha256": f"{index:02x}{slot:062x}"}
                for slot in range(count)
            ]
            for name, count in zip(CONDITIONS, (12, 12, 18), strict=True)
        }
        recipes.append({"video_id": video_id, "conditions": condition_recipes})
        frames.append({"video_id": video_id, "conditions": condition_frames})
        for condition in CONDITIONS:
            answers.append(
                {
                    "video_id": video_id,
                    "condition_id": condition,
                    "request_sha256": request_sha,
                    "status": "passed",
                    "answer": f"answer {index}",
                }
            )
    return {
        "requests": requests,
        "raw_results": raw,
        "normalized_results": normalized,
        "recipes": recipes,
        "frame_manifests": frames,
        "answers": answers,
        "output_path": tmp_path / "freeze.json",
    }


def test_freeze_requires_all_thirty_rows_and_ninety_answers(tmp_path: Path) -> None:
    fixture = _complete_freeze_fixture(tmp_path)
    fixture["answers"] = fixture["answers"][:-1]

    with pytest.raises(ValueError, match="90 complete answers"):
        freeze_three_condition_inference(**fixture)


def test_blind_packet_and_scoring_preserve_condition_hiding(tmp_path: Path) -> None:
    fixture = _complete_freeze_fixture(tmp_path)
    freeze = freeze_three_condition_inference(**fixture)
    references = {
        f"video-{index:02d}": f"reference {index}" for index in range(30)
    }
    packet, key = build_blind_grading_packet(
        freeze=freeze,
        answers=fixture["answers"],
        references=references,
        seed=20260801,
    )

    assert len(packet["rows"]) == 30
    assert "uniform-12" not in json.dumps(packet)
    assert "hybrid-12" not in json.dumps(packet)
    assert "combined-18" not in json.dumps(packet)
    judgments = {
        candidate["candidate_id"]: "C"
        for row in packet["rows"]
        for candidate in row["candidates"]
    }
    scored = score_blind_judgments(packet=packet, key=key, judgments=judgments)

    assert scored["completion"] == {"accepted": 90, "required": 90}
    assert scored["conditions"]["uniform-12"] == {"C": 30, "P": 0, "W": 0}
    assert scored["paired"]["hybrid-12"] == {"improved": 0, "regressed": 0, "tied": 30}


def test_validate_runtime_requests_binds_source_and_media(tmp_path: Path) -> None:
    source = _source(tmp_path)
    prepared_path = tmp_path / "prepared.json"
    prepared = prepare_frozen30_source(
        source_path=source,
        dataset_root=tmp_path,
        output_path=prepared_path,
        container_media_root=tmp_path / "runtime-media",
        expected_source_sha256=_sha256(source),
        duration_provider=lambda _: 12.0,
    )
    runtime_media = tmp_path / "runtime-media"
    runtime_media.mkdir()
    for row in prepared["requests"]:
        source_media = tmp_path / "media" / f"{row['video_id']}.mkv"
        (runtime_media / source_media.name).write_bytes(source_media.read_bytes())

    validated = validate_runtime_requests(
        prepared_path,
        expected_source_sha256=_sha256(source),
    )

    assert len(validated) == 30
    assert validated[0]["request_sha256"] == prepared["requests"][0]["request_sha256"]


def test_evidence_store_persists_partial_completion(tmp_path: Path) -> None:
    store = ThreeConditionEvidenceStore(tmp_path)
    store.checkpoint("selector:row-01")
    store.record_selector_attempt({"status": "accepted", "video_id": "row-01"})
    store.record_answer_attempt(
        {"status": "failed", "video_id": "row-01", "condition_id": "uniform-12"}
    )
    store.finalize(
        failure={"phase": "answer:row-01:uniform-12", "error_type": "RuntimeError"}
    )

    summary = json.loads((tmp_path / "three-condition-summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["completion"] == {
        "videos": {"accepted": 1, "attempted": 1, "required": 30},
        "answers": {"accepted": 0, "attempted": 1, "required": 90},
    }
    assert (tmp_path / "selector-attempts.jsonl").is_file()
    assert (tmp_path / "answer-attempts.jsonl").is_file()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_synthetic_canary_assets_and_real_frame_decoder(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    assets = build_synthetic_canary_assets(
        output_dir=tmp_path,
        ffmpeg_executable=ffmpeg,
    )
    assert Path(assets["video_path"]).is_file()
    assert len(assets["frame_paths"]) == 18
    decoder = make_ffmpeg_frame_decoder(ffmpeg)
    decoded = tmp_path / "decoded.png"

    decoder(Path(assets["video_path"]), 1.0, decoded)

    assert decoded.is_file()
    assert decoded.stat().st_size > 0
