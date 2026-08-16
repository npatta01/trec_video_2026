from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

import trec_video_2026.vqa_blind_frame_counting as blind_module
import trec_video_2026.vqa_count_frame_presence as reducer_module
from trec_video_2026.vqa_blind_frame_counting import (
    VisibleEpisodeCountSpec,
    build_frame_decision_request,
    parse_visible_episode_count_spec,
)
from trec_video_2026.vqa_submission import canonical_sha256

SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_blind_frame_counting.py"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _synthetic_inputs(root: Path) -> dict[str, Path]:
    artifact_root = root / "artifacts"
    artifact_root.mkdir()

    def bind_artifact(content: bytes) -> tuple[str, int]:
        digest = hashlib.sha256(content).hexdigest()
        (artifact_root / digest).write_bytes(content)
        return digest, len(content)

    spec_payload = {
        "schema_version": 1,
        "question": "How many times does the ball appear in the video?",
        "unit": "visible_appearance_episode",
        "scope": "full_video",
        "episode_rule": "maximal contiguous run of present samples",
        "maximum_sample_gap_seconds": 0.5,
        "every_decoded_frame": True,
        "cut_policy": "continuous_across_cuts",
        "occluded_identity": "unsupported",
    }
    inventory = {
        "schema_version": 1,
        "video_id": "Vaaaaaaaaaaaaaaaa",
        "source_sha256": "a" * 64,
        "source_size_bytes": 10_000,
        "decoder": {
            "name": "ffmpeg",
            "version": "7.1",
            "configuration_sha256": "9" * 64,
            "configuration_size_bytes": 500,
        },
        "duration_seconds": 1.0,
        "decoded_frame_count": 3,
        "frames": [
            {
                "frame_id": f"F{index}",
                "decoded_frame_index": index,
                "timestamp_seconds": index / 2,
                "artifact_sha256": "b" * 64,
                "artifact_size_bytes": 100 + index,
                "is_first_decoded_frame": index == 0,
                "is_last_decoded_frame": index == 2,
            }
            for index in range(3)
        ],
    }
    presentations = [
        {
            "presentation_id": "sheet-1",
            "presentation_sha256": "c" * 64,
            "artifact_size_bytes": 1000,
            "frame_ids": ["F0", "F1", "F2"],
            "mode": "numbered_contact_sheet",
        },
        *[
            {
                "presentation_id": f"individual-{index}",
                "presentation_sha256": "d" * 64,
                "artifact_size_bytes": 200 + index,
                "frame_ids": [f"F{index}"],
                "mode": "individual_frame",
            }
            for index in range(3)
        ],
    ]
    source_digest, source_size = bind_artifact(b"synthetic-video-source")
    inventory["source_sha256"] = source_digest
    inventory["source_size_bytes"] = source_size
    inventory["video_id"] = f"V{source_digest[:16]}"
    decoder_digest, decoder_size = bind_artifact(b"ffmpeg -vsync 0 -frame_pts 1")
    inventory["decoder"]["configuration_sha256"] = decoder_digest
    inventory["decoder"]["configuration_size_bytes"] = decoder_size
    for index, frame in enumerate(inventory["frames"]):
        digest, size = bind_artifact(f"frame-{index}".encode())
        frame["artifact_sha256"] = digest
        frame["artifact_size_bytes"] = size
    for index, presentation in enumerate(presentations):
        digest, size = bind_artifact(f"presentation-{index}".encode())
        presentation["presentation_id"] = f"P{digest}"
        presentation["presentation_sha256"] = digest
        presentation["artifact_size_bytes"] = size
    spec = parse_visible_episode_count_spec(spec_payload)
    assert isinstance(spec, VisibleEpisodeCountSpec)
    request = build_frame_decision_request(spec, inventory, presentations)
    states = ["present", "absent", "present"]
    primary = {
        "schema_version": 1,
        "pass_id": "primary",
        "request_complete_sha256": request.complete_sha256,
        "presentation_mode": "numbered_contact_sheet",
        "presentation_ids": [request.presentations[0].presentation_id],
        "decisions": [
            {"frame_id": f"F{index}", "target_state": state}
            for index, state in enumerate(states)
        ],
    }
    challenge = {
        "schema_version": 1,
        "pass_id": "challenge",
        "request_complete_sha256": request.complete_sha256,
        "presentation_mode": "individual_frame",
        "presentation_ids": [
            binding.presentation_id
            for binding in request.presentations
            if binding.mode == "individual_frame"
        ],
        "decisions": [
            {"frame_id": f"F{index}", "target_state": state}
            for index, state in enumerate(states)
        ],
    }
    payloads = {
        "spec": spec_payload,
        "inventory": inventory,
        "presentations": presentations,
        "primary": primary,
        "challenge": challenge,
    }
    paths: dict[str, Path] = {}
    for name, payload in payloads.items():
        path = root / f"{name}.json"
        _write_json(path, payload)
        paths[name] = path
    paths["artifact_root"] = artifact_root
    return paths


def _run_cli(inputs: dict[str, Path], output_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--spec",
            str(inputs["spec"]),
            "--inventory",
            str(inputs["inventory"]),
            "--presentations",
            str(inputs["presentations"]),
            "--primary",
            str(inputs["primary"]),
            "--challenge",
            str(inputs["challenge"]),
            "--artifact-root",
            str(inputs["artifact_root"]),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _snapshot_bytes(root: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(root.glob("*.json"))}


def _json_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_self_hash(payload: dict[str, object]) -> str:
    complete_sha256 = payload.pop("complete_sha256")
    assert isinstance(complete_sha256, str)
    assert canonical_sha256(payload) == complete_sha256
    return complete_sha256


def test_cli_freezes_byte_identical_authenticated_chain(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path)
    output_dir = tmp_path / "out"

    first = _run_cli(inputs, output_dir)
    assert first.returncode == 0, first.stderr
    before = _snapshot_bytes(output_dir)
    second = _run_cli(inputs, output_dir)

    assert second.returncode == 0, second.stderr
    assert _snapshot_bytes(output_dir) == before
    assert set(before) == {
        "certificate.freeze.json",
        "challenge.freeze.json",
        "ledger.freeze.json",
        "primary.freeze.json",
        "request.freeze.json",
        "result.freeze.json",
        "spec.freeze.json",
    }
    certificate = json.loads(
        (output_dir / "certificate.freeze.json").read_text(encoding="utf-8")
    )
    assert certificate["outcome"] == "exact"
    assert certificate["value"] == 2
    loaded = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in output_dir.glob("*.json")
    }
    complete_hashes = {
        name: _verify_self_hash(dict(payload)) for name, payload in loaded.items()
    }
    assert certificate["spec_complete_sha256"] == complete_hashes["spec.freeze.json"]
    assert (
        certificate["request_complete_sha256"] == complete_hashes["request.freeze.json"]
    )
    assert (
        certificate["primary_complete_sha256"] == complete_hashes["primary.freeze.json"]
    )
    assert (
        certificate["challenge_complete_sha256"]
        == complete_hashes["challenge.freeze.json"]
    )
    assert certificate["ledger_sha256"] == complete_hashes["ledger.freeze.json"]
    assert certificate["result_sha256"] == complete_hashes["result.freeze.json"]
    assert certificate["interface_source_sha256"] == _json_file_sha256(
        Path(blind_module.__file__)
    )
    assert certificate["reducer_source_sha256"] == _json_file_sha256(
        Path(reducer_module.__file__)
    )


@pytest.mark.parametrize(
    "case",
    [
        "missing_decision",
        "label_bearing_input",
        "request_hash_mutation",
        "artifact_mutation",
        "conflict",
    ],
)
def test_cli_rejects_invalid_chain_without_writing_certificate(
    tmp_path: Path, case: str
) -> None:
    inputs = _synthetic_inputs(tmp_path)
    output_dir = tmp_path / "out"

    if case == "missing_decision":
        primary = json.loads(inputs["primary"].read_text(encoding="utf-8"))
        primary["decisions"].pop()
        _write_json(inputs["primary"], primary)
    elif case == "label_bearing_input":
        inventory = json.loads(inputs["inventory"].read_text(encoding="utf-8"))
        inventory["reference_answer"] = 2
        _write_json(inputs["inventory"], inventory)
    elif case == "request_hash_mutation":
        primary = json.loads(inputs["primary"].read_text(encoding="utf-8"))
        primary["request_complete_sha256"] = "0" * 64
        _write_json(inputs["primary"], primary)
    elif case == "artifact_mutation":
        artifact = next(inputs["artifact_root"].iterdir())
        artifact.write_bytes(b"changed")
    elif case == "conflict":
        output_dir.mkdir()
        (output_dir / "request.freeze.json").write_text(
            '{"conflicting":true}\n', encoding="utf-8"
        )

    completed = _run_cli(inputs, output_dir)

    assert completed.returncode != 0
    assert not (output_dir / "certificate.freeze.json").exists()


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        nested_keys = set().union(
            *(_all_keys(nested) for nested in value.values()), set()
        )
        return set(value).union(nested_keys)
    if isinstance(value, list):
        return set().union(*(_all_keys(nested) for nested in value), set())
    return set()


def test_cli_freezes_no_label_or_scoring_keys(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path)
    output_dir = tmp_path / "out"
    completed = _run_cli(inputs, output_dir)
    assert completed.returncode == 0, completed.stderr

    keys: set[str] = set()
    for path in output_dir.glob("*.json"):
        keys.update(_all_keys(json.loads(path.read_text(encoding="utf-8"))))

    forbidden = {
        "reference",
        "label",
        "answer",
        "correctness",
        "score",
        "metric",
        "expected",
        "panel_role",
    }
    assert keys.isdisjoint(forbidden)


def test_cli_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path)
    primary_text = inputs["primary"].read_text(encoding="utf-8")
    inputs["primary"].write_text(
        primary_text.replace(
            '"target_state": "present"',
            '"target_state": "absent", "target_state": "present"',
            1,
        ),
        encoding="utf-8",
    )

    completed = _run_cli(inputs, tmp_path / "out")

    assert completed.returncode != 0
    assert "duplicate JSON key: target_state" in completed.stderr
    assert not (tmp_path / "out" / "certificate.freeze.json").exists()


def test_concurrent_conflicting_runs_cannot_mix_freezes(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    first_root.mkdir()
    first_inputs = _synthetic_inputs(first_root)
    second_root = tmp_path / "second"
    second_root.mkdir()
    second_inputs = _synthetic_inputs(second_root)
    for name in ("primary", "challenge"):
        payload = json.loads(second_inputs[name].read_text(encoding="utf-8"))
        for decision in payload["decisions"]:
            decision["target_state"] = "absent"
        _write_json(second_inputs[name], payload)
    output_dir = tmp_path / "out"

    processes = [
        subprocess.Popen(
            [
                sys.executable,
                str(SCRIPT),
                "--spec",
                str(inputs["spec"]),
                "--inventory",
                str(inputs["inventory"]),
                "--presentations",
                str(inputs["presentations"]),
                "--primary",
                str(inputs["primary"]),
                "--challenge",
                str(inputs["challenge"]),
                "--artifact-root",
                str(inputs["artifact_root"]),
                "--output-dir",
                str(output_dir),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for inputs in (first_inputs, second_inputs)
    ]
    results = [process.communicate(timeout=10) for process in processes]

    assert sorted(process.returncode for process in processes) == [0, 1]
    certificate = json.loads(
        (output_dir / "certificate.freeze.json").read_text(encoding="utf-8")
    )
    assert certificate["value"] in {0, 2}
    for stdout, stderr in results:
        assert not (stdout and stderr)
