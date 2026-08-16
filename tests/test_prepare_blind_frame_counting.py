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
    spec_payload = {
        "schema_version": 1,
        "spec_id": "synthetic-spec",
        "target": "ball visibly present in the frame",
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
        "video_id": "synthetic-video",
        "source_sha256": "a" * 64,
        "duration_seconds": 1.0,
        "decoded_frame_count": 3,
        "frames": [
            {
                "frame_id": f"F{index}",
                "decoded_frame_index": index,
                "timestamp_seconds": index / 2,
                "artifact_sha256": "b" * 64,
                "is_first_decoded_frame": index == 0,
                "is_last_decoded_frame": index == 2,
            }
            for index in range(3)
        ],
    }
    presentations = [
        {
            "presentation_id": "sheet",
            "presentation_sha256": "c" * 64,
            "frame_ids": ["F0", "F1", "F2"],
            "mode": "numbered_contact_sheet",
        },
        *[
            {
                "presentation_id": f"individual-{index}",
                "presentation_sha256": "d" * 64,
                "frame_ids": [f"F{index}"],
                "mode": "individual_frame",
            }
            for index in range(3)
        ],
    ]
    spec = parse_visible_episode_count_spec(spec_payload)
    assert isinstance(spec, VisibleEpisodeCountSpec)
    request = build_frame_decision_request(spec, inventory, presentations)
    states = ["present", "absent", "present"]
    primary = {
        "schema_version": 1,
        "pass_id": "primary",
        "request_complete_sha256": request.complete_sha256,
        "presentation_mode": "numbered_contact_sheet",
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
    ["missing_decision", "label_bearing_input", "request_hash_mutation", "conflict"],
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
