from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts/build_vqa_silver_freeze_manifest.py"
ARM_IDS = ("G/N", "G/F", "G/S", "H/N", "H/F", "H/S", "L/N", "L/F", "L/S")
FACTORIAL_IDS = (
    "4b-current",
    "4b-question-aware",
    "8b-current",
    "8b-question-aware",
)
SOURCE_IDS = FACTORIAL_IDS + ARM_IDS
POOL_IDS = ("A", "B", "C")
CELL_BY_ARM = {
    "G/N": "gn",
    "G/F": "gf",
    "G/S": "gs",
    "H/N": "hn",
    "H/F": "hf",
    "H/S": "hs",
    "L/N": "ln",
    "L/F": "lf",
    "L/S": "ls",
}
ANSWER_RUNTIME_IMAGE = (
    "docker.io/pytorch/pytorch@sha256:"
    "c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
)
JUDGE_RUNTIME_IMAGE = (
    "docker.io/qwenllm/qwen3-omni@sha256:"
    "71896cf6533abc610ac298f26289ff3a4ea4bd1fc835f370ece9e5c31e3181d8"
)


def load_builder():
    assert BUILDER_PATH.is_file(), "freeze-manifest builder is not implemented"
    spec = importlib.util.spec_from_file_location(
        "build_vqa_silver_freeze_manifest", BUILDER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_checksums(payload_root: Path) -> Path:
    files = sorted(
        path.relative_to(payload_root)
        for path in payload_root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    path = payload_root / "SHA256SUMS"
    path.write_text(
        "".join(
            f"{_sha256(payload_root / item)}  {item.as_posix()}\n" for item in files
        ),
        encoding="utf-8",
    )
    return path


def _source_identity(source_id: str) -> dict[str, object]:
    if source_id in ARM_IDS:
        condition_id = CELL_BY_ARM[source_id]
        model_id = "Qwen/Qwen3.5-4B"
        model_revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        answer_policy = "current"
    elif source_id.startswith("4b-"):
        condition_id = source_id
        model_id = "Qwen/Qwen3.5-4B"
        model_revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        answer_policy = source_id.removeprefix("4b-")
    else:
        condition_id = source_id
        model_id = "Qwen/Qwen3-VL-8B-Instruct"
        model_revision = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
        answer_policy = source_id.removeprefix("8b-")
    return {
        "source_id": source_id,
        "condition_id": condition_id,
        "model_id": model_id,
        "model_revision": model_revision,
        "answer_policy": answer_policy,
        "min_pixels_per_frame": 65536,
        "max_pixels_per_frame": 262144,
        "runtime": "transformers fixture; torch fixture",
        "runtime_image": ANSWER_RUNTIME_IMAGE,
        "transformers_revision": "cbf4d720ec734edb77d25452b2790c7e4be2f8d7",
        "do_sample": False,
        "enable_thinking": False,
        "max_new_tokens": 64,
    }


def _write_factorial_producer_bundle(
    root: Path, source_id: str, ids: tuple[str, ...]
) -> dict[str, Path]:
    identity = _source_identity(source_id)
    run_id = f"fixture-factorial-{source_id}"
    payload_root = root / "payload"
    condition_root = payload_root / "conditions" / source_id
    condition_root.mkdir(parents=True, exist_ok=True)
    requests_path = _write_json(
        condition_root / "requests.json",
        [
            {
                "run_id": run_id,
                "condition_id": source_id,
                "video_id": video_id,
                "question": "What is visible?",
                "frame_paths": [f"frames/{video_id}/frame.jpg"],
                "frame_timestamps": [0.0],
                "transcript": None,
            }
            for video_id in ids
        ],
    )
    prediction_rows = [
        {
            "status": "passed",
            "answer": f"{source_id} answer {index}",
            "elapsed_seconds": 0.25,
            "model_load_seconds": 1.5,
            "run_id": run_id,
            "condition_id": source_id,
            "video_id": video_id,
            "model_id": identity["model_id"],
            "model_revision": identity["model_revision"],
            "runtime": "transformers 5.15.0.dev0; torch 2.8.0",
            "answer_policy": identity["answer_policy"],
            "question_type": "general",
            "requested_min_pixels_per_frame": 65536,
            "requested_max_pixels_per_frame": 262144,
            "min_pixels_per_frame": 65536,
            "max_pixels_per_frame": 262144,
            "runtime_image": ANSWER_RUNTIME_IMAGE,
            "precision": "bfloat16",
            "device": "cuda:0 (NVIDIA CUDA)",
            "runtime_attestation": {
                "accelerator_backend": "nvidia-cuda",
                "parameter_tensor_count": 10,
                "parameter_devices": {"cuda:0": 10},
                "parameter_dtypes": {"torch.bfloat16": 10},
                "buffer_tensor_count": 2,
                "buffer_devices": {"cuda:0": 2},
                "buffer_dtypes": {"torch.float32": 1, "torch.int64": 1},
                "torch_cuda": "12.4",
                "torch_hip": None,
            },
            "peak_memory_bytes": 1024 + index,
            "error_type": None,
            "error_message": None,
        }
        for index, video_id in enumerate(ids)
    ]
    predictions_path = _write_json(
        condition_root / "predictions.json", prediction_rows
    )
    _write_json(
        condition_root / "condition-identity.json",
        {
            "condition": {
                "answer_policy": identity["answer_policy"],
                "condition_id": source_id,
                "max_pixels": 262144,
                "min_pixels": 65536,
                "model_id": identity["model_id"],
                "request_source": "shared-accepted-whisper",
                "revision": identity["model_revision"],
            },
            "deterministic_decoding": {
                "do_sample": False,
                "enable_thinking": False,
                "max_new_tokens": 64,
            },
            "expected_count": len(ids),
            "frame_count": 12,
            "input_uri": "hf://buckets/fixture/evidence",
            "manifest": "/dstack/run/configs/vqa/qwen35_vqa2025_475_label_free.json",
            "model_class": "multimodal",
            "profile_valid": True,
            "remote_uri": f"hf://buckets/test/output/{source_id}",
            "run_name": run_id,
            "runtime_image": ANSWER_RUNTIME_IMAGE,
        },
    )
    _write_json(
        condition_root / "timing.json",
        {
            "condition_id": source_id,
            "row_count": len(ids),
            "wall_seconds": 8.0,
            "prepare_wall_seconds": 2.0,
            "peak_memory_bytes": 1024 + len(ids) - 1,
            "model_load_seconds": 1.5,
            "runtime_identity": [
                [
                    "transformers 5.15.0.dev0; torch 2.8.0",
                    ANSWER_RUNTIME_IMAGE,
                    "cuda:0 (NVIDIA CUDA)",
                    "bfloat16",
                ]
            ],
        },
    )
    telemetry_path = condition_root / "telemetry" / "nvidia-smi.csv"
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_path.write_text(
        "timestamp,index,utilization.gpu [%],memory.used [MiB],memory.total [MiB]\n"
        "2026-08-09T00:00:00Z,0,50,100,1000\n"
        "2026-08-09T00:00:05Z,0,75,200,1000\n",
        encoding="utf-8",
    )
    _write_json(
        payload_root / "runtime.json",
        {
            "runtime": "transformers 5.15.0.dev0; torch 2.8.0",
            "runtime_image": ANSWER_RUNTIME_IMAGE,
            "transformers_version": "5.15.0.dev0",
            "transformers_revision": identity["transformers_revision"],
            "torch_version": "2.8.0",
            "torch_cuda": "12.4",
            "torch_hip": None,
            "cuda_device_name": "fixture NVIDIA",
            "cuda_bf16_supported": True,
            "dependencies": {
                "Pillow": "11.3.0",
                "soundfile": "0.13.1",
                "accelerate": "1.12.0",
                "av": "16.0.1",
                "imageio-ffmpeg": "0.6.0",
                "qwen-vl-utils": "0.0.14",
            },
        },
    )
    _write_json(
        payload_root / "status.json",
        {
            "schema_version": 1,
            "status": "passed",
            "mode": "condition",
            "run_name": run_id,
            "expected_count": len(ids),
            "frame_count": 12,
            "workload_returncode": 0,
            "remote_payload_sync": "frozen",
            "labels_joined": False,
        },
    )
    evidence_checksum_path = payload_root / "evidence" / "SHA256SUMS"
    evidence_checksum_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_checksum_path.write_text(
        "fixture nested evidence checksum bytes\n", encoding="utf-8"
    )
    checksum_path = _write_checksums(payload_root)
    terminal_path = _write_json(
        root / "terminal-status.json",
        {
            "schema_version": 1,
            "status": "passed",
            "payload_sha256": _sha256(checksum_path),
            "payload_uri": f"hf://buckets/test/output/{source_id}/payload",
            "run_name": run_id,
        },
    )
    return {
        "root": root,
        "request_path": requests_path,
        "prediction_path": predictions_path,
        "evidence_checksum_path": evidence_checksum_path,
        "checksum_path": checksum_path,
        "terminal_path": terminal_path,
    }


def _build_label_free_fixture(tmp_path: Path) -> dict[str, object]:
    ids = ("video000001", "video000002")
    panel_digest = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    protocol_path = _write_json(
        tmp_path / "selector-protocol.json",
        {"schema_version": 1, "label_policy": "label-free-selector"},
    )
    protocol_sha = _sha256(protocol_path)
    selector_rows = [
        {
            "video_id": ids[0],
            "protocol_sha256": protocol_sha,
            "selector": {"status": "localized"},
        },
        {
            "video_id": ids[1],
            "protocol_sha256": protocol_sha,
            "selector": {"status": "non_localizable"},
        },
    ]
    selector_path = _write_json(tmp_path / "selector.json", selector_rows)
    coverage_path = _write_json(
        tmp_path / "localized-coverage.json",
        {ids[0]: True, ids[1]: False},
    )

    evidence_root = tmp_path / "evidence"
    request_sets: dict[str, list[dict[str, object]]] = {}
    for arm_id, cell in CELL_BY_ARM.items():
        request_sets[cell] = [
            {
                "run_id": "fixture-run",
                "condition_id": cell,
                "video_id": video_id,
                "question": "What is visible?",
                "frame_paths": [f"frames/{video_id}/frame.jpg"],
                "frame_timestamps": [0.0],
                "transcript": None,
            }
            for video_id in ids
        ]
        _write_json(evidence_root / "requests" / f"{cell}.json", request_sets[cell])
    evidence_payload = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "canonical_id_digest": panel_digest,
        "selector_protocol_sha256": protocol_sha,
        "rows": [],
    }
    leakage_payload = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "rows": [],
        "references_loaded": False,
    }
    _write_json(evidence_root / "evidence.json", evidence_payload)
    _write_json(evidence_root / "leakage-audit.json", leakage_payload)
    marker_path = _write_json(
        evidence_root / "evidence-marker.json",
        {
            "schema_version": 1,
            "label_policy": "label-free-evidence",
            "example_count": len(ids),
            "canonical_id_digest": panel_digest,
            "checksum_file": "SHA256SUMS",
        },
    )
    summary = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "row_count": len(ids),
        "cell_count": 9,
        "cells": list(CELL_BY_ARM.values()),
        "request_counts": {cell: len(ids) for cell in CELL_BY_ARM.values()},
        "request_set_sha256": {
            cell: _canonical_sha256(rows) for cell, rows in request_sets.items()
        },
        "evidence_sha256": _canonical_sha256(evidence_payload),
        "leakage_audit_sha256": _canonical_sha256(leakage_payload),
        "canonical_id_digest": panel_digest,
        "protocol_sha256": protocol_sha,
        "selector_protocol_sha256": protocol_sha,
    }
    summary_path = _write_json(evidence_root / "summary.json", summary)
    evidence_checksums = _write_checksums(evidence_root)

    prediction_roots: dict[str, Path] = {}
    prediction_paths: dict[str, Path] = {}
    factorial_fixture: dict[str, dict[str, Path]] = {}
    for source_id in SOURCE_IDS:
        identity = _source_identity(source_id)
        root = tmp_path / "predictions" / source_id.replace("/", "-")
        if source_id in FACTORIAL_IDS:
            factorial_fixture[source_id] = _write_factorial_producer_bundle(
                root, source_id, ids
            )
            prediction_roots[source_id] = root
            prediction_paths[source_id] = factorial_fixture[source_id]["prediction_path"]
            continue
        payload = root / "payload"
        request_path = _write_json(
            payload / "requests.json",
            [
                {
                    "video_id": video_id,
                    "condition_id": identity["condition_id"],
                    "question": "What is visible?",
                }
                for video_id in ids
            ],
        )
        prediction_path = _write_json(
            payload / "predictions.json",
            {
                "predictions": [
                    {
                        "status": "passed",
                        "video_id": video_id,
                        "answer": f"{source_id} answer {index}",
                        "condition_id": identity["condition_id"],
                        "model_id": identity["model_id"],
                        "model_revision": identity["model_revision"],
                        "answer_policy": identity["answer_policy"],
                        "requested_min_pixels_per_frame": 65536,
                        "requested_max_pixels_per_frame": 262144,
                        "min_pixels_per_frame": 65536,
                        "max_pixels_per_frame": 262144,
                        "runtime": identity["runtime"],
                        "runtime_image": ANSWER_RUNTIME_IMAGE,
                    }
                    for index, video_id in enumerate(ids)
                ]
            },
        )
        request_sha = _sha256(request_path)
        prediction_sha = _sha256(prediction_path)
        producer_id = str(identity["condition_id"])
        task4_request_sha = (
            summary["request_set_sha256"][producer_id]
            if source_id in ARM_IDS
            else request_sha
        )
        completion_path = _write_json(
            payload / "completion.json",
            {
                "schema_version": 1,
                "status": "passed",
                "source_id": source_id,
                "condition_id": producer_id,
                "expected_count": len(ids),
                "evidence_marker_sha256": _sha256(marker_path),
                "task4_request_set_sha256": task4_request_sha,
                "request_sha256": request_sha,
                "prediction_sha256": prediction_sha,
                "labels_joined": False,
            },
        )
        checksum_path = _write_checksums(payload)
        _write_json(
            root / "terminal-status.json",
            {
                "schema_version": 1,
                "status": "passed",
                "source_id": source_id,
                "condition_id": producer_id,
                "payload_sha256": _sha256(checksum_path),
                "completion_sha256": _sha256(completion_path),
                "request_sha256": request_sha,
                "prediction_sha256": prediction_sha,
            },
        )
        prediction_roots[source_id] = root
        prediction_paths[source_id] = prediction_path

    election_roots: dict[str, Path] = {}
    ranking_paths: dict[str, Path] = {}
    sources_by_pool = {
        "A": FACTORIAL_IDS,
        "B": ARM_IDS,
        "C": SOURCE_IDS,
    }
    for pool_id in POOL_IDS:
        root = tmp_path / "elections" / pool_id
        payload = root / "payload"
        request_path = _write_json(
            payload / "requests.json",
            [
                {
                    "pool_id": pool_id,
                    "video_id": video_id,
                    "candidate_ids": [
                        f"{pool_id.lower()}-{index}"
                        for index, _source in enumerate(sources_by_pool[pool_id])
                    ],
                }
                for video_id in ids
            ],
        )
        ranking_path = _write_json(
            payload / "rankings.json",
            {
                "rows": [
                    {
                        "video_id": video_id,
                        "candidates": [
                            {
                                "candidate_id": f"{pool_id.lower()}-{index}",
                                "arm_id": source_id,
                                "answer": f"{source_id} answer {row_index}",
                            }
                            for index, source_id in enumerate(sources_by_pool[pool_id])
                        ],
                        "ranking": [
                            f"{pool_id.lower()}-{index}"
                            for index, _source in enumerate(sources_by_pool[pool_id])
                        ],
                    }
                    for row_index, video_id in enumerate(ids)
                ]
            },
        )
        checksum_path = _write_checksums(payload)
        _write_json(
            root / "terminal-status.json",
            {
                "schema_version": 1,
                "status": "passed",
                "pool_id": pool_id,
                "payload_sha256": _sha256(checksum_path),
                "request_sha256": _sha256(request_path),
                "ranking_sha256": _sha256(ranking_path),
                "judge_model_id": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
                "judge_model_revision": "2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b",
                "runtime_image": JUDGE_RUNTIME_IMAGE,
                "labels_joined": False,
            },
        )
        election_roots[pool_id] = root
        ranking_paths[pool_id] = ranking_path

    freeze_inputs = {
        "prediction_bundles": prediction_roots,
        "election_bundles": election_roots,
        "evidence_bundle": evidence_root,
        "selector": selector_path,
        "selector_protocol": protocol_path,
        "localized_coverage": coverage_path,
    }
    return {
        "ids": ids,
        "freeze_inputs": freeze_inputs,
        "prediction_paths": prediction_paths,
        "ranking_paths": ranking_paths,
        "evidence_checksums": evidence_checksums,
        "marker_path": marker_path,
        "summary_path": summary_path,
        "protocol_path": protocol_path,
        "selector_path": selector_path,
        "coverage_path": coverage_path,
        "factorial_fixture": factorial_fixture,
    }


def test_builder_binds_all_producer_judge_and_evidence_bytes(tmp_path: Path) -> None:
    builder = load_builder()
    fixture = _build_label_free_fixture(tmp_path)

    manifest = builder.build_freeze_manifest(fixture["freeze_inputs"], expected_rows=2)

    assert set(manifest) == {
        "schema_version",
        "manifest_type",
        "expected_rows",
        "panel_id_set_sha256",
        "verified_4b_current_source_prediction_sha256",
        "prediction_sources",
        "elections",
        "evidence",
    }
    assert set(manifest["prediction_sources"]) == set(SOURCE_IDS)
    assert set(manifest["elections"]) == set(POOL_IDS)
    for source_id, root in fixture["freeze_inputs"]["prediction_bundles"].items():
        payload = root / "payload"
        binding = manifest["prediction_sources"][source_id]
        assert binding["terminal_sha256"] == _sha256(root / "terminal-status.json")
        assert binding["payload_sha256"] == _sha256(payload / "SHA256SUMS")
        assert binding["checksum_sha256"] == _sha256(payload / "SHA256SUMS")
        if source_id in FACTORIAL_IDS:
            condition = payload / "conditions" / source_id
            assert binding["evidence_checksum_sha256"] == _sha256(
                payload / "evidence" / "SHA256SUMS"
            )
            assert binding["prediction_sha256"] == _sha256(
                condition / "predictions.json"
            )
            assert binding["request_sha256"] == _sha256(condition / "requests.json")
            assert "completion_sha256" not in binding
            assert "source_request_sha256" not in binding
        else:
            assert binding["completion_sha256"] == _sha256(payload / "completion.json")
            assert binding["prediction_sha256"] == _sha256(payload / "predictions.json")
            assert binding["request_sha256"] == _sha256(payload / "requests.json")
    for pool_id, root in fixture["freeze_inputs"]["election_bundles"].items():
        binding = manifest["elections"][pool_id]
        assert binding["terminal_sha256"] == _sha256(root / "terminal-status.json")
        assert binding["request_sha256"] == _sha256(root / "payload/requests.json")
        assert binding["ranking_sha256"] == _sha256(root / "payload/rankings.json")
    evidence = manifest["evidence"]
    assert evidence["checksum_sha256"] == _sha256(fixture["evidence_checksums"])
    assert evidence["marker_sha256"] == _sha256(fixture["marker_path"])
    assert evidence["summary_sha256"] == _sha256(fixture["summary_path"])
    assert evidence["protocol_sha256"] == _sha256(fixture["protocol_path"])
    assert evidence["selector_sha256"] == _sha256(fixture["selector_path"])
    assert evidence["coverage_sha256"] == _sha256(fixture["coverage_path"])
    assert set(evidence["request_file_sha256"]) == set(ARM_IDS)
    assert evidence["localized_count"] == 1
    assert evidence["non_localized_count"] == 1
    encoded = json.dumps(manifest, sort_keys=True)
    assert "video000001" not in encoded
    assert str(tmp_path) not in encoded


@pytest.mark.parametrize(
    "target",
    ["prediction", "prediction-terminal", "ranking", "judge-terminal"],
)
def test_builder_rejects_broken_remote_digest_chains(
    tmp_path: Path, target: str
) -> None:
    builder = load_builder()
    fixture = _build_label_free_fixture(tmp_path)
    if target == "prediction":
        fixture["prediction_paths"]["G/N"].write_text("[]\n", encoding="utf-8")
    elif target == "prediction-terminal":
        terminal = (
            fixture["freeze_inputs"]["prediction_bundles"]["G/N"]
            / "terminal-status.json"
        )
        payload = json.loads(terminal.read_text(encoding="utf-8"))
        payload["completion_sha256"] = "0" * 64
        _write_json(terminal, payload)
    elif target == "ranking":
        fixture["ranking_paths"]["A"].write_text("{}\n", encoding="utf-8")
    else:
        terminal = (
            fixture["freeze_inputs"]["election_bundles"]["A"] / "terminal-status.json"
        )
        payload = json.loads(terminal.read_text(encoding="utf-8"))
        payload["request_sha256"] = "0" * 64
        _write_json(terminal, payload)

    with pytest.raises(ValueError, match="checksum|digest|terminal|prediction|ranking"):
        builder.build_freeze_manifest(fixture["freeze_inputs"], expected_rows=2)


def test_builder_rejects_coverage_not_derived_from_frozen_selector(
    tmp_path: Path,
) -> None:
    builder = load_builder()
    fixture = _build_label_free_fixture(tmp_path)
    _write_json(
        fixture["coverage_path"],
        {"video000001": False, "video000002": False},
    )

    with pytest.raises(ValueError, match="coverage|selector"):
        builder.build_freeze_manifest(fixture["freeze_inputs"], expected_rows=2)


def test_builder_rejects_selector_protocol_substitution(tmp_path: Path) -> None:
    builder = load_builder()
    fixture = _build_label_free_fixture(tmp_path)
    selector = json.loads(fixture["selector_path"].read_text(encoding="utf-8"))
    selector[0]["protocol_sha256"] = "0" * 64
    _write_json(fixture["selector_path"], selector)

    with pytest.raises(ValueError, match="selector protocol"):
        builder.build_freeze_manifest(fixture["freeze_inputs"], expected_rows=2)


def test_builder_rejects_a_self_consistent_label_bearing_leakage_audit(
    tmp_path: Path,
) -> None:
    builder = load_builder()
    fixture = _build_label_free_fixture(tmp_path)
    evidence_root = fixture["freeze_inputs"]["evidence_bundle"]
    leakage_path = evidence_root / "leakage-audit.json"
    leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
    leakage["reference"] = "must never enter the freeze"
    _write_json(leakage_path, leakage)
    summary_path = evidence_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["leakage_audit_sha256"] = _canonical_sha256(leakage)
    _write_json(summary_path, summary)
    _write_checksums(evidence_root)

    with pytest.raises(ValueError, match="label-bearing|reference"):
        builder.build_freeze_manifest(fixture["freeze_inputs"], expected_rows=2)


def test_builder_rejects_a_symlinked_remote_bundle_root(tmp_path: Path) -> None:
    builder = load_builder()
    fixture = _build_label_free_fixture(tmp_path)
    real_root = fixture["freeze_inputs"]["prediction_bundles"]["G/N"]
    alias = tmp_path / "prediction-alias"
    alias.symlink_to(real_root, target_is_directory=True)
    fixture["freeze_inputs"]["prediction_bundles"]["G/N"] = alias

    with pytest.raises(ValueError, match="symlink"):
        builder.build_freeze_manifest(fixture["freeze_inputs"], expected_rows=2)


def test_builder_rejects_self_consistent_label_metadata_in_predictions(
    tmp_path: Path,
) -> None:
    builder = load_builder()
    fixture = _build_label_free_fixture(tmp_path)
    root = fixture["freeze_inputs"]["prediction_bundles"]["G/N"]
    payload_root = root / "payload"
    prediction_path = payload_root / "predictions.json"
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    predictions["metadata"] = {"reference": "must never enter the freeze"}
    _write_json(prediction_path, predictions)
    completion_path = payload_root / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["prediction_sha256"] = _sha256(prediction_path)
    _write_json(completion_path, completion)
    checksum_path = _write_checksums(payload_root)
    terminal_path = root / "terminal-status.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["prediction_sha256"] = _sha256(prediction_path)
    terminal["completion_sha256"] = _sha256(completion_path)
    terminal["payload_sha256"] = _sha256(checksum_path)
    _write_json(terminal_path, terminal)

    with pytest.raises(ValueError, match="label-bearing|reference"):
        builder.build_freeze_manifest(fixture["freeze_inputs"], expected_rows=2)


def test_verifier_rejects_post_manifest_byte_substitution(tmp_path: Path) -> None:
    builder = load_builder()
    fixture = _build_label_free_fixture(tmp_path)
    manifest_path = tmp_path / "freeze-manifest.json"
    builder.write_freeze_manifest(
        manifest_path,
        builder.build_freeze_manifest(fixture["freeze_inputs"], expected_rows=2),
    )
    fixture["coverage_path"].write_bytes(fixture["coverage_path"].read_bytes() + b" ")

    with pytest.raises(ValueError, match="manifest|coverage|bytes|identity"):
        builder.verify_freeze_manifest(
            manifest_path,
            fixture["freeze_inputs"],
            expected_rows=2,
        )


def test_verifier_rejects_post_manifest_factorial_prediction_substitution(
    tmp_path: Path,
) -> None:
    builder = load_builder()
    fixture = _build_label_free_fixture(tmp_path)
    manifest_path = tmp_path / "freeze-manifest.json"
    builder.write_freeze_manifest(
        manifest_path,
        builder.build_freeze_manifest(fixture["freeze_inputs"], expected_rows=2),
    )
    prediction_path = fixture["factorial_fixture"]["4b-current"]["prediction_path"]
    prediction_path.write_bytes(prediction_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="checksum|factorial|manifest|bytes"):
        builder.verify_freeze_manifest(
            manifest_path,
            fixture["freeze_inputs"],
            expected_rows=2,
        )


def test_verifier_rejects_post_manifest_factorial_nested_checksum_substitution(
    tmp_path: Path,
) -> None:
    builder = load_builder()
    fixture = _build_label_free_fixture(tmp_path)
    manifest_path = tmp_path / "freeze-manifest.json"
    builder.write_freeze_manifest(
        manifest_path,
        builder.build_freeze_manifest(fixture["freeze_inputs"], expected_rows=2),
    )
    nested_checksum = fixture["factorial_fixture"]["4b-current"][
        "evidence_checksum_path"
    ]
    nested_checksum.write_bytes(nested_checksum.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="manifest|bytes|identity"):
        builder.verify_freeze_manifest(
            manifest_path,
            fixture["freeze_inputs"],
            expected_rows=2,
        )


@pytest.mark.parametrize("target", ["terminal", "status"])
def test_builder_rejects_label_metadata_in_factorial_markers(
    tmp_path: Path, target: str
) -> None:
    builder = load_builder()
    fixture = _build_label_free_fixture(tmp_path)
    root = fixture["freeze_inputs"]["prediction_bundles"]["4b-current"]
    terminal_path = root / "terminal-status.json"
    if target == "terminal":
        marker_path = terminal_path
    else:
        marker_path = root / "payload" / "status.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["annotations"] = {"reference": "must never enter the freeze"}
    _write_json(marker_path, marker)
    if target == "status":
        checksum_path = _write_checksums(root / "payload")
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        terminal["payload_sha256"] = _sha256(checksum_path)
        _write_json(terminal_path, terminal)

    with pytest.raises(ValueError, match="label-bearing|reference|schema|status|terminal"):
        builder.build_freeze_manifest(fixture["freeze_inputs"], expected_rows=2)
