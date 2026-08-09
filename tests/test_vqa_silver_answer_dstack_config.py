from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run-vqa-silver-answer-dstack.sh"
TASK4_BUILDER = ROOT / "scripts/build_vqa_silver_evidence.py"
FULL_MANIFEST = ROOT / "configs/vqa/qwen35_vqa2025_475_label_free.json"
CONFIGS = {
    cell: ROOT / f"dstack/vqa-silver-answer-{cell}.dstack.yml"
    for cell in ("gn", "gf", "gs", "hn", "hf", "hs", "ln", "lf", "ls")
}
CELLS = tuple(CONFIGS)
RUN_ID = "vqa-silver-oracle-election-20260809-a1"
EVIDENCE_URI = (
    f"hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/{RUN_ID}/evidence"
)
OUTPUT_BUCKET_PREFIX = "Npatta01/trec-video-2026-vqa-silver-answer-20260809-a1"
RUNTIME_IMAGE = (
    "docker.io/pytorch/pytorch@sha256:"
    "c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
)
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TRANSFORMERS_REVISION = "cbf4d720ec734edb77d25452b2790c7e4be2f8d7"
CANONICAL_ID_DIGEST = "95a623c0add252541b9c81ff4910e949dc81db17482add2c467b761237b22b0f"
BUNDLE_DIGEST_PLACEHOLDER = "UNSET_TASK4_EVIDENCE_BUNDLE_SHA256"
MARKER_DIGEST_PLACEHOLDER = "UNSET_TASK4_EVIDENCE_MARKER_SHA256"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _output_bucket_id(cell: str) -> str:
    return f"{OUTPUT_BUCKET_PREFIX}-{cell}"


def _output_uri(cell: str) -> str:
    return f"hf://buckets/{_output_bucket_id(cell)}"


def _env(payload: dict[str, object]) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in payload["env"]:
        key, value = str(item).split("=", 1)
        values[key] = value
    return values


def _load_task4_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "vqa_silver_evidence_task4", TASK4_BUILDER
    )
    if spec is None or spec.loader is None:
        raise AssertionError("could not load the Task 4 evidence builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _task4_marker_keys() -> set[str]:
    tree = ast.parse(TASK4_BUILDER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "marker"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        keys = {
            key.value
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if keys:
            return keys
    raise AssertionError("could not read the Task 4 evidence marker contract")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _task4_request_matrix(
    builder: ModuleType,
    *,
    video_id: str,
    frame_paths: list[str],
) -> dict[str, dict[str, object]]:
    base_frames = [
        {
            "slot_kind": "global",
            "requested_time_seconds": float(index),
            "actual_pts_seconds": float(index),
            "stream_index": 0,
            "frame_index": index,
            "frame_path": frame_path,
        }
        for index, frame_path in enumerate(frame_paths)
    ]
    frame_plans = {
        "G": [dict(frame) for frame in base_frames],
        "H": [
            {**frame, "slot_kind": "local" if index < 6 else "global"}
            for index, frame in enumerate(base_frames)
        ],
        "L": [{**frame, "slot_kind": "local"} for frame in base_frames],
    }
    transcript_views = {
        "N": {"text": None},
        "F": {"text": "[00:00:00.000] Full transcript."},
        "S": {"text": "[00:00:00.000] Selected transcript."},
    }
    return builder.build_request_matrix(
        row={"video_id": video_id, "question": "What is shown?"},
        frame_plans=frame_plans,
        transcript_views=transcript_views,
        run_id=RUN_ID,
    )


def _write_input_bundle(root: Path, cell: str = "gn") -> Path:
    input_root = root / "input"
    requests_root = input_root / "requests"
    frames_root = input_root / "frames/shared"
    requests_root.mkdir(parents=True)
    frames_root.mkdir(parents=True)
    frame_paths: list[str] = []
    for frame_index in range(12):
        frame = frames_root / f"frame-{frame_index:02d}.png"
        frame.write_bytes(f"frame-{frame_index}".encode("utf-8"))
        frame_paths.append(f"/evidence/frames/shared/{frame.name}")

    builder = _load_task4_builder()
    manifest = json.loads(FULL_MANIFEST.read_text(encoding="utf-8"))
    requests_by_cell: dict[str, list[dict[str, object]]] = {name: [] for name in CELLS}
    for example in manifest["examples"]:
        matrix = _task4_request_matrix(
            builder,
            video_id=example["video_id"],
            frame_paths=frame_paths,
        )
        for name in CELLS:
            requests_by_cell[name].append(matrix[name])

    for name, rows in requests_by_cell.items():
        _write_json(requests_root / f"{name}.json", rows)

    evidence = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "canonical_id_digest": CANONICAL_ID_DIGEST,
        "selector_protocol_sha256": "a" * 64,
        "model_identity": {"model_id": MODEL_ID, "revision": MODEL_REVISION},
        "runtime_identity": {"image": RUNTIME_IMAGE},
        "prompt_source_identity": "frozen-current-prompt",
        "rows": [],
    }
    leakage_audit = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "rows": [],
        "references_loaded": False,
    }
    _write_json(input_root / "evidence.json", evidence)
    _write_json(input_root / "leakage-audit.json", leakage_audit)
    marker = {
        "schema_version": 1,
        "label_policy": "label-free-evidence",
        "example_count": 475,
        "canonical_id_digest": CANONICAL_ID_DIGEST,
        "checksum_file": "SHA256SUMS",
    }
    _write_json(input_root / "evidence-marker.json", marker)
    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "row_count": 475,
        "cell_count": 9,
        "cells": list(CELLS),
        "request_counts": {name: 475 for name in CELLS},
        "request_set_sha256": {
            name: _canonical_sha256(rows) for name, rows in requests_by_cell.items()
        },
        "evidence_sha256": _canonical_sha256(evidence),
        "leakage_audit_sha256": _canonical_sha256(leakage_audit),
        "canonical_id_digest": CANONICAL_ID_DIGEST,
        "protocol_sha256": "a" * 64,
        "selector_protocol_sha256": "a" * 64,
    }
    _write_json(input_root / "summary.json", summary)
    checksum_lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
        f"{path.relative_to(input_root).as_posix()}"
        for path in sorted(path for path in input_root.rglob("*") if path.is_file())
    ]
    (input_root / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    assert cell in requests_by_cell
    return input_root


def _evidence_pins(input_root: Path) -> dict[str, str]:
    return {
        "EXPECTED_EVIDENCE_BUNDLE_SHA256": hashlib.sha256(
            (input_root / "SHA256SUMS").read_bytes()
        ).hexdigest(),
        "EXPECTED_EVIDENCE_MARKER_SHA256": hashlib.sha256(
            (input_root / "evidence-marker.json").read_bytes()
        ).hexdigest(),
    }


def _rewrite_checksums(input_root: Path) -> None:
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
        f"{path.relative_to(input_root).as_posix()}"
        for path in sorted(
            path
            for path in input_root.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        )
    ]
    (input_root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fake_inference(path: Path) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "if os.environ.get('FAKE_INFERENCE_LOG'):\n"
        "    Path(os.environ['FAKE_INFERENCE_LOG']).write_text('called\\n')\n"
        "rows = json.load(sys.stdin)\n"
        "model = sys.argv[sys.argv.index('--model-id') + 1]\n"
        "revision = sys.argv[sys.argv.index('--revision') + 1]\n"
        "policy = sys.argv[sys.argv.index('--answer-policy') + 1]\n"
        "minimum = int(sys.argv[sys.argv.index('--min-pixels-per-frame') + 1])\n"
        "maximum = int(sys.argv[sys.argv.index('--max-pixels-per-frame') + 1])\n"
        "out = []\n"
        "for row in rows:\n"
        "    out.append({\n"
        "      'status': 'passed', 'answer': 'stable', 'elapsed_seconds': 0.01,\n"
        "      'model_load_seconds': 0.25, 'model_id': model,\n"
        "      'model_revision': revision,\n"
        "      'runtime': 'transformers 5.15.0.dev0; torch 2.8.0',\n"
        "      'runtime_image': '" + RUNTIME_IMAGE + "', 'precision': 'bfloat16',\n"
        "      'device': 'cuda:0 (NVIDIA CUDA)',\n"
        "      'runtime_attestation': {'parameter_tensor_count': 1,\n"
        "        'parameter_devices': {'cuda:0': 1},\n"
        "        'parameter_dtypes': {'torch.bfloat16': 1},\n"
        "        'buffer_tensor_count': 0, 'buffer_devices': {}, 'buffer_dtypes': {},\n"
        "        'accelerator_backend': 'nvidia-cuda', 'torch_cuda': '12.4',\n"
        "        'torch_hip': None},\n"
        "      'peak_memory_bytes': 11, 'error_type': None, 'error_message': None,\n"
        "      'answer_policy': policy, 'question_type': 'general',\n"
        "      'requested_min_pixels_per_frame': minimum,\n"
        "      'requested_max_pixels_per_frame': maximum,\n"
        "      'min_pixels_per_frame': minimum, 'max_pixels_per_frame': maximum,\n"
        "      'run_id': row['run_id'], 'condition_id': row['condition_id'],\n"
        "      'video_id': row['video_id']\n"
        "    })\n"
        "print(json.dumps(out))\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _fake_hf(path: Path) -> Path:
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\n\' "$*" >> "$FAKE_HF_LOG"\n'
        'case "$1/$2" in\n'
        "  buckets/list)\n"
        "    count=0\n"
        '    [[ ! -f "$FAKE_LIST_COUNT" ]] || count=$(<"$FAKE_LIST_COUNT")\n'
        "    count=$((count + 1))\n"
        '    printf \'%s\' "$count" > "$FAKE_LIST_COUNT"\n'
        '    if [[ "$count" == 1 ]]; then\n'
        "      printf '%s' \"${FAKE_REMOTE_INITIAL_LISTING:-}\"\n"
        "    else\n"
        "      printf '%s' \"${FAKE_REMOTE_FINAL_LISTING:-}\"\n"
        "    fi\n"
        "    ;;\n"
        "  buckets/create)\n"
        '    if [[ "${FAKE_CREATE_CONFLICT:-0}" == 1 ]]; then\n'
        "      printf '%s\\n' 'HTTP 409 Conflict: bucket already exists' >&2\n"
        "      exit 1\n"
        "    fi\n"
        '    if [[ -n "${FAKE_BUCKET_LEASE_ROOT:-}" ]]; then\n'
        '      mkdir -p "$FAKE_BUCKET_LEASE_ROOT"\n'
        "      lease_name=${3//\\//__}\n"
        '      if ! mkdir "$FAKE_BUCKET_LEASE_ROOT/$lease_name" 2>/dev/null; then\n'
        "        printf '%s\\n' 'HTTP 409 Conflict: bucket already exists' >&2\n"
        "        exit 1\n"
        "      fi\n"
        "    fi\n"
        "    printf 'hf://buckets/%s\\n' \"$3\"\n"
        "    ;;\n"
        "  buckets/sync)\n"
        '    if [[ "$3" == hf://* ]]; then\n'
        '      mkdir -p "$4"\n'
        '      cp -a "$FAKE_INPUT_ROOT/." "$4/"\n'
        "    else\n"
        '      mkdir -p "$FAKE_UPLOAD_ROOT"\n'
        '      cp -a "$3/." "$FAKE_UPLOAD_ROOT/"\n'
        "    fi\n"
        "    ;;\n"
        "  buckets/cp)\n"
        '    mkdir -p "$(dirname "$FAKE_TERMINAL_CAPTURE")"\n'
        '    cp "$3" "$FAKE_TERMINAL_CAPTURE"\n'
        "    ;;\n"
        "  *) exit 99 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _run_runner(
    tmp_path: Path,
    *,
    input_root: Path,
    cell: str = "gn",
    extra: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    fake_hf = _fake_hf(tmp_path / "hf")
    fake_inference = _fake_inference(tmp_path / "inference.py")
    env = {
        **os.environ,
        "HF_TOKEN": "test-token-not-for-output",
        "HF_CLI_BIN": str(fake_hf),
        "FAKE_INPUT_ROOT": str(input_root),
        "FAKE_HF_LOG": str(tmp_path / "hf.log"),
        "FAKE_LIST_COUNT": str(tmp_path / "hf-list-count"),
        "FAKE_UPLOAD_ROOT": str(tmp_path / "uploaded-payload"),
        "FAKE_TERMINAL_CAPTURE": str(tmp_path / "uploaded-terminal.json"),
        "FAKE_INFERENCE_LOG": str(tmp_path / "inference.log"),
        "EXPERIMENT_RUN_ID": RUN_ID,
        "ANSWER_CELL": cell,
        "HF_INPUT_URI": EVIDENCE_URI,
        "HF_OUTPUT_BUCKET_ID": _output_bucket_id(cell),
        "HF_REMOTE_URI": _output_uri(cell),
        "VQA_RUNTIME_IMAGE": RUNTIME_IMAGE,
        "VQA_MODE": "answer",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "MODEL_ID": MODEL_ID,
        "MODEL_REVISION": MODEL_REVISION,
        "ANSWER_POLICY": "current",
        "MIN_PIXELS_PER_FRAME": "65536",
        "MAX_PIXELS_PER_FRAME": "262144",
        "MAX_NEW_TOKENS": "64",
        "EXPECTED_COUNT": "475",
        "REQUEST_NAME": f"requests/{cell}.json",
        "PYTHON_BIN": os.environ.get("PYTHON", "python3"),
        "INPUT_DIR": str(tmp_path / "synced-input"),
        "OUTPUT_DIR": str(tmp_path / "output"),
        "INFERENCE_BIN": str(fake_inference),
        "VQA_TEST_MODE": "1",
        "VQA_DRY_RUN": "1",
        "VQA_SKIP_PUBLICATION": "1",
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        **_evidence_pins(input_root),
    }
    env.update(extra or {})
    return subprocess.run(
        ["bash", str(RUNNER)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_nine_configs_cover_exactly_one_unique_answer_cell_each() -> None:
    assert set(CONFIGS) == {
        path.name.removeprefix("vqa-silver-answer-").removesuffix(".dstack.yml")
        for path in CONFIGS.values()
    }
    names: set[str] = set()
    output_bucket_ids: set[str] = set()
    output_uris: set[str] = set()
    bundle_pins: set[str] = set()
    marker_pins: set[str] = set()
    for cell, path in CONFIGS.items():
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        env = _env(payload)
        assert payload["type"] == "task"
        assert payload["name"] == f"vqa-silver-answer-{cell}-20260809-a1"
        assert payload["name"] not in names
        names.add(payload["name"])
        assert env["EXPERIMENT_RUN_ID"] == RUN_ID
        assert env["ANSWER_CELL"] == cell
        assert env["HF_INPUT_URI"] == EVIDENCE_URI
        assert env["HF_OUTPUT_BUCKET_ID"] == _output_bucket_id(cell)
        assert env["HF_REMOTE_URI"] == _output_uri(cell)
        output_bucket_ids.add(env["HF_OUTPUT_BUCKET_ID"])
        output_uris.add(env["HF_REMOTE_URI"])
        bundle_pins.add(env["EXPECTED_EVIDENCE_BUNDLE_SHA256"])
        marker_pins.add(env["EXPECTED_EVIDENCE_MARKER_SHA256"])
        assert env["EXPECTED_COUNT"] == "475"
        assert env["REQUEST_NAME"] == f"requests/{cell}.json"
        assert env["MODEL_ID"] == MODEL_ID
        assert env["MODEL_REVISION"] == MODEL_REVISION
        assert env["ANSWER_POLICY"] == "current"
        assert env["MIN_PIXELS_PER_FRAME"] == "65536"
        assert env["MAX_PIXELS_PER_FRAME"] == "262144"
        assert env["MAX_NEW_TOKENS"] == "64"
        assert env["HF_HUB_DISABLE_TELEMETRY"] == "1"
        assert env["VQA_RUNTIME_IMAGE"] == RUNTIME_IMAGE
        assert payload["image"] == RUNTIME_IMAGE
        assert payload["backends"] == ["runpod", "vastai"]
        assert payload["spot_policy"] == "on-demand"
        assert payload["resources"]["gpu"] == "L40S:1:48GB.."
        assert "ports:" not in path.read_text(encoding="utf-8")
        assert "../data/" not in path.read_text(encoding="utf-8")
    assert len(output_bucket_ids) == len(output_uris) == 9
    assert len(bundle_pins) == 1
    assert len(marker_pins) == 1
    bundle_pin = next(iter(bundle_pins))
    marker_pin = next(iter(marker_pins))
    assert bundle_pin == BUNDLE_DIGEST_PLACEHOLDER or SHA256_RE.fullmatch(bundle_pin)
    assert marker_pin == MARKER_DIGEST_PLACEHOLDER or SHA256_RE.fullmatch(marker_pin)
    assert (bundle_pin == BUNDLE_DIGEST_PLACEHOLDER) == (
        marker_pin == MARKER_DIGEST_PLACEHOLDER
    )


def test_configs_mount_only_label_free_runtime_sources() -> None:
    required = {
        "../scripts/run-vqa-silver-answer-dstack.sh:/dstack/run/scripts/"
        "run-vqa-silver-answer-dstack.sh",
        "../src/trec_video_2026:/dstack/run/src/trec_video_2026",
    }
    for path in CONFIGS.values():
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        mappings = set(payload["files"])
        assert required <= mappings
        assert not any(
            "annotation" in item.lower() or "reference" in item.lower()
            for item in mappings
        )
        assert "vqa_transformers_runner.py" not in path.read_text(
            encoding="utf-8"
        ) or any(item.endswith("vqa_transformers_runner.py") for item in mappings)


def test_runner_reuses_frozen_transformers_runner_contract() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "vqa_transformers_runner" in text
    assert f'MODEL_ID="{MODEL_ID}"' in text
    assert f'MODEL_REVISION="{MODEL_REVISION}"' in text
    assert f'TRANSFORMERS_REVISION="{TRANSFORMERS_REVISION}"' in text
    assert 'ANSWER_POLICY="${ANSWER_POLICY:-current}"' in text
    assert '--answer-policy "$ANSWER_POLICY"' in text
    assert '--max-new-tokens "$MAX_NEW_TOKENS"' in text
    assert "--model-class multimodal" in text
    assert '--min-pixels-per-frame "$MIN_PIXELS_PER_FRAME"' in text
    assert '--max-pixels-per-frame "$MAX_PIXELS_PER_FRAME"' in text


def test_integration_fixture_reads_task4_request_and_marker_contract(
    tmp_path: Path,
) -> None:
    input_root = _write_input_bundle(tmp_path)
    builder = _load_task4_builder()
    rows = json.loads((input_root / "requests/gn.json").read_text(encoding="utf-8"))
    row = rows[0]
    expected = _task4_request_matrix(
        builder,
        video_id=row["video_id"],
        frame_paths=row["frame_paths"],
    )["gn"]
    marker = json.loads(
        (input_root / "evidence-marker.json").read_text(encoding="utf-8")
    )
    summary = json.loads((input_root / "summary.json").read_text(encoding="utf-8"))

    assert row == expected
    assert set(marker) == _task4_marker_keys()
    assert summary["request_set_sha256"]["gn"] == _canonical_sha256(rows)


def test_placeholder_digest_blocks_dstack_job_before_remote_access(
    tmp_path: Path,
) -> None:
    input_root = _write_input_bundle(tmp_path)
    result = _run_runner(
        tmp_path / "run",
        input_root=input_root,
        extra={
            "EXPECTED_EVIDENCE_BUNDLE_SHA256": BUNDLE_DIGEST_PLACEHOLDER,
            "EXPECTED_EVIDENCE_MARKER_SHA256": MARKER_DIGEST_PLACEHOLDER,
        },
    )

    assert result.returncode != 0
    assert "evidence" in result.stderr.lower() and "sha-256" in result.stderr.lower()
    assert not (tmp_path / "run/hf.log").exists()
    assert not (tmp_path / "run/inference.log").exists()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (
            {"HF_OUTPUT_BUCKET_ID": f"{OUTPUT_BUCKET_PREFIX}-gf"},
            "bucket identity",
        ),
        (
            {"HF_REMOTE_URI": "hf://buckets/Npatta01/unpinned-output"},
            "exact cell bucket identity",
        ),
    ],
)
def test_runner_rejects_unpinned_output_bucket_before_remote_access(
    tmp_path: Path,
    override: dict[str, str],
    message: str,
) -> None:
    input_root = _write_input_bundle(tmp_path)
    run_root = tmp_path / "run"
    result = _run_runner(run_root, input_root=input_root, extra=override)

    assert result.returncode != 0
    assert message in result.stderr.lower()
    assert not (run_root / "hf.log").exists()
    assert not (run_root / "inference.log").exists()


@pytest.mark.parametrize(
    ("pin_name", "message"),
    [
        ("EXPECTED_EVIDENCE_BUNDLE_SHA256", "bundle"),
        ("EXPECTED_EVIDENCE_MARKER_SHA256", "marker"),
    ],
)
def test_runner_rejects_stale_predeclared_evidence_digest_before_inference(
    tmp_path: Path,
    pin_name: str,
    message: str,
) -> None:
    input_root = _write_input_bundle(tmp_path)
    result = _run_runner(
        tmp_path / "run",
        input_root=input_root,
        extra={pin_name: "f" * 64},
    )

    assert result.returncode != 0
    assert message in result.stderr.lower()
    assert "digest" in result.stderr.lower()
    assert not (tmp_path / "run/inference.log").exists()


def test_self_consistent_mutation_cannot_replace_predeclared_bundle(
    tmp_path: Path,
) -> None:
    input_root = _write_input_bundle(tmp_path)
    frozen_pins = _evidence_pins(input_root)
    request_path = input_root / "requests/gn.json"
    request_path.write_text(
        request_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    _rewrite_checksums(input_root)
    checked = subprocess.run(
        ["sha256sum", "--check", "SHA256SUMS"],
        cwd=input_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 0

    result = _run_runner(
        tmp_path / "run",
        input_root=input_root,
        extra=frozen_pins,
    )

    assert result.returncode != 0
    assert "bundle digest mismatch" in result.stderr.lower()
    assert not (tmp_path / "run/inference.log").exists()


def test_runner_freezes_475_answers_with_checksums_and_telemetry(
    tmp_path: Path,
) -> None:
    input_root = _write_input_bundle(tmp_path)
    result = _run_runner(tmp_path / "run", input_root=input_root)
    assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / "run/output"
    predictions = json.loads((output / "predictions.json").read_text(encoding="utf-8"))
    assert len(predictions) == 475
    assert all(row["status"] == "passed" and row["answer"] for row in predictions)
    assert (output / "telemetry/nvidia-smi.csv").is_file()
    assert (output / "completion.json").is_file()
    assert (output / "SHA256SUMS").is_file()
    checked = subprocess.run(
        ["sha256sum", "--check", "SHA256SUMS"],
        cwd=output,
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert status["expected_count"] == 475
    assert status["labels_joined"] is False


def test_runner_rejects_label_bearing_request_before_inference_and_publish(
    tmp_path: Path,
) -> None:
    input_root = _write_input_bundle(tmp_path)
    request_path = input_root / "requests/gn.json"
    rows = json.loads(request_path.read_text(encoding="utf-8"))
    rows[0]["nested"] = {"reference_answer": "must never enter runtime"}
    _write_json(request_path, rows)
    summary_path = input_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["request_set_sha256"]["gn"] = _canonical_sha256(rows)
    _write_json(summary_path, summary)
    _rewrite_checksums(input_root)

    run_root = tmp_path / "run"
    result = _run_runner(
        run_root,
        input_root=input_root,
        extra={"VQA_SKIP_PUBLICATION": "0"},
    )

    assert result.returncode != 0
    assert "label" in result.stderr.lower() or "reference" in result.stderr.lower()
    assert not (run_root / "inference.log").exists()
    assert not (run_root / "uploaded-payload").exists()
    assert not (run_root / "uploaded-terminal.json").exists()
    status = json.loads((run_root / "output/status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"


def test_runner_refuses_existing_local_or_remote_output(tmp_path: Path) -> None:
    input_root = _write_input_bundle(tmp_path)
    local_root = tmp_path / "local-case"
    output = local_root / "output"
    output.mkdir(parents=True)
    (output / "existing").write_text("do not overwrite", encoding="utf-8")
    result = _run_runner(
        local_root,
        input_root=input_root,
        extra={"OUTPUT_DIR": str(output)},
    )
    assert result.returncode != 0
    assert "non-empty" in result.stderr

    remote_root = tmp_path / "remote-case"
    result = _run_runner(
        remote_root,
        input_root=input_root,
        extra={"FAKE_REMOTE_INITIAL_LISTING": f"{_output_bucket_id('gn')}\n"},
    )
    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert not (remote_root / "inference.log").exists()
    commands = (remote_root / "hf.log").read_text(encoding="utf-8").splitlines()
    assert len(commands) == 1
    assert commands[0].startswith("buckets list Npatta01 --search ")


def test_atomic_bucket_lease_409_cannot_upload_or_mix_same_cell_output(
    tmp_path: Path,
) -> None:
    input_root = _write_input_bundle(tmp_path)
    run_root = tmp_path / "run"
    result = _run_runner(
        run_root,
        input_root=input_root,
        extra={
            "VQA_SKIP_PUBLICATION": "0",
            "FAKE_CREATE_CONFLICT": "1",
        },
    )

    assert result.returncode != 0
    assert "409 conflict" in result.stderr.lower()
    assert "atomic" in result.stderr.lower() or "lease" in result.stderr.lower()
    assert (run_root / "inference.log").is_file()
    assert (run_root / "output/completion.json").is_file()
    commands = (run_root / "hf.log").read_text(encoding="utf-8").splitlines()
    create = f"buckets create {_output_bucket_id('gn')} --private --format quiet"
    assert commands.count(create) == 1
    assert not any("--exist-ok" in command for command in commands)
    assert sum(command.startswith("buckets list ") for command in commands) == 1
    assert not any(
        command.startswith(f"buckets sync {run_root / 'output'} ")
        for command in commands
    )
    assert not any(command.startswith("buckets cp ") for command in commands)
    assert not (run_root / "uploaded-payload").exists()
    assert not (run_root / "uploaded-terminal.json").exists()


def test_concurrent_same_cell_publishers_have_exactly_one_atomic_lease_winner(
    tmp_path: Path,
) -> None:
    input_root = _write_input_bundle(tmp_path / "bundle")
    lease_root = tmp_path / "provider-bucket-leases"

    def publish(run_name: str) -> tuple[Path, subprocess.CompletedProcess[str]]:
        run_root = tmp_path / run_name
        result = _run_runner(
            run_root,
            input_root=input_root,
            extra={
                "VQA_SKIP_PUBLICATION": "0",
                "FAKE_BUCKET_LEASE_ROOT": str(lease_root),
            },
        )
        return run_root, result

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(publish, ("publisher-a", "publisher-b")))

    winners = [(root, result) for root, result in outcomes if result.returncode == 0]
    losers = [(root, result) for root, result in outcomes if result.returncode != 0]
    assert len(winners) == len(losers) == 1

    winner_root, _ = winners[0]
    loser_root, loser_result = losers[0]
    assert "409 conflict" in loser_result.stderr.lower()
    assert (winner_root / "uploaded-payload/completion.json").is_file()
    assert (winner_root / "uploaded-terminal.json").is_file()
    assert not (loser_root / "uploaded-payload").exists()
    assert not (loser_root / "uploaded-terminal.json").exists()
    for run_root, _ in outcomes:
        assert (run_root / "inference.log").is_file()
        commands = (run_root / "hf.log").read_text(encoding="utf-8").splitlines()
        assert (
            commands.count(
                f"buckets create {_output_bucket_id('gn')} --private --format quiet"
            )
            == 1
        )
        assert not any("--exist-ok" in command for command in commands)


def test_success_publication_has_content_bound_completion_marker(
    tmp_path: Path,
) -> None:
    input_root = _write_input_bundle(tmp_path)
    pins = _evidence_pins(input_root)
    run_root = tmp_path / "run"
    result = _run_runner(
        run_root,
        input_root=input_root,
        extra={"VQA_SKIP_PUBLICATION": "0"},
    )
    assert result.returncode == 0, result.stdout + result.stderr

    uploaded = run_root / "uploaded-payload"
    completion_path = uploaded / "completion.json"
    terminal_path = run_root / "uploaded-terminal.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    identity = json.loads(
        (uploaded / "condition-identity.json").read_text(encoding="utf-8")
    )
    status = json.loads((uploaded / "status.json").read_text(encoding="utf-8"))
    prediction_sha = hashlib.sha256(
        (uploaded / "predictions.json").read_bytes()
    ).hexdigest()
    completion_sha = hashlib.sha256(completion_path.read_bytes()).hexdigest()
    payload_sha = hashlib.sha256((uploaded / "SHA256SUMS").read_bytes()).hexdigest()

    assert completion["status"] == "passed"
    assert completion["run_id"] == RUN_ID
    assert completion["answer_cell"] == "gn"
    assert completion["output_bucket_id"] == _output_bucket_id("gn")
    assert completion["output_uri"] == _output_uri("gn")
    assert (
        completion["evidence_bundle_sha256"] == pins["EXPECTED_EVIDENCE_BUNDLE_SHA256"]
    )
    assert (
        completion["evidence_marker_sha256"] == pins["EXPECTED_EVIDENCE_MARKER_SHA256"]
    )
    assert completion["prediction_sha256"] == prediction_sha
    assert identity["model_id"] == MODEL_ID
    assert identity["model_revision"] == MODEL_REVISION
    assert identity["runtime_image"] == RUNTIME_IMAGE
    assert identity["output_bucket_id"] == _output_bucket_id("gn")
    assert identity["output_uri"] == _output_uri("gn")
    assert status["output_bucket_id"] == identity["output_bucket_id"]
    assert status["output_uri"] == identity["output_uri"]
    assert status["labels_joined"] is False
    assert terminal["payload_sha256"] == payload_sha
    assert terminal["completion_sha256"] == completion_sha
    assert terminal["evidence_bundle_sha256"] == completion["evidence_bundle_sha256"]
    assert terminal["request_sha256"] == completion["request_sha256"]
    assert terminal["prediction_sha256"] == prediction_sha
    assert terminal["output_bucket_id"] == completion["output_bucket_id"]
    commands = (run_root / "hf.log").read_text(encoding="utf-8").splitlines()
    create = f"buckets create {_output_bucket_id('gn')} --private --format quiet"
    output_sync = f"buckets sync {run_root / 'output'} {_output_uri('gn')}/payload"
    terminal_cp = (
        f"buckets cp {run_root / 'output.terminal-status.json'} "
        f"{_output_uri('gn')}/terminal-status.json"
    )
    assert commands.count(create) == 1
    assert sum(command.startswith("buckets list ") for command in commands) == 1
    assert (
        commands.index(create)
        < commands.index(output_sync)
        < commands.index(terminal_cp)
    )
    assert not any("--exist-ok" in command for command in commands)


@pytest.mark.parametrize("cell", sorted(CONFIGS))
def test_each_config_uses_a_distinct_request_and_fresh_bucket(cell: str) -> None:
    payload = yaml.safe_load(CONFIGS[cell].read_text(encoding="utf-8"))
    env = _env(payload)
    assert env["ANSWER_CELL"] == cell
    assert env["REQUEST_NAME"] == f"requests/{cell}.json"
    assert env["HF_OUTPUT_BUCKET_ID"] == _output_bucket_id(cell)
    assert env["HF_REMOTE_URI"] == _output_uri(cell)
    namespace, bucket_name = env["HF_OUTPUT_BUCKET_ID"].split("/", 1)
    assert namespace == "Npatta01"
    assert re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,94}[a-z0-9])?", bucket_name)
    assert len(bucket_name) <= 96
