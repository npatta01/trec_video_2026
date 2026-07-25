#!/usr/bin/env bash
set -euo pipefail

readonly DEFAULT_MODEL_ID="Qwen/Qwen3.6-35B-A3B"
readonly DEFAULT_MODEL_REVISION="995ad96eacd98c81ed38be0c5b274b04031597b0"
readonly DEFAULT_PANEL_SHA256="d2c5515b8c7398f326931e2f993f3fd6eac8e70d9ef01d1a1bd4afcf3f6792e8"
readonly DEFAULT_REMOTE_ROOT="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments"
readonly DEFAULT_RUNTIME_IMAGE="docker.io/pytorch/pytorch@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca"
readonly TRANSFORMERS_REVISION="cbf4d720ec734edb77d25452b2790c7e4be2f8d7"
readonly TRANSFORMERS_VERSION="5.15.0.dev0"
readonly TRANSFORMERS_SOURCE_URL="https://github.com/huggingface/transformers/archive/cbf4d720ec734edb77d25452b2790c7e4be2f8d7.tar.gz"
readonly UV_VERSION="0.11.3"
readonly UV_REQUIREMENT="uv==0.11.3"
readonly RUNTIME_REQUIREMENTS="/dstack/run/configs/vqa/qwen36_a4_runtime_lock.txt"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
INPUT_REQUESTS="${INPUT_REQUESTS:-/dstack/input/requests.json}"
PANEL_MANIFEST="${PANEL_MANIFEST:-/dstack/input/failure_panel_12_inference.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/dstack/output}"
EXPERIMENT_RUN_NAME="${EXPERIMENT_RUN_NAME:-}"
EXPECTED_PANEL_SHA256="${EXPECTED_PANEL_SHA256:-$DEFAULT_PANEL_SHA256}"
MODEL_ID="${MODEL_ID:-$DEFAULT_MODEL_ID}"
MODEL_REVISION="${MODEL_REVISION:-$DEFAULT_MODEL_REVISION}"
HF_REMOTE_ROOT="${HF_REMOTE_ROOT:-$DEFAULT_REMOTE_ROOT}"
EXPECTED_EVIDENCE_ROOT="${EXPECTED_EVIDENCE_ROOT:-/dstack/input/evidence}"
EVIDENCE_ARTIFACT_URI="${EVIDENCE_ARTIFACT_URI:-}"
SOURCE_REQUESTS="${SOURCE_REQUESTS:-data/vqa/benchmarks/runs/native-audio-benchmark-v2/requests/qwen35-4b-12f-frozen30.json}"
STAGED_REQUESTS="${STAGED_REQUESTS:-data/vqa/benchmarks/followups/failure-panel-2026-07-24/dstack-input/requests.json}"
SOURCE_EVIDENCE_ROOT="${SOURCE_EVIDENCE_ROOT:-data/vqa/benchmarks/evidence}"
STAGED_EVIDENCE_ROOT="${STAGED_EVIDENCE_ROOT:-data/vqa/benchmarks/followups/failure-panel-2026-07-24/dstack-input/evidence}"
EXPECTED_RUNTIME_IMAGE="${EXPECTED_RUNTIME_IMAGE:-$DEFAULT_RUNTIME_IMAGE}"

die() {
  echo "error: $*" >&2
  exit 2
}

validate_static_inputs() {
  [[ "$PYTHON_BIN" = /* ]] || die "PYTHON_BIN must be an absolute path"
  [[ -x "$PYTHON_BIN" ]] || die "PYTHON_BIN is not executable: $PYTHON_BIN"
  [[ -f "$PANEL_MANIFEST" ]] || die "panel manifest is missing: $PANEL_MANIFEST"
  [[ -f "$INPUT_REQUESTS" ]] || die "request manifest is missing: $INPUT_REQUESTS"
  [[ "$EXPERIMENT_RUN_NAME" =~ ^[a-z0-9][a-z0-9-]*-[0-9]{8}T?[0-9]{0,6}-[a-z0-9]+$ ]] \
    || die "immutable experiment run name required (dated name plus unique suffix)"
  [[ "$EXPERIMENT_RUN_NAME" != *latest* ]] || die "immutable experiment run name cannot contain latest"
  [[ "$HF_REMOTE_ROOT" == hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments ]] \
    || die "unexpected Hugging Face Bucket root"
  mkdir -p "$OUTPUT_DIR"

  "$PYTHON_BIN" - "$PANEL_MANIFEST" "$EXPECTED_PANEL_SHA256" \
    "$INPUT_REQUESTS" "$EXPECTED_EVIDENCE_ROOT" "$OUTPUT_DIR/preflight-input.json" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

panel_path = Path(sys.argv[1])
expected_panel_hash = sys.argv[2]
requests_path = Path(sys.argv[3])
evidence_root = Path(sys.argv[4]).resolve()
record_path = Path(sys.argv[5])

actual_panel_hash = hashlib.sha256(panel_path.read_bytes()).hexdigest()
if actual_panel_hash != expected_panel_hash:
    raise SystemExit(
        "panel manifest SHA-256 mismatch: "
        f"expected {expected_panel_hash}, got {actual_panel_hash}"
    )

panel = json.loads(panel_path.read_text(encoding="utf-8"))
requests = json.loads(requests_path.read_text(encoding="utf-8"))
if not isinstance(requests, list) or len(requests) != 12:
    raise SystemExit("request manifest must contain exactly 12 requests")

forbidden = {
    "answer",
    "answers",
    "annotation",
    "annotations",
    "expected_answer",
    "ground_truth",
    "label",
    "labels",
    "reference",
    "reference_answer",
    "reference_answers",
}

def reject_labels(value: object, location: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in forbidden:
                raise SystemExit(
                    f"forbidden label-bearing field {key!r} at {location}"
                )
            reject_labels(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_labels(child, f"{location}[{index}]")

reject_labels(panel)
reject_labels(requests)

panel_ids = [row["video_id"] for row in panel["examples"]]
request_ids = [row.get("video_id") for row in requests]
if request_ids != panel_ids:
    raise SystemExit("request IDs/order do not match the frozen failure panel")

allowed = {
    "run_id",
    "condition_id",
    "video_id",
    "question",
    "frame_paths",
    "frame_timestamps",
    "frame_descriptions",
    "transcript",
}
for index, (request, example) in enumerate(zip(requests, panel["examples"], strict=True)):
    extra = set(request) - allowed
    if extra:
        raise SystemExit(f"request {index} has unsupported fields: {sorted(extra)}")
    if request.get("question") != example["question"]:
        raise SystemExit(f"request {index} question differs from frozen panel")
    if request.get("transcript") is not None:
        raise SystemExit(f"request {index} must be visual-only")
    paths = request.get("frame_paths")
    timestamps = request.get("frame_timestamps")
    if not isinstance(paths, list) or len(paths) != 12:
        raise SystemExit(f"request {index} must contain exactly 12 frames")
    if not isinstance(timestamps, list) or len(timestamps) != 12:
        raise SystemExit(f"request {index} must contain exactly 12 timestamps")
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.is_relative_to(evidence_root):
            raise SystemExit(f"frame path escapes derived evidence root: {path}")
        if not path.is_file():
            raise SystemExit(f"derived frame is missing: {path}")
        metadata_path = path.parent / "evidence.json"
        if not metadata_path.is_file():
            raise SystemExit(f"derived evidence metadata is missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            frame["path"]: frame["sha256"] for frame in metadata["frames"]
        }.get(path.name)
        if expected is None:
            raise SystemExit(f"frame is absent from evidence metadata: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"derived frame SHA-256 mismatch: {path}")

record = {
    "panel_manifest_sha256": actual_panel_hash,
    "requests_sha256": hashlib.sha256(requests_path.read_bytes()).hexdigest(),
    "request_count": len(requests),
    "video_ids": request_ids,
}
record_path.write_text(
    json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

prepare_local_requests() {
  [[ -f "$PANEL_MANIFEST" ]] || die "panel manifest is missing: $PANEL_MANIFEST"
  [[ -f "$SOURCE_REQUESTS" ]] || die "source requests are missing: $SOURCE_REQUESTS"
  mkdir -p "$(dirname "$STAGED_REQUESTS")"
  "$PYTHON_BIN" - "$PANEL_MANIFEST" "$SOURCE_REQUESTS" "$STAGED_REQUESTS" \
    "$EXPERIMENT_RUN_NAME" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

panel = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
source = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
destination = Path(sys.argv[3])
run_name = sys.argv[4]
by_id = {request["video_id"]: request for request in source}
staged = []
for example in panel["examples"]:
    video_id = example["video_id"]
    request = by_id[video_id]
    if request["question"] != example["question"]:
        raise SystemExit(f"question mismatch for {video_id}")
    frame_paths = [Path(path) for path in request["frame_paths"]]
    if len(frame_paths) != 12 or not all(path.is_file() for path in frame_paths):
        raise SystemExit(f"missing 12-frame derived evidence for {video_id}")
    staged.append(
        {
            "run_id": run_name,
            "condition_id": "qwen36-35b-a3b-12f-failure-panel",
            "video_id": video_id,
            "question": example["question"],
            "frame_paths": [
                f"/dstack/input/evidence/{video_id}/{path.name}"
                for path in frame_paths
            ],
            "frame_timestamps": request["frame_timestamps"],
            "frame_descriptions": None,
            "transcript": None,
        }
    )
destination.write_text(
    json.dumps(staged, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    encoding="utf-8",
)
print(destination)
PY
}

prepare_local_evidence() {
  [[ -f "$PANEL_MANIFEST" ]] || die "panel manifest is missing: $PANEL_MANIFEST"
  [[ -f "$SOURCE_REQUESTS" ]] || die "source requests are missing: $SOURCE_REQUESTS"
  mkdir -p "$STAGED_EVIDENCE_ROOT"
  "$PYTHON_BIN" - "$PANEL_MANIFEST" "$SOURCE_REQUESTS" \
    "$SOURCE_EVIDENCE_ROOT" "$STAGED_EVIDENCE_ROOT" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

panel = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
requests = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
source_root = Path(sys.argv[3]).resolve(strict=True)
destination_root_raw = Path(sys.argv[4])
destination_root = destination_root_raw.resolve()
by_id = {request["video_id"]: request for request in requests}
panel_ids = [example["video_id"] for example in panel["examples"]]
if len(panel_ids) != 12 or len(set(panel_ids)) != 12:
    raise SystemExit("frozen evidence panel must contain 12 unique IDs")

forbidden = {
    "answer", "answers", "annotation", "annotations", "expected_answer",
    "ground_truth", "label", "labels", "reference", "reference_answer",
    "reference_answers",
}

def reject_labels(value: object, location: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in forbidden:
                raise SystemExit(
                    f"forbidden label-bearing evidence field {key!r} at {location}"
                )
            reject_labels(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_labels(child, f"{location}[{index}]")

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def reject_symlink(path: Path, root: Path) -> None:
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise SystemExit(f"symlink is forbidden in evidence: {current}")

expected: dict[Path, Path] = {}
expected_directories = {Path(".")}
for video_id in panel_ids:
    if video_id not in by_id:
        raise SystemExit(f"missing source request for {video_id}")
    raw_paths = [Path(path) for path in by_id[video_id]["frame_paths"]]
    if len(raw_paths) != 12:
        raise SystemExit(f"expected 12 frames for {video_id}")
    for raw_path in raw_paths:
        reject_symlink(raw_path, source_root)
    frame_paths = [path.resolve(strict=True) for path in raw_paths]
    source_directory = frame_paths[0].parent
    if not source_directory.is_relative_to(source_root):
        raise SystemExit(f"source evidence escapes canonical cache: {source_directory}")
    if any(path.parent != source_directory for path in frame_paths):
        raise SystemExit(f"mixed evidence directories for {video_id}")
    metadata_path = source_directory / "evidence.json"
    reject_symlink(metadata_path, source_root)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    reject_labels(metadata, f"{video_id}.evidence.json")
    frame_records = metadata.get("frames")
    if not isinstance(frame_records, list) or len(frame_records) != 12:
        raise SystemExit(f"evidence metadata must contain 12 frames for {video_id}")
    metadata_hashes = {
        record["path"]: record["sha256"] for record in frame_records
    }
    if set(metadata_hashes) != {path.name for path in frame_paths}:
        raise SystemExit(f"evidence metadata frame set mismatch for {video_id}")
    for frame in frame_paths:
        if digest(frame) != metadata_hashes[frame.name]:
            raise SystemExit(f"source frame SHA-256 mismatch: {frame}")
        expected[Path(video_id) / frame.name] = frame
    expected[Path(video_id) / "evidence.json"] = metadata_path
    expected_directories.add(Path(video_id))

if destination_root_raw.is_symlink():
    raise SystemExit(f"staged evidence root cannot be a symlink: {destination_root_raw}")
destination_root.mkdir(parents=True, exist_ok=True)
existing = list(destination_root.iterdir())
if existing:
    actual_files = set()
    actual_directories = {Path(".")}
    for path in destination_root.rglob("*"):
        relative = path.relative_to(destination_root)
        if path.is_symlink():
            raise SystemExit(f"symlink is forbidden in staged evidence: {path}")
        if path.is_dir():
            actual_directories.add(relative)
        elif path.is_file():
            actual_files.add(relative)
        else:
            raise SystemExit(f"unsupported staged evidence entry: {path}")
    if actual_files != set(expected) or actual_directories != expected_directories:
        raise SystemExit("staged evidence destination is not an exact idempotent copy")
    for relative, source in expected.items():
        destination = destination_root / relative
        if not destination.samefile(source) or digest(destination) != digest(source):
            raise SystemExit(f"conflicting staged evidence file: {destination}")
else:
    for directory in sorted(expected_directories - {Path(".")}):
        (destination_root / directory).mkdir()
    for relative, source in expected.items():
        os.link(source, destination_root / relative)

files = [path for path in destination_root.rglob("*") if path.is_file()]
directories = [path for path in destination_root.iterdir() if path.is_dir()]
if len(files) != 156 or len(directories) != 12:
    raise SystemExit("staged evidence must be exactly 12 directories and 156 files")
print(destination_root)
PY
}

fetch_evidence() {
  command -v uvx >/dev/null || die "uvx is unavailable in the pinned image"
  [[ "$EVIDENCE_ARTIFACT_URI" == \
    hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/artifacts/failure-panel-12-frames-d2c5515b8c73 ]] \
    || die "unexpected immutable evidence artifact URI"
  mkdir -p "$EXPECTED_EVIDENCE_ROOT"
  uvx hf buckets sync "$EVIDENCE_ARTIFACT_URI" "$EXPECTED_EVIDENCE_ROOT"
}

verify_transformers_runtime() {
  mkdir -p "$OUTPUT_DIR"
  "$PYTHON_BIN" - "$OUTPUT_DIR/transformers-bootstrap.json" \
    "$TRANSFORMERS_VERSION" "$TRANSFORMERS_REVISION" \
    "$TRANSFORMERS_SOURCE_URL" <<'PY'
import json
import sys
from pathlib import Path

import transformers

record_path = Path(sys.argv[1])
expected_version, revision, source_url = sys.argv[2:5]
if transformers.__version__ != expected_version:
    raise SystemExit(
        "qwen3_5_moe runtime capability check failed: "
        f"expected transformers {expected_version}, got {transformers.__version__}"
    )
import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

try:
    config = AutoConfig.for_model("qwen3_5_moe")
except Exception as error:
    raise SystemExit(
        f"qwen3_5_moe runtime capability check failed: {error}"
    ) from error
if getattr(config, "model_type", None) != "qwen3_5_moe":
    raise SystemExit("qwen3_5_moe runtime capability check returned wrong model type")
if torch.__version__.split("+", 1)[0] != "2.10.0":
    raise SystemExit(f"expected torch 2.10.0, got {torch.__version__}")
if not torch.version.cuda or torch.version.hip:
    raise SystemExit("runtime torch build is not NVIDIA CUDA")
if not torch.cuda.is_available():
    raise SystemExit("live NVIDIA CUDA accelerator is unavailable")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("live NVIDIA CUDA accelerator does not support bfloat16")
if AutoProcessor is None or AutoModelForImageTextToText is None:
    raise SystemExit("required Transformers auto classes are unavailable")
record = {
    "transformers_version": transformers.__version__,
    "transformers_revision": revision,
    "transformers_source_url": source_url,
    "qwen3_5_moe_model_type": config.model_type,
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cuda_bf16_supported": torch.cuda.is_bf16_supported(),
    "cuda_device_name": torch.cuda.get_device_name(0),
    "python_executable": sys.executable,
    "auto_processor_import": AutoProcessor.__name__,
    "auto_model_import": AutoModelForImageTextToText.__name__,
}
payload = json.dumps(record, sort_keys=True) + "\n"
record_path.write_text(payload, encoding="utf-8")
print(payload, end="")
PY
}

bootstrap_transformers() {
  [[ "$PYTHON_BIN" = /* && -x "$PYTHON_BIN" ]] \
    || die "exact image Python is unavailable: $PYTHON_BIN"
  [[ "$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')" == "$PYTHON_BIN" ]] \
    || die "image Python executable identity mismatch"
  # The pinned Ubuntu 24.04 image marks apt Python as externally managed.
  # Install only the exact uv bootstrap into that image Python; the complete
  # dependency environment is then resolved and checked by uv below.
  "$PYTHON_BIN" -m pip install \
    --break-system-packages \
    --no-deps \
    "$UV_REQUIREMENT"
  [[ "$("$PYTHON_BIN" -c \
    'import importlib.metadata; print(importlib.metadata.version("uv"))')" == "$UV_VERSION" ]] \
    || die "uv version attestation failed"
  # The base image includes the unrelated Spin CLI, whose Click upper bound
  # conflicts with the pinned Hugging Face Hub CLI required by Transformers.
  # Spin is not used by this inference task, so remove it before resolving and
  # checking the model runtime.
  "$PYTHON_BIN" -m pip uninstall \
    --break-system-packages \
    --yes \
    spin
  "$PYTHON_BIN" - <<'PY'
import importlib.metadata

try:
    importlib.metadata.version("spin")
except importlib.metadata.PackageNotFoundError:
    pass
else:
    raise SystemExit("unused base-image spin package was not removed")
PY
  [[ -f "$RUNTIME_REQUIREMENTS" ]] || die "pinned runtime requirements are missing"
  "$PYTHON_BIN" -m uv pip install \
    --system \
    --break-system-packages \
    --python "$PYTHON_BIN" \
    --no-managed-python \
    --no-python-downloads \
    --reinstall-package transformers \
    --requirements "$RUNTIME_REQUIREMENTS"
  "$PYTHON_BIN" -m uv pip check --system --python "$PYTHON_BIN"
  verify_transformers_runtime
}

child_argv() {
  if [[ -n "${CHILD_EXECUTABLE:-}" ]]; then
    printf '%s\0' "$CHILD_EXECUTABLE"
    return
  fi
  printf '%s\0' \
    "$PYTHON_BIN" \
    "-m" "trec_video_2026.vqa_transformers_runner" \
    "--model-id" "$MODEL_ID" \
    "--revision" "$MODEL_REVISION" \
    "--model-class" "image-text" \
    "--max-new-tokens" "64"
}

write_preflight_record() {
  local -a argv
  mapfile -d '' -t argv < <(child_argv)
  set +e
  "${argv[@]}" --help >"$OUTPUT_DIR/child-help.stdout" 2>"$OUTPUT_DIR/child-help.stderr"
  local child_help_rc=$?
  set -e
  (( child_help_rc == 0 )) || die "real child argv preflight failed with return code $child_help_rc"

  "$PYTHON_BIN" - "$OUTPUT_DIR/preflight-input.json" "$OUTPUT_DIR/preflight.json" \
    "$MODEL_ID" "$MODEL_REVISION" "$child_help_rc" "${argv[@]}" <<'PY'
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

base = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
packages = {}
for name in ("torch", "transformers", "vllm", "uv"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
base.update(
    {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "model_id": sys.argv[3],
        "model_revision": sys.argv[4],
        "child_help_returncode": int(sys.argv[5]),
        "child_argv": sys.argv[6:],
        "packages": packages,
    }
)
Path(sys.argv[2]).write_text(
    json.dumps(base, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

remote_prefix_is_empty() {
  local remote_uri="$1"
  local listing
  set +e
  listing="$(uvx hf buckets list "$remote_uri" --format json 2>"$OUTPUT_DIR/remote-list.stderr")"
  local list_rc=$?
  set -e
  (( list_rc == 0 )) || die "could not verify immutable remote prefix (hf list rc=$list_rc)"
  [[ -z "$listing" ]]
}

preflight_local_evidence_upload() {
  command -v uvx >/dev/null || die "uvx is unavailable"
  [[ "$EVIDENCE_ARTIFACT_URI" == \
    hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/artifacts/failure-panel-12-frames-d2c5515b8c73 ]] \
    || die "unexpected immutable evidence artifact URI"
  mkdir -p "$OUTPUT_DIR"
  remote_prefix_is_empty "$EVIDENCE_ARTIFACT_URI" \
    || die "evidence artifact prefix already exists: $EVIDENCE_ARTIFACT_URI"
  [[ -d "$STAGED_EVIDENCE_ROOT" ]] || die "staged evidence root is missing"
}

validate_predictions() {
  "$PYTHON_BIN" - "$INPUT_REQUESTS" "$OUTPUT_DIR/predictions.json" \
    "$MODEL_ID" "$MODEL_REVISION" "$EXPECTED_RUNTIME_IMAGE" <<'PY'
import json
import os
import sys
from pathlib import Path

requests = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
try:
    results = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"prediction validation failed: invalid JSON: {error}")
model_id, revision, runtime_image = sys.argv[3:6]
if os.environ.get("VQA_RUNTIME_IMAGE") != runtime_image:
    raise SystemExit("prediction validation failed: VQA_RUNTIME_IMAGE is not exact")
if not isinstance(results, list) or len(results) != 12:
    raise SystemExit("prediction validation failed: expected exactly 12 rows")
expected_ids = [request["video_id"] for request in requests]
if [result.get("video_id") for result in results] != expected_ids:
    raise SystemExit("prediction validation failed: video IDs/order mismatch")
for index, result in enumerate(results):
    prefix = f"prediction validation failed at row {index}:"
    if result.get("status") != "passed":
        raise SystemExit(f"{prefix} status is not passed")
    answer = result.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise SystemExit(f"{prefix} answer is empty")
    if result.get("model_id") != model_id or result.get("model_revision") != revision:
        raise SystemExit(f"{prefix} model identity mismatch")
    if result.get("runtime_image") != runtime_image:
        raise SystemExit(f"{prefix} runtime image mismatch")
    if result.get("precision") != "bfloat16":
        raise SystemExit(f"{prefix} precision is not bfloat16")
    if result.get("device") != "cuda:0 (NVIDIA CUDA)":
        raise SystemExit(f"{prefix} device is not NVIDIA CUDA cuda:0")
    attestation = result.get("runtime_attestation")
    if not isinstance(attestation, dict):
        raise SystemExit(f"{prefix} runtime attestation is missing")
    if attestation.get("accelerator_backend") != "nvidia-cuda":
        raise SystemExit(f"{prefix} accelerator backend is not NVIDIA CUDA")
    parameter_devices = attestation.get("parameter_devices")
    if (
        not isinstance(parameter_devices, dict)
        or set(parameter_devices) != {"cuda:0"}
        or not isinstance(parameter_devices["cuda:0"], int)
        or parameter_devices["cuda:0"] <= 0
    ):
        raise SystemExit(f"{prefix} parameter placement is not exclusively cuda:0")
    parameter_dtypes = attestation.get("parameter_dtypes")
    if (
        not isinstance(parameter_dtypes, dict)
        or set(parameter_dtypes) != {"torch.bfloat16"}
        or not isinstance(parameter_dtypes["torch.bfloat16"], int)
        or parameter_dtypes["torch.bfloat16"] <= 0
    ):
        raise SystemExit(f"{prefix} parameters are not exclusively torch.bfloat16")
PY
}

run_workload() {
  command -v uvx >/dev/null || die "uvx is unavailable in the pinned image"
  local remote_uri="$HF_REMOTE_ROOT/$EXPERIMENT_RUN_NAME"
  remote_prefix_is_empty "$remote_uri" || die "remote run prefix already exists: $remote_uri"

  local -a argv
  mapfile -d '' -t argv < <(child_argv)
  set +e
  "${argv[@]}" <"$INPUT_REQUESTS" >"$OUTPUT_DIR/predictions.json" \
    2>"$OUTPUT_DIR/workload.stderr"
  local workload_rc=$?
  set -e
  if (( workload_rc == 0 )); then
    set +e
    validate_predictions 2>>"$OUTPUT_DIR/workload.stderr"
    local validation_rc=$?
    set -e
    if (( validation_rc != 0 )); then
      workload_rc=65
    fi
  fi

  "$PYTHON_BIN" - "$OUTPUT_DIR/terminal-status.json" "$workload_rc" 0 <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(
    json.dumps(
        {"workload_returncode": int(sys.argv[2]), "sync_returncode": int(sys.argv[3])},
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
  (
    cd "$OUTPUT_DIR"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >SHA256SUMS
  )

  set +e
  uvx hf buckets sync "$OUTPUT_DIR" "$remote_uri"
  local sync_rc=$?
  set -e
  "$PYTHON_BIN" - "$OUTPUT_DIR/terminal-status.json" "$workload_rc" "$sync_rc" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(
    json.dumps(
        {"workload_returncode": int(sys.argv[2]), "sync_returncode": int(sys.argv[3])},
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

  (( workload_rc == 0 )) || return "$workload_rc"
  return "$sync_rc"
}

case "${1:-}" in
  bootstrap-transformers)
    bootstrap_transformers
    ;;
  verify-transformers-runtime)
    verify_transformers_runtime
    ;;
  fetch-evidence)
    fetch_evidence
    ;;
  prepare-local)
    prepare_local_requests
    ;;
  prepare-evidence-local)
    prepare_local_evidence
    ;;
  preflight-evidence-upload-local)
    preflight_local_evidence_upload
    ;;
  --preflight)
    validate_static_inputs
    write_preflight_record
    ;;
  run)
    validate_static_inputs
    run_workload
    ;;
  *)
    die "usage: $0 bootstrap-transformers | verify-transformers-runtime | prepare-local | prepare-evidence-local | preflight-evidence-upload-local | fetch-evidence | --preflight | run"
    ;;
esac
