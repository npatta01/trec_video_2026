#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_RUN_NAME="qwen35-4b-12f-full-20260803-c1"
readonly EXPECTED_INPUT_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/qwen35-4b-12f-full-20260803-b1-inputs/inputs"
readonly EXPECTED_REMOTE_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/$EXPECTED_RUN_NAME-output"
readonly EXPECTED_RUNTIME_IMAGE="docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
readonly MODEL_ID="Qwen/Qwen3.5-4B"
readonly MODEL_REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
readonly TRANSFORMERS_REVISION="cbf4d720ec734edb77d25452b2790c7e4be2f8d7"
readonly TRANSFORMERS_VERSION="5.15.0.dev0"
readonly TRANSFORMERS_SOURCE_URL="https://github.com/huggingface/transformers/archive/$TRANSFORMERS_REVISION.tar.gz"
readonly UV_VERSION="0.11.3"
readonly HF_CLI_VERSION="1.26.0"
readonly EXPECTED_COUNT=475
readonly EXPECTED_FRAME_COUNT=12
readonly PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
readonly INPUT_DIR="${INPUT_DIR:-/dstack/input}"
readonly FRAME_DIR="${FRAME_DIR:-/dstack/frames}"
readonly OUTPUT_DIR="${OUTPUT_DIR:-/dstack/output}"
readonly MANIFEST="${MANIFEST:-/dstack/run/configs/vqa/qwen35_vqa2025_475_label_free.json}"

die() {
  echo "error: $*" >&2
  exit 2
}

hf_cli() {
  HF_TOKEN="${HF_TOKEN:?HF_TOKEN is required}" \
    uvx --from "huggingface-hub==$HF_CLI_VERSION" hf "$@"
}

verify_environment() {
  [[ -x "$PYTHON_BIN" ]] || die "missing image interpreter: $PYTHON_BIN"
  [[ "$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')" == "$PYTHON_BIN" ]] \
    || die "image interpreter identity mismatch"
  [[ "${EXPERIMENT_RUN_NAME:-}" == "$EXPECTED_RUN_NAME" ]] \
    || die "experiment name mismatch"
  [[ "${HF_INPUT_URI:-}" == "$EXPECTED_INPUT_URI" ]] \
    || die "input URI mismatch"
  [[ "${HF_REMOTE_URI:-}" == "$EXPECTED_REMOTE_URI" ]] \
    || die "remote URI mismatch"
  [[ "${VQA_RUNTIME_IMAGE:-}" == "$EXPECTED_RUNTIME_IMAGE" ]] \
    || die "runtime image mismatch"
  [[ -f "$MANIFEST" ]] || die "label-free manifest is missing: $MANIFEST"
  mkdir -p "$INPUT_DIR" "$FRAME_DIR" "$OUTPUT_DIR"
  "$PYTHON_BIN" - "$MANIFEST" "$OUTPUT_DIR/manifest-preflight.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
examples = manifest.get("examples")
if manifest.get("label_policy") != "label-free-inference-input":
    raise SystemExit("manifest label policy is not label-free")
if not isinstance(examples, list) or len(examples) != 475:
    raise SystemExit("manifest must contain exactly 475 examples")
forbidden = {
    "answer", "answers", "annotation", "annotations", "correct_answer",
    "incorrect_answers", "expected_answer", "ground_truth", "label", "labels",
    "reference", "reference_answer", "reference_answers",
}
def reject(value, location="$"):
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in forbidden:
                raise SystemExit(f"forbidden label-bearing field {key!r} at {location}")
            reject(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject(child, f"{location}[{index}]")
reject(manifest)
ids = []
for index, example in enumerate(examples):
    if set(example) != {"video_id", "question"}:
        raise SystemExit(f"manifest example {index} has unexpected fields")
    if not isinstance(example["video_id"], str) or not isinstance(example["question"], str):
        raise SystemExit(f"manifest example {index} has non-string fields")
    ids.append(example["video_id"])
if len(set(ids)) != 475:
    raise SystemExit("manifest IDs must be unique")
Path(sys.argv[2]).write_text(
    json.dumps({"example_count": len(ids), "video_ids": ids}, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

bootstrap_runtime() {
  "$PYTHON_BIN" -m pip install --no-cache-dir --only-binary=:all: "uv==$UV_VERSION"
  "$PYTHON_BIN" -m pip install --no-cache-dir \
    "transformers @ $TRANSFORMERS_SOURCE_URL" \
    "qwen-vl-utils[decord]==0.0.14" \
    "av==16.0.1" \
    "accelerate==1.12.0" \
    "imageio-ffmpeg==0.6.0"
  "$PYTHON_BIN" - "$OUTPUT_DIR/runtime.json" <<'PY'
import json
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig

if transformers.__version__ != "5.15.0.dev0":
    raise SystemExit(f"unexpected transformers version: {transformers.__version__}")
if not torch.cuda.is_available() or not torch.version.cuda or torch.version.hip:
    raise SystemExit("NVIDIA CUDA is unavailable")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("CUDA bfloat16 is unavailable")
config = AutoConfig.for_model("qwen3_5")
record = {
    "model_type": config.model_type,
    "transformers_version": transformers.__version__,
    "transformers_revision": "cbf4d720ec734edb77d25452b2790c7e4be2f8d7",
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_device_name": torch.cuda.get_device_name(0),
    "cuda_bf16_supported": True,
}
Path(sys.argv[1]).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, sort_keys=True))
PY
}

remote_prefix_is_empty() {
  local listing list_rc
  set +e
  listing="$(hf_cli buckets list "$EXPECTED_REMOTE_URI" --format quiet 2>"$OUTPUT_DIR/remote-list.stderr")"
  list_rc=$?
  set -e
  if (( list_rc != 0 )); then
    sed -n '1,120p' "$OUTPUT_DIR/remote-list.stderr" >&2
    die "could not inspect remote output prefix (hf rc=$list_rc)"
  fi
  [[ -z "$listing" ]] || die "remote output prefix already exists: $EXPECTED_REMOTE_URI"
}

verify_input_bundle() {
  "$PYTHON_BIN" - "$INPUT_DIR" "$OUTPUT_DIR/input-preflight.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath

root = Path(sys.argv[1]).resolve(strict=True)
contract_path = root / "contract.json"
marker_path = root / "INPUTS_COMPLETE.json"
contract = json.loads(contract_path.read_text(encoding="utf-8"))
marker = json.loads(marker_path.read_text(encoding="utf-8"))
objects = contract.get("objects")
if contract.get("schema_version") != 1 or contract.get("media_count") != 475:
    raise SystemExit("input contract identity mismatch")
if marker.get("schema_version") != 1 or marker.get("input_count") != 475:
    raise SystemExit("input marker identity mismatch")
if marker.get("object_count") != 477 or marker.get("media_count") != 475:
    raise SystemExit("input marker object count mismatch")
if not isinstance(objects, list) or len(objects) != 475:
    raise SystemExit("input contract must contain 475 media objects")

expected_paths = {"contract.json", "INPUTS_COMPLETE.json"}
digests = {}
for record in objects:
    relative_raw = record.get("path")
    video_id = record.get("video_id")
    expected_digest = record.get("sha256")
    expected_size = record.get("size_bytes")
    if not all(isinstance(value, str) for value in (relative_raw, video_id, expected_digest)):
        raise SystemExit("input contract has invalid object identity")
    relative = PurePosixPath(relative_raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise SystemExit(f"unsafe input path: {relative}")
    path = root.joinpath(*relative.parts).resolve(strict=True)
    if not path.is_relative_to(root):
        raise SystemExit(f"input path escapes root: {relative}")
    if path.suffix.lower() not in {".mkv", ".mp4", ".webm", ".mov"}:
        raise SystemExit(f"non-video object in media contract: {relative}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_digest = digest.hexdigest()
    if actual_digest != expected_digest or path.stat().st_size != expected_size:
        raise SystemExit(f"input media hash/size mismatch: {relative}")
    expected_paths.add(relative.as_posix())
    digests[relative.as_posix()] = actual_digest

actual_paths = {
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file()
}
if actual_paths != expected_paths:
    raise SystemExit(
        f"input object set mismatch: missing={sorted(expected_paths - actual_paths)} "
        f"extra={sorted(actual_paths - expected_paths)}"
    )
contract_digest = hashlib.sha256(contract_path.read_bytes()).hexdigest()
if marker.get("contract_sha256") != contract_digest:
    raise SystemExit("input contract hash does not match marker")
if marker.get("media_bytes") != sum((root / path).stat().st_size for path in digests):
    raise SystemExit("input media byte count does not match marker")
Path(sys.argv[2]).write_text(
    json.dumps(
        {
            "contract_sha256": contract_digest,
            "media_count": len(digests),
            "media_bytes": marker["media_bytes"],
            "object_count": len(actual_paths),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
}

prepare_input() {
  local started_ns finished_ns
  started_ns="$(date +%s%N)"
  hf_cli buckets sync "$EXPECTED_INPUT_URI" "$INPUT_DIR"
  verify_input_bundle
  "$PYTHON_BIN" /dstack/run/scripts/qwen35_prepare_requests.py \
    --manifest "$MANIFEST" \
    --input-root "$INPUT_DIR" \
    --frame-root "$FRAME_DIR" \
    --output "$OUTPUT_DIR/requests.json" \
    --run-id "$EXPECTED_RUN_NAME" \
    --condition-id qwen35-4b-12f-full \
    --count "$EXPECTED_COUNT" \
    --frame-count "$EXPECTED_FRAME_COUNT"
  finished_ns="$(date +%s%N)"
  "$PYTHON_BIN" - "$OUTPUT_DIR/prepare-summary.json" "$started_ns" "$finished_ns" <<'PY'
import json
import sys
from pathlib import Path
started = int(sys.argv[2])
finished = int(sys.argv[3])
Path(sys.argv[1]).write_text(
    json.dumps(
        {"prepare_wall_seconds": (finished - started) / 1_000_000_000},
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
}

run_workload() {
  local started_ns child_rc finished_ns
  started_ns="$(date +%s%N)"
  set +e
  "$PYTHON_BIN" -m trec_video_2026.vqa_transformers_runner \
    --model-id "$MODEL_ID" \
    --revision "$MODEL_REVISION" \
    --model-class image-text \
    --max-new-tokens 64 \
    < "$OUTPUT_DIR/requests.json" \
    > "$OUTPUT_DIR/predictions.json" \
    2> "$OUTPUT_DIR/inference.stderr"
  child_rc=$?
  set -e
  finished_ns="$(date +%s%N)"
  "$PYTHON_BIN" - "$OUTPUT_DIR/summary.json" "$child_rc" "$started_ns" "$finished_ns" <<'PY'
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
child_rc = int(sys.argv[2])
started_ns = int(sys.argv[3])
finished_ns = int(sys.argv[4])
payload = {
    "workload_returncode": child_rc,
    "wall_seconds": (finished_ns - started_ns) / 1_000_000_000,
}
prepare_summary = Path("/dstack/output/prepare-summary.json")
if prepare_summary.is_file():
    payload.update(json.loads(prepare_summary.read_text(encoding="utf-8")))
predictions = Path("/dstack/output/predictions.json")
if predictions.is_file():
    try:
        rows = json.loads(predictions.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        rows = None
    if isinstance(rows, list):
        payload["rows"] = len(rows)
        payload["statuses"] = {
            status: sum(row.get("status") == status for row in rows)
            for status in sorted({row.get("status") for row in rows})
        }
        elapsed = [
            row["elapsed_seconds"]
            for row in rows
            if isinstance(row.get("elapsed_seconds"), (int, float))
        ]
        if elapsed:
            payload["row_elapsed_seconds"] = {
                "sum": sum(elapsed),
                "mean": sum(elapsed) / len(elapsed),
                "min": min(elapsed),
                "max": max(elapsed),
            }
        payload["devices"] = sorted({row.get("device") for row in rows})
        payload["model_load_seconds"] = sorted(
            {row.get("model_load_seconds") for row in rows}
        )
Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  return "$child_rc"
}

validate_predictions() {
  "$PYTHON_BIN" - "$OUTPUT_DIR/requests.json" "$OUTPUT_DIR/predictions.json" <<'PY'
import json
import sys
from pathlib import Path

requests = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
predictions = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if not isinstance(predictions, list) or len(predictions) != 475:
    raise SystemExit("prediction count is not 475")
request_ids = [row["video_id"] for row in requests]
prediction_ids = [row.get("video_id") for row in predictions]
if prediction_ids != request_ids:
    raise SystemExit("prediction IDs/order do not match requests")
statuses = {row.get("status") for row in predictions}
if statuses != {"passed"}:
    raise SystemExit(f"prediction statuses are not all passed: {sorted(statuses)}")
if any(not isinstance(row.get("answer"), str) or not row["answer"].strip() for row in predictions):
    raise SystemExit("one or more predictions has an empty answer")
PY
}

finalize() {
  local workload_rc="$1"
  local sync_rc
  "$PYTHON_BIN" - "$OUTPUT_DIR/terminal-status.json" "$workload_rc" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(
    json.dumps({"workload_returncode": int(sys.argv[2])}, sort_keys=True) + "\n"
)
PY
  (cd "$OUTPUT_DIR" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
  set +e
  hf_cli buckets sync "$OUTPUT_DIR" "$EXPECTED_REMOTE_URI/output"
  sync_rc=$?
  set -e
  "$PYTHON_BIN" - "$OUTPUT_DIR/terminal-status.json" "$workload_rc" "$sync_rc" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "workload_returncode": int(sys.argv[2]),
            "sync_returncode": int(sys.argv[3]),
        },
        sort_keys=True,
    )
    + "\n"
)
PY
  (( workload_rc == 0 )) || return "$workload_rc"
  return "$sync_rc"
}

main() {
  local workload_rc
  mkdir -p "$OUTPUT_DIR"
  verify_environment
  bootstrap_runtime
  uvx hf version > "$OUTPUT_DIR/hf-version.txt"
  uvx hf auth whoami > "$OUTPUT_DIR/hf-whoami.txt"
  remote_prefix_is_empty
  prepare_input
  "$PYTHON_BIN" -m trec_video_2026.vqa_transformers_runner --help > "$OUTPUT_DIR/child-help.txt"
  set +e
  run_workload
  workload_rc=$?
  set -e
  if (( workload_rc == 0 )); then
    set +e
    validate_predictions
    workload_rc=$?
    set -e
  fi
  finalize "$workload_rc"
}

main "$@"
