#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_RUN_ID="vqa-silver-oracle-election-20260809-a1"
readonly EXPECTED_EVIDENCE_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/${EXPECTED_RUN_ID}/evidence"
readonly EXPECTED_OUTPUT_NAMESPACE="Npatta01"
readonly EXPECTED_OUTPUT_BUCKET_PREFIX="${EXPECTED_OUTPUT_NAMESPACE}/trec-video-2026-vqa-silver-answer-20260809-a1"
readonly EXPECTED_RUNTIME_IMAGE="docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
readonly EXPECTED_MODEL_ID="Qwen/Qwen3.5-4B"
readonly EXPECTED_MODEL_REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
readonly TRANSFORMERS_REVISION="cbf4d720ec734edb77d25452b2790c7e4be2f8d7"
readonly TRANSFORMERS_SOURCE_URL="https://github.com/huggingface/transformers/archive/${TRANSFORMERS_REVISION}.tar.gz"
readonly HF_CLI_VERSION="1.26.0"
readonly EXPECTED_COUNT=475
readonly EXPECTED_FRAME_COUNT=12
readonly EXPECTED_MIN_PIXELS=65536
readonly EXPECTED_MAX_PIXELS=262144
readonly EXPECTED_MAX_NEW_TOKENS=64
readonly EXPECTED_CANONICAL_ID_DIGEST="95a623c0add252541b9c81ff4910e949dc81db17482add2c467b761237b22b0f"

readonly PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
readonly INPUT_DIR="${INPUT_DIR:-/dstack/input}"
readonly OUTPUT_DIR="${OUTPUT_DIR:-/dstack/output}"
readonly TERMINAL_STATUS_PATH="${OUTPUT_DIR}.terminal-status.json"
readonly INFERENCE_BIN="${INFERENCE_BIN:-${PYTHON_BIN}}"

EXPERIMENT_RUN_ID="${EXPERIMENT_RUN_ID:-}"
ANSWER_CELL="${ANSWER_CELL:-}"
HF_INPUT_URI="${HF_INPUT_URI:-}"
HF_OUTPUT_BUCKET_ID="${HF_OUTPUT_BUCKET_ID:-}"
HF_REMOTE_URI="${HF_REMOTE_URI:-}"
EXPECTED_EVIDENCE_BUNDLE_SHA256="${EXPECTED_EVIDENCE_BUNDLE_SHA256:-}"
EXPECTED_EVIDENCE_MARKER_SHA256="${EXPECTED_EVIDENCE_MARKER_SHA256:-}"
VQA_RUNTIME_IMAGE="${VQA_RUNTIME_IMAGE:-}"
MODEL_ID="${MODEL_ID:-}"
MODEL_REVISION="${MODEL_REVISION:-}"
ANSWER_POLICY="${ANSWER_POLICY:-current}"
MIN_PIXELS_PER_FRAME="${MIN_PIXELS_PER_FRAME:-}"
MAX_PIXELS_PER_FRAME="${MAX_PIXELS_PER_FRAME:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
EXPECTED_COUNT_ENV="${EXPECTED_COUNT:-}"
REQUEST_NAME="${REQUEST_NAME:-}"
VQA_MODE="${VQA_MODE:-}"

HF_COMMAND=()
TELEMETRY_PID=""
CURRENT_CHILD_PID=""
RUNTIME_STRING=""
TORCH_CUDA_VERSION=""
REQUEST_SHA256=""
SOURCE_REQUEST_SHA256=""
EVIDENCE_BUNDLE_SHA256=""
EVIDENCE_MARKER_SHA256=""
EXECUTION_STARTED=0
FINALIZE_ATTEMPTED=0

die() { echo "error: $*" >&2; exit 2; }
event() { [[ -z "${EVENT_LOG:-}" ]] || printf '%s\n' "$1" >> "$EVENT_LOG"; }
require_value() { [[ -n "${!1:-}" ]] || die "$1 is required"; }

require_sha256_pin() {
  local name="$1" value="${!1:-}"
  [[ "$value" =~ ^[0-9a-f]{64}$ && "$value" != "$(printf '0%.0s' {1..64})" ]] || \
    die "$name must be a non-placeholder lowercase evidence SHA-256"
}

hf_cli() {
  ((${#HF_COMMAND[@]} > 0)) || die "Hugging Face client is not ready"
  if [[ -n "${HF_TOKEN:-}" && -z "${HF_CLI_BIN:-}" ]]; then
    HF_TOKEN="$HF_TOKEN" "${HF_COMMAND[@]}" "$@"
  else
    "${HF_COMMAND[@]}" "$@"
  fi
}

ensure_hf_cli() {
  event lightweight-client-start
  if [[ -n "${HF_CLI_BIN:-}" ]]; then
    command -v "$HF_CLI_BIN" >/dev/null 2>&1 || die "HF_CLI_BIN is not executable"
    HF_COMMAND=("$HF_CLI_BIN")
  elif command -v uvx >/dev/null 2>&1; then
    HF_COMMAND=(uvx --from "huggingface-hub==${HF_CLI_VERSION}" hf)
  elif command -v hf >/dev/null 2>&1; then
    HF_COMMAND=(hf)
  else
    "$PYTHON_BIN" -m pip install --no-cache-dir "huggingface_hub==${HF_CLI_VERSION}"
    command -v hf >/dev/null 2>&1 || die "pinned Hugging Face CLI is unavailable"
    HF_COMMAND=(hf)
  fi
  event lightweight-client-ready
}

validate_dispatch() {
  require_value HF_TOKEN
  [[ "$EXPERIMENT_RUN_ID" == "$EXPECTED_RUN_ID" ]] || die "experiment run identity mismatch"
  case "$ANSWER_CELL" in gn|gf|gs|hn|hf|hs|ln|lf|ls) ;; *) die "unsupported answer cell: $ANSWER_CELL" ;; esac
  [[ "$VQA_MODE" == "answer" ]] || die "VQA_MODE must be answer"
  [[ "$HF_INPUT_URI" == "$EXPECTED_EVIDENCE_URI" ]] || die "HF_INPUT_URI is not the immutable evidence URI"
  [[ "$HF_OUTPUT_BUCKET_ID" == "$EXPECTED_OUTPUT_BUCKET_PREFIX-$ANSWER_CELL" ]] || \
    die "HF_OUTPUT_BUCKET_ID is not the exact fresh cell bucket identity"
  [[ "$HF_REMOTE_URI" == "hf://buckets/$HF_OUTPUT_BUCKET_ID" ]] || \
    die "HF_REMOTE_URI is not bound to the exact cell bucket identity"
  require_sha256_pin EXPECTED_EVIDENCE_BUNDLE_SHA256
  require_sha256_pin EXPECTED_EVIDENCE_MARKER_SHA256
  [[ "$VQA_RUNTIME_IMAGE" == "$EXPECTED_RUNTIME_IMAGE" ]] || die "runtime image is not pinned"
  [[ "$MODEL_ID" == "$EXPECTED_MODEL_ID" && "$MODEL_REVISION" == "$EXPECTED_MODEL_REVISION" ]] || die "model identity is not frozen"
  [[ "$ANSWER_POLICY" == "current" ]] || die "answer policy is not the frozen current prompt"
  [[ "$MIN_PIXELS_PER_FRAME" == "$EXPECTED_MIN_PIXELS" && "$MAX_PIXELS_PER_FRAME" == "$EXPECTED_MAX_PIXELS" ]] || die "pixel bounds are not frozen"
  [[ "$MAX_NEW_TOKENS" == "$EXPECTED_MAX_NEW_TOKENS" ]] || die "max_new_tokens is not frozen"
  [[ "$EXPECTED_COUNT_ENV" == "$EXPECTED_COUNT" ]] || die "expected count is not 475"
  [[ "$REQUEST_NAME" == "requests/$ANSWER_CELL.json" ]] || die "request name is not bound to the selected cell"
  [[ "${HF_HUB_DISABLE_TELEMETRY:-}" == "1" ]] || die "Hugging Face telemetry must be disabled"
  if [[ "$PYTHON_BIN" == /* ]]; then
    [[ -x "$PYTHON_BIN" ]] || die "image Python is unavailable: $PYTHON_BIN"
    [[ "$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')" == "$PYTHON_BIN" ]] || die "image Python identity mismatch"
  else
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python is unavailable: $PYTHON_BIN"
  fi
  event dispatch-validated
}

refuse_nonempty_output() {
  [[ ! -L "$OUTPUT_DIR" ]] || die "OUTPUT_DIR must not be a symlink"
  if [[ -e "$OUTPUT_DIR" && ! -d "$OUTPUT_DIR" ]]; then die "OUTPUT_DIR is not a directory"; fi
  if [[ -d "$OUTPUT_DIR" ]] && find "$OUTPUT_DIR" -mindepth 1 -print -quit | grep -q .; then die "OUTPUT_DIR is non-empty"; fi
  [[ ! -e "$TERMINAL_STATUS_PATH" ]] || die "terminal status path already exists"
}

output_bucket_is_absent() {
  local bucket_name listing
  bucket_name="${HF_OUTPUT_BUCKET_ID#*/}"
  listing="$(hf_cli buckets list "$EXPECTED_OUTPUT_NAMESPACE" --search "$bucket_name" --format quiet)" || \
    die "could not inspect output bucket identity"
  if grep -Fxq "$HF_OUTPUT_BUCKET_ID" <<< "$listing"; then
    die "remote output bucket already exists: $HF_OUTPUT_BUCKET_ID"
  fi
}

acquire_output_bucket_lease() {
  local created_uri
  event output-bucket-lease-start
  if ! created_uri="$(hf_cli buckets create "$HF_OUTPUT_BUCKET_ID" --private --format quiet)"; then
    die "atomic output bucket lease failed (the provider may have returned 409 for a concurrent publisher)"
  fi
  [[ "$created_uri" == "$HF_REMOTE_URI" ]] || \
    die "created output bucket identity does not match the pinned URI"
  event output-bucket-lease-acquired
}

bootstrap_runtime() {
  event heavy-runtime-bootstrap
  mkdir -p "$OUTPUT_DIR"
  if [[ "${VQA_DRY_RUN:-0}" == "1" || "${VQA_TEST_MODE:-0}" == "1" ]]; then
    "$PYTHON_BIN" - "$OUTPUT_DIR/runtime.json" "$VQA_RUNTIME_IMAGE" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "runtime": "transformers 5.15.0.dev0; torch 2.8.0",
    "runtime_image": sys.argv[2],
    "transformers_version": "5.15.0.dev0",
    "transformers_revision": "cbf4d720ec734edb77d25452b2790c7e4be2f8d7",
    "torch_version": "2.8.0", "torch_cuda": "12.4", "torch_hip": None,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    return
  fi
  "$PYTHON_BIN" -m pip install --no-cache-dir \
    "transformers @ ${TRANSFORMERS_SOURCE_URL}" "qwen-vl-utils[decord]==0.0.14" \
    "av==16.0.1" "accelerate==1.12.0" "imageio-ffmpeg==0.6.0" \
    "Pillow==11.3.0" "soundfile==0.13.1"
  "$PYTHON_BIN" - "$OUTPUT_DIR/runtime.json" "$VQA_RUNTIME_IMAGE" <<'PY'
import json
import sys
from importlib.metadata import version
from pathlib import Path
import PIL
import accelerate
import av
import imageio_ffmpeg
import soundfile
import torch
import transformers
qwen_version = version("qwen-vl-utils")
if transformers.__version__ != "5.15.0.dev0" or qwen_version != "0.0.14":
    raise SystemExit("pinned Transformers/qwen-vl-utils runtime is unavailable")
if not torch.cuda.is_available() or not torch.version.cuda or torch.version.hip or not torch.cuda.is_bf16_supported():
    raise SystemExit("pinned NVIDIA CUDA BF16 runtime is unavailable")
record = {
    "runtime": f"transformers {transformers.__version__}; torch {torch.__version__}",
    "runtime_image": sys.argv[2], "transformers_version": transformers.__version__,
    "transformers_revision": "cbf4d720ec734edb77d25452b2790c7e4be2f8d7",
    "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
    "torch_hip": torch.version.hip, "cuda_device_name": torch.cuda.get_device_name(0),
    "cuda_bf16_supported": True,
    "dependencies": {"Pillow": PIL.__version__, "soundfile": soundfile.__version__,
                     "accelerate": accelerate.__version__, "av": av.__version__,
                     "imageio-ffmpeg": imageio_ffmpeg.__version__, "qwen-vl-utils": qwen_version},
}
Path(sys.argv[1]).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

load_runtime_provenance() {
  local values
  values="$("$PYTHON_BIN" - "$OUTPUT_DIR/runtime.json" "$VQA_RUNTIME_IMAGE" <<'PY'
import json
import sys
from pathlib import Path
record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (not isinstance(record.get("runtime"), str) or not record["runtime"].strip()
    or record.get("runtime_image") != sys.argv[2]
    or record.get("transformers_revision") != "cbf4d720ec734edb77d25452b2790c7e4be2f8d7"
    or not isinstance(record.get("torch_cuda"), str) or record.get("torch_hip") is not None):
    raise SystemExit("runtime provenance is not pinned")
print(record["runtime"])
print(record["torch_cuda"])
PY
)"
  RUNTIME_STRING="${values%%$'\n'*}"
  TORCH_CUDA_VERSION="${values#*$'\n'}"
}

verify_input_bundle() {
  if [[ -e "$INPUT_DIR" ]]; then
    [[ ! -L "$INPUT_DIR" ]] || die "input directory is a symlink"
    [[ -z "$(find "$INPUT_DIR" -mindepth 1 -print -quit)" ]] || die "input directory is not a fresh mount"
  fi
  mkdir -p "$INPUT_DIR"
  hf_cli buckets sync "$HF_INPUT_URI" "$INPUT_DIR" || die "immutable evidence sync failed"
  [[ -z "$(find "$INPUT_DIR" -type l -print -quit)" ]] || die "immutable evidence contains a symlink"
  [[ -f "$INPUT_DIR/SHA256SUMS" && ! -L "$INPUT_DIR/SHA256SUMS" ]] || \
    die "immutable evidence SHA256SUMS is missing"
  EVIDENCE_BUNDLE_SHA256="$(sha256sum "$INPUT_DIR/SHA256SUMS" | awk '{print $1}')"
  [[ "$EVIDENCE_BUNDLE_SHA256" == "$EXPECTED_EVIDENCE_BUNDLE_SHA256" ]] || \
    die "immutable evidence bundle digest mismatch"
  (cd "$INPUT_DIR" && sha256sum --strict --check SHA256SUMS >/dev/null) || \
    die "immutable evidence checksum verification failed"
  [[ -f "$INPUT_DIR/evidence-marker.json" && ! -L "$INPUT_DIR/evidence-marker.json" ]] || \
    die "immutable evidence marker is missing"
  EVIDENCE_MARKER_SHA256="$(sha256sum "$INPUT_DIR/evidence-marker.json" | awk '{print $1}')"
  [[ "$EVIDENCE_MARKER_SHA256" == "$EXPECTED_EVIDENCE_MARKER_SHA256" ]] || \
    die "immutable evidence marker digest mismatch"
  SOURCE_REQUEST_SHA256="$("$PYTHON_BIN" - "$INPUT_DIR" "$ANSWER_CELL" \
    "$EXPERIMENT_RUN_ID" "$EXPECTED_CANONICAL_ID_DIGEST" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
cell = sys.argv[2]
run_id = sys.argv[3]
expected_id_digest = sys.argv[4]
cells = ("gn", "gf", "gs", "hn", "hf", "hs", "ln", "lf", "ls")
sha256_re = re.compile(r"[0-9a-f]{64}")

def canonical_sha256(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

manifest_lines = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
listed = []
for line in manifest_lines:
    match = re.fullmatch(r"([0-9a-f]{64})  ([^\n]+)", line)
    if match is None:
        raise SystemExit("immutable evidence SHA256SUMS is not canonical")
    relative = Path(match.group(2))
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() == "SHA256SUMS":
        raise SystemExit("immutable evidence SHA256SUMS contains an unsafe path")
    listed.append(relative.as_posix())
if len(listed) != len(set(listed)):
    raise SystemExit("immutable evidence SHA256SUMS contains duplicate paths")
actual = sorted(
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file() and path.name != "SHA256SUMS"
)
if sorted(listed) != actual:
    raise SystemExit("immutable evidence inventory does not exactly match SHA256SUMS")

summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
summary_fields = {
    "schema_version",
    "run_id",
    "row_count",
    "cell_count",
    "cells",
    "request_counts",
    "request_set_sha256",
    "evidence_sha256",
    "leakage_audit_sha256",
    "canonical_id_digest",
    "protocol_sha256",
    "selector_protocol_sha256",
}
if not isinstance(summary, dict) or set(summary) != summary_fields:
    raise SystemExit("immutable evidence summary marker schema mismatch")
if (
    summary["schema_version"] != 1
    or summary["run_id"] != run_id
    or summary["row_count"] != 475
    or summary["cell_count"] != 9
    or summary["cells"] != list(cells)
    or summary["canonical_id_digest"] != expected_id_digest
    or summary["request_counts"] != {name: 475 for name in cells}
    or not isinstance(summary["request_set_sha256"], dict)
    or set(summary["request_set_sha256"]) != set(cells)
):
    raise SystemExit("immutable evidence summary marker identity mismatch")
for name in (
    "evidence_sha256",
    "leakage_audit_sha256",
    "canonical_id_digest",
    "protocol_sha256",
    "selector_protocol_sha256",
):
    if not isinstance(summary[name], str) or sha256_re.fullmatch(summary[name]) is None:
        raise SystemExit(f"immutable evidence summary marker has invalid {name}")

marker = json.loads((root / "evidence-marker.json").read_text(encoding="utf-8"))
if set(marker) != {
    "schema_version",
    "label_policy",
    "example_count",
    "canonical_id_digest",
    "checksum_file",
} or marker != {
    "schema_version": 1,
    "label_policy": "label-free-evidence",
    "example_count": 475,
    "canonical_id_digest": summary["canonical_id_digest"],
    "checksum_file": "SHA256SUMS",
}:
    raise SystemExit("immutable evidence marker identity mismatch")

evidence = json.loads((root / "evidence.json").read_text(encoding="utf-8"))
leakage = json.loads((root / "leakage-audit.json").read_text(encoding="utf-8"))
if canonical_sha256(evidence) != summary["evidence_sha256"]:
    raise SystemExit("immutable evidence payload identity mismatch")
if canonical_sha256(leakage) != summary["leakage_audit_sha256"]:
    raise SystemExit("immutable evidence leakage-audit identity mismatch")
if not isinstance(leakage, dict) or leakage.get("references_loaded") is not False:
    raise SystemExit("immutable evidence leakage audit is not label-free")
if (
    not isinstance(evidence, dict)
    or evidence.get("canonical_id_digest") != summary["canonical_id_digest"]
    or evidence.get("selector_protocol_sha256") != summary["selector_protocol_sha256"]
    or summary["protocol_sha256"] != summary["selector_protocol_sha256"]
):
    raise SystemExit("immutable evidence protocol identity mismatch")

selected_sha256 = None
for name in cells:
    request_path = root / "requests" / f"{name}.json"
    if request_path.is_symlink() or not request_path.is_file():
        raise SystemExit(f"immutable evidence request {name} is missing or not regular")
    rows = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 475:
        raise SystemExit(f"immutable evidence request {name} must contain 475 rows")
    digest = canonical_sha256(rows)
    expected = summary["request_set_sha256"][name]
    if not isinstance(expected, str) or sha256_re.fullmatch(expected) is None or digest != expected:
        raise SystemExit(f"immutable evidence request-set identity mismatch for {name}")
    if name == cell:
        selected_sha256 = digest
if selected_sha256 is None:
    raise SystemExit("selected answer request identity is missing")
print(selected_sha256)
PY
  )"
  chmod -R a-w "$INPUT_DIR"
}

prepare_requests() {
  "$PYTHON_BIN" - "$INPUT_DIR" "$OUTPUT_DIR/requests.json" "$REQUEST_NAME" \
    "$EXPERIMENT_RUN_ID" "$ANSWER_CELL" "$EXPECTED_CANONICAL_ID_DIGEST" \
    "$SOURCE_REQUEST_SHA256" <<'PY'
import hashlib
import json
import math
import re
import sys
from pathlib import Path

from trec_video_2026.vqa_experiments.silver_oracle import audit_label_free_request

root = Path(sys.argv[1]).resolve(strict=True)
output = Path(sys.argv[2])
request_path = root / sys.argv[3]
run_id, cell, expected_digest = sys.argv[4], sys.argv[5], sys.argv[6]
expected_request_sha256 = sys.argv[7]
video_id_re = re.compile(r"^[A-Za-z0-9_-]{11}$")

def canonical_sha256(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

payload = json.loads(request_path.read_text(encoding="utf-8"))
if not isinstance(payload, list) or len(payload) != 475:
    raise SystemExit("selected request count is not 475")
if canonical_sha256(payload) != expected_request_sha256:
    raise SystemExit("selected request no longer matches the Task 4 marker")
audit_label_free_request(payload, location=f"request[{cell}]")
allowed = {
    "run_id",
    "condition_id",
    "video_id",
    "question",
    "frame_paths",
    "frame_timestamps",
    "transcript",
}
rows, seen = [], set()
for index, row in enumerate(payload):
    if not isinstance(row, dict) or set(row) != allowed:
        raise SystemExit(f"request row {index} is not the exact label-free schema")
    if row["run_id"] != run_id or row["condition_id"] != cell:
        raise SystemExit(f"request row {index} has stale run/cell identity")
    video_id = row["video_id"]
    if not isinstance(video_id, str) or video_id in seen or video_id_re.fullmatch(video_id) is None:
        raise SystemExit(f"request row {index} has an invalid or duplicate video ID")
    seen.add(video_id)
    if not isinstance(row["question"], str) or not row["question"].strip():
        raise SystemExit(f"request row {index} has an empty question")
    paths, timestamps = row["frame_paths"], row["frame_timestamps"]
    if not isinstance(paths, list) or not isinstance(timestamps, list) or len(paths) != 12 or len(timestamps) != 12:
        raise SystemExit(f"request row {index} must contain twelve frames and timestamps")
    resolved = []
    for raw in paths:
        if not isinstance(raw, str):
            raise SystemExit(f"request row {index} frame path is not a string")
        marker = "/frames/"
        relative = Path(raw.split(marker, 1)[1] if marker in raw else raw.removeprefix("frames/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit(f"request row {index} contains an unsafe frame path")
        candidates = (root / "frames" / relative, root / "evidence" / "frames" / relative)
        frame = next((p.resolve(strict=True) for p in candidates if p.is_file()), None)
        if frame is None or not frame.is_relative_to(root) or frame.stat().st_size == 0:
            raise SystemExit(f"request row {index} frame path is outside immutable evidence")
        resolved.append(str(frame))
    if len(set(resolved)) != 12:
        raise SystemExit(f"request row {index} contains duplicate frames")
    for timestamp in timestamps:
        if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp) or timestamp < 0):
            raise SystemExit(f"request row {index} contains an invalid timestamp")
    transcript = row["transcript"]
    if transcript is not None and (not isinstance(transcript, str) or not transcript.strip()):
        raise SystemExit(f"request row {index} transcript is empty")
    if cell.endswith("n") and transcript is not None:
        raise SystemExit(f"no-transcript cell {cell} contains transcript bytes")
    rows.append({"run_id": run_id, "condition_id": cell, "video_id": video_id,
                 "question": row["question"], "frame_paths": resolved,
                 "frame_timestamps": [float(x) for x in timestamps], "transcript": transcript})
if hashlib.sha256(("\n".join(sorted(seen)) + "\n").encode()).hexdigest() != expected_digest:
    raise SystemExit("selected request ID digest does not match the frozen panel")
output.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  REQUEST_SHA256="$(sha256sum "$OUTPUT_DIR/requests.json" | awk '{print $1}')"
}

start_telemetry() {
  local child_pid="$1" telemetry_path="$2"
  mkdir -p "$(dirname "$telemetry_path")"
  printf '%s\n' 'timestamp,index,utilization.gpu [%],memory.used [MiB],memory.total [MiB]' > "$telemetry_path"
  (
    while kill -0 "$child_pid" 2>/dev/null; do
      sample="$(timeout 4s nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>&1 || true)"
      printf '%s,%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$sample" >> "$telemetry_path"
      [[ "${VQA_TEST_MODE:-0}" == "1" ]] || sleep 5
    done
  ) &
  TELEMETRY_PID=$!
}

kill_descendants() {
  local parent="$1" child
  while read -r child; do
    [[ -n "$child" ]] || continue
    kill_descendants "$child"
    kill "$child" 2>/dev/null || true
  done < <(ps -eo pid=,ppid= | awk -v parent="$parent" '$2 == parent {print $1}')
}
stop_telemetry() {
  if [[ -n "$TELEMETRY_PID" ]]; then
    kill_descendants "$TELEMETRY_PID"
    kill "$TELEMETRY_PID" 2>/dev/null || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
    TELEMETRY_PID=""
  fi
}
cleanup_runtime() {
  stop_telemetry
  if [[ -n "$CURRENT_CHILD_PID" ]]; then
    kill_descendants "$CURRENT_CHILD_PID"
    kill "$CURRENT_CHILD_PID" 2>/dev/null || true
    wait "$CURRENT_CHILD_PID" 2>/dev/null || true
    CURRENT_CHILD_PID=""
  fi
}

run_inference() {
  local started_ns finished_ns child_rc
  started_ns="$(date +%s%N)"
  "$INFERENCE_BIN" -m trec_video_2026.vqa_transformers_runner \
    --model-id "$MODEL_ID" --revision "$MODEL_REVISION" --model-class multimodal \
    --answer-policy "$ANSWER_POLICY" --min-pixels-per-frame "$MIN_PIXELS_PER_FRAME" \
    --max-pixels-per-frame "$MAX_PIXELS_PER_FRAME" --max-new-tokens "$MAX_NEW_TOKENS" \
    < "$OUTPUT_DIR/requests.json" > "$OUTPUT_DIR/predictions.json" 2> "$OUTPUT_DIR/inference.stderr" &
  CURRENT_CHILD_PID=$!
  start_telemetry "$CURRENT_CHILD_PID" "$OUTPUT_DIR/telemetry/nvidia-smi.csv"
  set +e
  wait "$CURRENT_CHILD_PID"
  child_rc=$?
  set -e
  CURRENT_CHILD_PID=""
  stop_telemetry
  finished_ns="$(date +%s%N)"
  (( child_rc == 0 )) || die "Qwen3.5 answer inference failed"
  "$PYTHON_BIN" - "$OUTPUT_DIR/requests.json" "$OUTPUT_DIR/predictions.json" "$RUNTIME_STRING" \
    "$VQA_RUNTIME_IMAGE" "$MODEL_ID" "$MODEL_REVISION" "$ANSWER_POLICY" "$MIN_PIXELS_PER_FRAME" \
    "$MAX_PIXELS_PER_FRAME" "$ANSWER_CELL" "$EXPERIMENT_RUN_ID" "$REQUEST_SHA256" "$TORCH_CUDA_VERSION" \
    "$EXPECTED_COUNT" "$started_ns" "$finished_ns" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path
requests = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
predictions = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if not isinstance(predictions, list) or len(predictions) != 475 or len(requests) != 475:
    raise SystemExit("prediction count is not exactly 475")
if [x.get("video_id") for x in predictions] != [x.get("video_id") for x in requests]:
    raise SystemExit("prediction IDs/order do not match selected requests")
fields = {"status","answer","elapsed_seconds","model_load_seconds","model_id","model_revision","runtime",
          "runtime_image","precision","device","runtime_attestation","peak_memory_bytes","error_type",
          "error_message","answer_policy","question_type","requested_min_pixels_per_frame",
          "requested_max_pixels_per_frame","min_pixels_per_frame","max_pixels_per_frame",
          "run_id","condition_id","video_id"}
for request, row in zip(requests, predictions, strict=True):
    if not isinstance(row, dict) or set(row) != fields:
        raise SystemExit("prediction schema is not the exact Transformers runner schema")
    if (row["run_id"] != sys.argv[11]
            or row["condition_id"] != sys.argv[10]
            or row["video_id"] != request["video_id"]):
        raise SystemExit("prediction request identity is stale")
    if (row["model_id"], row["model_revision"], row["answer_policy"]) != (sys.argv[5], sys.argv[6], sys.argv[7]):
        raise SystemExit("prediction model/prompt identity is stale")
    if row["runtime"] != sys.argv[3] or row["runtime_image"] != sys.argv[4] or row["precision"] != "bfloat16" or row["device"] != "cuda:0 (NVIDIA CUDA)":
        raise SystemExit("prediction runtime identity is not pinned")
    if (row["requested_min_pixels_per_frame"], row["requested_max_pixels_per_frame"],
        row["min_pixels_per_frame"], row["max_pixels_per_frame"]) != (int(sys.argv[8]), int(sys.argv[9])) * 2:
        raise SystemExit("prediction pixel bounds are not frozen")
    if row["status"] != "passed" or not isinstance(row["answer"], str) or not row["answer"].strip():
        raise SystemExit("prediction is incomplete or empty")
    for field in ("elapsed_seconds", "model_load_seconds"):
        value = row[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise SystemExit(f"invalid prediction {field}")
    if not isinstance(row["peak_memory_bytes"], int) or isinstance(row["peak_memory_bytes"], bool) or row["peak_memory_bytes"] < 0:
        raise SystemExit("invalid prediction peak memory")
    if row["error_type"] is not None or row["error_message"] is not None:
        raise SystemExit("passed prediction carries an error")
    attestation = row["runtime_attestation"]
    required = {"parameter_tensor_count","parameter_devices","parameter_dtypes","buffer_tensor_count",
                "buffer_devices","buffer_dtypes","accelerator_backend","torch_cuda","torch_hip"}
    if not isinstance(attestation, dict) or set(attestation) != required:
        raise SystemExit("runtime attestation schema is incomplete")
    if attestation["accelerator_backend"] != "nvidia-cuda" or attestation["torch_cuda"] != sys.argv[13] or attestation["torch_hip"] is not None:
        raise SystemExit("runtime attestation is not pinned NVIDIA CUDA")
    for label, count_key, devices_key, dtypes_key in (("parameter","parameter_tensor_count","parameter_devices","parameter_dtypes"),
                                                       ("buffer","buffer_tensor_count","buffer_devices","buffer_dtypes")):
        count, devices, dtypes = attestation[count_key], attestation[devices_key], attestation[dtypes_key]
        if not isinstance(count, int) or isinstance(count, bool) or count < (1 if label == "parameter" else 0):
            raise SystemExit(f"invalid {label} tensor count")
        if not isinstance(devices, dict) or not isinstance(dtypes, dict) or sum(devices.values()) != count or sum(dtypes.values()) != count:
            raise SystemExit(f"invalid {label} tensor attestation")
        if any(device != "cuda:0" for device in devices) or (label == "parameter" and any(dtype != "torch.bfloat16" for dtype in dtypes)):
            raise SystemExit(f"{label} tensors are not pinned")
timing = {
    "schema_version": 1, "answer_cell": sys.argv[10], "row_count": 475,
    "wall_seconds": (int(sys.argv[16]) - int(sys.argv[15])) / 1_000_000_000,
    "request_sha256": sys.argv[12], "prediction_sha256": hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest(),
    "runtime_identity": {"runtime": sys.argv[3], "runtime_image": sys.argv[4], "model_id": sys.argv[5],
                         "model_revision": sys.argv[6], "transformers_revision": "cbf4d720ec734edb77d25452b2790c7e4be2f8d7",
                         "answer_policy": sys.argv[7], "do_sample": False, "enable_thinking": False,
                         "max_new_tokens": 64, "min_pixels_per_frame": 65536, "max_pixels_per_frame": 262144},
}
Path(sys.argv[2]).with_name("timing.json").write_text(json.dumps(timing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

write_status() {
  "$PYTHON_BIN" - "$OUTPUT_DIR/condition-identity.json" "$OUTPUT_DIR/status.json" \
    "$OUTPUT_DIR/predictions.json" "$EVIDENCE_BUNDLE_SHA256" "$EVIDENCE_MARKER_SHA256" \
    "$SOURCE_REQUEST_SHA256" "$REQUEST_SHA256" "$RUNTIME_STRING" "$TORCH_CUDA_VERSION" \
    "$EXPERIMENT_RUN_ID" "$ANSWER_CELL" "$HF_INPUT_URI" "$HF_REMOTE_URI" \
    "$HF_OUTPUT_BUCKET_ID" \
    "$VQA_RUNTIME_IMAGE" "$MODEL_ID" "$MODEL_REVISION" "$ANSWER_POLICY" "$EXPECTED_COUNT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

prediction = Path(sys.argv[3])
prediction_sha = hashlib.sha256(prediction.read_bytes()).hexdigest()
identity = {
    "schema_version": 1, "run_id": sys.argv[10], "answer_cell": sys.argv[11],
    "model_id": sys.argv[16], "model_revision": sys.argv[17],
    "transformers_revision": "cbf4d720ec734edb77d25452b2790c7e4be2f8d7",
    "runtime_image": sys.argv[15], "runtime": sys.argv[8], "torch_cuda": sys.argv[9],
    "answer_policy": sys.argv[18],
    "deterministic_decoding": {"do_sample": False, "enable_thinking": False, "max_new_tokens": 64},
    "pixel_bounds": {"min": 65536, "max": 262144}, "expected_count": int(sys.argv[19]),
    "evidence_uri": sys.argv[12], "evidence_bundle_sha256": sys.argv[4],
    "evidence_checksums_sha256": sys.argv[4], "evidence_marker_sha256": sys.argv[5],
    "task4_request_set_sha256": sys.argv[6], "request_sha256": sys.argv[7],
    "prediction_sha256": prediction_sha, "labels_joined": False,
    "output_bucket_id": sys.argv[14], "output_uri": sys.argv[13],
}
Path(sys.argv[1]).write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
Path(sys.argv[2]).write_text(json.dumps({
    "schema_version": 1, "status": "passed", "run_id": sys.argv[10], "answer_cell": sys.argv[11],
    "expected_count": int(sys.argv[19]), "evidence_uri": sys.argv[12], "output_uri": sys.argv[13],
    "output_bucket_id": sys.argv[14],
    "evidence_bundle_sha256": sys.argv[4], "evidence_marker_sha256": sys.argv[5],
    "task4_request_set_sha256": sys.argv[6], "request_sha256": sys.argv[7],
    "prediction_sha256": prediction_sha, "labels_joined": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

write_completion() {
  "$PYTHON_BIN" - "$OUTPUT_DIR/completion.json" "$OUTPUT_DIR/predictions.json" \
    "$EVIDENCE_BUNDLE_SHA256" "$EVIDENCE_MARKER_SHA256" "$SOURCE_REQUEST_SHA256" \
    "$REQUEST_SHA256" "$EXPERIMENT_RUN_ID" "$ANSWER_CELL" "$HF_INPUT_URI" \
    "$HF_REMOTE_URI" "$HF_OUTPUT_BUCKET_ID" "$EXPECTED_COUNT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

prediction_sha = hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1, "status": "passed", "run_id": sys.argv[7],
    "answer_cell": sys.argv[8], "expected_count": int(sys.argv[12]),
    "evidence_uri": sys.argv[9], "output_uri": sys.argv[10],
    "output_bucket_id": sys.argv[11],
    "evidence_bundle_sha256": sys.argv[3], "evidence_marker_sha256": sys.argv[4],
    "task4_request_set_sha256": sys.argv[5], "request_sha256": sys.argv[6],
    "prediction_sha256": prediction_sha, "labels_joined": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

write_checksums() {
  (cd "$OUTPUT_DIR" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
  (cd "$OUTPUT_DIR" && sha256sum --check SHA256SUMS >/dev/null) || die "output SHA256SUMS verification failed"
}

write_terminal_status() {
  local status="$1" payload_checksum="$2" payload_uri="$3" completion_checksum="$4"
  "$PYTHON_BIN" - "$TERMINAL_STATUS_PATH" "$status" "$payload_checksum" "$payload_uri" \
    "$completion_checksum" "$EVIDENCE_BUNDLE_SHA256" "$EVIDENCE_MARKER_SHA256" \
    "$SOURCE_REQUEST_SHA256" "$REQUEST_SHA256" "$OUTPUT_DIR/predictions.json" \
    "$EXPERIMENT_RUN_ID" "$ANSWER_CELL" "$HF_OUTPUT_BUCKET_ID" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

prediction_sha = hashlib.sha256(Path(sys.argv[10]).read_bytes()).hexdigest()
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1, "status": sys.argv[2], "payload_sha256": sys.argv[3],
    "payload_uri": sys.argv[4], "completion_sha256": sys.argv[5],
    "evidence_bundle_sha256": sys.argv[6], "evidence_marker_sha256": sys.argv[7],
    "task4_request_set_sha256": sys.argv[8], "request_sha256": sys.argv[9],
    "prediction_sha256": prediction_sha, "run_id": sys.argv[11], "answer_cell": sys.argv[12],
    "output_bucket_id": sys.argv[13],
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

finalize_bundle() {
  local status="$1" rc="$2"
  (( FINALIZE_ATTEMPTED == 0 )) || return 0
  FINALIZE_ATTEMPTED=1
  cleanup_runtime
  if [[ "$status" != "passed" ]]; then
    mkdir -p "$OUTPUT_DIR"
    "$PYTHON_BIN" - "$OUTPUT_DIR/status.json" "$status" "$rc" "$ANSWER_CELL" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1, "status": sys.argv[2], "workload_returncode": int(sys.argv[3]),
    "answer_cell": sys.argv[4], "labels_joined": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    write_checksums
    event failure-finalized-locally
    return 0
  fi

  write_status
  write_completion
  write_checksums
  local payload_uri="${HF_REMOTE_URI%/}/payload"
  local payload_checksum completion_checksum
  payload_checksum="$(sha256sum "$OUTPUT_DIR/SHA256SUMS" | awk '{print $1}')"
  completion_checksum="$(sha256sum "$OUTPUT_DIR/completion.json" | awk '{print $1}')"
  write_terminal_status "$status" "$payload_checksum" "$payload_uri" "$completion_checksum"
  if [[ "${VQA_DRY_RUN:-0}" == "1" && "${VQA_SKIP_PUBLICATION:-0}" == "1" ]]; then return 0; fi
  acquire_output_bucket_lease
  hf_cli buckets sync "$OUTPUT_DIR" "$payload_uri" || die "answer payload publication failed"
  hf_cli buckets cp "$TERMINAL_STATUS_PATH" "${HF_REMOTE_URI%/}/terminal-status.json" || die "terminal marker publication failed"
}

on_exit() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  cleanup_runtime
  if (( EXECUTION_STARTED == 1 && FINALIZE_ATTEMPTED == 0 )); then finalize_bundle failed "$rc" >/dev/null 2>&1 || true; fi
  exit "$rc"
}
on_signal() { exit "$1"; }

main() {
  trap on_exit EXIT
  trap 'on_signal 130' INT
  trap 'on_signal 143' TERM
  validate_dispatch
  refuse_nonempty_output
  ensure_hf_cli
  output_bucket_is_absent
  EXECUTION_STARTED=1
  bootstrap_runtime
  load_runtime_provenance
  verify_input_bundle
  prepare_requests
  run_inference
  finalize_bundle passed 0
}
main "$@"
