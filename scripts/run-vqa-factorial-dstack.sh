#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_RUNTIME_IMAGE="docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
readonly TRANSFORMERS_REVISION="cbf4d720ec734edb77d25452b2790c7e4be2f8d7"
readonly TRANSFORMERS_SOURCE_URL="https://github.com/huggingface/transformers/archive/${TRANSFORMERS_REVISION}.tar.gz"
readonly HF_CLI_VERSION="1.26.0"
readonly PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
readonly BUILDER="${BUILDER:-/dstack/run/scripts/build_vqa_factorial_manifest.py}"
readonly PROFILE="${PROFILE:?PROFILE is required}"
readonly VQA_MODE="${VQA_MODE:?VQA_MODE is required}"
readonly INPUT_DIR="${INPUT_DIR:-/dstack/input}"
readonly FRAME_DIR="${FRAME_DIR:-/dstack/frames}"
readonly OUTPUT_DIR="${OUTPUT_DIR:-/dstack/output}"
readonly TERMINAL_STATUS_PATH="${OUTPUT_DIR}.terminal-status.json"
readonly INFERENCE_BIN="${INFERENCE_BIN:-$PYTHON_BIN}"
readonly CANARY_MANIFEST_SHA256="857d251870d24c4c5c7b43d94030a994bd4d8c39da4db1aa4385c1e01f7a7005"
readonly FULL_MANIFEST_SHA256="092aec23cfdcc370bbed8e776f7cc7c84752481da6ba0996983883955d3841e8"
readonly FULL_IDS_SHA256="95a623c0add252541b9c81ff4910e949dc81db17482add2c467b761237b22b0f"
readonly CANARY_IDS=(CtinYSH3nVw LKiaCf1QZFw IVqpg_1aWXU)

MANIFEST=""
PROFILE_MANIFEST=""
EXPECTED_COUNT=""
FRAME_COUNT=""
TELEMETRY_PID=""
CURRENT_CHILD_PID=""
HF_COMMAND=()
RUNTIME_STRING=""
TORCH_CUDA_VERSION=""
EXECUTION_STARTED=0
FINALIZE_ATTEMPTED=0
FINALIZED=0
PAYLOAD_PUBLISHED=0

event() {
  if [[ -n "${EVENT_LOG:-}" ]]; then
    printf '%s\n' "$1" >> "$EVENT_LOG"
  fi
}

die() {
  echo "error: $*" >&2
  exit 2
}

require_value() {
  local name="$1"
  [[ -n "${!name:-}" ]] || die "$name is required"
}

hf_cli() {
  ((${#HF_COMMAND[@]} > 0)) || die "Hugging Face client preflight did not complete"
  if [[ -n "${HF_TOKEN:-}" && -z "${HF_CLI_BIN:-}" ]]; then
    HF_TOKEN="$HF_TOKEN" "${HF_COMMAND[@]}" "$@"
  else
    "${HF_COMMAND[@]}" "$@"
  fi
}

refuse_nonempty_output() {
  if [[ -e "$OUTPUT_DIR" && ! -d "$OUTPUT_DIR" ]]; then
    die "OUTPUT_DIR exists but is not a directory: $OUTPUT_DIR"
  fi
  if [[ -d "$OUTPUT_DIR" ]] && find "$OUTPUT_DIR" -mindepth 1 -print -quit | grep -q .; then
    die "OUTPUT_DIR is non-empty: $OUTPUT_DIR"
  fi
  [[ ! -e "$TERMINAL_STATUS_PATH" ]] || die "terminal status path already exists: $TERMINAL_STATUS_PATH"
}

ensure_lightweight_hf_client() {
  event lightweight-client-start
  if [[ -n "${HF_CLI_BIN:-}" ]]; then
    command -v "$HF_CLI_BIN" >/dev/null 2>&1 || die "HF_CLI_BIN is not executable"
    HF_COMMAND=("$HF_CLI_BIN")
  elif command -v uvx >/dev/null 2>&1; then
    HF_COMMAND=(uvx --from "huggingface-hub==${HF_CLI_VERSION}" hf)
  elif command -v hf >/dev/null 2>&1; then
    HF_COMMAND=(hf)
  else
    "$PYTHON_BIN" -m pip install --no-cache-dir --only-binary=:all: "huggingface_hub==${HF_CLI_VERSION}"
    command -v hf >/dev/null 2>&1 || die "pinned Hugging Face CLI is unavailable after installation"
    HF_COMMAND=(hf)
  fi
  event lightweight-client-ready
}

json_field() {
  local json_path="$1"
  local field="$2"
  "$PYTHON_BIN" - "$json_path" "$field" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for part in sys.argv[2].split("."):
    value = value[part]
if isinstance(value, (dict, list)):
    print(json.dumps(value, sort_keys=True))
else:
    print(value)
PY
}

validate_dispatch() {
  [[ -f "$BUILDER" ]] || die "manifest/profile validator is missing: $BUILDER"
  [[ -f "$PROFILE" ]] || die "profile is missing: $PROFILE"
  [[ "$VQA_MODE" == "canary" || "$VQA_MODE" == "evidence" || "$VQA_MODE" == "condition" ]] \
    || die "unsupported VQA_MODE: $VQA_MODE"
  require_value EXPERIMENT_RUN_NAME
  require_value HF_INPUT_URI
  require_value HF_REMOTE_URI
  [[ "${VQA_RUNTIME_IMAGE:-}" == "$EXPECTED_RUNTIME_IMAGE" ]] \
    || die "runtime image is not the pinned digest"

  local profile_kind="$VQA_MODE"
  local expected_run="$EXPERIMENT_RUN_NAME"
  local expected_remote="$HF_REMOTE_URI"
  local condition_arg=()
  if [[ "$VQA_MODE" == "condition" ]]; then
    require_value CONDITION_ID
    local base_run base_remote
    base_run="$(json_field "$PROFILE" run_name)"
    base_remote="$(json_field "$PROFILE" remote_uri)"
    expected_run="${base_run}-${CONDITION_ID}"
    expected_remote="${base_remote%/}/${CONDITION_ID}"
    condition_arg=(--condition-id "$CONDITION_ID")
  fi
  "$PYTHON_BIN" "$BUILDER" --validate-profile "$PROFILE" --profile-kind "$profile_kind" \
    "${condition_arg[@]}" --expected-run-name "$expected_run" \
    --expected-input-uri "$HF_INPUT_URI" --expected-remote-uri "$expected_remote" \
    --expected-manifest "$(json_field "$PROFILE" manifest)" \
    >/dev/stderr
  [[ "$EXPERIMENT_RUN_NAME" == "$expected_run" ]] || die "run identity does not match profile"
  [[ "$HF_REMOTE_URI" == "$expected_remote" ]] || die "remote URI does not match profile"
  PROFILE_MANIFEST="$(json_field "$PROFILE" manifest)"
  MANIFEST="${MANIFEST_OVERRIDE:-$PROFILE_MANIFEST}"
  EXPECTED_COUNT="$(json_field "$PROFILE" expected_count)"
  FRAME_COUNT="$(json_field "$PROFILE" frame_count)"
  if [[ -z "${MANIFEST_OVERRIDE:-}" || "${VQA_TEST_MODE:-0}" != "1" ]]; then
    [[ "$MANIFEST" == /dstack/run/*.json && "$MANIFEST" != *".."* ]] \
      || die "validated manifest path is unsafe"
  else
    [[ "$MANIFEST" = /* && "$MANIFEST" != *".."* && -f "$MANIFEST" ]] \
      || die "manifest override path is unsafe"
  fi
  verify_manifest
  event dispatch-validated
}

remote_prefix_is_empty() {
  local listing
  if ! listing="$(hf_cli buckets list "$HF_REMOTE_URI" --format quiet)"; then
    die "could not inspect remote output prefix"
  fi
  [[ -z "$listing" ]] || die "remote output prefix already exists: $HF_REMOTE_URI"
}

bootstrap_runtime() {
  event heavy-runtime-bootstrap
  mkdir -p "$OUTPUT_DIR"
  if [[ "${VQA_DRY_RUN:-0}" == "1" || "${VQA_TEST_MODE:-0}" == "1" ]]; then
    "$PYTHON_BIN" - "$OUTPUT_DIR/runtime.json" "$EXPECTED_RUNTIME_IMAGE" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "runtime": "transformers 5.15.0.dev0; torch 2.8.0",
    "runtime_image": sys.argv[2],
    "torch_cuda": "12.4",
    "torch_hip": None,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    event heavy-runtime-bootstrap-skipped
    return
  fi
  "$PYTHON_BIN" -m pip install --no-cache-dir \
    "transformers @ ${TRANSFORMERS_SOURCE_URL}" \
    "qwen-vl-utils[decord]==0.0.14" \
    "av==16.0.1" \
    "accelerate==1.12.0" \
    "imageio-ffmpeg==0.6.0" \
    "Pillow==11.3.0" \
    "soundfile==0.13.1"
  "$PYTHON_BIN" - "$OUTPUT_DIR/runtime.json" <<'PY'
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

import PIL
import accelerate
import av
import imageio_ffmpeg
import qwen_vl_utils
import soundfile
import torch
import transformers

qwen_vl_utils_version = version("qwen-vl-utils")
if transformers.__version__ != "5.15.0.dev0":
    raise SystemExit(f"unexpected transformers version: {transformers.__version__}")
if qwen_vl_utils_version != "0.0.14":
    raise SystemExit(f"unexpected qwen-vl-utils version: {qwen_vl_utils_version}")
if not torch.cuda.is_available() or not torch.version.cuda or torch.version.hip:
    raise SystemExit("NVIDIA CUDA is unavailable")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("CUDA bfloat16 is unavailable")
if PIL.__version__ != "11.3.0" or soundfile.__version__ != "0.13.1":
    raise SystemExit("pinned Pillow or soundfile version is unavailable")
record = {
    "runtime": f"transformers {transformers.__version__}; torch {torch.__version__}",
    "runtime_image": os.environ.get("VQA_RUNTIME_IMAGE"),
    "transformers_version": transformers.__version__,
    "transformers_revision": "cbf4d720ec734edb77d25452b2790c7e4be2f8d7",
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torch_hip": torch.version.hip,
    "cuda_device_name": torch.cuda.get_device_name(0),
    "cuda_bf16_supported": True,
    "dependencies": {
        "Pillow": PIL.__version__,
        "soundfile": soundfile.__version__,
        "accelerate": accelerate.__version__,
        "av": av.__version__,
        "imageio-ffmpeg": imageio_ffmpeg.__version__,
        "qwen-vl-utils": qwen_vl_utils_version,
    },
}
Path(sys.argv[1]).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  event heavy-runtime-ready
}

load_runtime_provenance() {
  local values
  if ! values="$("$PYTHON_BIN" - "$OUTPUT_DIR/runtime.json" "$EXPECTED_RUNTIME_IMAGE" <<'PY'
import json
import sys
from pathlib import Path

record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
runtime = record.get("runtime")
runtime_image = record.get("runtime_image")
torch_cuda = record.get("torch_cuda")
if (
    not isinstance(runtime, str)
    or not runtime.strip()
    or "\n" in runtime
    or runtime_image != sys.argv[2]
    or not isinstance(torch_cuda, str)
    or not torch_cuda.strip()
    or "\n" in torch_cuda
    or record.get("torch_hip") is not None
):
    raise SystemExit("runtime.json does not contain the pinned NVIDIA runtime provenance")
print(runtime)
print(torch_cuda)
PY
)"; then
    die "runtime provenance is missing or invalid: $OUTPUT_DIR/runtime.json"
  fi
  RUNTIME_STRING="${values%%$'\n'*}"
  TORCH_CUDA_VERSION="${values#*$'\n'}"
  [[ -n "$RUNTIME_STRING" && -n "$TORCH_CUDA_VERSION" && "$RUNTIME_STRING" != "$TORCH_CUDA_VERSION" ]] \
    || die "runtime provenance values are incomplete"
  event runtime-provenance-bound
}

verify_manifest() {
  [[ -f "$MANIFEST" ]] || die "label-free manifest is missing: $MANIFEST"
  local args=(--validate-manifest "$MANIFEST" --expected-count "$EXPECTED_COUNT")
  if [[ "$VQA_MODE" == "canary" ]]; then
    args+=(--expected-sha256 "$CANARY_MANIFEST_SHA256")
    for id in "${CANARY_IDS[@]}"; do args+=(--expected-id "$id"); done
  else
    args+=(--expected-sha256 "$FULL_MANIFEST_SHA256" --expected-canonical-id-digest "$FULL_IDS_SHA256")
  fi
  "$PYTHON_BIN" "$BUILDER" "${args[@]}" >/dev/stderr \
    || die "manifest identity validation failed"
}

start_telemetry() {
  local child_pid="$1"
  local telemetry_path="$2"
  mkdir -p "$(dirname "$telemetry_path")"
  (
    if [[ ! -s "$telemetry_path" ]]; then
      printf '%s\n' 'timestamp,index,utilization.gpu [%],memory.used [MiB],memory.total [MiB]' \
        >> "$telemetry_path"
    fi
    while kill -0 "$child_pid" 2>/dev/null; do
      sample="$(timeout 4s nvidia-smi \
        --query-gpu=index,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits 2>&1 || true)"
      printf '%s,%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$sample" >> "$telemetry_path"
      sleep 5
    done
  ) &
  TELEMETRY_PID=$!
}

stop_telemetry() {
  if [[ -n "$TELEMETRY_PID" ]]; then
    kill_descendants "$TELEMETRY_PID"
    kill "$TELEMETRY_PID" 2>/dev/null || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
    TELEMETRY_PID=""
  fi
}

kill_descendants() {
  local parent="$1"
  local child
  while read -r child; do
    [[ -n "$child" ]] || continue
    kill_descendants "$child"
    kill "$child" 2>/dev/null || true
  done < <(ps -eo pid=,ppid= | awk -v parent="$parent" '$2 == parent {print $1}')
}

cleanup_runtime() {
  stop_telemetry
  if [[ -n "$CURRENT_CHILD_PID" ]]; then
    kill "$CURRENT_CHILD_PID" 2>/dev/null || true
    wait "$CURRENT_CHILD_PID" 2>/dev/null || true
    CURRENT_CHILD_PID=""
  fi
}

verify_475_media_bundle() {
  "$PYTHON_BIN" - "$INPUT_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath

root = Path(sys.argv[1]).resolve(strict=True)
contract = json.loads((root / "contract.json").read_text(encoding="utf-8"))
marker = json.loads((root / "INPUTS_COMPLETE.json").read_text(encoding="utf-8"))
objects = contract.get("objects")
if contract.get("schema_version") != 1 or contract.get("media_count") != 475:
    raise SystemExit("immutable input contract is not the 475-media bundle")
if marker.get("schema_version") != 1 or marker.get("input_count") != 475:
    raise SystemExit("immutable input marker is not the 475-media bundle")
if marker.get("object_count") != 477 or marker.get("media_count") != 475:
    raise SystemExit("immutable input marker object count mismatch")
if not isinstance(objects, list) or len(objects) != 475:
    raise SystemExit("immutable input contract must contain 475 media objects")
expected = {"contract.json", "INPUTS_COMPLETE.json"}
for record in objects:
    relative_raw = record.get("path")
    if not isinstance(relative_raw, str):
        raise SystemExit("media path is not a string")
    relative = PurePosixPath(relative_raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise SystemExit(f"unsafe media path: {relative}")
    path = root.joinpath(*relative.parts).resolve(strict=True)
    if not path.is_relative_to(root) or path.suffix.lower() not in {".mkv", ".mp4", ".webm", ".mov"}:
        raise SystemExit(f"media path escapes input root: {relative}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record.get("sha256") or path.stat().st_size != record.get("size_bytes"):
        raise SystemExit(f"media hash/size mismatch: {relative}")
    expected.add(relative.as_posix())
actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
if actual != expected:
    raise SystemExit("immutable input object set mismatch")
if hashlib.sha256((root / "contract.json").read_bytes()).hexdigest() != marker.get("contract_sha256"):
    raise SystemExit("immutable input contract checksum mismatch")
PY
}

prepare_evidence() {
  mkdir -p "$OUTPUT_DIR/evidence/frames" "$OUTPUT_DIR/evidence/audio" \
    "$OUTPUT_DIR/evidence/contact-sheets" "$OUTPUT_DIR/telemetry"
  hf_cli buckets sync "$HF_INPUT_URI" "$INPUT_DIR"
  [[ -f "$INPUT_DIR/contract.json" && -f "$INPUT_DIR/INPUTS_COMPLETE.json" ]] \
    || die "immutable input bundle markers are missing"
  verify_475_media_bundle
  "$PYTHON_BIN" /dstack/run/scripts/qwen35_prepare_matched_evidence.py \
    --manifest "$MANIFEST" --input-root "$INPUT_DIR" \
    --frame-root "$OUTPUT_DIR/evidence/frames" \
    --audio-root "$OUTPUT_DIR/evidence/audio" \
    --contact-sheet-root "$OUTPUT_DIR/evidence/contact-sheets" \
    --requests-output "$OUTPUT_DIR/evidence/requests.frames.json" \
    --evidence-output "$OUTPUT_DIR/evidence/evidence.json" \
    --run-id "$EXPERIMENT_RUN_NAME" --count "$EXPECTED_COUNT" \
    --frame-count "$FRAME_COUNT"
  local whisper_rc
  set +e
  "$PYTHON_BIN" /dstack/run/scripts/qwen35_whisper_cuda.py \
    --evidence "$OUTPUT_DIR/evidence/evidence.json" \
    --output "$OUTPUT_DIR/evidence/whisper.json" \
    --summary "$OUTPUT_DIR/evidence/whisper-summary.json" \
    --count "$EXPECTED_COUNT" --audio-root "$OUTPUT_DIR/evidence/audio" &
  CURRENT_CHILD_PID=$!
  start_telemetry "$CURRENT_CHILD_PID" "$OUTPUT_DIR/telemetry/nvidia-smi.csv"
  wait "$CURRENT_CHILD_PID"
  whisper_rc=$?
  CURRENT_CHILD_PID=""
  stop_telemetry
  set -e
  (( whisper_rc == 0 )) || die "Whisper evidence preparation failed"
  "$PYTHON_BIN" /dstack/run/scripts/qwen35_build_whisper_requests.py \
    --frames "$OUTPUT_DIR/evidence/requests.frames.json" \
    --whisper "$OUTPUT_DIR/evidence/whisper.json" \
    --output "$OUTPUT_DIR/evidence/requests.accepted-whisper.json" \
    --condition-id shared-accepted-whisper --count "$EXPECTED_COUNT" \
    --frame-root "$OUTPUT_DIR/evidence/frames"
  cp "$MANIFEST" "$OUTPUT_DIR/evidence/manifest.json"
}

write_evidence_marker() {
  "$PYTHON_BIN" - "$OUTPUT_DIR/evidence/evidence-marker.json" \
    "$OUTPUT_DIR/evidence/manifest.json" "$OUTPUT_DIR/evidence/requests.accepted-whisper.json" \
    "$OUTPUT_DIR/evidence/whisper.json" "$EXPECTED_COUNT" "$EXPERIMENT_RUN_NAME" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

marker_path = Path(sys.argv[1])
def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
payload = {
    "schema_version": 1,
    "label_policy": "label-free-evidence",
    "evidence_bundle_id": sys.argv[6],
    "example_count": int(sys.argv[5]),
    "manifest_sha256": digest(Path(sys.argv[2])),
    "requests_sha256": digest(Path(sys.argv[3])),
    "whisper_sha256": digest(Path(sys.argv[4])),
    "canonical_id_digest": (
        "95a623c0add252541b9c81ff4910e949dc81db17482add2c467b761237b22b0f"
        if int(sys.argv[5]) == 475 else None
    ),
    "checksum_file": "SHA256SUMS",
    "frame_policy": "12 uniform temporal midpoint requests, actual PTS recorded",
    "whisper_policy": "Whisper large-v3 accepted translations only",
}
marker_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

prepare_dry_run_evidence() {
  if [[ -n "${MANIFEST_OVERRIDE:-}" ]]; then
    MANIFEST="$MANIFEST_OVERRIDE"
  fi
  verify_manifest
  mkdir -p "$OUTPUT_DIR/evidence/frames" "$OUTPUT_DIR/evidence/audio" \
    "$OUTPUT_DIR/evidence/contact-sheets"
  "$PYTHON_BIN" - "$MANIFEST" "$OUTPUT_DIR/evidence" "$EXPERIMENT_RUN_NAME" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(sys.argv[2])
run_id = sys.argv[3]
requests = []
evidence = []
whisper_rows = []
for example in manifest["examples"]:
    video_id = example["video_id"]
    frame_paths = [
        str(root / "frames" / video_id / f"frame-{index:02d}.jpg")
        for index in range(1, 13)
    ]
    requests.append({
        "run_id": run_id,
        "condition_id": "shared-accepted-whisper",
        "video_id": video_id,
        "question": example["question"],
        "frame_paths": frame_paths,
        "frame_timestamps": [float(index) for index in range(12)],
        "transcript": f"transcript for {video_id}",
    })
    evidence.append({
        "video_id": video_id,
        "question": example["question"],
        "frame_count": 12,
        "frames": [
            {"path": f"frame-{index:02d}.jpg", "timestamp_seconds": float(index), "sha256": None}
            for index in range(1, 13)
        ],
        "audio": {"status": "extracted", "sha256": None},
    })
    whisper_rows.append({
        "status": "passed",
        "video_id": video_id,
        "transcript_included": True,
        "selected_transcript": f"transcript for {video_id}",
    })
(root / "requests.accepted-whisper.json").write_text(json.dumps(requests, indent=2) + "\n", encoding="utf-8")
(root / "evidence.json").write_text(json.dumps({"label_policy": "label-free-evidence", "examples": evidence}, indent=2) + "\n", encoding="utf-8")
(root / "whisper.json").write_text(json.dumps({"label_policy": "label-free-transcript-evidence", "results": whisper_rows}, indent=2) + "\n", encoding="utf-8")
Path(root / "whisper-summary.json").write_text(json.dumps({"example_count": len(requests), "accepted_count": len(requests)}) + "\n", encoding="utf-8")
PY
  cp "$MANIFEST" "$OUTPUT_DIR/evidence/manifest.json"
}

prepare_condition_evidence() {
  mkdir -p "$INPUT_DIR"
  hf_cli buckets sync "${HF_INPUT_URI%/}/payload" "$INPUT_DIR"
  [[ -f "$INPUT_DIR/evidence/evidence-marker.json" && -f "$INPUT_DIR/SHA256SUMS" ]] \
    || die "immutable evidence marker/checksum is missing"
  (cd "$INPUT_DIR" && sha256sum --check SHA256SUMS >/dev/null) \
    || die "immutable evidence checksum verification failed"
  "$PYTHON_BIN" - "$INPUT_DIR" "$MANIFEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
marker = json.loads((root / "evidence" / "evidence-marker.json").read_text(encoding="utf-8"))
if marker.get("schema_version") != 1 or marker.get("label_policy") != "label-free-evidence":
    raise SystemExit("evidence marker identity mismatch")
if marker.get("example_count") != 475 or marker.get("checksum_file") != "SHA256SUMS":
    raise SystemExit("evidence marker count/checksum mismatch")
if marker.get("manifest_sha256") != "092aec23cfdcc370bbed8e776f7cc7c84752481da6ba0996983883955d3841e8":
    raise SystemExit("evidence marker does not carry the pinned full manifest hash")
if marker.get("canonical_id_digest") != "95a623c0add252541b9c81ff4910e949dc81db17482add2c467b761237b22b0f":
    raise SystemExit("evidence marker does not carry the pinned canonical ID digest")
if marker.get("manifest_sha256") != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
    raise SystemExit("evidence manifest identity mismatch")
if len(manifest.get("examples", [])) != 475:
    raise SystemExit("condition manifest is not the canonical 475-row panel")
PY
  mkdir -p "$OUTPUT_DIR/evidence"
  cp "$INPUT_DIR/SHA256SUMS" "$OUTPUT_DIR/evidence/SHA256SUMS"
  cp -a "$INPUT_DIR/evidence/." "$OUTPUT_DIR/evidence/"
}

write_condition_requests() {
  local source="$1"
  local output="$2"
  local condition_id="$3"
  local request_source="$4"
  local evidence_root="$5"
  local condition_run_id="$6"
  "$PYTHON_BIN" - "$source" "$output" "$condition_id" "$request_source" "$evidence_root" "$condition_run_id" "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path, PurePosixPath

source = Path(sys.argv[1])
output = Path(sys.argv[2])
condition_id = sys.argv[3]
request_source = sys.argv[4]
evidence_root = Path(sys.argv[5])
condition_run_id = sys.argv[6]
manifest_path = Path(sys.argv[7])
rows = json.loads(source.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if not isinstance(rows, list):
    raise SystemExit("shared request file is not a list")
expected_ids = [row["video_id"] for row in manifest["examples"]]
if [row.get("video_id") for row in rows] != expected_ids:
    raise SystemExit("shared request IDs/order do not match the canonical manifest")
selected = []
allowed = {
    "run_id", "condition_id", "video_id", "question", "frame_paths",
    "frame_timestamps", "transcript",
}
for row in rows:
    if not isinstance(row, dict) or set(row) != allowed:
        raise SystemExit("shared request contains an unexpected or label-bearing field")
    if request_source not in {"shared-accepted-whisper", "shared-accepted-whisper-ocr-only"}:
        raise SystemExit("request_source is outside the closed vocabulary")
    if request_source == "shared-accepted-whisper-ocr-only" and row.get("video_id") != "IVqpg_1aWXU":
        continue
    rewritten = dict(row)
    rewritten["run_id"] = condition_run_id
    rewritten["condition_id"] = condition_id
    paths = []
    for frame_path in row.get("frame_paths", []):
        text = str(frame_path)
        marker = "/frames/"
        if marker not in text:
            raise SystemExit("frame path does not identify the shared evidence frame root")
        relative = PurePosixPath(text.split(marker, 1)[1])
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit("frame path contains an unsafe component")
        candidate = (evidence_root / "frames" / Path(*relative.parts)).resolve()
        if not candidate.is_relative_to((evidence_root / "frames").resolve()):
            raise SystemExit("frame path escapes the evidence root")
        paths.append(str(candidate))
    rewritten["frame_paths"] = paths
    selected.append(rewritten)
expected_count = 1 if request_source == "shared-accepted-whisper-ocr-only" else len(expected_ids)
if len(selected) != expected_count:
    raise SystemExit("condition request count does not match its immutable source binding")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

condition_info() {
  local condition_id="$1"
  local args=(--validate-profile "$PROFILE" --profile-kind "$VQA_MODE" --condition-id "$condition_id")
  if [[ "$VQA_MODE" == "canary" ]]; then
    args+=(--expected-run-name "$EXPERIMENT_RUN_NAME" --expected-input-uri "$HF_INPUT_URI" \
      --expected-remote-uri "$HF_REMOTE_URI" --expected-manifest "$PROFILE_MANIFEST")
  else
    local base_run base_remote
    base_run="$(json_field "$PROFILE" run_name)"
    base_remote="$(json_field "$PROFILE" remote_uri)"
    args+=(--expected-run-name "${base_run}-${condition_id}" \
      --expected-input-uri "$HF_INPUT_URI" --expected-remote-uri "${base_remote%/}/${condition_id}" \
      --expected-manifest "$PROFILE_MANIFEST")
  fi
  "$PYTHON_BIN" "$BUILDER" "${args[@]}"
}

run_condition() {
  local condition_id="$1"
  local profile_info condition_dir model_id revision answer_policy min_pixels max_pixels request_source
  profile_info="$(condition_info "$condition_id")"
  model_id="$(printf '%s' "$profile_info" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["condition"]["model_id"])')"
  revision="$(printf '%s' "$profile_info" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["condition"]["revision"])')"
  answer_policy="$(printf '%s' "$profile_info" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["condition"]["answer_policy"])')"
  min_pixels="$(printf '%s' "$profile_info" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["condition"]["min_pixels"])')"
  max_pixels="$(printf '%s' "$profile_info" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["condition"]["max_pixels"])')"
  request_source="$(printf '%s' "$profile_info" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["condition"]["request_source"])')"
  condition_dir="$OUTPUT_DIR/conditions/$condition_id"
  [[ ! -e "$condition_dir" ]] || die "condition output identity would overwrite an existing result: $condition_id"
  mkdir -p "$condition_dir/telemetry"
  if [[ "$VQA_MODE" == "canary" ]]; then
    write_condition_requests "$OUTPUT_DIR/evidence/requests.accepted-whisper.json" \
      "$condition_dir/requests.json" "$condition_id" "$request_source" "$OUTPUT_DIR/evidence" "$EXPERIMENT_RUN_NAME"
  else
    write_condition_requests "$INPUT_DIR/evidence/requests.accepted-whisper.json" \
      "$condition_dir/requests.json" "$condition_id" "$request_source" "$INPUT_DIR/evidence" "$EXPERIMENT_RUN_NAME"
  fi
  "$PYTHON_BIN" - "$condition_dir/condition-identity.json" "$profile_info" "$EXPECTED_RUNTIME_IMAGE" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(sys.argv[2])
payload["runtime_image"] = sys.argv[3]
payload["model_class"] = "multimodal"
payload["deterministic_decoding"] = {
    "do_sample": False,
    "enable_thinking": False,
    "max_new_tokens": 64,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  local started_ns finished_ns child_rc
  started_ns="$(date +%s%N)"
  set +e
  "$INFERENCE_BIN" -m trec_video_2026.vqa_transformers_runner \
    --model-id "$model_id" --revision "$revision" --model-class multimodal \
    --answer-policy "$answer_policy" --min-pixels-per-frame "$min_pixels" \
    --max-pixels-per-frame "$max_pixels" --max-new-tokens 64 \
    < "$condition_dir/requests.json" > "$condition_dir/predictions.json" \
    2> "$condition_dir/inference.stderr" &
  CURRENT_CHILD_PID=$!
  start_telemetry "$CURRENT_CHILD_PID" "$condition_dir/telemetry/nvidia-smi.csv"
  wait "$CURRENT_CHILD_PID"
  child_rc=$?
  CURRENT_CHILD_PID=""
  stop_telemetry
  set -e
  finished_ns="$(date +%s%N)"
  (( child_rc == 0 )) || die "condition process failed: $condition_id"
  "$PYTHON_BIN" - "$condition_dir/requests.json" "$condition_dir/predictions.json" \
    "$condition_dir/timing.json" "$started_ns" "$finished_ns" "$condition_id" \
    "$model_id" "$revision" "$answer_policy" "$min_pixels" "$max_pixels" "$EXPERIMENT_RUN_NAME" \
    "$EXPECTED_RUNTIME_IMAGE" "$request_source" "$EXPECTED_COUNT" \
    "$RUNTIME_STRING" "$TORCH_CUDA_VERSION" <<'PY'
import json
import math
import sys
from pathlib import Path

requests = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
predictions = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if not isinstance(predictions, list) or len(predictions) != len(requests):
    raise SystemExit("prediction count does not match selected requests")
expected_request_count = 1 if sys.argv[14] == "shared-accepted-whisper-ocr-only" else int(sys.argv[15])
if len(requests) != expected_request_count or len(predictions) != expected_request_count:
    raise SystemExit("prediction count does not match the immutable condition source")
if [row.get("video_id") for row in predictions] != [row.get("video_id") for row in requests]:
    raise SystemExit("prediction IDs/order do not match selected requests")
valid_question_types = {"count", "visible-text", "speech", "temporal", "causal-procedural", "general"}
producer_fields = {
    "status", "answer", "elapsed_seconds", "model_load_seconds", "model_id", "model_revision", "runtime",
    "runtime_image", "precision", "device", "runtime_attestation", "peak_memory_bytes",
    "error_type", "error_message", "answer_policy", "question_type",
    "requested_min_pixels_per_frame", "requested_max_pixels_per_frame",
    "min_pixels_per_frame", "max_pixels_per_frame", "run_id", "condition_id", "video_id",
}
for request, row in zip(requests, predictions, strict=True):
    if not isinstance(row, dict) or set(row) != producer_fields:
        raise SystemExit("prediction schema is not the exact producer schema")
    if row["run_id"] != sys.argv[12] or row["condition_id"] != sys.argv[6] or row["video_id"] != request["video_id"]:
        raise SystemExit("prediction request identity does not match the current condition run")
    if row["model_id"] != sys.argv[7] or row["model_revision"] != sys.argv[8] or row["answer_policy"] != sys.argv[9]:
        raise SystemExit("prediction model/revision/policy identity is stale or mutated")
    expected_bounds = (int(sys.argv[10]), int(sys.argv[11]))
    if (row["requested_min_pixels_per_frame"], row["requested_max_pixels_per_frame"], row["min_pixels_per_frame"], row["max_pixels_per_frame"]) != expected_bounds + expected_bounds:
        raise SystemExit("prediction pixel-bound controls do not match the condition")
    if row["runtime_image"] != sys.argv[13] or row["precision"] != "bfloat16" or row["device"] != "cuda:0 (NVIDIA CUDA)":
        raise SystemExit("prediction runtime identity is not the pinned NVIDIA BF16 runtime")
    if row["runtime"] != sys.argv[16]:
        raise SystemExit("prediction runtime does not match runtime.json provenance")
    elapsed = row["elapsed_seconds"]
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
        raise SystemExit("prediction elapsed_seconds is not finite and nonnegative")
    model_load = row["model_load_seconds"]
    if isinstance(model_load, bool) or not isinstance(model_load, (int, float)) or not math.isfinite(model_load) or model_load < 0:
        raise SystemExit("prediction model_load_seconds is not finite and nonnegative")
    peak_memory = row["peak_memory_bytes"]
    if isinstance(peak_memory, bool) or not isinstance(peak_memory, int) or peak_memory < 0:
        raise SystemExit("prediction peak_memory_bytes is not a nonnegative integer")
    if row["error_type"] is not None or row["error_message"] is not None:
        raise SystemExit("passed prediction carries an error")
    attestation = row["runtime_attestation"]
    attestation_fields = {
        "parameter_tensor_count", "parameter_devices", "parameter_dtypes",
        "buffer_tensor_count", "buffer_devices", "buffer_dtypes",
        "accelerator_backend", "torch_cuda", "torch_hip",
    }
    if not isinstance(attestation, dict) or set(attestation) != attestation_fields:
        raise SystemExit("prediction attestation schema is incomplete or has extra fields")
    parameter_count = attestation["parameter_tensor_count"]
    buffer_count = attestation["buffer_tensor_count"]
    if isinstance(parameter_count, bool) or not isinstance(parameter_count, int) or parameter_count <= 0:
        raise SystemExit("prediction parameter attestation count is invalid")
    if isinstance(buffer_count, bool) or not isinstance(buffer_count, int) or buffer_count < 0:
        raise SystemExit("prediction buffer attestation count is invalid")
    for label, count, devices, dtypes in (
        ("parameter", parameter_count, attestation["parameter_devices"], attestation["parameter_dtypes"]),
        ("buffer", buffer_count, attestation["buffer_devices"], attestation["buffer_dtypes"]),
    ):
        if not isinstance(devices, dict) or not isinstance(dtypes, dict):
            raise SystemExit(f"prediction {label} attestation maps are invalid")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (*devices.values(), *dtypes.values())):
            raise SystemExit(f"prediction {label} attestation map count is invalid")
        if sum(devices.values()) != count or sum(dtypes.values()) != count:
            raise SystemExit(f"prediction {label} attestation counts do not sum")
        if any(device != "cuda:0" for device in devices):
            raise SystemExit(f"prediction {label} tensors are not all on cuda:0")
        if label == "parameter" and any(dtype != "torch.bfloat16" for dtype in dtypes):
            raise SystemExit("prediction parameter tensors are not all bfloat16")
        if count and (not devices or not dtypes):
            raise SystemExit(f"prediction {label} attestation is empty")
        if not count and (devices or dtypes):
            raise SystemExit(f"prediction {label} zero-count attestation is not empty")
    if attestation["accelerator_backend"] != "nvidia-cuda" or attestation["torch_cuda"] != sys.argv[17] or attestation["torch_hip"] is not None:
        raise SystemExit("prediction accelerator attestation is not NVIDIA CUDA")
    if row["status"] != "passed" or not isinstance(row["answer"], str) or not row["answer"].strip():
        raise SystemExit("condition has an incomplete or empty prediction")
    if row["question_type"] not in valid_question_types:
        raise SystemExit("prediction question_type is invalid")
model_load_seconds = {row["model_load_seconds"] for row in predictions}
if len(model_load_seconds) != 1:
    raise SystemExit("model_load_seconds is not identical within the condition")
peak = [row["peak_memory_bytes"] for row in predictions]
payload = {
    "condition_id": sys.argv[6],
    "row_count": len(predictions),
    "wall_seconds": (int(sys.argv[5]) - int(sys.argv[4])) / 1_000_000_000,
    "peak_memory_bytes": max(peak) if peak else None,
    "model_load_seconds": next(iter(model_load_seconds)),
    "runtime_identity": sorted({
        (row.get("runtime"), row.get("runtime_image"), row.get("device"), row.get("precision"))
        for row in predictions
    }),
}
Path(sys.argv[3]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

write_final_status() {
  local status="$1"
  local workload_rc="$2"
  local sync_state="$3"
  "$PYTHON_BIN" - "$OUTPUT_DIR/status.json" "$status" "$VQA_MODE" \
    "$EXPERIMENT_RUN_NAME" "$EXPECTED_COUNT" "$FRAME_COUNT" "$workload_rc" "$sync_state" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "schema_version": 1,
    "status": sys.argv[2],
    "mode": sys.argv[3],
    "run_name": sys.argv[4],
    "expected_count": int(sys.argv[5]),
    "frame_count": int(sys.argv[6]),
    "workload_returncode": int(sys.argv[7]),
    "remote_payload_sync": sys.argv[8],
    "labels_joined": False,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

write_checksums() {
  (cd "$OUTPUT_DIR" && find . -type f ! -name SHA256SUMS ! -name terminal-status.json -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
  (cd "$OUTPUT_DIR" && sha256sum --check SHA256SUMS >/dev/null) \
    || die "local SHA256SUMS verification failed"
}

write_terminal_status() {
  local status="$1"
  local payload_checksum="$2"
  local payload_uri="$3"
  "$PYTHON_BIN" - "$TERMINAL_STATUS_PATH" "$status" "$payload_checksum" "$payload_uri" "$EXPERIMENT_RUN_NAME" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "schema_version": 1,
    "status": sys.argv[2],
    "payload_sha256": sys.argv[3],
    "payload_uri": sys.argv[4],
    "run_name": sys.argv[5],
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

finalize_bundle() {
  local requested_status="$1"
  local workload_rc="$2"
  (( FINALIZE_ATTEMPTED == 0 )) || return 0
  FINALIZE_ATTEMPTED=1
  cleanup_runtime
  write_final_status "$requested_status" "$workload_rc" frozen
  write_checksums
  if [[ "${VQA_DRY_RUN:-0}" == "1" && "${VQA_SKIP_PUBLICATION:-0}" == "1" ]]; then
    FINALIZED=1
    return 0
  fi
  local payload_uri="${HF_REMOTE_URI%/}/payload"
  local payload_checksum
  payload_checksum="$(sha256sum "$OUTPUT_DIR/SHA256SUMS" | awk '{print $1}')"
  local sync_rc=0
  event payload-sync-start
  set +e
  hf_cli buckets sync "$OUTPUT_DIR" "$payload_uri"
  sync_rc=$?
  set -e
  if (( sync_rc != 0 )); then
    write_final_status failed "$workload_rc" failed
    write_checksums
    event payload-sync-failed
    return 1
  fi
  PAYLOAD_PUBLISHED=1
  write_terminal_status "$requested_status" "$payload_checksum" "$payload_uri"
  event terminal-marker-start
  set +e
  hf_cli buckets cp "$TERMINAL_STATUS_PATH" "${HF_REMOTE_URI%/}/terminal-status.json"
  local marker_rc=$?
  set -e
  if (( marker_rc != 0 )); then
    event terminal-marker-failed
    return 1
  fi
  event terminal-marker-published
  FINALIZED=1
}

run_canary() {
  if [[ "${VQA_DRY_RUN:-0}" == "1" ]]; then
    prepare_dry_run_evidence
  else
    verify_manifest
    prepare_evidence
  fi
  write_evidence_marker
  local condition_id
  mapfile -t condition_ids < <("$PYTHON_BIN" - "$PROFILE" <<'PY'
import json
import sys
from pathlib import Path

for condition in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["conditions"]:
    print(condition["condition_id"])
PY
  )
  for condition_id in "${condition_ids[@]}"; do
    run_condition "$condition_id"
  done
  "$PYTHON_BIN" - "$OUTPUT_DIR/conditions/4b-current/predictions.json" \
    "$OUTPUT_DIR/conditions/4b-current-repeat/predictions.json" <<'PY'
import json
import sys
from pathlib import Path

first = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
repeat = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if [row.get("answer") for row in first] != [row.get("answer") for row in repeat]:
    raise SystemExit("4b-current repeat is not deterministic")
PY
}

run_evidence() {
  verify_manifest
  prepare_evidence
  write_evidence_marker
}

run_full_condition() {
  verify_manifest
  if [[ "${VQA_DRY_RUN:-0}" == "1" ]]; then
    die "full condition dry-run is not permitted because it would bypass immutable evidence"
  fi
  prepare_condition_evidence
  run_condition "$CONDITION_ID"
}

on_exit() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  cleanup_runtime
  if (( EXECUTION_STARTED == 1 && FINALIZE_ATTEMPTED == 0 )); then
    finalize_bundle failed "$rc" >/dev/null 2>&1 || true
  fi
  exit "$rc"
}

on_signal() {
  local signal_rc="$1"
  event "signal-${signal_rc}"
  exit "$signal_rc"
}

main() {
  trap on_exit EXIT
  trap 'on_signal 130' INT
  trap 'on_signal 143' TERM
  validate_dispatch
  refuse_nonempty_output
  ensure_lightweight_hf_client
  remote_prefix_is_empty
  EXECUTION_STARTED=1
  bootstrap_runtime
  load_runtime_provenance
  case "$VQA_MODE" in
    canary) run_canary ;;
    evidence) run_evidence ;;
    condition) run_full_condition ;;
  esac
  finalize_bundle passed 0 || die "durable output bundle publication failed"
}

main "$@"
