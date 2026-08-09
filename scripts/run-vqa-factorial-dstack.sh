#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_RUNTIME_IMAGE="docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
readonly TRANSFORMERS_REVISION="cbf4d720ec734edb77d25452b2790c7e4be2f8d7"
readonly TRANSFORMERS_SOURCE_URL="https://github.com/huggingface/transformers/archive/${TRANSFORMERS_REVISION}.tar.gz"
readonly UV_VERSION="0.11.3"
readonly HF_CLI_VERSION="1.26.0"
readonly PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
readonly BUILDER="${BUILDER:-/dstack/run/scripts/build_vqa_factorial_manifest.py}"
readonly PROFILE="${PROFILE:?PROFILE is required}"
readonly VQA_MODE="${VQA_MODE:?VQA_MODE is required}"
readonly INPUT_DIR="${INPUT_DIR:-/dstack/input}"
readonly FRAME_DIR="${FRAME_DIR:-/dstack/frames}"
readonly OUTPUT_DIR="${OUTPUT_DIR:-/dstack/output}"
readonly INFERENCE_BIN="${INFERENCE_BIN:-$PYTHON_BIN}"

MANIFEST=""
PROFILE_MANIFEST=""
EXPECTED_COUNT=""
FRAME_COUNT=""
TELEMETRY_PID=""
CURRENT_CHILD_PID=""

die() {
  echo "error: $*" >&2
  exit 2
}

require_value() {
  local name="$1"
  [[ -n "${!name:-}" ]] || die "$name is required"
}

hf_cli() {
  if [[ -n "${HF_CLI_BIN:-}" ]]; then
    "$HF_CLI_BIN" "$@"
  else
    HF_TOKEN="${HF_TOKEN:?HF_TOKEN is required}" \
      uvx --from "huggingface-hub==${HF_CLI_VERSION}" hf "$@"
  fi
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
  MANIFEST="$PROFILE_MANIFEST"
  EXPECTED_COUNT="$(json_field "$PROFILE" expected_count)"
  FRAME_COUNT="$(json_field "$PROFILE" frame_count)"
  [[ "$MANIFEST" == /dstack/run/*.json && "$MANIFEST" != *".."* ]] \
    || die "validated manifest path is unsafe"
  if [[ -n "${MANIFEST_OVERRIDE:-}" ]]; then
    MANIFEST="$MANIFEST_OVERRIDE"
    verify_manifest
  elif [[ -f "$MANIFEST" ]]; then
    verify_manifest
  fi
}

remote_prefix_is_empty() {
  local listing
  listing="$(hf_cli buckets list "$HF_REMOTE_URI" --format quiet 2>&1)" \
    || die "could not inspect remote output prefix"
  [[ -z "$listing" ]] || die "remote output prefix already exists: $HF_REMOTE_URI"
}

bootstrap_runtime() {
  mkdir -p "$OUTPUT_DIR"
  "$PYTHON_BIN" -m pip install --no-cache-dir --only-binary=:all: "uv==${UV_VERSION}"
  "$PYTHON_BIN" -m pip install --no-cache-dir \
    "transformers @ ${TRANSFORMERS_SOURCE_URL}" \
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

if transformers.__version__ != "5.15.0.dev0":
    raise SystemExit(f"unexpected transformers version: {transformers.__version__}")
if not torch.cuda.is_available() or not torch.version.cuda or torch.version.hip:
    raise SystemExit("NVIDIA CUDA is unavailable")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("CUDA bfloat16 is unavailable")
record = {
    "transformers_version": transformers.__version__,
    "transformers_revision": "cbf4d720ec734edb77d25452b2790c7e4be2f8d7",
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_device_name": torch.cuda.get_device_name(0),
    "cuda_bf16_supported": True,
}
Path(sys.argv[1]).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

verify_manifest() {
  [[ -f "$MANIFEST" ]] || die "label-free manifest is missing: $MANIFEST"
  "$PYTHON_BIN" "$BUILDER" --validate-manifest "$MANIFEST" --expected-count "$EXPECTED_COUNT" \
    >/dev/stderr
}

start_telemetry() {
  local child_pid="$1"
  local telemetry_path="$2"
  mkdir -p "$(dirname "$telemetry_path")"
  (
    if [[ ! -s "$telemetry_path" ]]; then
      printf 'timestamp,index,utilization.gpu [%],memory.used [MiB],memory.total [MiB]\n' \
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
    kill "$TELEMETRY_PID" 2>/dev/null || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
    TELEMETRY_PID=""
  fi
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
    --count "$EXPECTED_COUNT" &
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
    --condition-id shared-accepted-whisper --count "$EXPECTED_COUNT"
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
  hf_cli buckets sync "$HF_INPUT_URI" "$INPUT_DIR"
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
  "$PYTHON_BIN" - "$source" "$output" "$condition_id" "$request_source" "$evidence_root" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
output = Path(sys.argv[2])
condition_id = sys.argv[3]
request_source = sys.argv[4]
evidence_root = Path(sys.argv[5])
rows = json.loads(source.read_text(encoding="utf-8"))
if not isinstance(rows, list):
    raise SystemExit("shared request file is not a list")
selected = []
allowed = {
    "run_id", "condition_id", "video_id", "question", "frame_paths",
    "frame_timestamps", "transcript",
}
for row in rows:
    if not isinstance(row, dict) or set(row) != allowed:
        raise SystemExit("shared request contains an unexpected or label-bearing field")
    if request_source == "shared-accepted-whisper-ocr-only" and row.get("video_id") != "IVqpg_1aWXU":
        continue
    rewritten = dict(row)
    rewritten["condition_id"] = condition_id
    paths = []
    for frame_path in row.get("frame_paths", []):
        text = str(frame_path)
        marker = "/frames/"
        if marker not in text:
            raise SystemExit("frame path does not identify the shared evidence frame root")
        paths.append(str(evidence_root / "frames" / text.split(marker, 1)[1]))
    rewritten["frame_paths"] = paths
    selected.append(rewritten)
if request_source == "shared-accepted-whisper-ocr-only" and len(selected) != 1:
    raise SystemExit("OCR-only canary request must contain exactly one frozen row")
if not selected:
    raise SystemExit("condition request selection is empty")
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
      "$condition_dir/requests.json" "$condition_id" "$request_source" "$OUTPUT_DIR/evidence"
  else
    write_condition_requests "$INPUT_DIR/evidence/requests.accepted-whisper.json" \
      "$condition_dir/requests.json" "$condition_id" "$request_source" "$INPUT_DIR/evidence"
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
    "$condition_dir/timing.json" "$started_ns" "$finished_ns" "$condition_id" <<'PY'
import json
import sys
from pathlib import Path

requests = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
predictions = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if not isinstance(predictions, list) or len(predictions) != len(requests):
    raise SystemExit("prediction count does not match selected requests")
if [row.get("video_id") for row in predictions] != [row.get("video_id") for row in requests]:
    raise SystemExit("prediction IDs/order do not match selected requests")
if any(row.get("status") != "passed" or not str(row.get("answer", "")).strip() for row in predictions):
    raise SystemExit("condition has incomplete or empty predictions")
peak = [row.get("peak_memory_bytes") for row in predictions if isinstance(row.get("peak_memory_bytes"), int)]
payload = {
    "condition_id": sys.argv[6],
    "row_count": len(predictions),
    "wall_seconds": (int(sys.argv[5]) - int(sys.argv[4])) / 1_000_000_000,
    "peak_memory_bytes": max(peak) if peak else None,
    "runtime_identity": sorted({
        (row.get("runtime"), row.get("runtime_image")) for row in predictions
    }),
}
Path(sys.argv[3]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

write_final_status() {
  local status="$1"
  "$PYTHON_BIN" - "$OUTPUT_DIR/status.json" "$status" "$VQA_MODE" \
    "$EXPERIMENT_RUN_NAME" "$EXPECTED_COUNT" "$FRAME_COUNT" <<'PY'
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
    "labels_joined": False,
    "remote_sync": "pending-after-status-freeze",
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

finalize_bundle() {
  write_final_status passed
  (cd "$OUTPUT_DIR" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
  (cd "$OUTPUT_DIR" && sha256sum --check SHA256SUMS >/dev/null) \
    || die "local SHA256SUMS verification failed"
  if [[ "${VQA_DRY_RUN:-0}" != "1" ]]; then
    hf_cli buckets sync "$OUTPUT_DIR" "$HF_REMOTE_URI" \
      || die "durable output bundle sync failed"
  fi
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
  prepare_condition_evidence
  run_condition "$CONDITION_ID"
}

main() {
  trap cleanup_runtime EXIT
  validate_dispatch
  remote_prefix_is_empty
  if [[ "${VQA_DRY_RUN:-0}" == "1" ]]; then
    mkdir -p "$OUTPUT_DIR"
  else
    bootstrap_runtime
  fi
  case "$VQA_MODE" in
    canary) run_canary ;;
    evidence) run_evidence ;;
    condition) run_full_condition ;;
  esac
  finalize_bundle
}

main "$@"
