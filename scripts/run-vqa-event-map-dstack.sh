#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_RUNTIME_IMAGE="docker.io/vllm/vllm-openai:v0.20.0@sha256:77797441eae630c2e79eefa03957b3d61a278670f2a9928d64ce102e7a0790cc"
readonly EXPECTED_RUNTIME_IMAGE_DIGEST="sha256:77797441eae630c2e79eefa03957b3d61a278670f2a9928d64ce102e7a0790cc"
readonly EXPECTED_MODEL_ID="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
readonly EXPECTED_MODEL_REVISION="dc5f0b0bfddf8b6e0f5891475be9af05b80126fe"
readonly EXPECTED_INPUT_URI="hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/vqa-event-map-inputs-20260809-a3/inputs"
readonly EXPECTED_REMOTE_ROOT="hf://buckets/Npatta01/trec-video-2026-event-map-20260809"
readonly PYTHON_BIN="/usr/bin/python3"
readonly INPUT_DIR="/dstack/input"
readonly OUTPUT_DIR="/dstack/output"
readonly MAX_OUTPUT_TOKENS=768
readonly MIN_L40S_MEMORY_MIB=45000
readonly HF_CLI_VERSION="1.24.0"
readonly EXPECTED_PANEL_IDS_SHA256="95a623c0add252541b9c81ff4910e949dc81db17482add2c467b761237b22b0f"

MODE="${EXPERIMENT_MODE:-}"
RUN_ID="${EXPERIMENT_RUN_ID:-}"
CONDITIONS="${EVENT_MAP_CONDITIONS:-}"
SLICE_ID="${EVENT_MAP_SLICE:-}"
HF_INPUT_URI="${HF_INPUT_URI:-}"
HF_REMOTE_URI="${HF_REMOTE_URI:-}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-}"
RUNTIME_IMAGE_DIGEST="${RUNTIME_IMAGE_DIGEST:-}"
MODEL_ID="${MODEL_ID:-}"
MODEL_REVISION="${MODEL_REVISION:-}"
HF_ACCESS_TOKEN="${HF_TOKEN:-}"
unset HF_TOKEN

ATTEMPT_UUID="${ATTEMPT_UUID:-}"
EXPECTED_SOURCE_BUNDLE_SHA256="${EXPECTED_SOURCE_BUNDLE_SHA256:-}"
EXPECTED_CONTROLLER_SHA256="${EXPECTED_CONTROLLER_SHA256:-}"
EXPECTED_PROTOCOL_SHA256="${EXPECTED_PROTOCOL_SHA256:-}"
EXPECTED_REQUEST_FILE_SHA256="${EXPECTED_REQUEST_FILE_SHA256:-}"
EXPECTED_INFERENCE_MODULE_SHA256="${EXPECTED_INFERENCE_MODULE_SHA256:-}"
EXPECTED_METADATA_BUNDLE_SHA256="${EXPECTED_METADATA_BUNDLE_SHA256:-}"

SERVER_PID=""
TELEMETRY_PID=""
REMOTE_RESERVED="false"
FINALIZED="false"
FAILURE_STAGE="initializing"

die() {
  echo "error: $*" >&2
  return 2
}

require_sha256() {
  [[ "$2" =~ ^[0-9a-f]{64}$ ]] || die "$1 must be an exact SHA-256"
}

hf_cli() {
  HF_TOKEN="$HF_ACCESS_TOKEN" uvx --from "huggingface-hub==$HF_CLI_VERSION" hf "$@"
}

validate_identity() {
  case "$MODE:$RUN_ID:$CONDITIONS:$SLICE_ID" in
    canary:vqa-event-map-canary:v-map,av-map:) ;;
    full:vqa-event-map-v-map-chunk-00:v-map:00|full:vqa-event-map-v-map-chunk-01:v-map:01|full:vqa-event-map-v-map-chunk-02:v-map:02|full:vqa-event-map-v-map-chunk-03:v-map:03) ;;
    full:vqa-event-map-av-map-chunk-00:av-map:00|full:vqa-event-map-av-map-chunk-01:av-map:01|full:vqa-event-map-av-map-chunk-02:av-map:02|full:vqa-event-map-av-map-chunk-03:av-map:03) ;;
    *) die "unexpected immutable event-map identity" ;;
  esac
  [[ "$HF_INPUT_URI" == "$EXPECTED_INPUT_URI" ]] || die "unexpected immutable input URI"
  [[ "$HF_REMOTE_URI" == "$EXPECTED_REMOTE_ROOT-$RUN_ID" ]] || die "unexpected immutable output URI"
  [[ "$RUNTIME_IMAGE" == "$EXPECTED_RUNTIME_IMAGE" ]] || die "runtime image mismatch"
  [[ "$RUNTIME_IMAGE_DIGEST" == "$EXPECTED_RUNTIME_IMAGE_DIGEST" ]] || die "runtime image digest mismatch"
  [[ "$MODEL_ID" == "$EXPECTED_MODEL_ID" ]] || die "model identity mismatch"
  [[ "$MODEL_REVISION" == "$EXPECTED_MODEL_REVISION" ]] || die "model revision mismatch"
  [[ -n "$HF_ACCESS_TOKEN" ]] || die "HF_TOKEN secret is unavailable"
  [[ "$ATTEMPT_UUID" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] \
    || die "attempt UUID must be exact"
  require_sha256 EXPECTED_SOURCE_BUNDLE_SHA256 "$EXPECTED_SOURCE_BUNDLE_SHA256"
  require_sha256 EXPECTED_CONTROLLER_SHA256 "$EXPECTED_CONTROLLER_SHA256"
  require_sha256 EXPECTED_PROTOCOL_SHA256 "$EXPECTED_PROTOCOL_SHA256"
  require_sha256 EXPECTED_REQUEST_FILE_SHA256 "$EXPECTED_REQUEST_FILE_SHA256"
  require_sha256 EXPECTED_INFERENCE_MODULE_SHA256 "$EXPECTED_INFERENCE_MODULE_SHA256"
  require_sha256 EXPECTED_METADATA_BUNDLE_SHA256 "$EXPECTED_METADATA_BUNDLE_SHA256"
  [[ "$(uname -m)" == "x86_64" ]] || die "amd64 runtime is required"
  [[ -x "$PYTHON_BIN" ]] || die "pinned image interpreter is unavailable"
  [[ "$(sha256sum /dstack/run/src/trec_video_2026/vqa_experiments/nemotron3_omni_event_map.py | awk '{print $1}')" == "$EXPECTED_INFERENCE_MODULE_SHA256" ]] \
    || die "inference module digest mismatch"
}

verify_gpu() {
  local gpu_record gpu_name gpu_memory
  gpu_record="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1)" \
    || die "NVIDIA GPU telemetry is unavailable"
  IFS=',' read -r gpu_name gpu_memory <<<"$gpu_record"
  gpu_name="${gpu_name## }"
  gpu_name="${gpu_name%% }"
  gpu_memory="${gpu_memory## }"
  gpu_memory="${gpu_memory%% }"
  [[ "$gpu_name" == "L40S" || "$gpu_name" == "NVIDIA L40S" ]] \
    || die "GPU must be exact L40S"
  [[ "$gpu_memory" =~ ^[0-9]+$ && "$gpu_memory" -ge "$MIN_L40S_MEMORY_MIB" ]] \
    || die "L40S GPU reported memory is below the safe floor"
  printf '%s\n' "{\"gpu_name\":\"L40S\",\"gpu_memory_mib\":$gpu_memory}" >"$OUTPUT_DIR/gpu-identity.json"
}

bootstrap_runtime() {
  verify_gpu
  "$PYTHON_BIN" -m pip install --no-cache-dir "vllm[audio]==0.20.0" "uv==0.11.3" \
    >/dev/null 2>&1 || die "pinned runtime bootstrap failed"
  "$PYTHON_BIN" - "$OUTPUT_DIR/runtime.json" "$EXPECTED_RUNTIME_IMAGE" <<'PY' >/dev/null 2>&1
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import torch

if importlib.metadata.version("vllm") != "0.20.0":
    raise SystemExit("vLLM version mismatch")
if platform.machine() != "x86_64" or not torch.cuda.is_available() or not torch.version.cuda or torch.version.hip:
    raise SystemExit("CUDA L40S runtime attestation failed")
payload = {
    "schema_version": 1,
    "runtime_image": sys.argv[2],
    "vllm_version": importlib.metadata.version("vllm"),
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torch_hip": torch.version.hip,
    "gpu_name": torch.cuda.get_device_name(0),
    "gpu_memory_mib": int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)),
    "enable_thinking": False,
    "max_output_tokens": 768,
    "gpu_telemetry_path": "telemetry/nvidia-smi.csv",
}
Path(sys.argv[1]).write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
PY
}

verify_checksum_file() {
  local root="$1"
  "$PYTHON_BIN" - "$root" <<'PY' >/dev/null 2>&1 || die "input SHA256SUMS verification failed"
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath

root = Path(sys.argv[1]).resolve(strict=True)
checksum_path = root / "SHA256SUMS"
if not checksum_path.is_file() or checksum_path.is_symlink():
    raise SystemExit("checksum file missing")
entries = {}
for line in checksum_path.read_text(encoding="utf-8").splitlines():
    digest, separator, relative = line.partition("  ")
    pure = PurePosixPath(relative[2:]) if relative.startswith("./") else PurePosixPath("/")
    if separator != "  " or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise SystemExit("malformed checksum entry")
    if not relative.startswith("./") or pure.is_absolute() or ".." in pure.parts or relative in entries:
        raise SystemExit("unsafe checksum entry")
    if pure.parts[:1] == ("media",):
        raise SystemExit("media integrity must be bound by the media manifest")
    entries[relative] = digest
actual = {}
for path in root.rglob("*"):
    if path.is_symlink():
        raise SystemExit("input symlink is forbidden")
    if path.is_file() and path.name != "SHA256SUMS":
        relative = "./" + path.relative_to(root).as_posix()
        if relative.startswith("./media/"):
            continue
        actual[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
if set(actual) != set(entries):
    raise SystemExit("input checksum file does not cover exact bytes")
if actual != entries:
    raise SystemExit("input checksum mismatch")
PY
}

verify_label_free_bundle() {
  local request_file
  if [[ "$MODE" == "canary" ]]; then
    request_file="$INPUT_DIR/event-map-canary.jsonl"
  else
    request_file="$INPUT_DIR/${CONDITIONS}-chunk-${SLICE_ID}.jsonl"
  fi
  [[ -f "$INPUT_DIR/event-map-controller.json" && -f "$INPUT_DIR/event-map-protocol.json" ]] \
    || die "immutable event-map controller/protocol is missing"
  [[ -f "$request_file" ]] || die "immutable event-map request slice is missing"
  verify_checksum_file "$INPUT_DIR"
  [[ "$(sha256sum "$INPUT_DIR/SHA256SUMS" | awk '{print $1}')" == "$EXPECTED_METADATA_BUNDLE_SHA256" ]] \
    || die "metadata bundle digest mismatch"
  "$PYTHON_BIN" - "$INPUT_DIR" "$request_file" "$MODE" "$CONDITIONS" "$SLICE_ID" "$ATTEMPT_UUID" "$OUTPUT_DIR/preflight.json" \
    "$EXPECTED_SOURCE_BUNDLE_SHA256" "$EXPECTED_CONTROLLER_SHA256" "$EXPECTED_PROTOCOL_SHA256" \
    "$EXPECTED_REQUEST_FILE_SHA256" "$EXPECTED_INFERENCE_MODULE_SHA256" "$EXPECTED_PANEL_IDS_SHA256" <<'PY' >/dev/null 2>&1 \
    || die "event-map preflight identity, order, label, or media binding failed"
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath

root = Path(sys.argv[1]).resolve(strict=True)
request_path = Path(sys.argv[2]).resolve(strict=True)
(
    mode,
    condition,
    slice_id,
    attempt_uuid,
    preflight_path,
    expected_source_bundle,
    expected_controller,
    expected_protocol,
    expected_request,
    expected_module,
    expected_panel_ids,
) = sys.argv[3:]
controller_path = root / "event-map-controller.json"
media_manifest_path = root / "event-map-media-manifest.json"
protocol_path = root / "event-map-protocol.json"
controller = json.loads(controller_path.read_text(encoding="utf-8"))
media_manifest = json.loads(media_manifest_path.read_text(encoding="utf-8"))
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
forbidden = {"answer", "answers", "annotation", "annotations", "correctanswer", "expectedanswer", "groundtruth", "label", "labels", "reference", "metric", "metrics", "score", "scores", "prediction", "predictions"}
def reject(value):
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if normalized in forbidden or "reference" in normalized or "groundtruth" in normalized:
                raise SystemExit("label-bearing key")
            reject(child)
    elif isinstance(value, list):
        for child in value:
            reject(child)
reject(controller)
reject(protocol)
if controller.get("schema_version") != 1 or controller.get("label_policy") != "label-free-controller":
    raise SystemExit("controller identity mismatch")
rows = controller.get("rows")
if not isinstance(rows, list) or controller.get("panel", {}).get("row_count") != len(rows):
    raise SystemExit("controller row count mismatch")
if len(rows) == 475 and controller.get("panel", {}).get("shard_counts") != [119, 119, 119, 118]:
    raise SystemExit("controller shard counts mismatch")
protocol_without_hash = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
protocol_digest = hashlib.sha256(json.dumps(protocol_without_hash, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
if protocol.get("protocol_sha256") != protocol_digest:
    raise SystemExit("protocol digest mismatch")
if protocol.get("controller_sha256") != hashlib.sha256(controller_path.read_bytes()).hexdigest():
    raise SystemExit("controller digest mismatch")
if protocol["controller_sha256"] != expected_controller:
    raise SystemExit("controller digest is not the pinned artifact")
if protocol.get("media_manifest_sha256") != hashlib.sha256(media_manifest_path.read_bytes()).hexdigest():
    raise SystemExit("media manifest digest mismatch")
ids = [row.get("video_id") for row in rows]
ids_digest = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
if ids_digest != controller.get("panel", {}).get("ids_sha256"):
    raise SystemExit("canonical controller ID digest mismatch")
if ids_digest != expected_panel_ids:
    raise SystemExit("controller is not the pinned annotated-media panel")
source_bundle = hashlib.sha256(json.dumps({
    "controller_sha256": protocol["controller_sha256"],
    "media_manifest_sha256": protocol["media_manifest_sha256"],
}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
if source_bundle != protocol.get("source_bundle_sha256"):
    raise SystemExit("source bundle digest mismatch")
if source_bundle != expected_source_bundle:
    raise SystemExit("source bundle is not the pinned artifact")
if protocol.get("protocol_sha256") != expected_protocol:
    raise SystemExit("protocol is not the pinned artifact")
if protocol.get("inference_module_sha256") != expected_module:
    raise SystemExit("protocol inference module is not the pinned source")
request_bytes = request_path.read_bytes()
request_rows = [json.loads(line) for line in request_bytes.decode("utf-8").splitlines() if line.strip()]
if not request_rows:
    raise SystemExit("request file is empty")
if any(not isinstance(row, dict) for row in request_rows):
    raise SystemExit("request rows are malformed")
request_key = request_path.name
identity = protocol.get("request_files", {}).get(request_key)
if not isinstance(identity, dict) or identity.get("file_sha256") != hashlib.sha256(request_bytes).hexdigest() or identity.get("row_count") != len(request_rows):
    raise SystemExit("request file digest binding mismatch")
if identity["file_sha256"] != expected_request:
    raise SystemExit("request file is not the pinned slice")
if mode == "canary":
    if request_key != "event-map-canary.jsonl" or len(request_rows) != 8:
        raise SystemExit("paired canary identity mismatch")
    canary_ids = controller.get("panel", {}).get("canary_ids")
    if not isinstance(canary_ids, list) or len(canary_ids) != 4:
        raise SystemExit("canary panel identity is missing")
    expected = [(video_id, condition) for video_id in canary_ids for condition in ("v-map", "av-map")]
    actual = [(row.get("video_id"), row.get("condition_id")) for row in request_rows]
    if actual != expected:
        raise SystemExit("canary order mismatch")
else:
    if len(request_rows) != [119, 119, 119, 118][int(slice_id)]:
        raise SystemExit("full slice row count mismatch")
    all_ids = [row["video_id"] for row in controller["rows"]]
    expected_ids = all_ids[int(slice_id) * 119 : int(slice_id) * 119 + len(request_rows)]
    if [row.get("video_id") for row in request_rows] != expected_ids:
        raise SystemExit("full slice order mismatch")
    if any(row.get("condition_id") != condition for row in request_rows):
        raise SystemExit("full slice condition mismatch")
for row in request_rows:
    if set(row) != {"schema_version", "run_id", "condition_id", "slice_id", "video_id", "question", "question_stratum", "media_path", "media_sha256", "duration_seconds", "request_sha256"}:
        raise SystemExit("request schema mismatch")
    if hashlib.sha256(json.dumps({key: value for key, value in row.items() if key != "request_sha256"}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != row["request_sha256"]:
        raise SystemExit("request row hash mismatch")
    relative_media = PurePosixPath(row["media_path"])
    if relative_media.is_absolute() or relative_media.parts[:1] != ("media",) or len(relative_media.parts) != 2:
        raise SystemExit("request media path is unsafe")
    media = root / relative_media
    if not media.is_file() or media.is_symlink() or media.stat().st_size <= 0:
        raise SystemExit("source media bytes are missing")
    if hashlib.sha256(media.read_bytes()).hexdigest() != row["media_sha256"]:
        raise SystemExit("source media hash mismatch")
    if row["duration_seconds"] <= 0:
        raise SystemExit("source media duration is invalid")
if not isinstance(source_bundle, str) or len(source_bundle) != 64:
    raise SystemExit("source bundle digest is missing")
preflight_path.write_text(json.dumps({
    "schema_version": 1,
    "run_id": request_rows[0]["run_id"],
    "attempt_uuid": attempt_uuid,
    "source_bundle_sha256": source_bundle,
    "controller_sha256": protocol["controller_sha256"],
    "protocol_sha256": protocol["protocol_sha256"],
    "request_file_sha256": identity["file_sha256"],
    "row_count": len(request_rows),
    "condition_id": condition,
    "slice_id": slice_id,
}, sort_keys=True) + "\n", encoding="utf-8")
PY
  REQUEST_FILE="$request_file"
}

download_inputs() {
  mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"
  hf_cli buckets sync "$HF_INPUT_URI" "$INPUT_DIR" >/dev/null 2>&1 \
    || die "immutable input bundle download failed"
}

normalize_inputs() {
  mkdir -p "$INPUT_DIR/normalized"
  "$PYTHON_BIN" - "$INPUT_DIR/event-map-controller.json" "$REQUEST_FILE" "$INPUT_DIR" "$INPUT_DIR/normalized" "$INPUT_DIR/normalized-controller.json" <<'PY' >/dev/null 2>&1 \
    || die "media normalization or source/derived verification failed"
import json
import sys
from pathlib import Path

from trec_video_2026.vqa_experiments.nemotron3_omni_event_map import normalize_media_file

controller_path, requests_path, input_root, normalized_root, output_path = map(Path, sys.argv[1:])
controller = json.loads(controller_path.read_text(encoding="utf-8"))
request_rows = [
    json.loads(line)
    for line in requests_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
requested_ids = list(dict.fromkeys(row["video_id"] for row in request_rows))
controller_by_id = {row["video_id"]: row for row in controller["rows"]}
if not requested_ids or any(video_id not in controller_by_id for video_id in requested_ids):
    raise SystemExit("request media are not bound to the controller")
normalized_rows = []
for video_id in requested_ids:
    row = controller_by_id[video_id]
    source = input_root / row["media_path"]
    derived = normalized_root / f"{row['video_id']}.mp4"
    identity = normalize_media_file(
        source_path=source,
        derived_path=derived,
        expected_source_sha256=row["media_sha256"],
    )
    normalized_rows.append({
        **row,
        "normalized_media_path": f"normalized/{row['video_id']}.mp4",
        **identity,
    })
output_path.write_text(json.dumps({"schema_version": 1, "rows": normalized_rows}, sort_keys=True) + "\n", encoding="utf-8")
PY
}

reserve_remote_prefix() {
  local bucket_id created
  bucket_id="${HF_REMOTE_URI#hf://buckets/}"
  created="$(hf_cli buckets create "$bucket_id" --private --format quiet 2>/dev/null)" \
    || die "atomic private output reservation failed"
  [[ "$created" == "$HF_REMOTE_URI" ]] || die "atomic output identity mismatch"
  REMOTE_RESERVED="true"
}

start_telemetry() {
  mkdir -p "$OUTPUT_DIR/telemetry"
  {
    printf '%s\n' 'timestamp,gpu_name,memory_used_mib,memory_total_mib,utilization_gpu'
    while kill -0 "$SERVER_PID" 2>/dev/null; do
      nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null \
        | sed "s/^/$(date -u +%Y-%m-%dT%H:%M:%SZ),/" || true
      sleep 5
    done
  } >"$OUTPUT_DIR/telemetry/nvidia-smi.csv" &
  TELEMETRY_PID=$!
}

stop_telemetry() {
  if [[ -n "$TELEMETRY_PID" ]] && kill -0 "$TELEMETRY_PID" 2>/dev/null; then
    kill "$TELEMETRY_PID" 2>/dev/null || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
  fi
  TELEMETRY_PID=""
}

start_server() {
  local vllm_bin
  vllm_bin="$($PYTHON_BIN - <<'PY'
import os
import sysconfig
print(os.path.join(sysconfig.get_path("scripts"), "vllm"))
PY
)"
  [[ -x "$vllm_bin" ]] || die "pinned vLLM executable is unavailable"
  "$vllm_bin" serve "$EXPECTED_MODEL_ID" \
    --revision "$EXPECTED_MODEL_REVISION" \
    --host 127.0.0.1 --port 8000 \
    --max-model-len 32768 --max-num-seqs 1 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.90 --trust-remote-code \
    --allowed-local-media-path "$INPUT_DIR" \
    --limit-mm-per-prompt '{"video":1,"audio":1}' \
    --media-io-kwargs '{"video":{"fps":1,"num_frames":64}}' \
    --video-pruning-rate 0.5 \
    --kv-cache-dtype fp8 \
    --reasoning-parser nemotron_v3 \
    >"$OUTPUT_DIR/server.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 180); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      die "vLLM server exited before readiness"
    fi
    if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
from urllib.request import urlopen
with urlopen("http://127.0.0.1:8000/v1/models", timeout=2) as response:
    if response.status != 200:
        raise SystemExit(1)
PY
    then
      start_telemetry
      return 0
    fi
    sleep 5
  done
  die "vLLM server readiness timed out"
}

stop_server() {
  stop_telemetry
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}

run_workload() {
  local args=(
    -m trec_video_2026.vqa_experiments.nemotron3_omni_event_map run-file
    --requests "$REQUEST_FILE"
    --normalized-root "$INPUT_DIR/normalized"
    --normalized-controller "$INPUT_DIR/normalized-controller.json"
    --output "$OUTPUT_DIR/predictions.jsonl"
    --server-url "http://127.0.0.1:8000/v1/chat/completions"
    --runtime-json "$OUTPUT_DIR/runtime.json"
  )
  if [[ "${VLLM_RESPONSE_FORMAT_SUPPORTED:-1}" == "1" ]]; then
    args+=(--response-format-supported)
  fi
  set +e
  "$PYTHON_BIN" "${args[@]}" >/dev/null 2>&1
  WORKLOAD_RC=$?
  set -e
  return "$WORKLOAD_RC"
}

verify_outputs() {
  "$PYTHON_BIN" - "$OUTPUT_DIR/predictions.jsonl" "$OUTPUT_DIR/preflight.json" <<'PY' >/dev/null 2>&1 \
    || die "event-map output completion or row binding failed"
import json
import sys
from pathlib import Path

predictions = Path(sys.argv[1])
preflight = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) != preflight["row_count"]:
    raise SystemExit("prediction row count mismatch")
if any(row.get("status") != "passed" for row in rows):
    raise SystemExit("prediction status is incomplete")
if any(not isinstance(row.get("request_sha256"), str) or not isinstance(row.get("response_sha256"), str) for row in rows):
    raise SystemExit("prediction hash is missing")
PY
}

write_technical_status() {
  local exit_status="$1"
  mkdir -p "$OUTPUT_DIR"
  "$PYTHON_BIN" - "$OUTPUT_DIR/technical-status.json" "$RUN_ID" "$ATTEMPT_UUID" "$FAILURE_STAGE" "$exit_status" "$HF_REMOTE_URI" <<'PY' >/dev/null 2>&1 || true
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1,
    "status": "failed",
    "run_id": sys.argv[2],
    "attempt_uuid": sys.argv[3],
    "failure_stage": sys.argv[4],
    "exit_status": int(sys.argv[5]),
    "output_identity": sys.argv[6],
    "credentials_recorded": False,
}, sort_keys=True) + "\n", encoding="utf-8")
PY
}

prepare_payload_checksums() {
  (
    cd "$OUTPUT_DIR"
    find . -type f ! -name SHA256SUMS ! -name terminal-status.json \
      ! -name failure-status.json -print0 \
      | sort -z \
      | xargs -0 -r sha256sum >SHA256SUMS
    sha256sum -c SHA256SUMS >/dev/null
  )
}

publish_and_verify_payload() {
  hf_cli buckets sync "$OUTPUT_DIR" "$HF_REMOTE_URI" \
    --exclude terminal-status.json --exclude failure-status.json >/dev/null 2>&1 \
    || return 2
  local verify_dir
  verify_dir="$(mktemp -d /tmp/vqa-event-map-output-verify.XXXXXX)" || return 2
  if ! hf_cli buckets sync "$HF_REMOTE_URI" "$verify_dir" \
    --exclude terminal-status.json --exclude failure-status.json >/dev/null 2>&1; then
    rm -rf -- "$verify_dir"
    return 2
  fi
  if ! (cd "$verify_dir" && sha256sum -c SHA256SUMS >/dev/null); then
    rm -rf -- "$verify_dir"
    return 2
  fi
  rm -rf -- "$verify_dir"
}

finalize_failure() {
  local exit_status="$1"
  stop_server
  rm -f "$OUTPUT_DIR/server.log" "$OUTPUT_DIR/terminal-status.json"
  write_technical_status "$exit_status"
  prepare_payload_checksums || return 2
  local payload_checksum row_count
  payload_checksum="$(sha256sum "$OUTPUT_DIR/SHA256SUMS" | awk '{print $1}')"
  row_count=0
  if [[ -f "$OUTPUT_DIR/predictions.jsonl" ]]; then
    row_count="$(wc -l <"$OUTPUT_DIR/predictions.jsonl")"
  fi
  publish_and_verify_payload || return 2
  "$PYTHON_BIN" - "$OUTPUT_DIR/failure-status.json" "$RUN_ID" "$ATTEMPT_UUID" \
    "$HF_REMOTE_URI" "$payload_checksum" "$row_count" "$OUTPUT_DIR/preflight.json" <<'PY' >/dev/null 2>&1 \
    || return 2
import json
import sys
from pathlib import Path

preflight = json.loads(Path(sys.argv[7]).read_text(encoding="utf-8"))
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1,
    "status": "failed",
    "run_id": sys.argv[2],
    "attempt_uuid": sys.argv[3],
    "output_identity": sys.argv[4],
    "payload_checksum": sys.argv[5],
    "row_count": int(sys.argv[6]),
    "source_bundle_sha256": preflight["source_bundle_sha256"],
    "controller_sha256": preflight["controller_sha256"],
    "protocol_sha256": preflight["protocol_sha256"],
    "request_file_sha256": preflight["request_file_sha256"],
}, sort_keys=True) + "\n", encoding="utf-8")
PY
  hf_cli buckets cp "$OUTPUT_DIR/failure-status.json" \
    "$HF_REMOTE_URI/failure-status.json" >/dev/null 2>&1
}

finalize_success() {
  local row_count="$1"
  stop_server
  rm -f "$OUTPUT_DIR/server.log" "$OUTPUT_DIR/failure-status.json"
  prepare_payload_checksums || die "payload checksum finalization failed"
  local payload_checksum
  payload_checksum="$(sha256sum "$OUTPUT_DIR/SHA256SUMS" | awk '{print $1}')"
  publish_and_verify_payload || die "private payload publication verification failed"
  "$PYTHON_BIN" - "$OUTPUT_DIR/terminal-status.json" "$RUN_ID" "$ATTEMPT_UUID" "$HF_REMOTE_URI" "$payload_checksum" "$row_count" "$OUTPUT_DIR/preflight.json" <<'PY' >/dev/null 2>&1 \
    || die "terminal marker finalization failed"
import json
import sys
from pathlib import Path

preflight = json.loads(Path(sys.argv[7]).read_text(encoding="utf-8"))
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1,
    "status": "complete",
    "run_id": sys.argv[2],
    "attempt_uuid": sys.argv[3],
    "output_identity": sys.argv[4],
    "payload_checksum": sys.argv[5],
    "row_count": int(sys.argv[6]),
    "source_bundle_sha256": preflight["source_bundle_sha256"],
    "controller_sha256": preflight["controller_sha256"],
    "protocol_sha256": preflight["protocol_sha256"],
    "request_file_sha256": preflight["request_file_sha256"],
}, sort_keys=True) + "\n", encoding="utf-8")
PY
  hf_cli buckets cp "$OUTPUT_DIR/terminal-status.json" \
    "$HF_REMOTE_URI/terminal-status.json" >/dev/null 2>&1 \
    || die "terminal marker publication failed"
}

on_exit() {
  local exit_status="$?"
  trap - EXIT
  stop_server
  if (( exit_status != 0 )) && [[ "$FINALIZED" != "true" ]]; then
    if [[ "$REMOTE_RESERVED" == "true" ]]; then
      set +e
      finalize_failure "$exit_status"
      set -e
    else
      write_technical_status "$exit_status"
    fi
  fi
  exit "$exit_status"
}

main() {
  mkdir -p "$OUTPUT_DIR"
  trap on_exit EXIT
  FAILURE_STAGE="identity"
  validate_identity
  FAILURE_STAGE="runtime-bootstrap"
  bootstrap_runtime
  FAILURE_STAGE="input-download"
  download_inputs
  FAILURE_STAGE="input-preflight"
  verify_label_free_bundle
  FAILURE_STAGE="output-reservation"
  reserve_remote_prefix
  FAILURE_STAGE="media-normalization"
  normalize_inputs
  FAILURE_STAGE="server-startup"
  start_server
  FAILURE_STAGE="inference"
  run_workload || die "event-map inference failed"
  FAILURE_STAGE="output-verification"
  verify_outputs
  FAILURE_STAGE="artifact-finalization"
  finalize_success "$(wc -l <"$OUTPUT_DIR/predictions.jsonl")"
  FINALIZED="true"
}

main "$@"
