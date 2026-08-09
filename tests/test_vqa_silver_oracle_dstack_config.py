from __future__ import annotations

import hashlib
import re
import shlex
import subprocess
from pathlib import Path

import pytest

from trec_video_2026.vqa_experiments.qwen3_omni_structured import (
    MODEL_ID,
    MODEL_REVISION,
    RUNTIME_IMAGE,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "vqa-silver-oracle-election-20260809-b4"
CONFIGS = {
    "canary-b4": Path("dstack/vqa-silver-oracle-selector-canary.dstack.yml"),
    **{
        f"chunk-{index:02d}": Path(
            f"dstack/vqa-silver-oracle-selector-chunk-{index:02d}.dstack.yml"
        )
        for index in range(4)
    },
}
RUNNER = Path("scripts/run-vqa-silver-oracle-selector-dstack.sh")
MEDIA_URI = (
    "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/artifacts/"
    "vqa2025-training-media-475-20260809-a1/media"
)
CONTROLLER_URI = (
    "hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/"
    f"{RUN_ID}/input-controller"
)


def test_selector_configs_are_pinned_private_bounded_tasks() -> None:
    for slice_id, path in CONFIGS.items():
        text = path.read_text(encoding="utf-8")
        expected_duration = "45m" if slice_id.startswith("canary") else "200m"
        expected_name = (
            "vqa-silver-selector-canary-20260809-b4"
            if slice_id == "canary-b4"
            else f"vqa-silver-selector-{slice_id}-20260809-b4"
        )
        assert "type: task" in text
        assert f"name: {expected_name}" in text
        assert f"image: {RUNTIME_IMAGE}" in text
        assert f"SELECTOR_SLICE={slice_id}" in text
        assert f"MODEL_ID={MODEL_ID}" in text
        assert f"MODEL_REVISION={MODEL_REVISION}" in text
        assert (
            "EXPECTED_PROTOCOL_SHA256="
            "a63dace536d03b8c99062011faa472f37de418901fe8e55a7e3bffb40312901c"
        ) in text
        attempt_match = re.search(
            r"EXECUTION_ATTEMPT_ID=([0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
            r"[89ab][0-9a-f]{3}-[0-9a-f]{12})",
            text,
        )
        assert attempt_match is not None
        assert "HF_TOKEN=${{ secrets.hf_token }}" in text
        assert "ports:" not in text
        assert "spot_policy: on-demand" in text
        assert "backends: [runpod, vastai]" in text
        assert "name: H100" in text
        assert "memory: 80GB.." in text
        assert "max_price: 3.35" in text
        assert f"max_duration: {expected_duration}" in text
        assert "on_events: [no-capacity]" in text
        assert "duration: 30m" in text
        assert "../src/trec_video_2026:/dstack/run/src/trec_video_2026" in text
        assert (
            "../scripts/run-vqa-silver-oracle-selector-dstack.sh:"
            "/dstack/run/scripts/run-vqa-silver-oracle-selector-dstack.sh"
        ) in text
        assert "../data/" not in text


def test_selector_runner_fetches_only_slice_media_and_syncs_private_outputs() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert f'readonly EXPECTED_RUN_ID="{RUN_ID}"' in text
    assert f'readonly EXPECTED_MEDIA_URI="{MEDIA_URI}"' in text
    assert f'readonly EXPECTED_CONTROLLER_URI="{CONTROLLER_URI}"' in text
    assert RUNTIME_IMAGE in text
    assert MODEL_ID in text
    assert MODEL_REVISION in text
    assert "readarray -t requested_media" in text
    assert 'include_args+=(--include "$basename")' in text
    assert 'hf_cli buckets sync "$MEDIA_URI" "$INPUT_ROOT/media"' in text
    assert 'hf_cli buckets sync "$CONTROLLER_URI" "$INPUT_ROOT/controller"' in text
    assert "sha256sum -c SHA256SUMS" in text
    assert 'chmod -R a-w "$INPUT_ROOT"' in text
    assert "qwen3_omni_structured" in text
    assert 'hf_cli buckets sync "$OUTPUT_DIR" "$REMOTE_OUTPUT_URI"' in text
    assert 'readonly PREFLIGHT_MARKER="/dstack/runtime-preflight-marker.json"' in text
    assert 'cp "$PREFLIGHT_MARKER" "$OUTPUT_DIR/runtime-preflight-marker.json"' in text
    assert 'readonly WORKLOAD_STDOUT="/dstack/workload.stdout.json"' in text
    assert '>"$WORKLOAD_STDOUT"' in text
    assert '>"$OUTPUT_DIR/workload.stdout.json"' not in text
    assert "verify_source_bundle" in text
    assert "source-bundle-identity.json" in text
    assert 'readonly TELEMETRY_CSV="/dstack/nvidia-smi.csv"' in text
    assert "memory.total" in text
    assert "memory.free" in text
    assert 'cp "$TELEMETRY_CSV" "$OUTPUT_DIR/nvidia-smi.csv"' in text
    assert "--delete" not in text
    assert "dstack logs -d" not in text


def test_selector_slice_request_names_and_output_prefixes_are_disjoint() -> None:
    runner = RUNNER.read_text(encoding="utf-8")

    assert 'canary-b4) REQUEST_NAME="selector-canary.jsonl"' in runner
    for index in range(4):
        chunk = f"chunk-{index:02d}"
        assert f'{chunk}) REQUEST_NAME="selector-{chunk}.jsonl"' in runner
    assert "selector/$SELECTOR_SLICE/$EXECUTION_ATTEMPT_ID" in runner


def test_four_h100_max_duration_and_price_stay_under_billed_cap() -> None:
    max_hours_per_shard = 3 + 20 / 60
    four_shard_max_cost = 4 * max_hours_per_shard * 3.35

    assert max_hours_per_shard > 3.25
    assert four_shard_max_cost <= 45.0
    assert four_shard_max_cost == pytest.approx(44.666666666666664)


def test_runner_main_arms_finalization_only_after_authenticated_runtime_preflight() -> (
    None
):
    runner = RUNNER.read_text(encoding="utf-8")
    match = re.search(r"(?ms)^main\(\) \{\n(?P<body>.*?)^\}", runner)
    assert match is not None
    main_body = match.group("body")

    source_check = main_body.index("verify_source_bundle")
    input_check = main_body.index("fetch_slice_inputs")
    runtime_preflight = main_body.index("run_runtime_preflight")
    arm = main_body.index('FINALIZE_READY="true"')
    workload = main_body.index("run_selector")
    assert source_check < input_check < runtime_preflight < arm < workload


def test_runtime_preflight_failure_cannot_call_output_publication(
    tmp_path: Path,
) -> None:
    call_log = tmp_path / "hf-calls.log"
    probe = f"""
source {shlex.quote(str(RUNNER.resolve()))}
mkdir() {{ :; }}
validate_environment() {{
  REQUEST_NAME="selector-canary.jsonl"
  REMOTE_OUTPUT_URI="hf://private/output"
}}
bootstrap_hf_cli() {{ :; }}
hf_cli() {{
  printf '%s\\n' "$*" >>"$CALL_LOG"
  if [[ "$*" == buckets\\ list* ]]; then
    printf '[]'
  fi
}}
verify_source_bundle() {{ :; }}
fetch_slice_inputs() {{ :; }}
run_runtime_preflight() {{
  FINALIZE_READY="true"
  return 37
}}
runtime_preflight_marker_is_valid() {{ return 1; }}
start_telemetry() {{ :; }}
run_selector() {{ :; }}
main
"""

    completed = subprocess.run(
        ["bash", "-c", probe],
        cwd=ROOT,
        env={"CALL_LOG": str(call_log)},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 37, completed.stderr
    calls = call_log.read_text(encoding="utf-8")
    assert "buckets sync /dstack/output" not in calls


def test_selector_configs_pin_exact_submitted_source_bundle() -> None:
    script = RUNNER.read_bytes()
    files = {RUNNER.as_posix(): script}
    source_root = Path("src/trec_video_2026")
    for path in source_root.rglob("*.py"):
        files[path.as_posix()] = path.read_bytes()
    digest = hashlib.sha256()
    for relative, payload in sorted(files.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    expected_source = digest.hexdigest()

    revisions = set()
    attempt_ids = set()
    for path in CONFIGS.values():
        text = path.read_text(encoding="utf-8")
        assert f"EXPECTED_SOURCE_BUNDLE_SHA256={expected_source}" in text
        match = re.search(r"SUBMITTED_CODE_REVISION=([0-9a-f]{40})", text)
        assert match is not None
        revisions.add(match.group(1))
        attempt_match = re.search(
            r"EXECUTION_ATTEMPT_ID=([0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
            r"[89ab][0-9a-f]{3}-[0-9a-f]{12})",
            text,
        )
        assert attempt_match is not None
        attempt_ids.add(attempt_match.group(1))
    assert len(revisions) == 1
    assert revisions == {"f204a490f935d2dec1643c16e11bf0c784b7b495"}
    assert len(attempt_ids) == len(CONFIGS)
