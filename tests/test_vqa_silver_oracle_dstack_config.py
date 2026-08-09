from __future__ import annotations

import hashlib
import re
from pathlib import Path

from trec_video_2026.vqa_experiments.qwen3_omni_structured import (
    MODEL_ID,
    MODEL_REVISION,
    RUNTIME_IMAGE,
)

RUN_ID = "vqa-silver-oracle-election-20260809-a1"
CONFIGS = {
    "canary": Path("dstack/vqa-silver-oracle-selector-canary.dstack.yml"),
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
        expected_duration = "45m" if slice_id == "canary" else "80m"
        assert "type: task" in text
        assert f"name: vqa-silver-selector-{slice_id}-20260809-a1" in text
        assert f"image: {RUNTIME_IMAGE}" in text
        assert f"SELECTOR_SLICE={slice_id}" in text
        assert f"MODEL_ID={MODEL_ID}" in text
        assert f"MODEL_REVISION={MODEL_REVISION}" in text
        assert "HF_TOKEN=${{ secrets.hf_token }}" in text
        assert "ports:" not in text
        assert "spot_policy: on-demand" in text
        assert "backends: [runpod, vastai]" in text
        assert "name: [H100, H200, B200, RTXPRO6000WS]" in text
        assert "memory: 90GB.." in text
        assert "max_price: 4.75" in text
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
    assert 'readarray -t requested_media' in text
    assert 'include_args+=(--include "$basename")' in text
    assert 'hf_cli buckets sync "$MEDIA_URI" "$INPUT_ROOT/media"' in text
    assert 'hf_cli buckets sync "$CONTROLLER_URI" "$INPUT_ROOT/controller"' in text
    assert 'sha256sum -c SHA256SUMS' in text
    assert 'chmod -R a-w "$INPUT_ROOT"' in text
    assert "qwen3_omni_structured" in text
    assert 'hf_cli buckets sync "$OUTPUT_DIR" "$REMOTE_OUTPUT_URI"' in text
    assert 'readonly WORKLOAD_STDOUT="/dstack/workload.stdout.json"' in text
    assert '>"$WORKLOAD_STDOUT"' in text
    assert '>"$OUTPUT_DIR/workload.stdout.json"' not in text
    assert "verify_source_bundle" in text
    assert "source-bundle-identity.json" in text
    assert "--delete" not in text
    assert "dstack logs -d" not in text


def test_selector_slice_request_names_and_output_prefixes_are_disjoint() -> None:
    runner = RUNNER.read_text(encoding="utf-8")

    assert 'REQUEST_NAME="selector-canary.jsonl"' in runner
    for index in range(4):
        chunk = f"chunk-{index:02d}"
        assert f'{chunk}) REQUEST_NAME="selector-{chunk}.jsonl"' in runner
    assert 'selector/$SELECTOR_SLICE' in runner


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
    for path in CONFIGS.values():
        text = path.read_text(encoding="utf-8")
        assert f"EXPECTED_SOURCE_BUNDLE_SHA256={expected_source}" in text
        match = re.search(r"SUBMITTED_CODE_REVISION=([0-9a-f]{40})", text)
        assert match is not None
        revisions.add(match.group(1))
    assert len(revisions) == 1
    assert revisions != {"0" * 40}
