from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts import overlay_vqa_deadline_gemini_rescue as overlay


def _canonical(value: dict[str, object]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def _write_artifact(path: Path, body: dict[str, object]) -> str:
    value = dict(body)
    value["_artifact_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.write_text(data, encoding="utf-8")
    return hashlib.sha256(data.encode()).hexdigest()


def _base_rows() -> list[dict[str, object]]:
    rows = [
        {
            "query_id": query_id,
            "video_id": f"video-{query_id}.webm",
            "status": "succeeded",
            "answer": f"base answer {query_id}",
            "elapsed_seconds": 1.0,
        }
        for query_id in range(1, 2_444)
    ]
    for query_id, reason in overlay.EXPECTED_REASON_BY_QUERY.items():
        rows[query_id - 1].update(
            status="upload_failed" if reason == "provider_upload_failed" else "failed",
            answer="",
            failure_reason=reason,
        )
    return rows


def _write_inputs(tmp_path: Path) -> tuple[Path, str, Path]:
    base_root = tmp_path / "base"
    rescue_root = tmp_path / "rescue"
    base_root.mkdir()
    rescue_root.mkdir()
    base_path = base_root / "predictions-freeze.json"
    rows = _base_rows()
    manifest_rows = [
        {
            "query_id": query_id,
            "video_id": f"video-{query_id}.webm",
            "question": f"question {query_id}",
            "media_sha256": f"{query_id:064x}"[-64:],
            "duration_seconds": float(query_id),
            "status": "ready",
        }
        for query_id in range(1, 2_444)
    ]
    prompt = "Answer from native audio and video."
    input_binding = {"path": "/private/requests.json", "sha256": "a" * 64}
    base_manifest_sha = _write_artifact(
        base_root / "manifest.json",
        {
            "schema_version": 1,
            "status": "inputs_frozen",
            "model": "gemini-3.7-flash",
            "prompt": prompt,
            "input": input_binding,
            "rows": manifest_rows,
        },
    )
    base_sha = _write_artifact(
        base_path,
        {
            "schema_version": 1,
            "status": "predictions_frozen",
            "model": "gemini-3.7-flash",
            "manifest_sha256": base_manifest_sha,
            "row_count": len(rows),
            "completed_count": 2_443 - len(overlay.RESCUE_QUERY_IDS),
            "rows": rows,
        },
    )
    rescue_rows = []
    for index, query_id in enumerate(overlay.RESCUE_QUERY_IDS):
        succeeded = index < 2
        rescue_rows.append(
            {
                "query_id": query_id,
                "video_id": f"video-{query_id}.webm",
                "status": "succeeded" if succeeded else "failed",
                "answer": f"rescued answer {query_id}" if succeeded else "",
                "elapsed_seconds": 2.0,
                "failure_reason": None if succeeded else "empty_text",
            }
        )
    rescue_path = tmp_path / "rescue.json"
    rescue_path = rescue_root / "predictions-freeze.json"
    rescue_manifest_sha = _write_artifact(
        rescue_root / "manifest.json",
        {
            "schema_version": 1,
            "status": "inputs_frozen",
            "model": "gemini-3.7-flash",
            "prompt": prompt,
            "input": input_binding,
            "base_predictions_freeze_sha256": base_sha,
            "base_manifest_sha256": base_manifest_sha,
            "selected_query_ids": list(overlay.RESCUE_QUERY_IDS),
            "model_settings": {
                "video_fps": 4.0,
                "media_resolution": "MEDIA_RESOLUTION_LOW",
                "thinking_level": "HIGH",
                "max_output_tokens": 8_192,
            },
            "rows": [
                manifest_rows[query_id - 1] | {"input_sha256": input_binding["sha256"]}
                for query_id in overlay.RESCUE_QUERY_IDS
            ],
        },
    )
    _write_artifact(
        rescue_path,
        {
            "schema_version": 1,
            "status": "predictions_frozen",
            "model": "gemini-3.7-flash",
            "manifest_sha256": rescue_manifest_sha,
            "row_count": len(rescue_rows),
            "completed_count": 2,
            "base_predictions_freeze_sha256": base_sha,
            "selected_query_ids": list(overlay.RESCUE_QUERY_IDS),
            "max_output_tokens": 8_192,
            "rows": rescue_rows,
        },
    )
    return base_path, base_sha, rescue_path


def test_overlay_replaces_only_successful_rescue_rows(tmp_path: Path) -> None:
    base_path, base_sha, rescue_path = _write_inputs(tmp_path)
    output_path = tmp_path / "overlay.json"

    result = overlay.overlay_rescue(
        base_freeze=base_path,
        rescue_freeze=rescue_path,
        output_path=output_path,
        expected_base_sha256=base_sha,
    )

    assert result["status"] == "predictions_frozen"
    assert result["row_count"] == 2_443
    assert result["rescue_success_count"] == 2
    assert result["rescue_preserved_fallback_count"] == 25
    assert result["completed_count"] == 2_443 - 27 + 2
    rows = {row["query_id"]: row for row in result["rows"]}
    first, second, third = overlay.RESCUE_QUERY_IDS[:3]
    assert rows[first]["answer"] == f"rescued answer {first}"
    assert rows[second]["answer"] == f"rescued answer {second}"
    assert rows[third]["answer"] == ""
    assert rows[third]["failure_reason"] == overlay.EXPECTED_REASON_BY_QUERY[third]
    assert result["base_predictions_freeze_sha256"] == base_sha
    assert result["rescue_predictions_freeze_sha256"] == hashlib.sha256(
        rescue_path.read_bytes()
    ).hexdigest()
    assert json.loads(output_path.read_text())["_artifact_sha256"]


@pytest.mark.parametrize(
    "mutation",
    ["wrong_base", "missing_row", "wrong_video", "tokens", "running", "nested_label"],
)
def test_overlay_rejects_invalid_source_bindings(
    tmp_path: Path, mutation: str
) -> None:
    base_path, base_sha, rescue_path = _write_inputs(tmp_path)
    if mutation == "wrong_base":
        expected = "0" * 64
    else:
        expected = base_sha
        rescue = json.loads(rescue_path.read_text())
        rescue.pop("_artifact_sha256")
        if mutation == "missing_row":
            rescue["rows"].pop()
        elif mutation == "wrong_video":
            rescue["rows"][0]["video_id"] = "wrong.webm"
        elif mutation == "tokens":
            rescue["max_output_tokens"] = 2_048
        elif mutation == "running":
            rescue["rows"][0]["status"] = "running"
        else:
            rescue["rows"][0]["metadata"] = {"ground_truth": "private"}
        _write_artifact(rescue_path, rescue)

    with pytest.raises(overlay.OverlayError):
        overlay.overlay_rescue(
            base_freeze=base_path,
            rescue_freeze=rescue_path,
            output_path=tmp_path / "out.json",
            expected_base_sha256=expected,
        )


@pytest.mark.parametrize(
    "mutation", ["prompt", "input", "question", "nested_label"]
)
def test_overlay_rejects_self_consistent_forged_rescue_manifest(
    tmp_path: Path, mutation: str
) -> None:
    base_path, base_sha, rescue_path = _write_inputs(tmp_path)
    manifest_path = rescue_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("_artifact_sha256")
    if mutation == "prompt":
        manifest["prompt"] = "changed prompt"
    elif mutation == "input":
        manifest["input"]["sha256"] = "b" * 64
    elif mutation == "question":
        manifest["rows"][0]["question"] = "changed question"
    else:
        manifest["metadata"] = {"ground_truth": "private"}
    new_manifest_sha = _write_artifact(manifest_path, manifest)
    rescue = json.loads(rescue_path.read_text())
    rescue.pop("_artifact_sha256")
    rescue["manifest_sha256"] = new_manifest_sha
    _write_artifact(rescue_path, rescue)

    with pytest.raises(overlay.OverlayError):
        overlay.overlay_rescue(
            base_freeze=base_path,
            rescue_freeze=rescue_path,
            output_path=tmp_path / "out.json",
            expected_base_sha256=base_sha,
        )
