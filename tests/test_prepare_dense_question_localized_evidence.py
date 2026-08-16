from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import hashlib
import subprocess

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "prepare_dense_question_localized_evidence.py"
spec = importlib.util.spec_from_file_location(
    "prepare_dense_question_localized_evidence", SCRIPT
)
if spec is None or spec.loader is None:
    raise ImportError(f"cannot load preparation module from {SCRIPT}")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _source_fixture(tmp_path: Path, *, insufficient_rows: int = 1) -> dict[str, Path]:
    root = tmp_path / "run"
    root.mkdir()
    protocol = root / "protocol.json"
    maps = root / "maps.json"
    audit = root / "audit.json"
    rows = []
    for index in range(insufficient_rows):
        video_id = f"video-{index + 1}"
        rows.append(
            {
                "audit_index": index + 1,
                "video_id": video_id,
                "question": "What happens next?",
                "role": "insufficient_or_ambiguous",
                "verdict": "insufficient_or_ambiguous",
                "evidence_line_numbers": [1],
                "exposure_flags": {"grounded_v2_24": False},
                "frozen_map": {
                    "lines": [{"line_number": 1, "claim": "SCENE | 00:00-00:03 | A person moves."}],
                },
            }
        )
    _write_json(protocol, {"rows": rows})
    _write_json(maps, [{"video_id": row["video_id"], "map": "SCENE | 00:00-00:03 | A person moves."} for row in rows])
    _write_json(audit, {"rows": rows})
    return {
        "root": root,
        "source_protocol": protocol,
        "source_maps": maps,
        "corrected_audit": audit,
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _completed_fixture(tmp_path: Path, *, visual_targets: int = 2, controls: int = 2) -> dict[str, Path]:
    root = tmp_path / "completed"
    root.mkdir()
    protocol = root / "protocol.json"
    maps = root / "maps.json"
    audit = root / "audit.json"
    protocol_rows = []
    map_rows = []
    audit_rows = []
    source = root / "source.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=blue:s=160x120:r=10",
            "-t", "4", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
            "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", str(source),
        ],
        check=True,
        capture_output=True,
    )
    source_sha = _sha(source)
    total = visual_targets + controls
    for index in range(total):
        video_id = f"video-{index + 1}"
        role = "insufficient_or_ambiguous" if index < visual_targets else "supported_control"
        protocol_rows.append(
            {
                "video_id": video_id,
                "question": "What does the person do?",
                "source_path": str(source),
                "source_sha256": source_sha,
                "duration_seconds": 4.0,
            }
        )
        map_rows.append({"video_id": video_id, "map": "SCENE | 00:00.000-00:04.000 | A person moves.\nSTATE | 00:00.000-00:04.000 | person | standing."})
        audit_rows.append(
            {
                "audit_index": index + 1,
                "video_id": video_id,
                "question": "What does the person do?",
                "role": role,
                "verdict": role,
                "evidence_line_numbers": [1],
                "exposure_flags": {"grounded_v2_24": index == 0},
            }
        )
    _write_json(protocol, {"rows": protocol_rows})
    _write_json(maps, map_rows)
    _write_json(audit, {"rows": audit_rows})
    return {"root": root, "source_protocol": protocol, "source_maps": maps, "corrected_audit": audit}


def _complete_manual_audit(fixture: dict[str, Path]) -> None:
    manual = fixture["root"] / "manual-modality-audit.json"
    payload = json.loads(manual.read_text())
    for row in payload["rows"]:
        row.update(
            {
                "modality": "brief_visual_action" if row["role"] == "target" else None,
                "rationale": "The action is directly visible in the localized clip.",
                "map_line_numbers": [1],
                "window_start_seconds": 0.0,
                "window_end_seconds": 2.0,
                "include": True,
                "visually_eligible": row["role"] == "control",
            }
        )
    manual.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_modality_template_requires_committed_treatment_freeze(tmp_path: Path) -> None:
    fixture = _source_fixture(tmp_path)
    with pytest.raises(ValueError, match="treatment freeze"):
        prep.prepare_modality_audit(**fixture)


def test_freeze_panel_requires_all_17_insufficiency_rows_classified(tmp_path: Path) -> None:
    fixture = _source_fixture(tmp_path, insufficient_rows=17)
    prep.freeze_treatment(**fixture)
    prep.prepare_modality_audit(**fixture)
    manual_audit = fixture["root"] / "manual-modality-audit.json"
    payload = json.loads(manual_audit.read_text())
    for row in payload["rows"][:16]:
        row.update(
            {
                "modality": "brief_visual_action",
                "rationale": "The action is visible in the localized clip.",
                "map_line_numbers": [1],
                "window_start_seconds": 0.0,
                "window_end_seconds": 2.0,
                "include": True,
            }
        )
    manual_audit.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="17"):
        prep.freeze_panel(**fixture)


def test_selection_uses_first_eight_visual_targets_and_first_four_controls(tmp_path: Path) -> None:
    fixture = _completed_fixture(tmp_path, visual_targets=9, controls=5)
    prep.freeze_treatment(**fixture)
    prep.prepare_modality_audit(**fixture)
    _complete_manual_audit(fixture)
    prep.freeze_panel(**fixture)
    selection = json.loads((fixture["root"] / "selection-freeze.json").read_text())
    assert selection["target_count"] == 8
    assert selection["control_count"] == 4
    assert selection["planned_call_count"] == 24
    assert selection["selected_rows"] == list(range(1, 9)) + [10, 11, 12, 13]


def test_both_arms_bind_the_same_clip_bytes(tmp_path: Path) -> None:
    fixture = _completed_fixture(tmp_path, visual_targets=2, controls=2)
    prep.freeze_treatment(**fixture)
    prep.prepare_modality_audit(**fixture)
    _complete_manual_audit(fixture)
    prep.freeze_panel(**fixture)
    rows = json.loads((fixture["root"] / "media-manifest.json").read_text())["rows"]
    assert all(row["arms"]["4fps"]["clip_sha256"] == row["arms"]["12fps"]["clip_sha256"] for row in rows)
    assert all(row["clip_path"].endswith(".mp4") for row in rows)


def test_panel_rejects_invalid_window_before_media_generation(tmp_path: Path) -> None:
    fixture = _completed_fixture(tmp_path)
    prep.freeze_treatment(**fixture)
    prep.prepare_modality_audit(**fixture)
    _complete_manual_audit(fixture)
    manual = fixture["root"] / "manual-modality-audit.json"
    payload = json.loads(manual.read_text())
    payload["rows"][0]["window_end_seconds"] = 1.5
    manual.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="between 2 and 6"):
        prep.freeze_panel(**fixture)


def test_provider_rows_do_not_project_private_labels(tmp_path: Path) -> None:
    fixture = _completed_fixture(tmp_path)
    prep.freeze_treatment(**fixture)
    prep.prepare_modality_audit(**fixture)
    _complete_manual_audit(fixture)
    prep.freeze_panel(**fixture)
    rows = json.loads((fixture["root"] / "provider-rows.json").read_text())["rows"]
    assert all("direct_answer" not in row and "reference" not in row for row in rows)
    assert all("exposure_flags" not in row and "modality" not in row for row in rows)


def test_selection_uses_only_brief_visual_action_targets(tmp_path: Path) -> None:
    fixture = _completed_fixture(tmp_path, visual_targets=9, controls=4)
    prep.freeze_treatment(**fixture)
    prep.prepare_modality_audit(**fixture)
    _complete_manual_audit(fixture)
    manual = fixture["root"] / "manual-modality-audit.json"
    payload = json.loads(manual.read_text())
    payload["rows"][0]["modality"] = "fine_spatial_detail"
    manual.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    prep.freeze_panel(**fixture)
    selection = json.loads((fixture["root"] / "selection-freeze.json").read_text())
    assert selection["selected_rows"][:8] == list(range(2, 10))


def test_provider_rows_project_only_selected_map_lines(tmp_path: Path) -> None:
    fixture = _completed_fixture(tmp_path)
    prep.freeze_treatment(**fixture)
    prep.prepare_modality_audit(**fixture)
    _complete_manual_audit(fixture)
    prep.freeze_panel(**fixture)
    rows = json.loads((fixture["root"] / "provider-rows.json").read_text())["rows"]
    assert all(row["map_lines"] == [{"line_number": 1, "text": "SCENE | 00:00.000-00:04.000 | A person moves."}] for row in rows)


def test_minimum_panel_failure_writes_frozen_zero_call_feasibility(tmp_path: Path) -> None:
    fixture = _completed_fixture(tmp_path, visual_targets=1, controls=1)
    prep.freeze_treatment(**fixture)
    prep.prepare_modality_audit(**fixture)
    _complete_manual_audit(fixture)
    with pytest.raises(ValueError, match="fewer than 2"):
        prep.freeze_panel(**fixture)
    feasibility = json.loads((fixture["root"] / "feasibility-freeze.json").read_text())
    assert feasibility["status"] == "frozen"
    assert feasibility["provider_calls_made"] == 0
    assert not (fixture["root"] / "media-manifest.json").exists()


def test_audit_exposure_drift_is_rejected_before_clip_generation(tmp_path: Path) -> None:
    fixture = _completed_fixture(tmp_path)
    prep.freeze_treatment(**fixture)
    prep.prepare_modality_audit(**fixture)
    _complete_manual_audit(fixture)
    manual = fixture["root"] / "manual-modality-audit.json"
    payload = json.loads(manual.read_text())
    payload["rows"][0]["exposure_flags"]["grounded_v2_24"] = not payload["rows"][0]["exposure_flags"]["grounded_v2_24"]
    manual.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="exposure"):
        prep.freeze_panel(**fixture)


def test_authorize_rejects_manual_audit_hash_drift(tmp_path: Path) -> None:
    fixture = _completed_fixture(tmp_path)
    prep.freeze_treatment(**fixture)
    prep.prepare_modality_audit(**fixture)
    _complete_manual_audit(fixture)
    prep.freeze_panel(**fixture)
    manual = fixture["root"] / "manual-modality-audit.json"
    payload = json.loads(manual.read_text())
    payload["rows"][0]["rationale"] = "changed after panel freeze"
    manual.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="manual modality audit hash"):
        prep.authorize(**fixture)


def test_mid_panel_failure_leaves_resumable_incomplete_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _completed_fixture(tmp_path)
    prep.freeze_treatment(**fixture)
    prep.prepare_modality_audit(**fixture)
    _complete_manual_audit(fixture)
    original = prep._make_clip
    calls = 0

    def fail_on_second(root: Path, row: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic clip failure")
        return original(root, row)

    monkeypatch.setattr(prep, "_make_clip", fail_on_second)
    with pytest.raises(RuntimeError, match="synthetic clip failure"):
        prep.freeze_panel(**fixture)
    assert (fixture["root"] / "panel-preparation-incomplete.json").exists()
    assert not (fixture["root"] / "media-manifest.json").exists()
    assert not (fixture["root"] / "clips").exists()
    monkeypatch.setattr(prep, "_make_clip", original)
    prep.freeze_panel(**fixture)
    assert (fixture["root"] / "media-manifest.json").exists()
    assert len(list((fixture["root"] / "clips").glob("*.mp4"))) == 4
