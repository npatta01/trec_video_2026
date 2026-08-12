from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from trec_video_2026.vqa_frontier_phase0_audit import (
    build_audit_packet,
    freeze_pilot_audit,
    freeze_screen_audit,
    validate_audit_packet,
)
from trec_video_2026.vqa_frontier_phase0_panel import (
    CATEGORIES,
    CONDITIONS,
    PHASE0_RUN_ID,
    QWEN_CONDITION_ID,
    REFERENCE_POLICY,
    SELECTION_POLICY,
)
from trec_video_2026.vqa_frontier_phase0_runner import (
    authorize_remaining,
    freeze_pilot,
    freeze_screen,
    prepare_run,
    run_pilot,
    run_screen_row,
)
from trec_video_2026.vqa_submission import canonical_sha256
from trec_video_2026.vqa_experiments.temporal_bottleneck import freeze_json_artifact


SENTINEL_QUESTION = "sentinel private question"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture_media(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:r=4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=8000",
            "-t",
            "2",
            "-c:v",
            "libvpx-vp9",
            "-c:a",
            "libopus",
            "-b:v",
            "80k",
            "-b:a",
            "16k",
            str(path),
        ],
        check=True,
    )


def _map_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "perceived_duration_seconds": 2.0,
        "global_description": "A blue fixture frame.",
        "entities": [],
        "events": [
            {
                "event_id": "E1",
                "start_seconds": 0.0,
                "end_seconds": 1.0,
                "keyframe_seconds": [0.5],
                "description": "The fixture is visible.",
                "entity_ids": [],
                "actions": ["remain visible"],
                "relations": [],
                "state_before": "visible",
                "state_change": "none",
                "state_after": "visible",
                "visual_observations": ["blue frame"],
                "speech_observations": [],
                "sound_observations": [],
                "visible_text": [],
                "confidence": 0.9,
                "uncertainties": [],
            },
            {
                "event_id": "E2",
                "start_seconds": 1.0,
                "end_seconds": 2.0,
                "keyframe_seconds": [1.5],
                "description": "The fixture remains visible.",
                "entity_ids": [],
                "actions": ["remain visible"],
                "relations": [],
                "state_before": "visible",
                "state_change": "none",
                "state_after": "visible",
                "visual_observations": ["blue frame"],
                "speech_observations": [],
                "sound_observations": [],
                "visible_text": [],
                "confidence": 0.9,
                "uncertainties": [],
            },
        ],
        "global_uncertainties": [],
    }


def _answer_payload(*, guided: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "top1_answer": "The fixture remains visible.",
        "claims": [
            {
                "claim_id": "C1",
                "text": "The fixture remains visible.",
                "supporting_spans": [
                    {"start_seconds": 0.0, "end_seconds": 1.0}
                ],
                "supporting_modalities": ["video"],
                "map_event_ids": ["E1"] if guided else [],
            }
        ],
        "could_be_wrong_because": [],
    }


class LocalTransport:
    provider_name = "openrouter"
    response_format_policy = "prompt-json-local-validation-v3-exact-fence"

    def __init__(self) -> None:
        from trec_video_2026.vqa_frontier_phase0_runner import ANSWER_POLICY, MAP_POLICY

        self.call_count = 0
        self.policy_identity: dict[str, object] = {
            "neutral_map": MAP_POLICY,
            "direct_answer": ANSWER_POLICY,
            "map_guided_answer": ANSWER_POLICY,
        }

    def upload(self, path: Path, *, mime_type: str) -> object:
        from trec_video_2026.vqa_submission_gemini import UploadedFileRecord

        return UploadedFileRecord(
            file_id=f"fixture-{self.call_count}",
            mime_type=mime_type,
            sha256=_sha256(path),
        )

    def delete(self, file_id: str) -> None:
        return None

    def estimate_cost_usd(self, provider: str, row: object) -> float:
        return 0.001

    def generate(self, *, model: str, parts: list[object], schema: object) -> object:
        from trec_video_2026.vqa_submission_gemini import GenerationResult, TokenUsage

        self.call_count += 1
        prompt = str(parts[1])
        if "frontier-neutral-map-v" in prompt:
            payload = _map_payload()
        else:
            payload = _answer_payload(guided="Neutral map: supplied" in prompt)
        return GenerationResult(
            payload=payload,
            raw_response=json.dumps(payload, separators=(",", ":")),
            usage=TokenUsage(input_tokens=100, output_tokens=50),
            provider_metadata={
                "model": "google/gemini-3.1-pro-preview",
                "reported_cost_usd": 0.001,
            },
        )


def _panel_row(
    ordinal: int,
    phase: str,
    video_id: str,
    media_sha256: str,
    size_bytes: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ordinal": ordinal,
        "phase": phase,
        "video_id": video_id,
        "question": SENTINEL_QUESTION,
        "category": "temporal",
        "evidence_topology": "single",
        "selection_rationale": "synthetic local fixture",
        "media_relative_path": f"fixture/{video_id}.webm",
        "media_sha256": media_sha256,
        "size_bytes": size_bytes,
        "duration_seconds": 2.0,
        "has_audio": True,
        "has_video": True,
        "condition_ids": ["neutral_map"]
        if phase == "pilot"
        else list(CONDITIONS),
    }
    if phase == "screen":
        row["qwen_condition_id"] = QWEN_CONDITION_ID
        row["qwen_status"] = "passed"
    return row


def _write_prepared(tmp_path: Path, *, phase: str) -> Path:
    source = tmp_path / "source.webm"
    _write_fixture_media(source)
    media_sha256 = _sha256(source)
    rows = [_panel_row(1, "pilot", "pilot-video", media_sha256, source.stat().st_size)]
    rows.extend(
        _panel_row(
            index + 2,
            "screen",
            f"screen-video-{index + 1}",
            media_sha256,
            source.stat().st_size,
        )
        for index in range(7)
    )
    for index, row in enumerate(rows[1:]):
        row["category"] = CATEGORIES[index]
    panel = {
        "schema_version": 1,
        "run_id": PHASE0_RUN_ID,
        "selection_policy": SELECTION_POLICY,
        "reference_policy": REFERENCE_POLICY,
        "conditions": list(CONDITIONS),
        "pilot_count": 1,
        "screen_count": 7,
        "bindings": {
            "source_panel_sha256": "b" * 64,
            "qwen_predictions_sha256": "c" * 64,
            "selected_ordered_ids_sha256": canonical_sha256(
                [row["video_id"] for row in rows]
            ),
            "source_media_sha256": canonical_sha256(
                [
                    {"video_id": row["video_id"], "source_sha256": media_sha256}
                    for row in rows
                ]
            ),
            "selection_policy": SELECTION_POLICY,
        },
        "examples": rows,
    }
    panel_path = tmp_path / "panel.json"
    panel_path.write_text(json.dumps(panel), encoding="utf-8")
    prepared = prepare_run(panel_path, tmp_path / "run")
    fixture_root = prepared.path / "fixture-media"
    fixture_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        shutil.copyfile(source, fixture_root / f"{row['video_id']}.webm")

    run_pilot(prepared, LocalTransport())
    freeze_pilot(prepared)
    if phase == "pilot":
        return prepared.path

    pilot_audit = tmp_path / "pilot-audit"
    pilot_audit.mkdir()
    pilot_artifact = pilot_audit / "audit.json"
    pilot_artifact.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": prepared.run_id,
                "panel_sha256": prepared.panel_sha256,
                "phase": "pilot",
                "decision": "usable_for_screen",
            }
        ),
        encoding="utf-8",
    )
    pilot_freeze = freeze_json_artifact(
        pilot_artifact,
        bindings={
            "run_id": prepared.run_id,
            "panel_sha256": prepared.panel_sha256,
            "phase": "pilot",
            "decision": "usable_for_screen",
        },
    )
    (pilot_audit / "freeze.json").write_text(json.dumps(pilot_freeze), encoding="utf-8")
    from trec_video_2026.vqa_frontier_phase0_runner import authorize_screen

    authorize_screen(prepared, pilot_audit / "freeze.json")
    run_screen_row(
        prepared,
        ordinal=1,
        transport_factory=lambda _condition: LocalTransport(),
    )
    authorize_remaining(prepared)
    for ordinal in range(2, 8):
        run_screen_row(
            prepared,
            ordinal=ordinal,
            transport_factory=lambda _condition: LocalTransport(),
        )
    freeze_screen(prepared)
    return prepared.path


def _map_verdict(
    video_id: str,
    *,
    usable: str = "usable_for_screen",
    answer_critical_evidence: str = "not_assessed",
) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "event_checks": [
            {
                "event_id": "E1",
                "status": "confirmed",
                "boundary_status": "confirmed",
                "notes": "fixture event checked",
            },
            {
                "event_id": "E2",
                "status": "confirmed",
                "boundary_status": "confirmed",
                "notes": "fixture event checked",
            },
        ],
        "coverage": "complete",
        "entity_identity": "confirmed",
        "ordering": "confirmed",
        "visible_text": "unverifiable",
        "speech_and_sound": "unverifiable",
        "answer_critical_evidence": answer_critical_evidence,
        "overall_decision": usable,
        "notes": "fixture map audit",
    }


def _answer_verdict(video_id: str, condition_id: str) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "condition_id": condition_id,
        "question": SENTINEL_QUESTION,
        "frozen_answer": "The fixture remains visible.",
        "claim_checks": [
            {
                "claim_id": "C1",
                "native_span_support": "supported",
                "guided_map_event_citation": "valid"
                if condition_id == "map_guided_answer"
                else "not_applicable",
                "notes": "fixture claim checked",
            }
        ],
        "unsupported_detail": False,
        "failure_causes": [],
        "supported_top1": True,
        "notes": "fixture answer audit",
    }


def _write_verdicts(packet: Any, *, complete: bool = True) -> None:
    rows = json.loads(packet.blind_manifest.read_text(encoding="utf-8"))["rows"]
    phase = json.loads(packet.packet_manifest.read_text(encoding="utf-8"))["phase"]
    map_rows = [
        _map_verdict(
            row["video_id"],
            answer_critical_evidence=(
                "present" if phase == "screen" else "not_assessed"
            ),
        )
        for row in rows
    ]
    answer_rows = []
    if phase == "screen":
        answer_rows = [
            _answer_verdict(row["video_id"], condition)
            for row in rows
            for condition in ("direct_answer", "map_guided_answer")
        ]
    if not complete:
        map_rows[0]["event_checks"] = map_rows[0]["event_checks"][:-1]
    packet.map_verdicts.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in map_rows),
        encoding="utf-8",
    )
    packet.answer_verdicts.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in answer_rows),
        encoding="utf-8",
    )


def test_blind_manifest_contains_map_but_no_question_or_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import trec_video_2026.vqa_frontier_phase0_audit as audit_module

    monkeypatch.setattr(audit_module, "CANONICAL_DATA_ROOT", tmp_path)
    prepared = _write_prepared(tmp_path, phase="screen")
    packet = build_audit_packet(prepared, phase="screen", output=tmp_path / "audit")
    payload = json.loads(packet.blind_manifest.read_text(encoding="utf-8"))
    rendered = json.dumps(payload).lower()
    assert SENTINEL_QUESTION not in rendered
    assert "top1_answer" not in rendered
    assert "reference" not in rendered
    assert "events" in rendered


def test_real_media_manifest_binds_pts_hashes_and_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import trec_video_2026.vqa_frontier_phase0_audit as audit_module

    monkeypatch.setattr(audit_module, "CANONICAL_DATA_ROOT", tmp_path)
    prepared = _write_prepared(tmp_path, phase="pilot")
    packet = build_audit_packet(prepared, phase="pilot", output=tmp_path / "audit")
    manifest = json.loads(packet.private_manifest.read_text(encoding="utf-8"))
    row = manifest["rows"][0]
    assert row["source_media_sha256"]
    assert row["native_media_path"]
    assert row["ffmpeg_version"].startswith("ffmpeg version")
    assert len(row["overview_frames"]) == 6
    assert all(frame["desired_pts"] >= 0 for frame in row["overview_frames"])
    assert all(frame["actual_pts"] is not None for frame in row["overview_frames"])
    assert all(len(frame["frame_sha256"]) == 64 for frame in row["overview_frames"])
    assert len(row["event_keyframes"]) == 2


def test_build_preserves_six_bound_overview_source_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trec_video_2026.vqa_frontier_phase0_audit as audit_module

    monkeypatch.setattr(audit_module, "CANONICAL_DATA_ROOT", tmp_path)
    prepared = _write_prepared(tmp_path, phase="pilot")

    packet = build_audit_packet(
        prepared, phase="pilot", output=tmp_path / "audit"
    )

    private = json.loads(packet.private_manifest.read_text(encoding="utf-8"))
    frames = private["rows"][0]["overview_frames"]
    assert [frame["path"] for frame in frames] == [
        f"rows/pilot-video/overview-frames/{index:04d}.jpg"
        for index in range(1, 7)
    ]
    assert all((packet.root / frame["path"]).is_file() for frame in frames)


def test_nonzero_event_frame_records_source_timeline_pts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trec_video_2026.vqa_frontier_phase0_audit as audit_module

    monkeypatch.setattr(audit_module, "CANONICAL_DATA_ROOT", tmp_path)
    prepared = _write_prepared(tmp_path, phase="pilot")

    packet = build_audit_packet(
        prepared, phase="pilot", output=tmp_path / "audit"
    )

    private = json.loads(packet.private_manifest.read_text(encoding="utf-8"))
    frame = next(
        item
        for item in private["rows"][0]["event_keyframes"]
        if item["event_id"] == "E2"
    )
    assert frame["desired_pts"] == 1.5
    assert frame["actual_pts"] == pytest.approx(1.5, abs=0.01)


def test_screen_audit_requires_every_map_event_and_answer_claim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import trec_video_2026.vqa_frontier_phase0_audit as audit_module

    monkeypatch.setattr(audit_module, "CANONICAL_DATA_ROOT", tmp_path)
    prepared = _write_prepared(tmp_path, phase="screen")
    packet = build_audit_packet(prepared, phase="screen", output=tmp_path / "audit")
    _write_verdicts(packet, complete=False)
    with pytest.raises(ValueError, match="every neutral-map event"):
        freeze_screen_audit(packet.root)


def test_freeze_binds_result_media_manifests_and_verdicts_immutably(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import trec_video_2026.vqa_frontier_phase0_audit as audit_module

    monkeypatch.setattr(audit_module, "CANONICAL_DATA_ROOT", tmp_path)
    prepared = _write_prepared(tmp_path, phase="pilot")
    packet = build_audit_packet(prepared, phase="pilot", output=tmp_path / "audit")
    _write_verdicts(packet)
    freeze = freeze_pilot_audit(packet.root)
    assert freeze["status"] == "frozen"
    assert len(freeze["artifact_sha256"]) == 64
    assert len(freeze["bindings_sha256"]) == 64
    assert set(freeze["bindings"]) == {"run_id", "panel_sha256", "phase", "decision"}
    assert json.loads((packet.root / "audit.json").read_text())["decision"] == "usable_for_screen"
    with pytest.raises(FileExistsError):
        freeze_pilot_audit(packet.root)

    assert validate_audit_packet(packet.root)["frozen"] is True


@pytest.mark.parametrize("mutation", ["delete", "replace"])
def test_validate_frozen_audit_rejects_changed_evidence_jpeg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    import trec_video_2026.vqa_frontier_phase0_audit as audit_module

    monkeypatch.setattr(audit_module, "CANONICAL_DATA_ROOT", tmp_path)
    prepared = _write_prepared(tmp_path, phase="pilot")
    packet = build_audit_packet(
        prepared, phase="pilot", output=tmp_path / "audit"
    )
    _write_verdicts(packet)
    freeze_pilot_audit(packet.root)
    private = json.loads(packet.private_manifest.read_text(encoding="utf-8"))
    frame = packet.root / private["rows"][0]["overview_frames"][0]["path"]
    if mutation == "delete":
        frame.unlink()
    else:
        frame.write_bytes(b"replacement-jpeg")

    with pytest.raises(ValueError, match="evidence media"):
        validate_audit_packet(packet.root)


def test_validate_frozen_audit_rejects_symlinked_evidence_jpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trec_video_2026.vqa_frontier_phase0_audit as audit_module

    monkeypatch.setattr(audit_module, "CANONICAL_DATA_ROOT", tmp_path)
    prepared = _write_prepared(tmp_path, phase="pilot")
    packet = build_audit_packet(
        prepared, phase="pilot", output=tmp_path / "audit"
    )
    _write_verdicts(packet)
    freeze_pilot_audit(packet.root)
    private = json.loads(packet.private_manifest.read_text(encoding="utf-8"))
    frame = packet.root / private["rows"][0]["overview_frames"][0]["path"]
    replacement = packet.root / "replacement.jpg"
    shutil.copyfile(frame, replacement)
    frame.unlink()
    frame.symlink_to(replacement)

    with pytest.raises(ValueError, match="symlink|unsafe"):
        validate_audit_packet(packet.root)


def test_validate_rejects_missing_event_keyframe_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trec_video_2026.vqa_frontier_phase0_audit as audit_module

    monkeypatch.setattr(audit_module, "CANONICAL_DATA_ROOT", tmp_path)
    prepared = _write_prepared(tmp_path, phase="pilot")
    packet = build_audit_packet(
        prepared, phase="pilot", output=tmp_path / "audit"
    )
    private = json.loads(packet.private_manifest.read_text(encoding="utf-8"))
    media = private["rows"][0]["media"]
    media["event_keyframes"] = media["event_keyframes"][:-1]
    packet_manifest = json.loads(
        packet.packet_manifest.read_text(encoding="utf-8")
    )
    packet_manifest["rows"][0]["media_sha256"] = canonical_sha256(media)
    packet.private_manifest.write_text(json.dumps(private), encoding="utf-8")
    packet.packet_manifest.write_text(
        json.dumps(packet_manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="event keyframe records"):
        validate_audit_packet(packet.root)


def test_validate_rejects_missing_answer_citation_frame_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trec_video_2026.vqa_frontier_phase0_audit as audit_module

    monkeypatch.setattr(audit_module, "CANONICAL_DATA_ROOT", tmp_path)
    prepared = _write_prepared(tmp_path, phase="screen")
    packet = build_audit_packet(
        prepared, phase="screen", output=tmp_path / "audit"
    )
    private = json.loads(packet.private_manifest.read_text(encoding="utf-8"))
    media = private["rows"][0]["media"]
    media["answer_citations"]["direct_answer"] = media[
        "answer_citations"
    ]["direct_answer"][:-1]
    packet_manifest = json.loads(
        packet.packet_manifest.read_text(encoding="utf-8")
    )
    packet_manifest["rows"][0]["media_sha256"] = canonical_sha256(media)
    packet.private_manifest.write_text(json.dumps(private), encoding="utf-8")
    packet.packet_manifest.write_text(
        json.dumps(packet_manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="answer citation records"):
        validate_audit_packet(packet.root)


def test_validate_rejects_evidence_pts_far_from_requested_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trec_video_2026.vqa_frontier_phase0_audit as audit_module

    monkeypatch.setattr(audit_module, "CANONICAL_DATA_ROOT", tmp_path)
    prepared = _write_prepared(tmp_path, phase="pilot")
    packet = build_audit_packet(
        prepared, phase="pilot", output=tmp_path / "audit"
    )
    private = json.loads(packet.private_manifest.read_text(encoding="utf-8"))
    media = private["rows"][0]["media"]
    media["event_keyframes"][0]["actual_pts"] = 1.9
    packet_manifest = json.loads(
        packet.packet_manifest.read_text(encoding="utf-8")
    )
    packet_manifest["rows"][0]["media_sha256"] = canonical_sha256(media)
    packet.private_manifest.write_text(json.dumps(private), encoding="utf-8")
    packet.packet_manifest.write_text(
        json.dumps(packet_manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="timestamp"):
        validate_audit_packet(packet.root)


def test_pilot_usable_decision_requires_complete_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trec_video_2026.vqa_frontier_phase0_audit as audit_module

    monkeypatch.setattr(audit_module, "CANONICAL_DATA_ROOT", tmp_path)
    prepared = _write_prepared(tmp_path, phase="pilot")
    packet = build_audit_packet(
        prepared, phase="pilot", output=tmp_path / "audit"
    )
    _write_verdicts(packet)
    verdict = json.loads(packet.map_verdicts.read_text(encoding="utf-8"))
    verdict["coverage"] = "partial"
    packet.map_verdicts.write_text(json.dumps(verdict) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="usable_for_screen.*coverage"):
        freeze_pilot_audit(packet.root)


@pytest.mark.parametrize(
    "field",
    [
        "event_status",
        "boundary_status",
        "entity_identity",
        "ordering",
        "visible_text",
        "speech_and_sound",
    ],
)
def test_pilot_usable_decision_rejects_any_contradicted_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    import trec_video_2026.vqa_frontier_phase0_audit as audit_module

    monkeypatch.setattr(audit_module, "CANONICAL_DATA_ROOT", tmp_path)
    prepared = _write_prepared(tmp_path, phase="pilot")
    packet = build_audit_packet(
        prepared, phase="pilot", output=tmp_path / "audit"
    )
    _write_verdicts(packet)
    verdict = json.loads(packet.map_verdicts.read_text(encoding="utf-8"))
    if field == "event_status":
        verdict["event_checks"][0]["status"] = "contradicted"
    elif field == "boundary_status":
        verdict["event_checks"][0]["boundary_status"] = "contradicted"
    else:
        verdict[field] = "contradicted"
    packet.map_verdicts.write_text(json.dumps(verdict) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="usable_for_screen.*contradiction"):
        freeze_pilot_audit(packet.root)


def test_output_symlink_escape_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import trec_video_2026.vqa_frontier_phase0_audit as audit_module

    monkeypatch.setattr(audit_module, "CANONICAL_DATA_ROOT", tmp_path / "private")
    audit_module.CANONICAL_DATA_ROOT.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = audit_module.CANONICAL_DATA_ROOT / "escape"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="canonical private data root|symlink"):
        build_audit_packet(tmp_path / "missing-prepared", phase="pilot", output=link / "audit")


def test_build_rejects_symlinked_fixture_media_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trec_video_2026.vqa_frontier_phase0_audit as audit_module

    monkeypatch.setattr(audit_module, "CANONICAL_DATA_ROOT", tmp_path)
    prepared = _write_prepared(tmp_path, phase="pilot")
    fixture_root = prepared / "fixture-media"
    moved_fixture_root = tmp_path / "moved-fixture-media"
    fixture_root.rename(moved_fixture_root)
    fixture_root.symlink_to(moved_fixture_root, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        build_audit_packet(
            prepared, phase="pilot", output=tmp_path / "audit"
        )
