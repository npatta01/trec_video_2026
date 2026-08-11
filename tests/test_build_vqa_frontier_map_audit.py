from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_vqa_frontier_map_audit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_vqa_frontier_map_audit", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


audit_module = _load_module()
build_audit_packet = audit_module.build_audit_packet
freeze_audit = audit_module.freeze_audit
verify_audit_freeze = audit_module.verify_audit_freeze


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_synthetic_jpeg(path: Path, label: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xd8synthetic-audit-media:" + label.encode("utf-8") + b"\xff\xd9")
    return path


def _base_row(tmp_path: Path, *, provider: str = "gemini", video_id: str = "video-001") -> dict:
    overview = _write_synthetic_jpeg(tmp_path / "media" / video_id / "overview.jpg", "overview")
    keyframes = [
        _write_synthetic_jpeg(tmp_path / "media" / video_id / "frames" / "0001.jpg", "kf1"),
        _write_synthetic_jpeg(tmp_path / "media" / video_id / "frames" / "0002.jpg", "kf2"),
    ]
    return {
        "provider": provider,
        "video_id": video_id,
        "panel_row_id": f"{provider}:{video_id}",
        "category": "temporal_reasoning",
        "question": "What does the person pick up after opening the drawer?",
        "top1_answer": "a red mug",
        "support_span": {"start": 1.25, "end": 4.5},
        "relevance": "high",
        "could_be_wrong": "maybe handle is hidden",
        "provider_audio_state": "provider_reported",
        "map": {
            "schema_version": 1,
            "question_interpretation": {
                "meaning": "Identify the object picked up after the drawer opens.",
                "required_modalities": ["visual"],
                "required_evidence": "the object in the person's hand",
            },
            "events": [
                {
                    "event_id": "E1",
                    "start_seconds": 1.0,
                    "end_seconds": 2.0,
                    "timestamp": 1.25,
                    "keyframe_seconds": [1.25],
                    "scene": "kitchen",
                    "description": "person opens drawer",
                    "entities": ["person", "drawer"],
                    "actions": ["opens drawer"],
                    "relations": ["person interacts with drawer"],
                    "state_before": "drawer closed",
                    "state_change": "drawer opens",
                    "state_after": "drawer open",
                    "visual_observations": ["drawer front moves outward"],
                    "speech_observations": ["absent"],
                    "sound_observations": ["absent"],
                    "visible_text": ["absent"],
                    "question_relevance": "setup only",
                    "confidence": 0.95,
                    "gaps": [],
                },
                {
                    "event_id": "E2",
                    "start_seconds": 2.0,
                    "end_seconds": 4.0,
                    "keyframe_seconds": [3.75],
                    "scene": "kitchen",
                    "timestamp": 3.75,
                    "description": "person picks up red mug",
                    "entities": ["person", "red mug"],
                    "actions": ["picks up mug"],
                    "relations": ["person holds red mug"],
                    "state_before": "mug in drawer",
                    "state_change": "person lifts mug",
                    "state_after": "mug in person's hand",
                    "visual_observations": ["red mug is visible in hand"],
                    "speech_observations": ["absent"],
                    "sound_observations": ["absent"],
                    "visible_text": ["absent"],
                    "question_relevance": "direct answer evidence",
                    "confidence": 0.9,
                    "gaps": ["handle partly occluded"],
                },
            ],
            "support_spans": [
                {
                    "event_ids": ["E2"],
                    "start_seconds": 2.0,
                    "end_seconds": 4.0,
                    "modalities": ["visual"],
                    "claim": "E2 shows the red mug in the person's hand.",
                }
            ],
            "top1_answer": "a red mug",
            "support_explanation": "E2 shows the red mug in the person's hand.",
            "could_be_wrong_because": ["the handle is partly occluded"],
        },
        "overview_path": str(overview),
        "keyframe_paths": [str(path) for path in keyframes],
        "source_hashes": {
            str(overview): _sha256(overview),
            **{str(path): _sha256(path) for path in keyframes},
        },
    }


def make_frozen_run(tmp_path: Path, *, rows: list[dict] | None = None) -> dict:
    if rows is None:
        rows = [_base_row(tmp_path)]
    return {
        "panel_manifest": {
            "panel_id": "synthetic-frontier-map-panel",
            "row_count": len(rows),
            "video_ids": [row["video_id"] for row in rows],
        },
        "provider_results": rows,
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _verdict(provider: str, video_id: str, view: str, *, usable_map_decision: str = "usable") -> dict:
    row = {
        "provider": provider,
        "video_id": video_id,
        "view": view,
        "notes": "synthetic private reviewer note",
    }
    if view == "blind":
        row |= {
            "event_coverage": 5,
            "order": 5,
            "entities": 5,
            "actions": 5,
            "state_changes": 4,
            "visible_text": "unverified",
            "timestamp_boundary_error": False,
            "answer_changing_false_claims": False,
        }
    else:
        row |= {
            "question_understanding": 5,
            "modality_choice": "visual",
            "evidence_presence": 5,
            "evidence_localization": 4,
            "support_citation": 5,
            "usable_map_decision": usable_map_decision,
            "confidence": 4,
        }
    return row


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def make_complete_audit_fixture(
    tmp_path: Path,
    *,
    row_count: int = 3,
    providers: tuple[str, ...] = ("gemini",),
) -> Path:
    rows = [
        _base_row(tmp_path, provider=provider, video_id=f"video-{idx:03d}")
        for provider in providers
        for idx in range(row_count)
    ]
    packet = build_audit_packet(make_frozen_run(tmp_path, rows=rows), output=tmp_path / "audit")
    blind_verdicts = []
    aware_verdicts = []
    for row in rows:
        blind_verdicts.append(_verdict(row["provider"], row["video_id"], "blind"))
        aware_verdicts.append(_verdict(row["provider"], row["video_id"], "aware"))
    _write_jsonl(packet.verdicts_blind, blind_verdicts)
    _write_jsonl(packet.verdicts_aware, aware_verdicts)
    return packet.root


def _write_provider_run_fixture(path: Path, frozen_run: dict) -> Path:
    path.mkdir()
    artifact = path / "provider-results.json"
    artifact_payload = {
        "schema_version": 1,
        "provider_results": frozen_run["provider_results"],
        "panel_manifest": frozen_run["panel_manifest"],
    }
    artifact.write_text(json.dumps(artifact_payload, sort_keys=True), encoding="utf-8")
    freeze = {
        "schema_version": 1,
        "status": "frozen",
        "artifact_sha256": _sha256(artifact),
        "bindings": {"schema_version": 1},
        "bindings_sha256": _sha256_json({"schema_version": 1}),
    }
    (path / "freeze.json").write_text(json.dumps(freeze, sort_keys=True), encoding="utf-8")
    (path / "frozen-run.json").write_text(
        json.dumps(frozen_run, sort_keys=True), encoding="utf-8"
    )
    return path


def _sha256_json(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_blind_packet_masks_question_answer_support_and_relevance(tmp_path: Path) -> None:
    packet = build_audit_packet(make_frozen_run(tmp_path), output=tmp_path / "audit")

    blind = _read_json(packet.root / "rows" / "gemini" / "video-001" / "blind.json")
    text = json.dumps(blind).lower()

    for forbidden in ("question", "top1_answer", "support_span", "relevance", "could_be_wrong"):
        assert forbidden not in text


def test_blind_packet_uses_canonical_mask_and_preserves_observational_chronology(
    tmp_path: Path,
) -> None:
    packet = build_audit_packet(make_frozen_run(tmp_path), output=tmp_path / "audit")
    blind = _read_json(packet.root / "rows" / "gemini" / "video-001" / "blind.json")
    expected = audit_module.mask_question_fields(_base_row(tmp_path)["map"])

    assert blind["map"] == expected
    assert [event["event_id"] for event in blind["map"]["events"]] == ["E1", "E2"]
    assert [event["state_change"] for event in blind["map"]["events"]] == [
        "drawer opens",
        "person lifts mug",
    ]
    assert all("question_relevance" not in event for event in blind["map"]["events"])
    assert "question_interpretation" not in blind["map"]
    assert "support_spans" not in blind["map"]
    assert "top1_answer" not in blind["map"]
    assert "support_explanation" not in blind["map"]
    assert "could_be_wrong_because" not in blind["map"]


def test_aware_packet_keeps_question_context_without_reference_text(tmp_path: Path) -> None:
    packet = build_audit_packet(make_frozen_run(tmp_path), output=tmp_path / "audit")

    aware = _read_json(packet.root / "rows" / "gemini" / "video-001" / "aware.json")
    text = json.dumps(aware).lower()

    assert aware["question"] == "What does the person pick up after opening the drawer?"
    assert aware["top1_answer"] == "a red mug"
    assert aware["support_span"] == {"start": 1.25, "end": 4.5}
    assert "reference" not in text
    assert "correct_answer" not in text


def test_packet_copies_synthetic_overview_and_keyframes_with_auditable_timestamps(
    tmp_path: Path,
) -> None:
    packet = build_audit_packet(make_frozen_run(tmp_path), output=tmp_path / "audit")
    row_dir = packet.root / "rows" / "gemini" / "video-001"

    assert (row_dir / "overview.jpg").read_bytes().startswith(b"\xff\xd8synthetic")
    assert sorted(path.name for path in (row_dir / "event-keyframes").iterdir()) == [
        "0001.jpg",
        "0002.jpg",
    ]
    blind = _read_json(row_dir / "blind.json")
    assert blind["event_timestamps"] == [1.25, 3.75]


def test_failed_provider_row_has_no_fabricated_map_content_and_requires_unusable(
    tmp_path: Path,
) -> None:
    row = _base_row(tmp_path)
    row["failure_code"] = "provider_timeout"
    row["map"] = None
    row["top1_answer"] = None
    packet = build_audit_packet(make_frozen_run(tmp_path, rows=[row]), output=tmp_path / "audit")
    row_dir = packet.root / "rows" / "gemini" / "video-001"

    blind = _read_json(row_dir / "blind.json")
    aware = _read_json(row_dir / "aware.json")

    assert blind["map_payload_present"] is False
    assert aware["map_payload_present"] is False
    assert blind["failure_code"] == "provider_timeout"
    assert "events" not in blind
    assert "support" not in aware
    _write_jsonl(packet.verdicts_blind, [_verdict("gemini", "video-001", "blind")])
    _write_jsonl(packet.verdicts_aware, [_verdict("gemini", "video-001", "aware")])
    with pytest.raises(ValueError, match="unusable"):
        freeze_audit(packet.root)
    _write_jsonl(packet.verdicts_aware, [_verdict("gemini", "video-001", "aware", usable_map_decision="unusable")])
    freeze = freeze_audit(packet.root)
    assert freeze["rows"][0]["failure_code"] == "provider_timeout"
    assert freeze["rows"][0]["map_payload_present"] is False


def test_failed_provider_row_requires_frozen_failure_code_when_map_absent(
    tmp_path: Path,
) -> None:
    row = _base_row(tmp_path)
    row["map"] = None

    with pytest.raises(ValueError, match="failure_code"):
        build_audit_packet(make_frozen_run(tmp_path, rows=[row]), output=tmp_path / "audit")


def test_failed_provider_row_strips_fabricated_answer_support_and_map_fields(
    tmp_path: Path,
) -> None:
    row = _base_row(tmp_path)
    row["failure_code"] = "provider_timeout"
    row["map"] = None
    row["top1_answer"] = "fabricated red mug"
    row["support_span"] = {"start": 1.0, "end": 2.0}
    row["support_spans"] = [{"start": 1.0, "end": 2.0}]
    row["support_explanation"] = "fabricated support"
    row["explanation"] = "fabricated explanation"

    packet = build_audit_packet(make_frozen_run(tmp_path, rows=[row]), output=tmp_path / "audit")
    row_dir = packet.root / "rows" / "gemini" / "video-001"
    blind_text = json.dumps(_read_json(row_dir / "blind.json")).lower()
    aware_text = json.dumps(_read_json(row_dir / "aware.json")).lower()

    for serialized in (blind_text, aware_text):
        assert "\"events\"" not in serialized
        for forbidden in (
            "top1_answer",
            "fabricated red mug",
            "support_span",
            "support_spans",
            "support_explanation",
            "fabricated support",
            "explanation",
            "fabricated explanation",
        ):
            assert forbidden not in serialized


def test_failed_aware_packet_keeps_question_for_unusable_review_only(
    tmp_path: Path,
) -> None:
    row = _base_row(tmp_path)
    row["failure_code"] = "provider_timeout"
    row["map"] = None
    row["top1_answer"] = "fabricated red mug"
    row["support_span"] = {"start": 1.0, "end": 2.0}
    row["support_spans"] = [{"start": 1.0, "end": 2.0}]
    row["support_explanation"] = "fabricated support"
    row["explanation"] = "fabricated explanation"
    row["could_be_wrong"] = "fabricated uncertainty"

    packet = build_audit_packet(make_frozen_run(tmp_path, rows=[row]), output=tmp_path / "audit")
    aware = _read_json(packet.root / "rows" / "gemini" / "video-001" / "aware.json")
    serialized = json.dumps(aware).lower()

    assert aware["question"] == "What does the person pick up after opening the drawer?"
    assert aware["failure_code"] == "provider_timeout"
    assert aware["map_payload_present"] is False
    for forbidden in (
        "top1_answer",
        "fabricated red mug",
        "support_span",
        "support_spans",
        "support_explanation",
        "fabricated support",
        "explanation",
        "fabricated explanation",
        "could_be_wrong",
        "fabricated uncertainty",
        "\"map\"",
        "\"events\"",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    "provider_audio_state",
    ["provider_reported", "not_claimed", "unverified"],
)
def test_audio_state_allows_only_frozen_private_boundary_values(
    tmp_path: Path,
    provider_audio_state: str,
) -> None:
    row = _base_row(tmp_path)
    row["provider_audio_state"] = provider_audio_state

    packet = build_audit_packet(make_frozen_run(tmp_path, rows=[row]), output=tmp_path / "audit")

    aware = _read_json(packet.root / "rows" / "gemini" / "video-001" / "aware.json")
    assert aware["provider_audio_state"] == provider_audio_state


def test_audio_state_rejects_values_that_would_imply_unverified_audio_access(
    tmp_path: Path,
) -> None:
    row = _base_row(tmp_path)
    row["provider_audio_state"] = "transcribed_by_auditor"

    with pytest.raises(ValueError, match="provider_audio_state"):
        build_audit_packet(make_frozen_run(tmp_path, rows=[row]), output=tmp_path / "audit")


def test_source_hashes_and_symlink_inputs_are_enforced(tmp_path: Path) -> None:
    row = _base_row(tmp_path)
    overview = Path(row["overview_path"])
    overview.write_bytes(overview.read_bytes() + b"stale")

    with pytest.raises(ValueError, match="source hash"):
        build_audit_packet(make_frozen_run(tmp_path, rows=[row]), output=tmp_path / "audit-stale")

    linked = tmp_path / "linked-overview.jpg"
    linked.symlink_to(overview)
    fresh = _base_row(tmp_path, video_id="video-002")
    fresh["overview_path"] = str(linked)
    fresh["source_hashes"][str(linked)] = _sha256(overview)

    with pytest.raises(ValueError, match="symlink"):
        build_audit_packet(make_frozen_run(tmp_path, rows=[fresh]), output=tmp_path / "audit-link")


def test_audit_freeze_requires_two_views_for_every_provider_row(tmp_path: Path) -> None:
    audit = make_complete_audit_fixture(tmp_path, row_count=35, providers=("gemini", "reka"))

    freeze = freeze_audit(audit)

    assert freeze["bindings"]["blind_verdicts"] == 70
    assert freeze["bindings"]["aware_verdicts"] == 70


def test_audit_freeze_rejects_missing_or_duplicate_view_verdicts(tmp_path: Path) -> None:
    rows = [
        _base_row(tmp_path, provider="gemini", video_id="video-001"),
        _base_row(tmp_path, provider="gemini", video_id="video-002"),
    ]
    packet = build_audit_packet(make_frozen_run(tmp_path, rows=rows), output=tmp_path / "audit")
    _write_jsonl(
        packet.verdicts_blind,
        [
            _verdict("gemini", "video-001", "blind"),
            _verdict("gemini", "video-001", "blind"),
            _verdict("gemini", "video-002", "blind"),
        ],
    )
    _write_jsonl(packet.verdicts_aware, [_verdict("gemini", "video-001", "aware")])

    with pytest.raises(ValueError, match="duplicate blind verdict"):
        freeze_audit(packet.root)

    _write_jsonl(packet.verdicts_blind, [_verdict("gemini", "video-001", "blind")])
    with pytest.raises(ValueError, match="missing verdict"):
        freeze_audit(packet.root)


def test_audit_freeze_writes_complete_hash_bound_boundary(tmp_path: Path) -> None:
    audit = make_complete_audit_fixture(tmp_path, row_count=2, providers=("gemini", "reka"))

    freeze = freeze_audit(audit)
    persisted = _read_json(audit / "freeze.json")

    assert persisted == freeze
    assert freeze["frozen"] is True
    assert freeze["bindings"]["provider_rows"] == 4
    assert len(freeze["rows"]) == 4
    assert freeze["bindings"]["packet_manifest_sha256"] == _sha256(audit / "packet-manifest.json")
    assert freeze["bindings"]["blind_verdicts_sha256"] == _sha256(audit / "verdicts.blind.jsonl")
    assert freeze["bindings"]["aware_verdicts_sha256"] == _sha256(audit / "verdicts.aware.jsonl")
    assert "packet_row_sha256" in freeze["rows"][0]


def test_audit_freeze_verification_accepts_untampered_artifacts(tmp_path: Path) -> None:
    audit = make_complete_audit_fixture(tmp_path, row_count=1)
    freeze = freeze_audit(audit)

    assert verify_audit_freeze(audit) == freeze


@pytest.mark.parametrize(
    "relative_path",
    [
        "rows/gemini/video-000/blind.json",
        "rows/gemini/video-000/aware.json",
        "rows/gemini/video-000/overview.jpg",
        "rows/gemini/video-000/event-keyframes/0001.jpg",
        "packet-manifest.json",
        "blind-manifest.json",
        "aware-manifest.json",
        "verdicts.blind.jsonl",
        "verdicts.aware.jsonl",
    ],
)
def test_post_freeze_tampering_of_each_artifact_class_is_rejected(
    tmp_path: Path, relative_path: str
) -> None:
    audit = make_complete_audit_fixture(tmp_path, row_count=1)
    freeze_audit(audit)
    target = audit / relative_path
    target.write_bytes(target.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="hash mismatch"):
        verify_audit_freeze(audit)


def test_cli_build_validate_freeze_verifies_prepared_run_and_includes_terminal_failure(
    tmp_path: Path,
) -> None:
    success = _base_row(tmp_path, video_id="video-000")
    failure = _base_row(tmp_path, video_id="video-001")
    failure["map"] = None
    failure["failure_code"] = "provider_timeout"
    frozen_run = make_frozen_run(tmp_path, rows=[success, failure])
    prepared = _write_provider_run_fixture(tmp_path / "prepared", frozen_run)
    output = tmp_path / "audit"

    def run(*args: str) -> dict:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    assert run("build", "--prepared", str(prepared), "--output", str(output))["row_count"] == 2
    assert run("validate", "--audit", str(output))["row_count"] == 2
    assert (output / "rows" / "gemini" / "video-001" / "blind.json").is_file()
    assert _read_json(output / "rows" / "gemini" / "video-001" / "aware.json")[
        "failure_code"
    ] == "provider_timeout"
    _write_jsonl(
        output / "verdicts.blind.jsonl",
        [_verdict("gemini", "video-000", "blind"), _verdict("gemini", "video-001", "blind")],
    )
    _write_jsonl(
        output / "verdicts.aware.jsonl",
        [
            _verdict("gemini", "video-000", "aware"),
            _verdict("gemini", "video-001", "aware", usable_map_decision="unusable"),
        ],
    )
    assert run("validate", "--audit", str(output))["row_count"] == 2
    assert run("freeze", "--audit", str(output))["frozen"] is True


def test_cli_build_rejects_stale_provider_freeze_before_copying_media(
    tmp_path: Path,
) -> None:
    prepared = _write_provider_run_fixture(
        tmp_path / "prepared",
        make_frozen_run(tmp_path, rows=[_base_row(tmp_path, video_id="video-000")]),
    )
    artifact = prepared / "provider-results.json"
    artifact.write_bytes(artifact.read_bytes() + b"\n")
    output = tmp_path / "audit"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "build",
            "--prepared",
            str(prepared),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "provider freeze artifact hash mismatch" in json.loads(result.stdout)["error"]
    assert not output.exists()


def test_build_refuses_nonempty_or_symlink_output_root(tmp_path: Path) -> None:
    frozen_run = make_frozen_run(tmp_path)
    output = tmp_path / "audit"
    output.mkdir()
    (output / "existing.txt").write_text("preserve me", encoding="utf-8")

    with pytest.raises(ValueError, match="overwrite"):
        build_audit_packet(frozen_run, output=output)

    linked_output = tmp_path / "linked-audit"
    linked_output.symlink_to(output, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        build_audit_packet(frozen_run, output=linked_output)


def test_freeze_verification_rejects_manifest_path_escape(tmp_path: Path) -> None:
    audit = make_complete_audit_fixture(tmp_path, row_count=1)
    freeze_audit(audit)
    manifest = _read_json(audit / "packet-manifest.json")
    manifest["rows"][0]["blind_manifest"] = "../escaped.json"
    (audit / "packet-manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="path escape"):
        verify_audit_freeze(audit)
