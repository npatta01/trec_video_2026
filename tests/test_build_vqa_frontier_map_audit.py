from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

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
            "events": [
                {
                    "timestamp": 1.25,
                    "description": "person opens drawer",
                    "entities": ["person", "drawer"],
                },
                {
                    "timestamp": 3.75,
                    "description": "person picks up red mug",
                    "entities": ["person", "red mug"],
                },
            ],
            "support": [{"start": 1.25, "end": 4.5, "reason": "object interaction"}],
            "explanation": "The visible action changes from opening to picking up the mug.",
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


def test_blind_packet_masks_question_answer_support_and_relevance(tmp_path: Path) -> None:
    packet = build_audit_packet(make_frozen_run(tmp_path), output=tmp_path / "audit")

    blind = _read_json(packet.root / "rows" / "gemini" / "video-001" / "blind.json")
    text = json.dumps(blind).lower()

    for forbidden in ("question", "top1_answer", "support_span", "relevance", "could_be_wrong"):
        assert forbidden not in text


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
