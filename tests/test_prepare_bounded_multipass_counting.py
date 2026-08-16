from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


SCRIPT = Path("scripts/prepare_bounded_multipass_counting.py")
SPEC = importlib.util.spec_from_file_location("prepare_bounded", SCRIPT)
assert SPEC and SPEC.loader
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


REQUIRED = {
    "failure_action": 4,
    "failure_static": 3,
    "failure_other_temporal": 1,
    "control_temporal": 2,
    "control_static": 2,
}


def synthetic_rows() -> list[dict[str, object]]:
    rows = []
    ordinal = 0
    for stratum, count in REQUIRED.items():
        status = "wrong" if stratum.startswith("failure") else "correct"
        for index in range(count + 2):
            rows.append(
                {
                    "ordinal": ordinal,
                    "video_id": f"{stratum}-{index}",
                    "question": f"Question for {stratum} candidate {index}?",
                    "status": status,
                    "reference_answer": str(index + 1),
                    "reference_count": index + 1,
                    "baseline_prediction": str(index),
                    "prediction_count": index,
                    "prediction_candidates": [index],
                }
            )
            ordinal += 1
    return rows


def routing(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "bounded-multipass-routing-v1",
        "rows": [
            {
                "video_id": row["video_id"],
                "stratum": str(row["video_id"]).rsplit("-", 1)[0],
                "supported": True,
                "question_only_rationale": "synthetic supported route",
            }
            for row in rows
        ],
    }


def test_select_panel_has_exact_predeclared_strata_and_is_order_independent() -> None:
    rows = synthetic_rows()
    first = prepare.select_panel(rows=rows, routing=routing(rows), excluded_ids=set())
    second = prepare.select_panel(
        rows=list(reversed(rows)), routing=routing(rows), excluded_ids=set()
    )
    assert [row["video_id"] for row in first] == [row["video_id"] for row in second]
    assert len(first) == 12
    observed = {}
    for row in first:
        observed[row["stratum"]] = observed.get(row["stratum"], 0) + 1
    assert observed == REQUIRED


def test_select_panel_applies_exclusions_before_hash_ordering() -> None:
    rows = synthetic_rows()
    initial = prepare.select_panel(rows=rows, routing=routing(rows), excluded_ids=set())
    excluded = {initial[0]["video_id"]}
    selected = prepare.select_panel(
        rows=rows, routing=routing(rows), excluded_ids=excluded
    )
    assert not excluded & {row["video_id"] for row in selected}
    assert len(selected) == 12


def test_select_panel_rejects_duplicate_video_ids() -> None:
    rows = synthetic_rows()
    with pytest.raises(ValueError, match="duplicate video_id"):
        prepare.select_panel(
            rows=[*rows, dict(rows[0])], routing=routing(rows), excluded_ids=set()
        )


def test_select_panel_rejects_missing_required_stratum() -> None:
    rows = [
        row
        for row in synthetic_rows()
        if not str(row["video_id"]).startswith("failure_action")
    ]
    with pytest.raises(ValueError, match="failure_action"):
        prepare.select_panel(rows=rows, routing=routing(rows), excluded_ids=set())


def test_select_panel_rejects_status_stratum_mismatch() -> None:
    rows = synthetic_rows()
    rows[0]["status"] = "correct"
    with pytest.raises(ValueError, match="status"):
        prepare.select_panel(rows=rows, routing=routing(rows), excluded_ids=set())


def test_project_label_blind_removes_outcome_fields() -> None:
    rows = synthetic_rows()
    panel = prepare.select_panel(rows=rows, routing=routing(rows), excluded_ids=set())
    requests = prepare.project_label_blind(panel)
    assert len(requests) == 12
    rendered = json.dumps(requests, sort_keys=True).lower()
    for forbidden in (
        "reference",
        "baseline",
        "prediction",
        "correct",
        "status",
        "stratum",
        "panel_role",
    ):
        assert forbidden not in rendered
    assert set(requests[0]) == {"ordinal", "question", "video_id"}


def test_write_freeze_is_byte_identical_or_refuses_conflict(tmp_path: Path) -> None:
    payload = {"schema_version": "fixture-v1", "rows": [{"video_id": "v1"}]}
    path = tmp_path / "freeze.json"
    first = prepare.write_self_hashed_once(path, payload)
    second = prepare.write_self_hashed_once(path, payload)
    assert first == second
    assert json.loads(path.read_text())["complete_sha256"] == first["complete_sha256"]
    with pytest.raises(FileExistsError, match="conflicting"):
        prepare.write_self_hashed_once(path, {"schema_version": "fixture-v2"})


def test_load_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"rows": [], "rows": []}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        prepare.load_json(path)


def test_bind_media_recomputes_source_and_delivery_hashes(tmp_path: Path) -> None:
    source = tmp_path / "v1.mkv"
    delivery = tmp_path / "v1.webm"
    source.write_bytes(b"source-video")
    delivery.write_bytes(b"delivery-video")
    panel = [
        {
            "ordinal": 1,
            "video_id": "v1",
            "question": "How many jumps?",
            "stratum": "failure_action",
        }
    ]
    prior = {
        "rows": [
            {
                "video_id": "v1",
                "source_path": str(source),
                "source_sha256": hashlib.sha256(b"source-video").hexdigest(),
                "delivery_path": str(delivery),
                "delivery_sha256": hashlib.sha256(b"delivery-video").hexdigest(),
                "delivery_mime_type": "video/webm",
                "duration_seconds": 4.0,
            }
        ]
    }
    requests = prepare.bind_media(panel=panel, prior_requests=prior)
    assert requests == [
        {
            "ordinal": 1,
            "video_id": "v1",
            "question": "How many jumps?",
            "source_path": str(source),
            "source_sha256": hashlib.sha256(b"source-video").hexdigest(),
            "delivery_path": str(delivery),
            "delivery_sha256": hashlib.sha256(b"delivery-video").hexdigest(),
            "delivery_mime_type": "video/webm",
            "duration_seconds": 4.0,
        }
    ]


def test_bind_media_rejects_hash_mutation_or_missing_row(tmp_path: Path) -> None:
    source = tmp_path / "v1.mkv"
    delivery = tmp_path / "v1.webm"
    source.write_bytes(b"source-video")
    delivery.write_bytes(b"delivery-video")
    panel = [{"ordinal": 1, "video_id": "v1", "question": "How many jumps?"}]
    prior = {
        "rows": [
            {
                "video_id": "v1",
                "source_path": str(source),
                "source_sha256": "0" * 64,
                "delivery_path": str(delivery),
                "delivery_sha256": hashlib.sha256(b"delivery-video").hexdigest(),
                "delivery_mime_type": "video/webm",
                "duration_seconds": 4.0,
            }
        ]
    }
    with pytest.raises(ValueError, match="source_sha256"):
        prepare.bind_media(panel=panel, prior_requests=prior)
    with pytest.raises(ValueError, match="missing media binding"):
        prepare.bind_media(panel=panel, prior_requests={"rows": []})


def test_bind_media_from_roots_accepts_recorded_mp4_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    delivery_root = tmp_path / "deliveries"
    source_root.mkdir()
    delivery = delivery_root / "v1/delivery-media/v1.mp4"
    delivery.parent.mkdir(parents=True)
    (source_root / "v1.mkv").write_bytes(b"source")
    delivery.write_bytes(b"delivery")
    monkeypatch.setattr(prepare, "_probe_duration", lambda _path: 4.0)
    rows = prepare.bind_media_from_roots(
        panel=[{"ordinal": 1, "video_id": "v1", "question": "How many?"}],
        source_root=source_root,
        delivery_root=delivery_root,
    )
    assert rows[0]["delivery_path"] == str(delivery)
    assert rows[0]["delivery_mime_type"] == "video/mp4"
