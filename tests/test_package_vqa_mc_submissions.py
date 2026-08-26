from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import package_vqa_mc_submissions as packager  # noqa: E402


def _submission(task_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "Q_ID": row["Q_ID"],
            "Video_ID": row["Video_ID"],
            "Answers": [
                {"Rank": rank, "Answer": option["option"]}
                for rank, option in enumerate(row["Options"], start=1)
            ],
        }
        for row in task_rows
    ]


@pytest.fixture
def candidate_fixture(tmp_path: Path) -> dict[str, object]:
    task = [
        {
            "Q_ID": 1,
            "Video_ID": "one.webm",
            "Question": "What happens?",
            "Options": [{"option": "A"}, {"option": "B"}, {"option": "C"}],
        },
        {
            "Q_ID": 2,
            "Video_ID": "missing",
            "Question": "Who arrives?",
            "Options": [{"option": "D"}, {"option": "E"}],
        },
    ]
    sources: dict[str, Path] = {}
    for tag in ("g37q36_mc_text", "g37_mc_target", "g37_mc_direct"):
        path = tmp_path / f"{tag}.json"
        path.write_text(json.dumps(_submission(task)), encoding="utf-8")
        sources[tag] = path
    parent_freeze = {
        "rows": [
            {"query_id": 1, "video_id": "one.webm", "status": "passed"},
            {"query_id": 2, "video_id": "missing", "status": "unavailable"},
        ]
    }
    composite = {
        "rows": [
            {"query_id": 7, "video_id": "bad.webm", "status": "invalid"},
        ]
    }
    return {
        "task": task,
        "sources": sources,
        "parent_freeze": parent_freeze,
        "composite": composite,
        "output": tmp_path / "package",
    }


def test_validate_mc_submission_requires_exact_complete_released_options(
    candidate_fixture: dict[str, object],
) -> None:
    submission = _submission(candidate_fixture["task"])
    submission[0]["Answers"][2]["Answer"] = "invented"

    with pytest.raises(ValueError, match="released options"):
        packager.validate_mc_submission(
            submission=submission,
            task_rows=candidate_fixture["task"],
            expected_count=2,
        )


def test_package_candidates_writes_three_exact_uploads_and_checksums(
    candidate_fixture: dict[str, object],
) -> None:
    result = packager.package_candidates(
        task_rows=candidate_fixture["task"],
        sources=candidate_fixture["sources"],
        output_root=candidate_fixture["output"],
        expected_count=2,
    )

    assert result["run_count"] == 3
    for tag, source in candidate_fixture["sources"].items():
        packaged = candidate_fixture["output"] / tag / "submission.json"
        assert packaged.read_bytes() == source.read_bytes()
        assert result["runs"][tag]["sha256"] == hashlib.sha256(
            source.read_bytes()
        ).hexdigest()


def test_build_exception_tracker_separates_unavailable_and_persistent_invalid(
    candidate_fixture: dict[str, object],
) -> None:
    tracker = packager.build_exception_tracker(
        parent_freeze=candidate_fixture["parent_freeze"],
        composite_repair_freeze=candidate_fixture["composite"],
        weak_query_ids={2, 7},
    )

    assert tracker["counts"] == {
        "media_unavailable": 1,
        "persistent_provider_invalid": 1,
        "v2_fallback": 2,
        "v3_fallback": 2,
    }
    assert tracker["exceptions"] == [
        {
            "Q_ID": 2,
            "Video_ID": "missing",
            "affected_runs": ["g37_mc_target", "g37_mc_direct"],
            "fallback": "g37q36_mc_text",
            "status": "media_unavailable",
        },
        {
            "Q_ID": 7,
            "Video_ID": "bad.webm",
            "affected_runs": ["g37_mc_target", "g37_mc_direct"],
            "fallback": "g37q36_mc_text",
            "status": "persistent_provider_invalid",
        },
    ]


def test_legacy_packager_requires_explicit_audit_flag(
    candidate_fixture: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir()
    legacy_catalog = private_root / "catalog.json"
    legacy_catalog_bytes = b'{"status":"locally_validated_not_uploaded"}\n'
    legacy_catalog.write_bytes(legacy_catalog_bytes)
    monkeypatch.setattr(packager, "PRIVATE_ROOT", private_root)
    monkeypatch.setattr(packager, "_task2", lambda: candidate_fixture["task"])
    monkeypatch.setattr(
        packager,
        "package_candidates",
        lambda **_: {"runs": {}},
    )
    monkeypatch.setattr(packager, "_json", lambda _: {})
    monkeypatch.setattr(packager, "_weak_query_ids", lambda: set())
    monkeypatch.setattr(
        packager,
        "build_exception_tracker",
        lambda **_: {"counts": {}},
    )
    monkeypatch.setattr(sys, "argv", ["package_vqa_mc_submissions"])
    with pytest.raises(SystemExit):
        packager.main()
    assert legacy_catalog.read_bytes() == legacy_catalog_bytes
    assert sorted(path.name for path in private_root.iterdir()) == ["catalog.json"]

    monkeypatch.setattr(
        sys,
        "argv",
        ["package_vqa_mc_submissions", "--legacy-audit"],
    )
    assert packager.main() == 0
    assert legacy_catalog.read_bytes() == legacy_catalog_bytes
    audit_catalog = json.loads(
        (private_root / "superseded-audit-catalog.json").read_text()
    )
    assert audit_catalog["status"] == "superseded_audit_only"
    assert audit_catalog["upload_forbidden"] is True
