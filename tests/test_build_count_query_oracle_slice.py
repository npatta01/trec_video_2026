from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

from trec_video_2026.vqa_counting_evidence import validate_label_blind
from trec_video_2026.vqa_submission import canonical_sha256

SCRIPT = Path(__file__).parents[1] / "scripts" / "build_count_query_oracle_slice.py"
SPEC = importlib.util.spec_from_file_location("count_query_oracle_slice", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _with_hash(payload: dict[str, object]) -> dict[str, object]:
    result = deepcopy(payload)
    result["complete_sha256"] = canonical_sha256(result)
    return result


def source_fixture() -> tuple[dict[str, object], dict[str, object], dict[str, str]]:
    panel_rows = []
    request_rows = []
    strata = {}
    stratum_names = ("physical", "appearance", "concurrent", "transition")
    for role, count, prefix in (
        ("confirmed_failure", 26, "failure"),
        ("correct_control", 12, "control"),
    ):
        for index in range(count):
            video_id = f"{prefix}-{index:02}"
            panel_rows.append(
                {
                    "video_id": video_id,
                    "panel_role": role,
                    "status": "wrong" if prefix == "failure" else "correct",
                    "question": f"How many objects in {video_id}?",
                    "reference_answer": "two",
                    "reference_count": 2,
                    "baseline_prediction": "3" if prefix == "failure" else "2",
                    "prediction_count": 3 if prefix == "failure" else 2,
                    "prediction_candidates": [3 if prefix == "failure" else 2],
                    "ordinal": index,
                }
            )
            request_rows.append(
                {
                    "video_id": video_id,
                    "question": f"How many objects in {video_id}?",
                    "duration_seconds": 10.0,
                    "frozen_map": f"map for {video_id}",
                    "source_path": f"/private/{video_id}.mkv",
                    "source_sha256": f"source-{video_id}",
                    "delivery_mime_type": "video/webm",
                    "delivery_sha256": f"delivery-{video_id}",
                    "file_name": f"files/{video_id}",
                    "file_uri": f"https://example.invalid/{video_id}",
                    "file_state": "ACTIVE",
                    "expiration_time": "2026-08-15T00:00:00Z",
                }
            )
            strata[video_id] = stratum_names[index % len(stratum_names)]
    panel = _with_hash(
        {
            "schema_version": 1,
            "selection_policy": "fixture",
            "score_denominators": {
                "confirmed_failures": 26,
                "correct_controls": 12,
            },
            "rows": panel_rows,
        }
    )
    requests = _with_hash(
        {
            "schema_version": 1,
            "prompt_version": "fixture",
            "canary_video_ids": [
                "failure-00",
                "failure-01",
                "failure-02",
                "failure-03",
                "failure-04",
                "control-00",
            ],
            "ordered_video_ids_sha256": canonical_sha256(
                [row["video_id"] for row in request_rows]
            ),
            "rows": request_rows,
        }
    )
    return panel, requests, strata


def test_selection_is_balanced_hash_ranked_and_order_invariant() -> None:
    panel, requests, strata = source_fixture()
    design_ids = requests["canary_video_ids"]

    first = MODULE.build_oracle_slice(
        panel=panel,
        requests=requests,
        design_video_ids=design_ids,
        semantic_strata=strata,
    )
    reversed_panel = deepcopy(panel)
    reversed_panel["rows"].reverse()
    reversed_panel["complete_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in reversed_panel.items()
            if key != "complete_sha256"
        }
    )
    reversed_requests = deepcopy(requests)
    reversed_requests["rows"].reverse()
    reversed_requests["ordered_video_ids_sha256"] = canonical_sha256(
        [row["video_id"] for row in reversed_requests["rows"]]
    )
    reversed_requests["complete_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in reversed_requests.items()
            if key != "complete_sha256"
        }
    )
    second = MODULE.build_oracle_slice(
        panel=reversed_panel,
        requests=reversed_requests,
        design_video_ids=design_ids,
        semantic_strata=strata,
    )

    selected = first["selection"]["rows"]
    assert sum(row["panel_role"] == "confirmed_failure" for row in selected) == 8
    assert sum(row["panel_role"] == "correct_control" for row in selected) == 4
    assert not set(design_ids) & {row["video_id"] for row in selected}
    assert {row["semantic_stratum"] for row in selected} == set(strata.values())
    assert [row["video_id"] for row in selected] == [
        row["video_id"] for row in second["selection"]["rows"]
    ]


def test_authoring_outputs_are_recursively_label_blind() -> None:
    panel, requests, strata = source_fixture()
    artifacts = MODULE.build_oracle_slice(
        panel=panel,
        requests=requests,
        design_video_ids=requests["canary_video_ids"],
        semantic_strata=strata,
    )

    validate_label_blind(artifacts["authoring_requests"])
    validate_label_blind(artifacts["authoring_scaffolds"])
    assert all(
        set(row) == {"video_id", "query", "graph"}
        for row in artifacts["authoring_scaffolds"]["rows"]
    )
    assert all(
        "panel_role" not in row
        and "reference_answer" not in row
        and "baseline_prediction" not in row
        for row in artifacts["authoring_requests"]["rows"]
    )
