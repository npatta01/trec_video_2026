"""Public CPU-only contract tests for temporal required-moment evidence."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from trec_video_2026.vqa_experiments.temporal_bottleneck import (
    ELIGIBILITY_LABELABLE_VISUAL,
    ELIGIBILITY_NOT_VISUALLY_ANSWERABLE,
    ROLE_ADJUDICATOR,
    ROLE_PRIMARY_REVIEWER,
    SCHEMA_VERSION,
    annotation_agreement,
    score_selector_recall,
    validate_annotation,
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@pytest.fixture
def panel() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "panel_id": "synthetic-temporal-2",
        "examples": [
            {"video_id": "alpha", "duration_seconds": 10.0},
            {"video_id": "beta", "duration_seconds": 4.0},
        ],
    }


@pytest.fixture
def valid_annotation(panel: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "reviewer": "reviewer-a",
        "role": ROLE_PRIMARY_REVIEWER,
        "rubric_revision": "temporal-required-moment-v1",
        "packet_sha256": _canonical_sha256(panel),
        "started_at": "2026-07-30T09:00:00Z",
        "ended_at": "2026-07-30T09:30:00Z",
        "rows": [
            {
                "video_id": "alpha",
                "eligibility": ELIGIBILITY_LABELABLE_VISUAL,
                "sufficient_sets": [[[1.0, 2.0], [4.0, 5.0]], [[7.0, 8.0]]],
                "preferred_set_index": 0,
            },
            {
                "video_id": "beta",
                "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE,
            },
        ],
    }


def test_validator_normalizes_duration_bounded_closed_intervals_in_panel_order(
    panel: dict, valid_annotation: dict
) -> None:
    annotation = validate_annotation(
        valid_annotation, panel, expected_role=ROLE_PRIMARY_REVIEWER
    )

    assert [row["video_id"] for row in annotation["rows"]] == ["alpha", "beta"]
    assert annotation["rows"][0]["sufficient_sets"] == [
        [[1.0, 2.0], [4.0, 5.0]],
        [[7.0, 8.0]],
    ]
    assert annotation["reviewer"] == "reviewer-a"
    assert annotation["rubric_revision"] == "temporal-required-moment-v1"
    assert annotation["started_at"] == "2026-07-30T09:00:00Z"
    assert annotation["ended_at"] == "2026-07-30T09:30:00Z"
    assert annotation["annotation_sha256"] == _canonical_sha256(valid_annotation)


def test_validator_revalidates_unchanged_normalized_annotation_idempotently(
    panel: dict, valid_annotation: dict
) -> None:
    normalized = validate_annotation(
        valid_annotation, panel, expected_role=ROLE_PRIMARY_REVIEWER
    )

    revalidated = validate_annotation(
        normalized, panel, expected_role=ROLE_PRIMARY_REVIEWER
    )

    assert revalidated["annotation_sha256"] == normalized["annotation_sha256"]
    assert revalidated["rows"] == normalized["rows"]


def test_validator_rejects_normalized_duration_mutated_after_panel_binding(
    panel: dict, valid_annotation: dict
) -> None:
    normalized = validate_annotation(
        valid_annotation, panel, expected_role=ROLE_PRIMARY_REVIEWER
    )
    normalized["rows"][0]["duration_seconds"] = 100.0

    with pytest.raises(ValueError, match="duration_seconds.*frozen panel"):
        validate_annotation(normalized, panel, expected_role=ROLE_PRIMARY_REVIEWER)


def test_validator_rejects_annotation_that_omits_a_frozen_panel_row(
    panel: dict, valid_annotation: dict
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload["rows"] = payload["rows"][:1]

    with pytest.raises(ValueError, match="panel rows"):
        validate_annotation(payload, panel, expected_role=ROLE_PRIMARY_REVIEWER)


def test_validator_rejects_annotation_that_reorders_frozen_panel_ids(
    panel: dict, valid_annotation: dict
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload["rows"].reverse()

    with pytest.raises(ValueError, match="frozen panel order"):
        validate_annotation(payload, panel, expected_role=ROLE_PRIMARY_REVIEWER)


def test_validator_rejects_annotation_with_duplicate_video_ids(
    panel: dict, valid_annotation: dict
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload["rows"][1]["video_id"] = "alpha"

    with pytest.raises(ValueError, match="duplicate"):
        validate_annotation(payload, panel, expected_role=ROLE_PRIMARY_REVIEWER)


def test_validator_rejects_labelable_row_with_an_empty_sufficient_set(
    panel: dict, valid_annotation: dict
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload["rows"][0]["sufficient_sets"] = [[]]

    with pytest.raises(ValueError, match="empty"):
        validate_annotation(payload, panel, expected_role=ROLE_PRIMARY_REVIEWER)


def test_validator_rejects_sufficient_set_with_more_than_six_intervals(
    panel: dict, valid_annotation: dict
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload["rows"][0]["sufficient_sets"] = [
        [[0.0, 0.1], [1.0, 1.1], [2.0, 2.1], [3.0, 3.1], [4.0, 4.1], [5.0, 5.1], [6.0, 6.1]]
    ]
    payload["rows"][0]["preferred_set_index"] = 0

    with pytest.raises(ValueError, match="at most six"):
        validate_annotation(payload, panel, expected_role=ROLE_PRIMARY_REVIEWER)


@pytest.mark.parametrize(
    ("interval", "message"),
    [([-0.1, 1.0], "nonnegative"), ([3.0, 3.0], "end"), ([3.0, 2.0], "end")],
)
def test_validator_rejects_negative_or_reversed_interval_bounds(
    panel: dict, valid_annotation: dict, interval: list[float], message: str
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload["rows"][0]["sufficient_sets"] = [[interval]]

    with pytest.raises(ValueError, match=message):
        validate_annotation(payload, panel, expected_role=ROLE_PRIMARY_REVIEWER)


def test_validator_rejects_interval_ending_beyond_frozen_duration(
    panel: dict, valid_annotation: dict
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload["rows"][0]["sufficient_sets"] = [[[9.9, 10.1]]]

    with pytest.raises(ValueError, match="duration"):
        validate_annotation(payload, panel, expected_role=ROLE_PRIMARY_REVIEWER)


def test_validator_rejects_unknown_eligibility_value(
    panel: dict, valid_annotation: dict
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload["rows"][1]["eligibility"] = "probably_labelable"

    with pytest.raises(ValueError, match="eligibility"):
        validate_annotation(payload, panel, expected_role=ROLE_PRIMARY_REVIEWER)


def test_validator_rejects_intervals_on_an_ineligible_row(
    panel: dict, valid_annotation: dict
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload["rows"][1]["sufficient_sets"] = [[[0.0, 1.0]]]

    with pytest.raises(ValueError, match="ineligible"):
        validate_annotation(payload, panel, expected_role=ROLE_PRIMARY_REVIEWER)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("packet_sha256", "0" * 64, "packet_sha256"),
        ("role", "adjudicator", "role"),
    ],
)
def test_validator_rejects_mismatched_packet_hash_or_role(
    panel: dict, valid_annotation: dict, field: str, value: str, message: str
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        validate_annotation(payload, panel, expected_role=ROLE_PRIMARY_REVIEWER)


@pytest.mark.parametrize(
    "forbidden_key",
    ["answer", "reference", "prediction", "grade", "condition", "frame_timestamps"],
)
def test_validator_rejects_forbidden_label_or_condition_key_at_any_depth(
    panel: dict, valid_annotation: dict, forbidden_key: str
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload["rows"][0]["sufficient_sets"][0][0].append({forbidden_key: "secret"})

    with pytest.raises(ValueError, match=forbidden_key):
        validate_annotation(payload, panel, expected_role=ROLE_PRIMARY_REVIEWER)


@pytest.mark.parametrize(
    "forbidden_key",
    ["prior_prediction", "previous_grade", "newPrediction", "selectedFrameTimestamps"],
)
def test_validator_rejects_compound_blinding_keys_but_allows_provenance_names(
    panel: dict, valid_annotation: dict, forbidden_key: str
) -> None:
    payload = copy.deepcopy(valid_annotation)
    payload["provenance"] = {
        "requested_time_seconds": 1.0,
        "extraction_command_identity": "ffmpeg-extract-v1",
        "prompt_source_identity": "temporal-panel-prompt-v1",
        "request_sha256": "b" * 64,
        "runtime_identity": "cpu-contract-v1",
        "answerable_media_count": 1,
        forbidden_key: "secret",
    }

    with pytest.raises(ValueError, match=forbidden_key):
        validate_annotation(payload, panel, expected_role=ROLE_PRIMARY_REVIEWER)


def _primary_annotation(panel: dict, reviewer: str, rows: list[dict]) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "reviewer": reviewer,
        "role": ROLE_PRIMARY_REVIEWER,
        "rubric_revision": "temporal-required-moment-v1",
        "packet_sha256": _canonical_sha256(panel),
        "rows": rows,
    }


def _labelable(video_id: str, sufficient_sets: list[list[list[float]]]) -> dict:
    return {
        "video_id": video_id,
        "eligibility": ELIGIBILITY_LABELABLE_VISUAL,
        "sufficient_sets": sufficient_sets,
        "preferred_set_index": 0,
    }


def test_agreement_reports_identical_preferred_span_as_perfect_temporal_iou(
    panel: dict,
) -> None:
    rows = [
        _labelable("alpha", [[[1.0, 3.0]]]),
        {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
    ]

    result = annotation_agreement(
        _primary_annotation(panel, "reviewer-a", rows),
        _primary_annotation(panel, "reviewer-b", copy.deepcopy(rows)),
        panel,
    )

    assert result["rows"][0]["temporal_iou"] == 1.0
    assert result["rows"][0]["flags"] == []


def test_agreement_reports_disjoint_preferred_spans_as_zero_temporal_iou(
    panel: dict,
) -> None:
    left = [
        _labelable("alpha", [[[1.0, 2.0]]]),
        {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
    ]
    right = [
        _labelable("alpha", [[[3.0, 4.0]]]),
        {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
    ]

    result = annotation_agreement(
        _primary_annotation(panel, "reviewer-a", left),
        _primary_annotation(panel, "reviewer-b", right),
        panel,
    )

    assert result["rows"][0]["temporal_iou"] == 0.0
    assert "temporal_iou_below_threshold" in result["rows"][0]["flags"]


def test_agreement_uses_union_without_double_counting_for_partial_overlap(
    panel: dict,
) -> None:
    left = [
        _labelable("alpha", [[[1.0, 5.0]]]),
        {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
    ]
    right = [
        _labelable("alpha", [[[3.0, 7.0]]]),
        {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
    ]

    result = annotation_agreement(
        _primary_annotation(panel, "reviewer-a", left),
        _primary_annotation(panel, "reviewer-b", right),
        panel,
    )

    assert result["rows"][0]["temporal_iou"] == pytest.approx(2 / 6)


def test_agreement_flags_preferred_set_topology_mismatch(
    panel: dict,
) -> None:
    left = [
        _labelable("alpha", [[[1.0, 2.0], [3.0, 4.0]]]),
        {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
    ]
    right = [
        _labelable("alpha", [[[1.0, 4.0]]]),
        {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
    ]

    result = annotation_agreement(
        _primary_annotation(panel, "reviewer-a", left),
        _primary_annotation(panel, "reviewer-b", right),
        panel,
    )

    assert result["rows"][0]["structural_match"] is False
    assert "topology_mismatch" in result["rows"][0]["flags"]


def test_agreement_flags_eligibility_mismatch(
    panel: dict,
) -> None:
    left = [
        _labelable("alpha", [[[1.0, 2.0]]]),
        {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
    ]
    right = [
        {"video_id": "alpha", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
        {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
    ]

    result = annotation_agreement(
        _primary_annotation(panel, "reviewer-a", left),
        _primary_annotation(panel, "reviewer-b", right),
        panel,
    )

    assert result["rows"][0]["structural_match"] is False
    assert "eligibility_mismatch" in result["rows"][0]["flags"]


def test_agreement_flags_matched_boundary_drift_above_one_second(panel: dict) -> None:
    left = [
        _labelable("alpha", [[[1.0, 2.0]]]),
        {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
    ]
    right = [
        _labelable("alpha", [[[2.1, 3.1]]]),
        {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
    ]

    result = annotation_agreement(
        _primary_annotation(panel, "reviewer-a", left),
        _primary_annotation(panel, "reviewer-b", right),
        panel,
    )

    assert result["rows"][0]["maximum_boundary_drift_seconds"] == pytest.approx(1.1)
    assert "boundary_drift_above_one_second" in result["rows"][0]["flags"]


def _ten_row_panel() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "panel_id": "synthetic-temporal-10",
        "examples": [
            {"video_id": f"video-{index}", "duration_seconds": 10.0}
            for index in range(10)
        ],
    }


def test_agreement_accepts_exactly_eight_of_ten_structural_matches() -> None:
    agreement_panel = _ten_row_panel()
    left = [_labelable(f"video-{index}", [[[1.0, 2.0]]]) for index in range(10)]
    right = copy.deepcopy(left)
    right[8]["sufficient_sets"] = [[[1.0, 2.0], [3.0, 4.0]]]
    right[9]["sufficient_sets"] = [[[1.0, 2.0], [3.0, 4.0]]]

    result = annotation_agreement(
        _primary_annotation(agreement_panel, "reviewer-a", left),
        _primary_annotation(agreement_panel, "reviewer-b", right),
        agreement_panel,
    )

    assert result["structural_match_count"] == 8
    assert result["structural_threshold_passes"] is True
    assert result["agreement_passes"] is True


def test_agreement_fails_when_median_temporal_iou_is_below_half() -> None:
    agreement_panel = _ten_row_panel()
    left = [_labelable(f"video-{index}", [[[1.0, 2.0]]]) for index in range(10)]
    right = copy.deepcopy(left)
    for index in range(6):
        right[index]["sufficient_sets"] = [[[3.0, 4.0]]]

    result = annotation_agreement(
        _primary_annotation(agreement_panel, "reviewer-a", left),
        _primary_annotation(agreement_panel, "reviewer-b", right),
        agreement_panel,
    )

    assert result["median_temporal_iou"] == 0.0
    assert result["median_temporal_iou_passes"] is False
    assert result["agreement_passes"] is False


def _adjudicated_annotation(panel: dict, rows: list[dict]) -> dict:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "reviewer": "adjudicator-c",
        "role": ROLE_ADJUDICATOR,
        "rubric_revision": "temporal-required-moment-v1",
        "packet_sha256": _canonical_sha256(panel),
        "rows": rows,
    }
    return validate_annotation(payload, panel, expected_role=ROLE_ADJUDICATOR)


def _frame(actual_pts_seconds: float, frame_index: int) -> dict:
    return {
        "actual_pts_seconds": actual_pts_seconds,
        "frame_index": frame_index,
        "time_base": "1/1000",
        "sha256": f"{frame_index:064x}",
        "requested_time_seconds": actual_pts_seconds,
        "extraction_command_identity": "ffmpeg-extract-v1",
        "prompt_source_identity": "temporal-panel-prompt-v1",
        "request_sha256": "b" * 64,
        "runtime_identity": "cpu-contract-v1",
    }


def test_selector_recall_distinguishes_strict_from_half_second_boundary_hits(
    panel: dict,
) -> None:
    annotation = _adjudicated_annotation(
        panel,
        [
            _labelable("alpha", [[[1.0, 1.1]]]),
            {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
        ],
    )
    frames = {"alpha": [_frame(0.5, 1)]}

    strict = score_selector_recall(annotation, frames, tolerance_seconds=0.0)
    primary = score_selector_recall(annotation, frames, tolerance_seconds=0.5)

    assert strict["row_recall"] == 0.0
    assert strict["missed_video_ids"] == ["alpha"]
    assert primary["row_recall"] == 1.0
    assert primary["missed_video_ids"] == []


def test_selector_recall_requires_every_conjunctive_interval_to_hit(panel: dict) -> None:
    annotation = _adjudicated_annotation(
        panel,
        [
            _labelable("alpha", [[[1.0, 1.1], [4.0, 4.1]]]),
            {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
        ],
    )

    result = score_selector_recall(
        annotation, {"alpha": [_frame(1.0, 1)]}, tolerance_seconds=0.0
    )

    assert result["component_recall"] == 0.5
    assert result["row_recall"] == 0.0
    assert result["missed_video_ids"] == ["alpha"]


def test_selector_recall_accepts_any_complete_alternative_sufficient_set(
    panel: dict,
) -> None:
    annotation = _adjudicated_annotation(
        panel,
        [
            _labelable("alpha", [[[1.0, 1.1]], [[5.0, 5.1]]]),
            {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
        ],
    )

    result = score_selector_recall(
        annotation, {"alpha": [_frame(5.0, 1)]}, tolerance_seconds=0.0
    )

    assert result["component_recall"] == 0.5
    assert result["row_recall"] == 1.0
    assert result["rows"][0]["recalled_sufficient_set_indexes"] == [1]


def test_selector_recall_excludes_ineligible_rows_from_recall_denominators(
    panel: dict,
) -> None:
    annotation = _adjudicated_annotation(
        panel,
        [
            _labelable("alpha", [[[1.0, 1.1]]]),
            {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
        ],
    )

    result = score_selector_recall(
        annotation, {"alpha": [_frame(1.0, 1)]}, tolerance_seconds=0.0
    )

    assert result["eligible_row_count"] == 1
    assert result["component_count"] == 1
    assert result["row_recall"] == 1.0
    assert result["rows"][1]["included_in_recall"] is False


def test_selector_recall_rejects_requested_timestamp_substituted_for_actual_pts(
    panel: dict,
) -> None:
    annotation = _adjudicated_annotation(
        panel,
        [
            _labelable("alpha", [[[1.0, 1.1]]]),
            {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
        ],
    )
    requested_only = {
        "alpha": [
            {
                "requested_time_seconds": 1.0,
                "frame_index": 1,
                "time_base": "1/1000",
                "sha256": "a" * 64,
            }
        ]
    }

    with pytest.raises(ValueError, match="actual_pts_seconds"):
        score_selector_recall(annotation, requested_only, tolerance_seconds=0.0)


def test_selector_recall_rejects_annotation_mutated_after_hash_validation(
    panel: dict,
) -> None:
    annotation = _adjudicated_annotation(
        panel,
        [
            _labelable("alpha", [[[1.0, 1.1]]]),
            {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
        ],
    )
    annotation["rows"][0]["sufficient_sets"] = [[[5.0, 5.1]]]

    with pytest.raises(ValueError, match="annotation_sha256"):
        score_selector_recall(
            annotation, {"alpha": [_frame(5.0, 1)]}, tolerance_seconds=0.0
        )


def test_selector_recall_rejects_normalized_duration_mutated_after_validation(
    panel: dict,
) -> None:
    annotation = _adjudicated_annotation(
        panel,
        [
            _labelable("alpha", [[[1.0, 1.1]]]),
            {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
        ],
    )
    annotation["rows"][0]["duration_seconds"] = 100.0

    with pytest.raises(ValueError, match="normalized_annotation_sha256"):
        score_selector_recall(
            annotation, {"alpha": [_frame(1.0, 1)]}, tolerance_seconds=0.0
        )


@pytest.mark.parametrize(
    "missing_field",
    [
        "requested_time_seconds",
        "extraction_command_identity",
        "prompt_source_identity",
        "request_sha256",
        "runtime_identity",
    ],
)
def test_selector_recall_requires_complete_preinference_frame_provenance(
    panel: dict, missing_field: str
) -> None:
    annotation = _adjudicated_annotation(
        panel,
        [
            _labelable("alpha", [[[1.0, 1.1]]]),
            {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
        ],
    )
    incomplete_frame = _frame(1.0, 1)
    incomplete_frame.pop(missing_field)

    with pytest.raises(ValueError, match=missing_field):
        score_selector_recall(
            annotation, {"alpha": [incomplete_frame]}, tolerance_seconds=0.0
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("requested_time_seconds", float("nan")),
        ("extraction_command_identity", ""),
        ("prompt_source_identity", ""),
        ("request_sha256", "not-a-sha256"),
        ("runtime_identity", {}),
    ],
)
def test_selector_recall_rejects_empty_or_invalid_frame_provenance(
    panel: dict, field: str, invalid_value: object
) -> None:
    annotation = _adjudicated_annotation(
        panel,
        [
            _labelable("alpha", [[[1.0, 1.1]]]),
            {"video_id": "beta", "eligibility": ELIGIBILITY_NOT_VISUALLY_ANSWERABLE},
        ],
    )
    invalid_frame = _frame(1.0, 1)
    invalid_frame[field] = invalid_value

    with pytest.raises((TypeError, ValueError), match=field):
        score_selector_recall(
            annotation, {"alpha": [invalid_frame]}, tolerance_seconds=0.0
        )
