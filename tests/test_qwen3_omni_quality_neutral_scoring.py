import copy
import json

import pytest

from trec_video_2026.vqa_experiments import qwen3_omni_quality_scoring as a1_scoring
from trec_video_2026.vqa_experiments import (
    qwen3_omni_quality_neutral_scoring as neutral_scoring,
)


def _a2_rows() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (
            neutral_scoring.DEFAULT_RUN_DIRECTORY / "predictions.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]


def _failed_row() -> dict[str, object]:
    return next(row for row in _a2_rows() if row["status"] == "failed")


def test_terminal_fenced_json_recovery_is_narrow_and_terminal():
    """Catches a post-hoc parser that accepts prose or content after the fence."""
    raw = (
        "<think>diagnostic reasoning</think>\n\n"
        "```json\n"
        '{\n  "answer": "fixture answer",\n  "evidence": "fixture evidence"\n}\n'
        "```"
    )

    assert neutral_scoring._parse_terminal_fenced_json(raw) == {
        "answer": "fixture answer",
        "evidence": "fixture evidence",
    }
    for invalid in (
        raw + "\ntrailing prose",
        raw.replace("```json", "```"),
        raw + "\n" + raw,
        raw.replace('"evidence": "fixture evidence"', '"extra": "value"'),
    ):
        with pytest.raises(ValueError, match="terminal fenced JSON"):
            neutral_scoring._parse_terminal_fenced_json(invalid)


@pytest.mark.parametrize(
    "mutation",
    ("video_id", "condition_id", "raw_output", "raw_output_sha256", "status"),
)
def test_post_hoc_recovery_accepts_only_the_pinned_failed_row(mutation: str):
    """Catches recovery broadening beyond the one immutable A2 parser failure."""
    row = copy.deepcopy(_failed_row())
    if mutation == "video_id":
        row["video_id"] = "Mi-NpnCBegw"
    elif mutation == "condition_id":
        row["condition_id"] = "frames-only"
    elif mutation == "raw_output":
        row["raw_output"] += "\n"
    elif mutation == "raw_output_sha256":
        row["modality_attestation"]["generation"]["raw_output_sha256"] = "0" * 64
    else:
        row["status"] = "passed"

    with pytest.raises(ValueError, match="pinned A2 recovery"):
        neutral_scoring._recover_pinned_failure(row)


def test_a2_strict_failure_stays_ineligible_while_diagnostic_recovers_one_row():
    """Catches post-hoc recovery being misrepresented as strict run success."""
    result = neutral_scoring._default_score()

    assert result["strict_result"] == {
        "eligible": False,
        "run_succeeded": False,
        "strict_passed_rows": 19,
        "strict_failed_rows": 1,
        "failed_video_id": "NkXPPJoZ-h8",
        "failed_condition_id": "frames-native-audio",
        "failure_type": "output_parse_error",
    }
    assert result["post_hoc_semantic_diagnostic"]["eligible_for_strict_gate"] is False
    assert result["post_hoc_semantic_diagnostic"]["recovered_rows"] == 1
    recovered = [
        row for row in result["examples"] if row["diagnostic_recovery"] is not None
    ]
    assert [(row["video_id"], row["condition_id"]) for row in recovered] == [
        ("NkXPPJoZ-h8", "frames-native-audio")
    ]
    assert recovered[0]["diagnostic_recovery"]["raw_output_sha256"] == (
        "0dbb246367b2abdd7e1810d39597acf2cab9326c453e54db3299d07dd719872a"
    )


def test_a2_strict_pass_count_requires_each_passed_row_modality_contract():
    """Catches counting parser success as strict success after attestation drift."""
    row = copy.deepcopy(next(row for row in _a2_rows() if row["status"] == "passed"))
    assert neutral_scoring._passed_row_is_strict(row) is True

    row["modality_attestation"]["vision_element_count"] = 11

    assert neutral_scoring._passed_row_is_strict(row) is False


def test_existing_strict_scorer_still_rejects_a2_incomplete_run():
    """Catches weakening A1's complete-run refusal to accommodate A2 diagnostics."""
    contract = a1_scoring._load_json(neutral_scoring.DEFAULT_CONTRACT_PATH)

    with pytest.raises(ValueError, match="complete frozen run"):
        a1_scoring._require_complete_frozen_run(
            run_dir=neutral_scoring.DEFAULT_RUN_DIRECTORY,
            contract=contract,
            expected_run_id=neutral_scoring.RUN_ID,
            expected_predictions_sha256=neutral_scoring.PREDICTIONS_SHA256,
            expected_contract_sha256=a1_scoring.QUALITY_CONTRACT_SHA256,
        )


def test_a2_official_scores_and_a1_comparison_are_frozen():
    """Catches metric or comparison drift in the published A2 diagnostic."""
    result = neutral_scoring._default_score()

    assert result["aggregate"]["frames-only"] == {
        "correct": 3,
        "partial": 3,
        "wrong": 4,
        "normalized_exact_match": 0.1,
        "token_f1": pytest.approx(0.22947809722558646),
    }
    assert result["aggregate"]["frames-native-audio"] == {
        "correct": 5,
        "partial": 2,
        "wrong": 3,
        "normalized_exact_match": 0.1,
        "token_f1": pytest.approx(0.220422693837328),
    }
    assert result["aggregate"]["all_conditions"] == {
        "correct": 8,
        "partial": 5,
        "wrong": 7,
        "normalized_exact_match": 0.1,
        "token_f1": pytest.approx(0.2249503955314572),
    }
    assert result["paired_deltas"]["audio_minus_frames"]["token_f1"] == (
        pytest.approx(-0.00905540338825848)
    )
    assert result["comparison_to_a1"]["frames-native-audio"] == {
        "correct_delta": 2,
        "partial_delta": 0,
        "wrong_delta": -2,
        "normalized_exact_match_delta": 0.0,
        "token_f1_delta": pytest.approx(-0.0395951868327266),
    }


def test_a2_sanitized_result_omits_reference_text():
    """Catches annotation answer text leaking into the tracked A2 artifact."""
    result = neutral_scoring._default_score()

    assert all("reference" not in row for row in result["examples"])
    assert all("correct_answer" not in row for row in result["examples"])
    assert all("reference_sha256" in row for row in result["examples"])
