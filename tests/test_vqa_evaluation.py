from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from trec_video_2026.vqa_evaluation import (
    LOCAL_DIAGNOSTICS_MODE,
    NIST_2025_PARITY_MODE,
    NIST_2026_VALIDATE_AG_MODE,
    discover_submission_csvs,
    evaluate_saved_predictions,
    evaluate_local_answer,
    execute_nist_invocation,
    normalized_exact_match,
    nist_style_semantic_metrics,
    prepare_nist_invocation,
    summarize_paired_conditions,
    token_f1,
)


def write_csv(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def make_scorer_checkout(tmp_path: Path) -> Path:
    scorer = tmp_path / "nist-vqa"
    scorer.mkdir()
    (scorer / "eval.vqa.ag.py").write_text("print('parity')\n", encoding="utf-8")
    (scorer / "validate.ag.run.py").write_text(
        "print('validator')\n", encoding="utf-8"
    )
    return scorer


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ("  A DOG! ", "a dog", 1.0),
        ("The dog", "dog", 1.0),
        ("two dogs", "three dogs", 0.0),
    ],
)
def test_normalized_exact_match_ignores_case_articles_punctuation_and_spacing(
    prediction: str, reference: str, expected: float
) -> None:
    assert normalized_exact_match(prediction, reference) == expected


def test_token_f1_uses_token_multiplicity() -> None:
    assert token_f1("red red ball", "red ball ball") == pytest.approx(2 / 3)


def test_token_f1_handles_empty_answers() -> None:
    assert token_f1("", "") == 1.0
    assert token_f1("", "answer") == 0.0


def test_local_evaluation_labels_mode_and_missing_semantic_metrics() -> None:
    result = evaluate_local_answer("A person jumps", "person jumping")

    assert result == {
        "mode": LOCAL_DIAGNOSTICS_MODE,
        "normalized_exact_match": 0.0,
        "token_f1": pytest.approx(0.5),
        "sts": None,
        "meteor": None,
        "bertscore_f1": None,
        "ndcg": None,
    }


def test_local_evaluation_preserves_available_semantic_metrics() -> None:
    result = evaluate_local_answer(
        "dog", "dog", semantic_metrics={"meteor": 0.9, "bertscore_f1": 0.95}
    )

    assert result["meteor"] == 0.9
    assert result["bertscore_f1"] == 0.95
    assert result["sts"] is None
    assert result["ndcg"] is None


def test_nist_style_semantic_metrics_use_normalized_text_and_no_ndcg() -> None:
    calls = {}

    def tokenize(text: str) -> list[str]:
        return text.split()

    def meteor(references, hypothesis):
        calls["meteor"] = (references, hypothesis)
        return 0.25

    class Scores:
        def tolist(self):
            return [0.75]

    def bert(predictions, references, *, lang):
        calls["bert"] = (predictions, references, lang)
        return Scores(), Scores(), Scores()

    result = nist_style_semantic_metrics(
        ["  A DOG  "],
        ["The Dog"],
        word_tokenize=tokenize,
        meteor_score=meteor,
        bert_score=bert,
    )

    assert result == [{"meteor": 0.25, "bertscore_f1": 0.75, "sts": None, "ndcg": None}]
    assert calls["meteor"] == ([['the', 'dog']], ['a', 'dog'])
    assert calls["bert"] == (["a dog"], ["the dog"], "en")


def test_saved_predictions_regenerate_aggregate_and_stratum_scores(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(
        json.dumps(
            [
                {
                    "condition_id": "audio-off",
                    "video_id": "speech-1",
                    "status": "passed",
                    "answer": "A dog",
                    "semantic_metrics": {"meteor": 0.8},
                    "request": {
                        "video_id": "speech-1",
                        "question": "What is shown?",
                    },
                },
                {
                    "condition_id": "audio-off",
                    "video_id": "sound-1",
                    "status": "passed",
                    "answer": "piano",
                    "semantic_metrics": {"meteor": 0.4},
                    "request": {
                        "video_id": "sound-1",
                        "question": "What is heard?",
                    },
                },
            ]
        ),
        encoding="utf-8",
    )

    report = evaluate_saved_predictions(
        predictions_path,
        references={"speech-1": "dog", "sound-1": "violin"},
        strata={"speech-1": "speech", "sound-1": "sound"},
    )

    assert report["aggregate"] == {
        "example_count": 2,
        "completed_count": 2,
        "completion_rate": 1.0,
        "normalized_exact_match": 0.5,
        "token_f1": 0.5,
        "meteor": pytest.approx(0.6),
    }
    assert report["strata"]["speech"]["normalized_exact_match"] == 1.0
    assert report["strata"]["sound"]["token_f1"] == 0.0
    assert report["examples"][0]["reference"] == "dog"
    assert "reference" not in report["examples"][0]["request"]
    assert evaluate_saved_predictions(
        predictions_path,
        references={"speech-1": "dog", "sound-1": "violin"},
        strata={"speech-1": "speech", "sound-1": "sound"},
    ) == report


def test_saved_prediction_runner_requests_reject_reference_and_scoring_fields(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "leaky.json"
    predictions_path.write_text(
        json.dumps(
            [
                {
                    "condition_id": "candidate",
                    "video_id": "video-1",
                    "status": "passed",
                    "answer": "dog",
                    "request": {
                        "video_id": "video-1",
                        "question": "What?",
                        "reference_answer": "dog",
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Runner request contains unrecognized fields"):
        evaluate_saved_predictions(
            predictions_path,
            references={"video-1": "dog"},
            strata={"video-1": "control"},
        )


@pytest.mark.parametrize(
    "request_update",
    [
        {"ground_truth": "dog"},
        {"evaluation_metrics": {"exact_match": 1.0}},
        {"frame_descriptions": [[{"f1": 1.0}]]},
    ],
)
def test_saved_runner_requests_enforce_canonical_schema_recursively(
    tmp_path: Path, request_update: dict[str, object]
) -> None:
    predictions_path = tmp_path / "leaky-schema.json"
    request = {
        "run_id": "run-1",
        "condition_id": "candidate",
        "video_id": "video-1",
        "question": "What?",
        "frame_paths": [],
        "frame_timestamps": [],
        "frame_descriptions": None,
        "transcript": None,
        "media_path": None,
        "audio_path": None,
        **request_update,
    }
    predictions_path.write_text(
        json.dumps(
            [
                {
                    "condition_id": "candidate",
                    "video_id": "video-1",
                    "status": "passed",
                    "answer": "dog",
                    "request": request,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Runner request"):
        evaluate_saved_predictions(
            predictions_path,
            references={"video-1": "dog"},
            strata={"video-1": "control"},
        )


def test_saved_runner_request_accepts_the_established_schema(tmp_path: Path) -> None:
    predictions_path = tmp_path / "canonical.json"
    request = {
        "run_id": "run-1",
        "condition_id": "candidate",
        "video_id": "video-1",
        "question": "What?",
        "frame_paths": ["/tmp/frame.png"],
        "frame_timestamps": [0.5],
        "frame_descriptions": ["a dog"],
        "transcript": None,
        "media_path": "/tmp/video.mkv",
        "audio_path": None,
    }
    predictions_path.write_text(
        json.dumps(
            [
                {
                    "condition_id": "candidate",
                    "video_id": "video-1",
                    "status": "passed",
                    "answer": "dog",
                    "request": request,
                }
            ]
        ),
        encoding="utf-8",
    )

    report = evaluate_saved_predictions(
        predictions_path,
        references={"video-1": "dog"},
        strata={"video-1": "control"},
    )

    assert report["examples"][0]["request"] == request


@pytest.mark.parametrize(
    ("field", "request_value"),
    [("condition_id", "other-condition"), ("video_id", "other-video")],
)
def test_saved_runner_request_identity_must_match_enclosing_attempt(
    tmp_path: Path, field: str, request_value: str
) -> None:
    predictions_path = tmp_path / "identity-mismatch.json"
    request = {
        "run_id": "run-1",
        "condition_id": "candidate",
        "video_id": "video-1",
        "question": "What?",
    }
    request[field] = request_value
    predictions_path.write_text(
        json.dumps(
            [
                {
                    "condition_id": "candidate",
                    "video_id": "video-1",
                    "status": "passed",
                    "answer": "dog",
                    "request": request,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=f"Attempt/request identity mismatch for {field}"):
        evaluate_saved_predictions(
            predictions_path,
            references={"video-1": "dog"},
            strata={"video-1": "control"},
        )


def test_superseded_jsonl_attempt_request_identity_is_still_validated(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "identity-attempts.jsonl"
    attempts = [
        {
            "condition_id": "candidate",
            "video_id": "video-1",
            "status": "failed",
            "answer": None,
            "request": {
                "condition_id": "candidate",
                "video_id": "nested-other-video",
                "question": "What?",
            },
        },
        {
            "condition_id": "candidate",
            "video_id": "video-1",
            "status": "passed",
            "answer": "dog",
        },
    ]
    predictions_path.write_text(
        "".join(json.dumps(attempt) + "\n" for attempt in attempts),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="identity mismatch for video_id"):
        evaluate_saved_predictions(
            predictions_path,
            references={"video-1": "dog"},
            strata={"video-1": "control"},
        )


@pytest.mark.parametrize(
    "mutation",
    ["missing_audio_path", "string_timestamp", "non_string_question"],
)
def test_full_canonical_runner_requests_validate_required_fields_and_types(
    tmp_path: Path, mutation: str
) -> None:
    predictions_path = tmp_path / "invalid-canonical.json"
    request: dict[str, object] = {
        "run_id": "run-1",
        "condition_id": "candidate",
        "video_id": "video-1",
        "question": "What?",
        "frame_paths": ["/tmp/frame.png"],
        "frame_timestamps": [0.5],
        "frame_descriptions": ["a dog"],
        "transcript": None,
        "media_path": "/tmp/video.mkv",
        "audio_path": None,
    }
    if mutation == "missing_audio_path":
        del request["audio_path"]
    elif mutation == "string_timestamp":
        request["frame_timestamps"] = ["0.5"]
    else:
        request["question"] = 12
    predictions_path.write_text(
        json.dumps(
            [
                {
                    "condition_id": "candidate",
                    "video_id": "video-1",
                    "status": "passed",
                    "answer": "dog",
                    "request": request,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical RunnerRequest"):
        evaluate_saved_predictions(
            predictions_path,
            references={"video-1": "dog"},
            strata={"video-1": "control"},
        )


def test_saved_jsonl_attempts_keep_a_success_over_a_later_failure(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "attempts.jsonl"
    attempts = [
        {"condition_id": "candidate", "video_id": "v1", "status": "failed", "answer": None},
        {"condition_id": "candidate", "video_id": "v1", "status": "passed", "answer": "dog"},
        {"condition_id": "candidate", "video_id": "v1", "status": "passed", "answer": "  "},
        {"condition_id": "candidate", "video_id": "v1", "status": "timed_out", "answer": None},
    ]
    predictions_path.write_text(
        "".join(json.dumps(attempt) + "\n" for attempt in attempts),
        encoding="utf-8",
    )

    report = evaluate_saved_predictions(
        predictions_path,
        references={"v1": "dog"},
        strata={"v1": "speech"},
    )

    assert len(report["examples"]) == 1
    assert report["examples"][0]["status"] == "passed"
    assert report["examples"][0]["normalized_exact_match"] == 1.0


def test_paired_summary_rejects_identical_condition_ids() -> None:
    with pytest.raises(ValueError, match="must be different"):
        summarize_paired_conditions(
            [],
            baseline_condition_id="same",
            treatment_condition_id="same",
        )


def test_paired_audio_summary_reports_deltas_wins_losses_and_seeded_intervals(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "paired.json"
    predictions_path.write_text(
        json.dumps(
            [
                {"condition_id": "off", "video_id": "v1", "status": "passed", "answer": "wrong", "semantic_metrics": {"meteor": 0.1}},
                {"condition_id": "on", "video_id": "v1", "status": "passed", "answer": "dog", "semantic_metrics": {"meteor": 0.9}},
                {"condition_id": "off", "video_id": "v2", "status": "passed", "answer": "cat", "semantic_metrics": {"meteor": 0.8}},
                {"condition_id": "on", "video_id": "v2", "status": "passed", "answer": "wrong", "semantic_metrics": {"meteor": 0.2}},
                {"condition_id": "off", "video_id": "v3", "status": "passed", "answer": "bird", "semantic_metrics": {"meteor": 0.7}},
                {"condition_id": "on", "video_id": "v3", "status": "passed", "answer": "bird", "semantic_metrics": {"meteor": 0.7}},
            ]
        ),
        encoding="utf-8",
    )
    scored = evaluate_saved_predictions(
        predictions_path,
        references={"v1": "dog", "v2": "cat", "v3": "bird"},
        strata={"v1": "audio", "v2": "audio", "v3": "control"},
    )

    first = summarize_paired_conditions(
        scored["examples"],
        baseline_condition_id="off",
        treatment_condition_id="on",
        bootstrap_iterations=500,
        bootstrap_seed=17,
    )
    second = summarize_paired_conditions(
        scored["examples"],
        baseline_condition_id="off",
        treatment_condition_id="on",
        bootstrap_iterations=500,
        bootstrap_seed=17,
    )

    assert first == second
    assert first["paired_count"] == 3
    assert first["exact_match_wins"] == 1
    assert first["exact_match_losses"] == 1
    assert first["metric_deltas"] == {
        "normalized_exact_match": pytest.approx(0.0),
        "token_f1": pytest.approx(0.0),
        "meteor": pytest.approx(0.06666666666666667),
    }
    assert first["bootstrap_intervals"]["meteor"] == {
        "lower": pytest.approx(-0.6),
        "upper": pytest.approx(0.8),
        "confidence": 0.95,
        "iterations": 500,
        "seed": 17,
        "exploratory": True,
    }
    assert first["strata"]["audio"]["exact_match_wins"] == 1
    assert first["strata"]["audio"]["exact_match_losses"] == 1
    assert first["strata"]["control"]["metric_deltas"]["meteor"] == 0.0


def test_csv_discovery_excludes_scorer_outputs(tmp_path: Path) -> None:
    for name in (
        "submission.csv",
        "submission_diagnostic.csv",
        "submission_per_response.csv",
        "evaluation_results.csv",
        "notes.txt",
    ):
        (tmp_path / name).write_text("", encoding="utf-8")

    assert discover_submission_csvs(tmp_path) == [tmp_path / "submission.csv"]


def test_parity_invocation_rejects_scorer_output_as_submission(tmp_path: Path) -> None:
    scorer = make_scorer_checkout(tmp_path)
    references = tmp_path / "references.csv"
    submission = tmp_path / "evaluation_results.csv"
    write_csv(references, [[1, "video", "What?", "dog"]])
    write_csv(submission, [[1, "video", 1, "dog", 0.2]])

    with pytest.raises(ValueError, match="looks like scorer output"):
        prepare_nist_invocation(
            NIST_2025_PARITY_MODE,
            scorer_dir=scorer,
            references_path=references,
            run_path=submission,
            isolated_dir=tmp_path / "isolated",
        )


def test_parity_invocation_stages_one_csv_and_uses_argv_not_a_shell(
    tmp_path: Path,
) -> None:
    scorer = make_scorer_checkout(tmp_path)
    references = tmp_path / "references.csv"
    submission = tmp_path / "candidate.csv"
    write_csv(references, [[1, "video", "What?", "a dog"]])
    write_csv(submission, [[1, "video", 1, "a dog", 0.2]])

    invocation = prepare_nist_invocation(
        NIST_2025_PARITY_MODE,
        scorer_dir=scorer,
        references_path=references,
        run_path=submission,
        isolated_dir=tmp_path / "isolated",
    )

    assert invocation.mode == NIST_2025_PARITY_MODE
    assert invocation.command == (
        sys.executable,
        str((scorer / "eval.vqa.ag.py").resolve()),
        str(references.resolve()),
        str(invocation.run_dir.resolve()),
    )
    assert invocation.cwd == scorer.resolve()
    assert invocation.uses_shell is False
    assert discover_submission_csvs(invocation.run_dir) == [
        invocation.run_dir / "candidate.csv"
    ]
    assert (invocation.run_dir / "candidate.csv").read_bytes() == submission.read_bytes()


def test_execute_nist_invocation_uses_sanitized_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer = make_scorer_checkout(tmp_path)
    script = scorer / "eval.vqa.ag.py"
    script.write_text(
        "import json, os\n"
        "print(json.dumps({'cwd': os.getcwd(), 'secret': os.getenv('VQA_TEST_SECRET')}))\n",
        encoding="utf-8",
    )
    references = tmp_path / "references.csv"
    submission = tmp_path / "candidate.csv"
    write_csv(references, [[1, "video", "What?", "dog"]])
    write_csv(submission, [[1, "video", 1, "dog", 0.2]])
    invocation = prepare_nist_invocation(
        NIST_2025_PARITY_MODE,
        scorer_dir=scorer,
        references_path=references,
        run_path=submission,
        isolated_dir=tmp_path / "isolated",
    )
    monkeypatch.setenv("VQA_TEST_SECRET", "must-not-leak")

    result = execute_nist_invocation(invocation, timeout_seconds=2)

    payload = json.loads(result.stdout)
    assert result.mode == NIST_2025_PARITY_MODE
    assert result.returncode == 0
    assert payload == {"cwd": str(scorer.resolve()), "secret": None}


def test_parity_invocation_rejects_nonempty_isolation_directory(tmp_path: Path) -> None:
    scorer = make_scorer_checkout(tmp_path)
    references = tmp_path / "references.csv"
    submission = tmp_path / "candidate.csv"
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    (isolated / "old.csv").write_text("old", encoding="utf-8")
    write_csv(references, [[1, "video", "What?", "dog"]])
    write_csv(submission, [[1, "video", 1, "dog", 0.2]])

    with pytest.raises(ValueError, match="must be empty"):
        prepare_nist_invocation(
            NIST_2025_PARITY_MODE,
            scorer_dir=scorer,
            references_path=references,
            run_path=submission,
            isolated_dir=isolated,
        )


@pytest.mark.parametrize("unsafe_answer", ['bad " quote', "$(touch nope)", "`id`"])
def test_parity_invocation_rejects_shell_active_answer_text(
    tmp_path: Path, unsafe_answer: str
) -> None:
    scorer = make_scorer_checkout(tmp_path)
    references = tmp_path / "references.csv"
    submission = tmp_path / "candidate.csv"
    write_csv(references, [[1, "video", "What?", "dog"]])
    write_csv(submission, [[1, "video", 1, unsafe_answer, 0.2]])

    with pytest.raises(ValueError, match="unsafe for the upstream scorer"):
        prepare_nist_invocation(
            NIST_2025_PARITY_MODE,
            scorer_dir=scorer,
            references_path=references,
            run_path=submission,
            isolated_dir=tmp_path / "isolated",
        )


def test_validator_invocation_has_distinct_mode_and_report(tmp_path: Path) -> None:
    scorer = make_scorer_checkout(tmp_path)
    topics = tmp_path / "topics.json"
    submission = tmp_path / "submission.json"
    topics.write_text("[]", encoding="utf-8")
    submission.write_text("[]", encoding="utf-8")

    invocation = prepare_nist_invocation(
        NIST_2026_VALIDATE_AG_MODE,
        scorer_dir=scorer,
        references_path=topics,
        run_path=submission,
        isolated_dir=tmp_path / "isolated",
    )

    assert invocation.mode == NIST_2026_VALIDATE_AG_MODE
    assert invocation.command[-2:] == (
        "--report-json",
        str(invocation.report_path.resolve()),
    )
    assert invocation.command[1] == str((scorer / "validate.ag.run.py").resolve())


def test_unknown_nist_mode_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported NIST mode"):
        prepare_nist_invocation(
            "official-2026-score",
            scorer_dir=tmp_path,
            references_path=tmp_path / "references",
            run_path=tmp_path / "run",
            isolated_dir=tmp_path / "isolated",
        )
