from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_vqa_grouped_rewrite_canary as runner  # noqa: E402


QUERY_IDS = (1, 310, 1563, 2404, 2421)


def _write_json(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def canary_inputs(tmp_path: Path) -> dict[str, Path]:
    questions = [
        {
            "query_id": query_id,
            "video_id": f"video-{query_id}",
            "question": f"What happens in scene {query_id}?",
            "source_sha256": f"source-{query_id}",
            "frame_paths": [f"private/frame-{query_id}.jpg"],
        }
        for query_id in QUERY_IDS
    ]
    gemini = [
        {
            "query_id": query_id,
            "video_id": f"video-{query_id}",
            "status": "succeeded",
            "answer": f"gemini answer {query_id}",
            "usage": {"prompt_tokens": 10, "output_tokens": 2},
            "raw_response_sha256": "provider-detail",
        }
        for query_id in QUERY_IDS
    ]
    qwen = [
        {
            "query_id": query_id,
            "video_id": f"video-{query_id}",
            "condition_id": condition,
            "status": "succeeded",
            "answer": f"{condition} answer {query_id}",
            "raw_response_sha256": "provider-detail",
        }
        for query_id in QUERY_IDS
        for condition in ("video-plus-transcript", "native-video")
    ]
    baseline = [
        {
            "Q_ID": query_id,
            "Video_ID": f"video-{query_id}",
            "Answers": [
                {
                    "Rank": 1,
                    "Answer": f"control answer {query_id}",
                    "Time": 0.0,
                },
                {
                    "Rank": 2,
                    "Answer": f"speculative tail {query_id}",
                    "Time": 99.0,
                },
            ],
            "labels": {"answer": "must not enter request"},
        }
        for query_id in QUERY_IDS
    ]
    fusion = [
        {
            "Q_ID": query_id,
            "Video_ID": f"video-{query_id}",
            "Answers": [
                {
                    "Rank": 1,
                    "Answer": (
                        f"gemini answer {query_id}"
                        if query_id in {1, 2421}
                        else f"video-plus-transcript answer {query_id}"
                    ),
                    "Time": 0.0,
                },
                {
                    "Rank": 2,
                    "Answer": f"fusion tail {query_id}",
                    "Time": 5.0,
                },
            ],
        }
        for query_id in QUERY_IDS
    ]
    provenance = {
        "rows": [
            {
                "Q_ID": query_id,
                "Video_ID": f"video-{query_id}",
                "selected_sources": [
                    "gemini" if query_id in {1, 2421} else "qwen_video_plus_transcript",
                    "baseline",
                ],
                "baseline_ranks_used": [1, 2],
                "metrics": {"hidden_label": 1.0},
            }
            for query_id in QUERY_IDS
        ],
        "annotations": {"must not enter request": True},
    }
    return {
        "questions": _write_json(tmp_path / "questions.json", {"rows": questions}),
        "gemini": _write_json(tmp_path / "gemini.json", {"rows": gemini}),
        "qwen": _write_json(tmp_path / "qwen.json", {"rows": qwen}),
        "baseline": _write_json(tmp_path / "baseline.json", baseline),
        "fusion": _write_json(tmp_path / "fusion.json", fusion),
        "provenance": _write_json(tmp_path / "provenance.json", provenance),
    }


def _selection_kwargs(paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "questions_path": paths["questions"],
        "gemini_overlay_path": paths["gemini"],
        "qwen_aggregate_path": paths["qwen"],
        "qwen35_control_path": paths["baseline"],
        "fusion_submission_path": paths["fusion"],
        "fusion_provenance_path": paths["provenance"],
    }


def test_select_rows_uses_exact_panel_rank_one_sources_and_no_forbidden_fields(
    canary_inputs: dict[str, Path],
) -> None:
    rows = runner.select_rows(**_selection_kwargs(canary_inputs))

    assert [row["query_id"] for row in rows] == list(QUERY_IDS)
    assert [row["rank1_source"] for row in rows] == [
        "gemini_overlay",
        "qwen_video_plus_transcript",
        "qwen_video_plus_transcript",
        "qwen_video_plus_transcript",
        "gemini_overlay",
    ]
    assert all(len(row["source_candidates"]) == 4 for row in rows)
    assert all(
        candidate["original_rank"] == 1
        for row in rows
        for candidate in row["source_candidates"]
    )
    serialized = json.dumps(rows, sort_keys=True)
    for forbidden in ("speculative tail", "fusion tail", "must not enter request"):
        assert forbidden not in serialized
    def keys(value: Any) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(child) for child in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(child) for child in value))
        return set()
    assert keys(rows).isdisjoint(
        {"labels", "annotations", "metrics", "frame_paths", "transcript", "baseline_ranks_used", "raw_response_sha256"}
    )


def test_build_request_is_one_batched_text_only_request_with_fixed_protocol(
    canary_inputs: dict[str, Path],
) -> None:
    rows = runner.select_rows(**_selection_kwargs(canary_inputs))
    request = runner.build_request(rows)

    assert request["model"] == "gemini-3.7-flash"
    assert request["seed"] == 42
    assert request["max_output_tokens"] == 8192
    assert request["response_mime_type"] == "application/json"
    assert request["must_separate_source_pairs"] == [
        {"query_id": query_id, "source_ids": list(pair)}
        for query_id, pair in runner.MUST_SEPARATE_SOURCE_PAIRS.items()
    ]
    assert request["request_count"] == 1
    assert len(request["queries"]) == 5
    assert all(query["requested_answer_count"] == 10 for query in request["queries"])
    assert "video_id" not in json.dumps(request).lower()
    assert "media" not in json.dumps(request).lower()


def _valid_response(rows: list[dict[str, Any]]) -> dict[str, Any]:
    queries = []
    for row in rows:
        source_ids = [candidate["source_id"] for candidate in row["source_candidates"]]
        groups = [
            {"group_id": f"{row['query_id']}-g{index}", "source_ids": [source_id]}
            for index, source_id in enumerate(source_ids, start=1)
        ]
        rank1_group = next(
            group["group_id"]
            for group in groups
            if row["rank1_source"] in group["source_ids"]
        )
        answers = [
            {
                "rank": 1,
                "answer": row["rank1_answer"],
                "source_group": rank1_group,
            }
        ]
        answers.extend(
            {
                "rank": rank,
                "answer": f"{row['query_id']} rewritten answer {rank}",
                "source_group": groups[(rank - 1) % len(groups)]["group_id"],
            }
            for rank in range(2, 11)
        )
        queries.append(
            {
                "query_id": row["query_id"],
                "groups": groups,
                "answers": answers,
            }
        )
    return {"queries": queries}


def _valid_strategy_response(
    rows: list[dict[str, Any]], strategy: str
) -> dict[str, Any]:
    if strategy == runner.STRATEGY_SEMANTIC_REWRITE_A:
        return _valid_response(rows)
    queries = []
    for row in rows:
        groups = [
            {"group_id": f"{row['query_id']}-g{index}", "source_ids": [candidate["source_id"]]}
            for index, candidate in enumerate(row["source_candidates"], start=1)
        ]
        rank1_group = next(
            group["group_id"]
            for group in groups
            if row["rank1_source"] in group["source_ids"]
        )
        answers = [
            {"rank": 1, "answer": row["rank1_answer"], "source_group": rank1_group}
        ]
        for group in groups:
            candidate = next(
                item
                for item in row["source_candidates"]
                if item["source_id"] in group["source_ids"]
            )
            if candidate["answer"] != answers[0]["answer"]:
                answers.append(
                    {
                        "rank": len(answers) + 1,
                        "answer": candidate["answer"],
                        "source_group": group["group_id"],
                    }
                )
        while len(answers) < 10:
            answers.append(
                {
                    "rank": len(answers) + 1,
                    "answer": f"{row['query_id']} hedge answer {len(answers) + 1}",
                    "source_group": groups[(len(answers) - 1) % len(groups)]["group_id"],
                }
            )
        queries.append(
            {"query_id": row["query_id"], "groups": groups, "answers": answers[:10]}
        )
    return {"queries": queries}


def test_validate_response_preserves_rank_one_groups_sources_and_ten_distinct_answers(
    canary_inputs: dict[str, Path],
) -> None:
    rows = runner.select_rows(**_selection_kwargs(canary_inputs))
    validated = runner.validate_response(_valid_response(rows), rows)

    assert validated["status"] == "validated"
    assert validated["query_count"] == 5
    assert validated["answer_count"] == 50
    assert validated["source_group_coverage"] == "complete"


def test_semantic_strategy_allows_differently_worded_equivalent_sources(
    canary_inputs: dict[str, Path],
) -> None:
    rows = runner.select_rows(**_selection_kwargs(canary_inputs))
    payload = _valid_response(rows)
    first = payload["queries"][0]
    first["groups"][0]["source_ids"].append(rows[0]["source_candidates"][1]["source_id"])
    first["groups"].pop(1)
    for answer in first["answers"]:
        if answer["source_group"] == "1-g2":
            answer["source_group"] = "1-g1"

    validated = runner.validate_response(
        payload, rows, strategy=runner.STRATEGY_SEMANTIC_REWRITE_A
    )
    assert validated["status"] == "validated"


def test_validator_rejects_a_predeclared_conflict_pair_merge(
    canary_inputs: dict[str, Path],
) -> None:
    rows = runner.select_rows(**_selection_kwargs(canary_inputs))
    payload = _valid_response(rows)
    query = next(item for item in payload["queries"] if item["query_id"] == 1563)
    pair = runner.MUST_SEPARATE_SOURCE_PAIRS[1563]
    first_group = next(
        group for group in query["groups"] if pair[0] in group["source_ids"]
    )
    second_group = next(
        group for group in query["groups"] if pair[1] in group["source_ids"]
    )
    first_group["source_ids"].append(pair[1])
    query["groups"].remove(second_group)
    for answer in query["answers"]:
        if answer["source_group"] == second_group["group_id"]:
            answer["source_group"] = first_group["group_id"]

    with pytest.raises(ValueError, match="must remain separate"):
        runner.validate_response(payload, rows)


def test_every_group_must_be_represented_by_an_answer(
    canary_inputs: dict[str, Path],
) -> None:
    rows = runner.select_rows(**_selection_kwargs(canary_inputs))
    payload = _valid_response(rows)
    unused_group = payload["queries"][0]["groups"][-1]["group_id"]
    for answer in payload["queries"][0]["answers"]:
        if answer["source_group"] == unused_group:
            answer["source_group"] = payload["queries"][0]["groups"][0]["group_id"]
    with pytest.raises(ValueError, match="represented"):
        runner.validate_response(payload, rows)


def test_exact_group_hedge_requires_an_exact_source_answer_per_group(
    canary_inputs: dict[str, Path],
) -> None:
    rows = runner.select_rows(**_selection_kwargs(canary_inputs))
    payload = _valid_strategy_response(rows, runner.STRATEGY_EXACT_GROUP_HEDGE_B)
    runner.validate_response(payload, rows, strategy=runner.STRATEGY_EXACT_GROUP_HEDGE_B)

    group = payload["queries"][0]["groups"][1]["group_id"]
    answer = next(item for item in payload["queries"][0]["answers"] if item["source_group"] == group)
    answer["answer"] = "a safe paraphrase that is not exact"
    with pytest.raises(ValueError, match="exact source"):
        runner.validate_response(payload, rows, strategy=runner.STRATEGY_EXACT_GROUP_HEDGE_B)


def test_verified_paraphrase_strategy_freezes_two_stage_protocol_and_isolated_verifier(
    canary_inputs: dict[str, Path],
) -> None:
    rows = runner.select_rows(**_selection_kwargs(canary_inputs))
    request = runner.build_request(rows, strategy=runner.STRATEGY_VERIFIED_PARAPHRASE_C)

    assert request["strategy_id"] == runner.STRATEGY_VERIFIED_PARAPHRASE_C
    assert request["request_count"] == 2
    assert request["response_json_schema"] == runner.RESPONSE_JSON_SCHEMA
    assert request["verifier_response_json_schema"] == runner.VERIFIER_RESPONSE_JSON_SCHEMA
    verifier_request = runner.build_verifier_request(
        rows,
        _valid_strategy_response(rows, runner.STRATEGY_VERIFIED_PARAPHRASE_C),
        strategy=runner.STRATEGY_VERIFIED_PARAPHRASE_C,
    )
    assert verifier_request["strategy_id"] == runner.STRATEGY_VERIFIED_PARAPHRASE_C
    for item in verifier_request["items"]:
        assert set(item) == {"query_id", "group_id", "question", "source_candidates", "answers"}
        assert len(item["source_candidates"]) >= 1
        assert all(set(answer) == {"answer_id", "answer_sha256", "text"} for answer in item["answers"])
        assert "instructions" not in item
        assert "rank" not in json.dumps(item)


def test_verifier_accepts_synonym_and_rejects_unsupported_claim(
    canary_inputs: dict[str, Path],
) -> None:
    rows = runner.select_rows(**_selection_kwargs(canary_inputs))
    generated = runner.validate_response(
        _valid_strategy_response(rows, runner.STRATEGY_VERIFIED_PARAPHRASE_C),
        rows,
        strategy=runner.STRATEGY_VERIFIED_PARAPHRASE_C,
    )
    verifier_request = runner.build_verifier_request(
        rows, generated, strategy=runner.STRATEGY_VERIFIED_PARAPHRASE_C
    )
    payload = {
        "verifications": [
            {
                "query_id": item["query_id"],
                "group_id": item["group_id"],
                "answers": [
                    {
                        "answer_id": answer["answer_id"],
                        "answer_sha256": answer["answer_sha256"],
                        "entailed": True,
                        "unsupported_claims": [],
                        "supporting_source_ids": [item["source_candidates"][0]["source_id"]],
                        "supporting_quotes": [
                            {
                                "source_id": item["source_candidates"][0]["source_id"],
                                "quote": item["source_candidates"][0]["answer"],
                            }
                        ],
                    }
                    for answer in item["answers"]
                ],
            }
            for item in verifier_request["items"]
        ]
    }
    runner.validate_verifier_response(payload, verifier_request)
    payload["verifications"][0]["answers"][0]["entailed"] = False
    payload["verifications"][0]["answers"][0]["unsupported_claims"] = [
        {"category": "manner", "claim": "slowly"}
    ]
    rejected = runner.validate_verifier_response(payload, verifier_request)
    assert rejected["status"] == "verification_rejected"
    assert rejected["verification_counts"]["total"] == 50
    assert rejected["verification_counts"]["rejected"] == 1
    assert rejected["unsupported_claim_categories"] == {"manner": 1}


def test_validate_response_rejects_cross_query_and_rank_one_rewrite(
    canary_inputs: dict[str, Path],
) -> None:
    rows = runner.select_rows(**_selection_kwargs(canary_inputs))
    payload = _valid_response(rows)

    payload["queries"][0]["answers"][0]["answer"] = "rank one from another query"
    with pytest.raises(ValueError, match="rank 1"):
        runner.validate_response(payload, rows)

    payload = _valid_response(rows)
    payload["queries"][0]["answers"][1]["source_group"] = payload["queries"][1]["groups"][0]["group_id"]
    with pytest.raises(ValueError, match="source group"):
        runner.validate_response(payload, rows)


def test_validate_response_rejects_missing_source_and_duplicate_answers(
    canary_inputs: dict[str, Path],
) -> None:
    rows = runner.select_rows(**_selection_kwargs(canary_inputs))
    payload = _valid_response(rows)
    payload["queries"][0]["groups"] = payload["queries"][0]["groups"][:-1]
    with pytest.raises(ValueError, match="source candidate"):
        runner.validate_response(payload, rows)

    payload = _valid_response(rows)
    payload["queries"][0]["answers"][2]["answer"] = payload["queries"][0]["answers"][1]["answer"]
    with pytest.raises(ValueError, match="distinct"):
        runner.validate_response(payload, rows)


def test_cost_caps_are_hard_for_estimate_and_observed_usage() -> None:
    rows = [
        {
            "query_id": query_id,
            "question": "What happens?",
            "rank1_answer": "answer",
            "rank1_source": "control",
            "source_candidates": [
                {"source_id": "control", "original_rank": 1, "answer": "answer"}
            ],
        }
        for query_id in QUERY_IDS
    ]
    request = runner.build_request(rows)
    assert request["estimated_cost_usd"] < 0.10

    payload = _valid_response(rows)
    payload["usage"] = {"prompt_tokens": 200_000, "output_tokens": 8_192}
    with pytest.raises(ValueError, match="cost"):
        runner.validate_response(payload, rows)


def test_dry_run_does_not_create_private_output_root(
    canary_inputs: dict[str, Path], tmp_path: Path
) -> None:
    output_root = tmp_path / "private-run"
    result = runner.run_canary(
        **_selection_kwargs(canary_inputs),
        output_root=output_root,
        run=False,
    )

    assert result["status"] == "dry_run"
    assert result["query_count"] == 5
    assert result["request_count"] == 1
    assert result["requested_answer_count"] == 50
    assert result["estimated_cost_usd"] < 0.10
    assert not output_root.exists()


def test_real_sdk_response_is_allowlisted_and_usage_is_billed_conservatively(
    canary_inputs: dict[str, Path],
) -> None:
    rows = runner.select_rows(**_selection_kwargs(canary_inputs))
    raw = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=json.dumps(_valid_response(rows)))],
                ),
                finish_reason=types.FinishReason.STOP,
                safety_ratings=[],
            )
        ],
        model_version="gemini-test",
        response_id="response-id",
        prompt_feedback=types.GenerateContentResponsePromptFeedback(
            block_reason=types.BlockedReason.BLOCKED_REASON_UNSPECIFIED,
            safety_ratings=[],
        ),
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100,
            candidates_token_count=40,
            total_token_count=170,
            thoughts_token_count=30,
        ),
    )
    payload = runner._response_payload(raw)
    assert payload["usage"] == {
        "prompt_tokens": 100,
        "response_tokens": 40,
        "total_tokens": 170,
        "thoughts_tokens": 30,
        "output_tokens": 70,
    }
    assert runner._observed_usage(payload) == (100, 70)
    safe = runner._safe_raw(raw)
    assert safe["usageMetadata"] == payload["usage"]
    assert safe["candidates"][0]["content"]["parts"][0]["text"]
    assert safe["candidates"][0]["finishReason"] == "STOP"
    assert "safetyRatings" in safe["candidates"][0]
    assert safe["promptFeedback"]["blockReason"] == "BLOCKED_REASON_UNSPECIFIED"
    assert "sdkHttpResponse" not in safe


def test_strategy_request_schema_and_distinct_default_roots(
    canary_inputs: dict[str, Path],
) -> None:
    rows = runner.select_rows(**_selection_kwargs(canary_inputs))
    request_a = runner.build_request(rows, strategy=runner.STRATEGY_SEMANTIC_REWRITE_A)
    request_b = runner.build_request(rows, strategy=runner.STRATEGY_EXACT_GROUP_HEDGE_B)

    assert request_a["strategy_id"] == runner.STRATEGY_SEMANTIC_REWRITE_A
    assert request_b["strategy_id"] == runner.STRATEGY_EXACT_GROUP_HEDGE_B
    assert request_a["response_json_schema"] == runner.RESPONSE_JSON_SCHEMA
    assert request_a["config"]["response_json_schema"] == runner.RESPONSE_JSON_SCHEMA
    assert set(request_a["response_json_schema"]["properties"]) == {"queries"}
    assert request_a["response_json_schema"]["additionalProperties"] is False
    assert request_a["response_json_schema"] == request_b["response_json_schema"]
    assert runner.PRIVATE_RUN_ROOTS[runner.STRATEGY_VERIFIED_PARAPHRASE_C] not in {
        runner.PRIVATE_RUN_ROOTS[runner.STRATEGY_SEMANTIC_REWRITE_A],
        runner.PRIVATE_RUN_ROOTS[runner.STRATEGY_EXACT_GROUP_HEDGE_B],
    }
    assert runner.PRIVATE_RUN_ROOTS[runner.STRATEGY_SEMANTIC_REWRITE_A] != runner.PRIVATE_RUN_ROOTS[
        runner.STRATEGY_EXACT_GROUP_HEDGE_B
    ]
    assert runner.STRATEGY_SEMANTIC_REWRITE_A in str(
        runner.PRIVATE_RUN_ROOTS[runner.STRATEGY_SEMANTIC_REWRITE_A]
    )
    assert runner.STRATEGY_EXACT_GROUP_HEDGE_B in str(
        runner.PRIVATE_RUN_ROOTS[runner.STRATEGY_EXACT_GROUP_HEDGE_B]
    )


def test_live_call_uses_the_frozen_response_schema_in_provider_config(
    canary_inputs: dict[str, Path],
) -> None:
    rows = runner.select_rows(**_selection_kwargs(canary_inputs))
    request = runner.build_request(rows, strategy=runner.STRATEGY_SEMANTIC_REWRITE_A)

    class Models:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def generate_content(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(kwargs)
            return {"text": json.dumps(_valid_response(rows))}

    models = Models()
    runner._live_call(SimpleNamespace(models=models), request)
    assert len(models.calls) == 1
    config = models.calls[0]["config"]
    assert config.response_json_schema == request["response_json_schema"]
    sent = json.loads(models.calls[0]["contents"])
    assert set(sent) == {"instructions", "queries"}
    assert all(set(query) == {"query_id", "question", "rank1_source", "rank1_answer", "source_candidates", "requested_answer_count"} for query in sent["queries"])


def test_dry_run_both_strategies_freezes_all_six_source_bindings(
    canary_inputs: dict[str, Path], tmp_path: Path
) -> None:
    results = {}
    for strategy in (
        runner.STRATEGY_SEMANTIC_REWRITE_A,
        runner.STRATEGY_EXACT_GROUP_HEDGE_B,
    ):
        result = runner.run_canary(
            **_selection_kwargs(canary_inputs),
            output_root=tmp_path / strategy,
            strategy=strategy,
            run=False,
        )
        results[strategy] = result
        assert result["status"] == "dry_run"
        assert result["strategy_id"] == strategy
        assert len(result["source_artifacts"]) == 6
        assert all(set(item) == {"name", "path", "sha256"} for item in result["source_artifacts"])
        assert result["estimated_cost_usd"] < 0.10
        assert not (tmp_path / strategy).exists()
    assert results[runner.STRATEGY_SEMANTIC_REWRITE_A]["source_artifacts"] == results[
        runner.STRATEGY_EXACT_GROUP_HEDGE_B
    ]["source_artifacts"]


def test_dry_run_verified_paraphrase_is_two_stage_and_under_combined_cap(
    canary_inputs: dict[str, Path], tmp_path: Path
) -> None:
    result = runner.run_canary(
        **_selection_kwargs(canary_inputs),
        output_root=tmp_path / runner.STRATEGY_VERIFIED_PARAPHRASE_C,
        strategy=runner.STRATEGY_VERIFIED_PARAPHRASE_C,
        run=False,
    )

    assert result["status"] == "dry_run"
    assert result["request_count"] == 2
    assert result["estimated_total_cost_usd"] < 0.10
    assert not (tmp_path / runner.STRATEGY_VERIFIED_PARAPHRASE_C).exists()


def test_verified_paraphrase_fake_client_freezes_both_stage_chain(
    canary_inputs: dict[str, Path], tmp_path: Path
) -> None:
    rows = runner.select_rows(**_selection_kwargs(canary_inputs))
    generated = _valid_strategy_response(rows, runner.STRATEGY_VERIFIED_PARAPHRASE_C)

    class Models:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def generate_content(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return {
                    "text": json.dumps(generated),
                    "usage": {"prompt_tokens": 100, "output_tokens": 100},
                }
            sent = json.loads(kwargs["contents"])
            verifications = []
            for item in sent["items"]:
                source = item["source_candidates"][0]
                verifications.append(
                    {
                        "query_id": item["query_id"],
                        "group_id": item["group_id"],
                        "answers": [
                            {
                                "answer_id": answer["answer_id"],
                                "answer_sha256": answer["answer_sha256"],
                                "entailed": True,
                                "unsupported_claims": [],
                                "supporting_source_ids": [source["source_id"]],
                                "supporting_quotes": [
                                    {
                                        "source_id": source["source_id"],
                                        "quote": source["answer"],
                                    }
                                ],
                            }
                            for answer in item["answers"]
                        ],
                    }
                )
            return {
                "text": json.dumps({"verifications": verifications}),
                "usage": {"prompt_tokens": 100, "output_tokens": 100},
            }

    models = Models()
    root = tmp_path / runner.STRATEGY_VERIFIED_PARAPHRASE_C
    summary = runner.run_canary(
        **_selection_kwargs(canary_inputs),
        output_root=root,
        strategy=runner.STRATEGY_VERIFIED_PARAPHRASE_C,
        client=SimpleNamespace(models=models),
        run=True,
        created_at="2026-08-17T00:00:00Z",
    )
    assert len(models.calls) == 2
    assert summary["request_count"] == 2
    assert summary["verifier_answer_count"] == 50
    assert (root / "verifier-request.json").exists()
    assert (root / "verifier-raw-response.json").exists()
    assert (root / "verifier-parsed-response.json").exists()
    assert (root / "terminal.json").exists()


def test_source_artifact_binding_changes_when_input_changes(
    canary_inputs: dict[str, Path],
) -> None:
    before = runner._source_artifacts(**_selection_kwargs(canary_inputs))
    canary_inputs["questions"].write_text(
        canary_inputs["questions"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    after = runner._source_artifacts(**_selection_kwargs(canary_inputs))
    assert before[0]["path"] == after[0]["path"]
    assert before[0]["sha256"] != after[0]["sha256"]
    with pytest.raises(ValueError, match="changed"):
        runner._assert_source_artifacts_unchanged(
            before, **_selection_kwargs(canary_inputs)
        )
