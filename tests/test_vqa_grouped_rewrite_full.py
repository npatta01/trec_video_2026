from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_vqa_grouped_rewrite_full as runner  # noqa: E402


def _write(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, Path]:
    query_ids = range(1, 13)
    questions = [
        {"query_id": qid, "video_id": f"video-{qid}", "question": f"What happens {qid}?"}
        for qid in query_ids
    ]
    gemini = [
        {"query_id": qid, "video_id": f"video-{qid}", "status": "succeeded", "answer": f"gemini answer {qid}", "time": 1.0}
        for qid in query_ids
    ]
    qwen = [
        {"query_id": qid, "condition_id": condition, "status": "succeeded", "answer": f"{condition} answer {qid}", "time": 2.0}
        for qid in query_ids
        for condition in ("video-plus-transcript", "native-video")
    ]
    control = [
        {"Q_ID": qid, "Video_ID": f"video-{qid}", "Answers": [{"Rank": 1, "Answer": f"control answer {qid}", "Time": 3.0}]}
        for qid in query_ids
    ]
    fusion = [
        {
            "Q_ID": qid,
            "Video_ID": f"video-{qid}",
            "Answers": [
                {"Rank": rank, "Answer": (f"gemini answer {qid}" if rank == 1 else f"fusion answer {qid} {rank}"), "Time": float(rank)}
                for rank in range(1, 11)
            ],
        }
        for qid in query_ids
    ]
    provenance = [
        {"Q_ID": qid, "Video_ID": f"video-{qid}", "selected_sources": ["gemini"]}
        for qid in query_ids
    ]
    return {
        "questions": _write(tmp_path / "questions.json", {"rows": questions}),
        "gemini": _write(tmp_path / "gemini.json", {"rows": gemini}),
        "qwen": _write(tmp_path / "qwen.json", {"rows": qwen}),
        "control": _write(tmp_path / "control.json", control),
        "fusion": _write(tmp_path / "fusion.json", fusion),
        "provenance": _write(tmp_path / "provenance.json", provenance),
        "prior_root": tmp_path / "prior-root",
    }


def _kwargs(paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "questions": paths["questions"],
        "gemini": paths["gemini"],
        "qwen": paths["qwen"],
        "control": paths["control"],
        "fusion": paths["fusion"],
        "provenance": paths["provenance"],
    }


def _rows(paths: dict[str, Path]) -> list[dict[str, Any]]:
    return runner.load_rows(**_kwargs(paths), expected_query_count=12)


def _run_kwargs(paths: dict[str, Path], *, expected_query_count: int = 12) -> dict[str, Any]:
    paths["prior_root"].mkdir(parents=True, exist_ok=True)
    return {
        **_kwargs(paths),
        "prior_root": paths["prior_root"],
        "exposed_batch_count": 0,
        "expected_query_count": expected_query_count,
    }


def _generated(batch: list[dict[str, Any]]) -> dict[str, Any]:
    queries = []
    for row in batch:
        groups = [
            {"group_id": f"{row['query_id']}-g{index}", "source_ids": [candidate["source_id"]]}
            for index, candidate in enumerate(row["source_candidates"], start=1)
        ]
        answers = []
        for index, candidate in enumerate(row["source_candidates"], start=1):
            answers.append({"rank": index, "answer": candidate["answer"], "source_group": groups[index - 1]["group_id"]})
        first = row["source_candidates"][0]["answer"]
        for suffix in ("the", "a", "in", "with", "for", "to"):
            answers.append({"rank": len(answers) + 1, "answer": f"{suffix} {first}", "source_group": groups[0]["group_id"]})
        queries.append({"query_id": row["query_id"], "groups": groups, "answers": answers})
    return {"queries": queries}


def test_loads_full_panel_and_batches_resumably(inputs: dict[str, Path]) -> None:
    rows = _rows(inputs)
    assert len(rows) == 12
    assert all(len(row["source_candidates"]) == 4 for row in rows)
    grouped = runner.batches(rows)
    assert [len(batch) for batch in grouped] == [5, 5, 2]
    assert runner.source_artifacts(**_kwargs(inputs))[0]["name"] == "questions"


def test_generator_only_contract(inputs: dict[str, Path]) -> None:
    rows = _rows(inputs)
    batch = runner.batches(rows)[0]
    request = runner.build_generator_request(batch, 1)
    generated = runner.validate_generator(_generated(batch), batch)
    assert request["max_output_tokens"] == 8_192
    assert request["strategy_id"] == "grouped_unique_generator_only"
    assert request["stage"] == "generator"
    assert "verifier" not in json.dumps(request).lower()
    query_schema = request["response_json_schema"]["properties"]["queries"]
    group_schema = query_schema["items"]["properties"]["groups"]["items"]
    answer_schema = query_schema["items"]["properties"]["answers"]["items"]
    assert query_schema["minItems"] == query_schema["maxItems"] == 5
    assert group_schema["additionalProperties"] is False
    assert set(group_schema["properties"]) == {"group_id", "source_ids"}
    assert answer_schema["additionalProperties"] is False
    assert set(answer_schema["properties"]) == {"rank", "answer", "source_group"}
    assert generated["answer_count"] == 50


def test_provider_config_receives_exact_generator_schema(inputs: dict[str, Path]) -> None:
    batch = runner.batches(_rows(inputs))[0]
    request = runner.build_generator_request(batch, 1)

    class Models:
        def generate_content(self, **kwargs: Any) -> dict[str, Any]:
            self.kwargs = kwargs
            return {"text": "{}"}

    models = Models()
    runner._call(SimpleNamespace(models=models), request)
    schema = models.kwargs["config"].response_json_schema
    assert schema == request["response_json_schema"]
    assert schema["properties"]["queries"]["minItems"] == 5


def test_full_dry_run_is_provider_free_and_under_global_cap(inputs: dict[str, Path], tmp_path: Path) -> None:
    result = runner.run_full(**_run_kwargs(inputs), output_root=tmp_path / "full")
    assert result["status"] == "dry_run"
    assert result["query_count"] == 12
    assert result["batch_count"] == 3
    assert result["request_count"] == 3
    assert result["estimated_cost_usd"] < 25
    assert not (tmp_path / "full").exists()


def test_custom_cost_cap_is_bound_and_gates_before_provider_dispatch(
    inputs: dict[str, Path], tmp_path: Path
) -> None:
    dry = runner.run_full(
        **_run_kwargs(inputs),
        output_root=tmp_path / "dry",
        max_cost_usd=5.0,
    )
    assert dry["max_cost_usd"] == 5.0
    assert dry["estimated_cost_usd"] < dry["max_cost_usd"]

    class Models:
        calls = 0

        def generate_content(self, **_: Any) -> dict[str, Any]:
            self.calls += 1
            raise AssertionError("cost cap must gate before provider dispatch")

    models = Models()
    with pytest.raises(ValueError, match="estimated full-panel cost exceeds run cap"):
        runner.run_full(
            **_run_kwargs(inputs),
            output_root=tmp_path / "blocked",
            max_cost_usd=0.000001,
            run=True,
            client=SimpleNamespace(models=models),
        )
    assert models.calls == 0


def test_subset_and_cost_cap_are_available_through_cli(
    inputs: dict[str, Path], capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    args = [
        "--questions",
        str(inputs["questions"]),
        "--gemini",
        str(inputs["gemini"]),
        "--qwen",
        str(inputs["qwen"]),
        "--control",
        str(inputs["control"]),
        "--fusion",
        str(inputs["fusion"]),
        "--provenance",
        str(inputs["provenance"]),
        "--prior-root",
        str(inputs["prior_root"]),
        "--output-root",
        str(tmp_path / "cli"),
        "--exposed-batch-count",
        "0",
        "--expected-query-count",
        "12",
        "--max-cost-usd",
        "5",
    ]
    inputs["prior_root"].mkdir()

    assert runner.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["query_count"] == 12
    assert result["max_cost_usd"] == 5.0


@pytest.mark.parametrize("invalid_cap", [True, 0.0, -1.0, float("nan"), float("inf"), 50.01])
def test_invalid_custom_cost_caps_fail_closed(inputs: dict[str, Path], invalid_cap: float) -> None:
    with pytest.raises(ValueError, match="max-cost-usd must be finite"):
        runner.run_full(**_run_kwargs(inputs), max_cost_usd=invalid_cap)


def test_fallback_preserves_existing_objects(inputs: dict[str, Path]) -> None:
    rows = _rows(inputs)
    organized = runner._organize(rows, {}, {})
    assert organized[0] == rows[0]["fallback"]
    assert all(answer["Rank"] == index for index, answer in enumerate(organized[0]["Answers"], start=1))


def test_generated_times_use_one_measured_batch_elapsed_value(inputs: dict[str, Path]) -> None:
    rows = _rows(inputs)
    generated = _generated([rows[0]])["queries"][0]
    organized = runner._organize(rows[:1], {1: generated}, {1: 2.5})
    assert {answer["Time"] for answer in organized[0]["Answers"]} == {2.5}


def test_generator_rejects_duplicate_sources_and_bool_identity(inputs: dict[str, Path]) -> None:
    batch = runner.batches(_rows(inputs))[0]
    payload = _generated(batch)
    payload["queries"][0]["groups"][0]["source_ids"].append(
        payload["queries"][0]["groups"][0]["source_ids"][0]
    )
    with pytest.raises(ValueError, match="source coverage"):
        runner.validate_generator(payload, batch)

    payload = _generated(batch)
    payload["queries"][0]["query_id"] = True
    with pytest.raises(ValueError, match="query identity"):
        runner.validate_generator(payload, batch)

    payload = _generated(batch)
    payload["queries"][0]["answers"][0]["rank"] = True
    with pytest.raises(ValueError, match="answer fields"):
        runner.validate_generator(payload, batch)

    payload = _generated(batch)
    payload["queries"][0]["groups"][0]["answer"] = "legacy extra field"
    with pytest.raises(ValueError, match="group schema"):
        runner.validate_generator(payload, batch)

    payload = _generated(batch)
    payload["queries"][0]["groups"][0]["group_id"] = 1
    with pytest.raises(ValueError, match="group identity"):
        runner.validate_generator(payload, batch)

    payload = _generated(batch)
    payload["queries"][0]["answers"][0] = "legacy answer string"
    with pytest.raises(ValueError, match="answer schema"):
        runner.validate_generator(payload, batch)


def test_prior_root_exposure_and_one_batch_smoke_resume(
    inputs: dict[str, Path], tmp_path: Path
) -> None:
    prior = inputs["prior_root"]
    prior.mkdir(parents=True)
    (prior / "batch-0001").mkdir()
    _write(prior / "batch-0001" / "failed.json", {"status": "failed", "accounted_cost_usd": 1.25})

    class Models:
        def __init__(self) -> None:
            self.calls = 0

        def generate_content(self, **kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            body = json.loads(kwargs["contents"])
            queries = []
            for query in body["queries"]:
                groups = [
                    {"group_id": f"{query['query_id']}-g{index}", "source_ids": [candidate["source_id"]]}
                    for index, candidate in enumerate(query["source_candidates"], start=1)
                ]
                answers = [
                    {"rank": index, "answer": candidate["answer"], "source_group": groups[index - 1]["group_id"]}
                    for index, candidate in enumerate(query["source_candidates"], start=1)
                ]
                first = query["source_candidates"][0]["answer"]
                for suffix in ("the", "a", "in", "with", "for", "to"):
                    answers.append({"rank": len(answers) + 1, "answer": f"{suffix} {first}", "source_group": groups[0]["group_id"]})
                queries.append({"query_id": query["query_id"], "groups": groups, "answers": answers})
            return {"text": json.dumps({"queries": queries}), "usage": {"prompt_tokens": 100, "output_tokens": 100}}

    root = tmp_path / "a2"
    smoke_models = Models()
    smoke = runner.run_full(
        **_kwargs(inputs),
        prior_root=prior,
        exposed_batch_count=1,
        stop_after_fresh_batches=1,
        output_root=root,
        run=True,
        client=SimpleNamespace(models=smoke_models),
        expected_query_count=12,
    )
    assert smoke["status"] == "stopped"
    assert smoke_models.calls == 1
    assert (root / "manifest.json").exists()
    assert (root / "batch-0001" / "exposed-fallback.json").exists()
    exposed = json.loads((root / "batch-0001" / "exposed-fallback.json").read_text(encoding="utf-8"))
    assert exposed["source_root"] == str(prior.resolve())
    assert exposed["prior_root_inventory_sha256"] == runner._sha_bytes(runner._canonical(runner.prior_root_inventory(prior)))
    assert exposed["prior_accounting"]["declared_terminal_cost_usd"] == 1.25
    assert not (root / "final-organizer.json").exists()

    resume_models = Models()
    completed = runner.run_full(
        **_kwargs(inputs),
        prior_root=prior,
        exposed_batch_count=1,
        output_root=root,
        run=True,
        client=SimpleNamespace(models=resume_models),
        expected_query_count=12,
    )
    assert completed["status"] == "complete"
    assert resume_models.calls == 1


def test_paid_malformed_response_falls_back_and_persists_reservation(
    inputs: dict[str, Path], tmp_path: Path
) -> None:
    class Models:
        def __init__(self) -> None:
            self.calls = 0

        def generate_content(self, **_: Any) -> dict[str, Any]:
            self.calls += 1
            return {"text": "not-json", "usage": {"prompt_tokens": 100, "output_tokens": 100}}

    models = Models()
    root = tmp_path / "full"
    result = runner.run_full(
        **_run_kwargs(inputs),
        output_root=root,
        run=True,
        client=SimpleNamespace(models=models),
    )
    assert result["status"] == "complete"
    assert result["fallback_query_count"] == 12
    assert result["accounted_cost_usd"] > 0
    assert models.calls == 3
    assert (root / "batch-0001" / "attempt-0001" / "raw-response.json").exists()
    assert (root / "batch-0001" / "attempt-0001" / "error.json").exists()
    assert (root / "batch-0001" / "failed.json").exists()

    resumed = Models()
    resumed_result = runner.run_full(
        **_run_kwargs(inputs),
        output_root=root,
        run=True,
        client=SimpleNamespace(models=resumed),
    )
    assert resumed.calls == 0
    assert resumed_result["status"] == "complete"


def test_manifest_drift_is_rejected_before_provider_dispatch(
    inputs: dict[str, Path], tmp_path: Path
) -> None:
    root = tmp_path / "drift"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"schema_version": 1}) + "\n", encoding="utf-8")

    class Models:
        calls = 0

        def generate_content(self, **_: Any) -> dict[str, Any]:
            self.calls += 1
            raise AssertionError("manifest drift must gate before provider dispatch")

    models = Models()
    with pytest.raises(ValueError, match="manifest/config/source hash drift"):
        runner.run_full(
            **_run_kwargs(inputs),
            output_root=root,
            run=True,
            client=SimpleNamespace(models=models),
        )
    assert models.calls == 0


def test_interrupted_attempt_becomes_fallback_without_provider_repeat(
    inputs: dict[str, Path], tmp_path: Path
) -> None:
    inputs["prior_root"].mkdir(parents=True, exist_ok=True)
    artifacts = runner.source_artifacts(**_kwargs(inputs))
    prior_inventory = runner.prior_root_inventory(inputs["prior_root"])
    prior_accounting = runner.prior_root_accounting(inputs["prior_root"])
    rows_for_manifest = _rows(inputs)
    requests_for_manifest = [runner.build_generator_request(batch, index + 1) for index, batch in enumerate(runner.batches(rows_for_manifest))]
    manifest = {
        "schema_version": 1,
        "status": "inputs_frozen",
        "strategy_id": runner.STRATEGY,
        "query_count": 12,
        "batch_count": 3,
        "batch_size": runner.BATCH_SIZE,
        "model": runner.MODEL,
        "seed": runner.SEED,
        "generator_max_output_tokens": runner.GENERATOR_MAX_OUTPUT_TOKENS,
        "max_concurrency": runner.MAX_BATCH_CONCURRENCY,
        "global_max_cost_usd": runner.GLOBAL_MAX_COST_USD,
        "prior_root": str(inputs["prior_root"].resolve()),
        "prior_root_inventory": prior_inventory,
        "prior_root_inventory_sha256": runner._sha_bytes(runner._canonical(prior_inventory)),
        "prior_accounting": prior_accounting,
        "prior_accounting_sha256": runner._sha_bytes(runner._canonical(prior_accounting)),
        "exposed_batch_count": 0,
        "exposed_batch_ids": [],
        "exposed_batch_bindings": [],
        "generator_response_schema_sha256": runner._sha_bytes(runner._canonical(runner._generator_response_schema(runner.BATCH_SIZE))),
        "request_bundle_sha256": runner._sha_bytes(runner._canonical(requests_for_manifest)),
        "source_artifacts": artifacts,
        "source_artifacts_sha256": runner._sha_bytes(runner._canonical(artifacts)),
    }
    root = tmp_path / "interrupted"
    for batch_id in (1, 2, 3):
        (root / f"batch-{batch_id:04d}" / "attempt-0001").mkdir(parents=True)
    (root / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    for batch_id in (1, 2, 3):
        (root / f"batch-{batch_id:04d}" / "attempt-0001" / "request.json").write_text("{}\n", encoding="utf-8")

    class Models:
        calls = 0

        def generate_content(self, **_: Any) -> dict[str, Any]:
            self.calls += 1
            raise AssertionError("ambiguous attempts must not call provider")

    models = Models()
    result = runner.run_full(
        **_run_kwargs(inputs),
        output_root=root,
        run=True,
        client=SimpleNamespace(models=models),
    )
    assert models.calls == 0
    assert result["fallback_query_count"] == 12
    assert (root / "batch-0001" / "failed.json").exists()


def test_partial_finalization_reuses_valid_final_without_provider_call(
    inputs: dict[str, Path], tmp_path: Path
) -> None:
    class Models:
        def __init__(self) -> None:
            self.calls = 0

        def generate_content(self, **kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            body = json.loads(kwargs["contents"])
            queries = []
            for query in body["queries"]:
                groups = [
                    {"group_id": f"{query['query_id']}-g{index}", "source_ids": [candidate["source_id"]]}
                    for index, candidate in enumerate(query["source_candidates"], start=1)
                ]
                answers = [
                    {"rank": index, "answer": candidate["answer"], "source_group": groups[index - 1]["group_id"]}
                    for index, candidate in enumerate(query["source_candidates"], start=1)
                ]
                first = query["source_candidates"][0]["answer"]
                for suffix in ("the", "a", "in", "with", "for", "to"):
                    answers.append({"rank": len(answers) + 1, "answer": f"{suffix} {first}", "source_group": groups[0]["group_id"]})
                queries.append({"query_id": query["query_id"], "groups": groups, "answers": answers})
            return {
                "text": json.dumps({"queries": queries}),
                "usage": {"prompt_tokens": 100, "output_tokens": 100},
            }

    root = tmp_path / "partial"
    first_models = Models()
    first = runner.run_full(
        **_run_kwargs(inputs),
        output_root=root,
        run=True,
        client=SimpleNamespace(models=first_models),
    )
    assert first["status"] == "complete"
    assert first_models.calls == 3
    (root / "summary.json").unlink()

    resumed_models = Models()
    recovered = runner.run_full(
        **_run_kwargs(inputs),
        output_root=root,
        run=True,
        client=SimpleNamespace(models=resumed_models),
    )
    assert resumed_models.calls == 0
    assert recovered["status"] == "complete"
    assert (root / "summary.json").exists()
    assert recovered["final_sha256"] == runner._sha_file(root / "final-organizer.json")
    assert recovered["provenance_sha256"] == runner._sha_file(root / "final-provenance.json")


def test_complete_summary_hashes_are_verified_before_reuse(
    inputs: dict[str, Path], tmp_path: Path
) -> None:
    class Models:
        calls = 0

        def generate_content(self, **_: Any) -> dict[str, Any]:
            self.calls += 1
            raise AssertionError("corrupt complete root must not dispatch")

    root = tmp_path / "corrupt"
    inputs["prior_root"].mkdir(parents=True, exist_ok=True)
    artifacts = runner.source_artifacts(**_kwargs(inputs))
    prior_inventory = runner.prior_root_inventory(inputs["prior_root"])
    prior_accounting = runner.prior_root_accounting(inputs["prior_root"])
    rows_for_manifest = _rows(inputs)
    requests_for_manifest = [runner.build_generator_request(batch, index + 1) for index, batch in enumerate(runner.batches(rows_for_manifest))]
    manifest = {
        "schema_version": 1,
        "status": "inputs_frozen",
        "strategy_id": runner.STRATEGY,
        "query_count": 12,
        "batch_count": 3,
        "batch_size": runner.BATCH_SIZE,
        "model": runner.MODEL,
        "seed": runner.SEED,
        "generator_max_output_tokens": runner.GENERATOR_MAX_OUTPUT_TOKENS,
        "max_concurrency": runner.MAX_BATCH_CONCURRENCY,
        "global_max_cost_usd": runner.GLOBAL_MAX_COST_USD,
        "prior_root": str(inputs["prior_root"].resolve()),
        "prior_root_inventory": prior_inventory,
        "prior_root_inventory_sha256": runner._sha_bytes(runner._canonical(prior_inventory)),
        "prior_accounting": prior_accounting,
        "prior_accounting_sha256": runner._sha_bytes(runner._canonical(prior_accounting)),
        "exposed_batch_count": 0,
        "exposed_batch_ids": [],
        "exposed_batch_bindings": [],
        "generator_response_schema_sha256": runner._sha_bytes(runner._canonical(runner._generator_response_schema(runner.BATCH_SIZE))),
        "request_bundle_sha256": runner._sha_bytes(runner._canonical(requests_for_manifest)),
        "source_artifacts": artifacts,
        "source_artifacts_sha256": runner._sha_bytes(runner._canonical(artifacts)),
    }
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    final_rows = runner._organize(_rows(inputs), {}, {})
    _write(root / "final-organizer.json", final_rows)
    _write(root / "final-provenance.json", runner._build_provenance(_rows(inputs), final_rows))
    _write(
        root / "summary.json",
        {
            "status": "complete",
            "final_sha256": "wrong",
            "provenance_sha256": runner._sha_file(root / "final-provenance.json"),
        },
    )
    models = Models()
    with pytest.raises(ValueError, match="summary hash"):
        runner.run_full(
            **_run_kwargs(inputs),
            output_root=root,
            run=True,
            client=SimpleNamespace(models=models),
        )
    assert models.calls == 0
