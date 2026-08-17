from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_vqa_grouped_rewrite_full as runner  # noqa: E402
import salvage_vqa_grouped_rewrite_full as salvage  # noqa: E402


def _write(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def source_root(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    qids = range(1, 5)
    questions = [{"query_id": qid, "video_id": f"v{qid}", "question": f"What happens {qid}?"} for qid in qids]
    gemini = [{"query_id": qid, "video_id": f"v{qid}", "status": "succeeded", "answer": f"gemini {qid}", "time": 1.0} for qid in qids]
    qwen = [
        {"query_id": qid, "condition_id": condition, "status": "succeeded", "answer": f"{condition} {qid}", "time": 2.0}
        for qid in qids
        for condition in ("video-plus-transcript", "native-video")
    ]
    control = [{"Q_ID": qid, "Video_ID": f"v{qid}", "Answers": [{"Rank": 1, "Answer": f"control {qid}", "Time": 3.0}]} for qid in qids]
    fusion = [{"Q_ID": qid, "Video_ID": f"v{qid}", "Answers": [{"Rank": rank, "Answer": f"fusion {qid} {rank}", "Time": float(rank)} for rank in range(1, 11)]} for qid in qids]
    fusion[0]["Answers"][0]["Answer"] = "gemini 1"
    fusion[1]["Answers"][0]["Answer"] = "gemini 2"
    fusion[2]["Answers"][0]["Answer"] = "gemini 3"
    fusion[3]["Answers"][0]["Answer"] = "gemini 4"
    provenance = [{"Q_ID": qid, "Video_ID": f"v{qid}", "selected_sources": ["gemini"]} for qid in qids]
    paths = {
        "questions": _write(tmp_path / "questions.json", {"rows": questions}),
        "gemini": _write(tmp_path / "gemini.json", {"rows": gemini}),
        "qwen": _write(tmp_path / "qwen.json", {"rows": qwen}),
        "control": _write(tmp_path / "control.json", control),
        "fusion": _write(tmp_path / "fusion.json", fusion),
        "provenance": _write(tmp_path / "provenance.json", provenance),
    }
    root = tmp_path / "a3"
    rows = runner.load_rows(**paths, expected_query_count=4)
    batches = runner.batches(rows, batch_size=2)
    requests = [runner.build_generator_request(batch, index + 1) for index, batch in enumerate(batches)]
    artifacts = runner.source_artifacts(**paths)
    manifest = {
        "schema_version": 1,
        "strategy_id": runner.STRATEGY,
        "query_count": 4,
        "batch_count": 2,
        "batch_size": 2,
        "model": runner.MODEL,
        "seed": runner.SEED,
        "generator_max_output_tokens": runner.GENERATOR_MAX_OUTPUT_TOKENS,
        "source_artifacts": artifacts,
        "source_artifacts_sha256": runner._sha_bytes(runner._canonical(artifacts)),
        "request_bundle_sha256": runner._sha_bytes(runner._canonical(requests)),
        "generator_response_schema_sha256": runner._sha_bytes(runner._canonical(runner._generator_response_schema(2))),
    }
    _write(root / "manifest.json", manifest)

    def generated(batch: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
        queries = []
        for row in batch:
            groups = [{"group_id": f"g{index}", "source_ids": [item["source_id"]]} for index, item in enumerate(row["source_candidates"], 1)]
            answers = [{"rank": index, "answer": item["answer"], "source_group": groups[index - 1]["group_id"]} for index, item in enumerate(row["source_candidates"], 1)]
            first = row["source_candidates"][0]["answer"]
            for suffix in ("the", "a", "in", "with", "for", "to"):
                answers.append({"rank": len(answers) + 1, "answer": f"{prefix} {suffix} {first}", "source_group": groups[0]["group_id"]})
            queries.append({"query_id": row["query_id"], "groups": groups, "answers": answers})
        return {"queries": queries}

    accepted_payload = generated(batches[0], "accepted")
    request_artifact_hash = runner._sha_bytes(runner._canonical(artifacts))
    accepted_request = dict(requests[0])
    accepted_request.update(source_artifacts=artifacts, source_artifacts_sha256=request_artifact_hash)
    failed_request = dict(requests[1])
    failed_request.update(source_artifacts=artifacts, source_artifacts_sha256=request_artifact_hash)
    _write(root / "batch-0001" / "attempt-0001" / "request.json", accepted_request)
    _write(root / "batch-0001" / "attempt-0001" / "parsed-response.json", runner.validate_generator(accepted_payload, batches[0]))
    _write(root / "batch-0001" / "accepted.json", {"status": "accepted", "elapsed_seconds": 2.0, "queries": accepted_payload["queries"]})
    failed_payload = generated([batches[1][0]], "salvaged")
    raw = {"provider_response": {"candidates": [{"content": {"parts": [{"text": json.dumps(failed_payload)}]}}]}}
    _write(root / "batch-0002" / "attempt-0001" / "request.json", failed_request)
    _write(root / "batch-0002" / "attempt-0001" / "raw-response.json", raw)
    _write(root / "batch-0002" / "failed.json", {"status": "failed", "elapsed_seconds": 0.0, "accounted_cost_usd": 0.1})

    accepted_by_qid = {query["query_id"]: query for query in accepted_payload["queries"]}
    source_final_rows = []
    for row in rows:
        final_row = dict(row["fallback"])
        generated_query = accepted_by_qid.get(row["query_id"])
        if generated_query is not None:
            final_row["Answers"] = [
                {"Rank": answer["rank"], "Answer": answer["answer"], "Time": 2.0}
                for answer in generated_query["answers"]
            ]
        source_final_rows.append(final_row)
    source_provenance = runner._build_provenance(rows, source_final_rows)
    final_hash = runner._sha_file(_write(root / "final-organizer.json", source_final_rows))
    provenance_hash = runner._sha_file(_write(root / "final-provenance.json", source_provenance))
    _write(
        root / "summary.json",
        {
            "status": "complete",
            "final_sha256": final_hash,
            "provenance_sha256": provenance_hash,
            "source_artifacts_sha256": manifest["source_artifacts_sha256"],
        },
    )
    return root, paths


def test_salvage_mixed_batch_preserves_accepted_and_fallbacks_invalid(
    source_root: tuple[Path, dict[str, Path]], tmp_path: Path
) -> None:
    root, paths = source_root
    result = salvage.salvage_run(source_root=root, output_root=tmp_path / "salvage", source_paths=paths, expected_query_count=4)
    assert result["status"] == "complete"
    assert result["salvaged_query_count"] == 3
    assert result["fallback_query_count"] == 1
    final = json.loads((tmp_path / "salvage" / "final-organizer.json").read_text(encoding="utf-8"))
    assert final[0]["Answers"][0]["Answer"] == "gemini 1"
    assert final[2]["Answers"][0]["Answer"] == "gemini 3"
    assert final[2]["Answers"][4]["Answer"].startswith("salvaged ")
    assert final[3]["Answers"][0]["Answer"] == "gemini 4"


def test_salvage_malformed_raw_falls_back(source_root: tuple[Path, dict[str, Path]], tmp_path: Path) -> None:
    root, paths = source_root
    raw_path = root / "batch-0002" / "attempt-0001" / "raw-response.json"
    _write(raw_path, {"provider_response": {"candidates": [{"content": {"parts": [{"text": "not json"}]}}]}})
    result = salvage.salvage_run(source_root=root, output_root=tmp_path / "salvage", source_paths=paths, expected_query_count=4)
    assert result["salvaged_query_count"] == 2
    assert result["fallback_query_count"] == 2


def test_salvage_rejects_source_hash_drift_and_overwrite(source_root: tuple[Path, dict[str, Path]], tmp_path: Path) -> None:
    root, paths = source_root
    output = tmp_path / "salvage"
    salvage.salvage_run(source_root=root, output_root=output, source_paths=paths, expected_query_count=4)
    with pytest.raises(FileExistsError):
        salvage.salvage_run(source_root=root, output_root=output, source_paths=paths, expected_query_count=4)
    paths["questions"].write_text(paths["questions"].read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source artifact hash"):
        salvage.salvage_run(source_root=root, output_root=tmp_path / "drift", source_paths=paths, expected_query_count=4)


def test_salvage_requires_complete_terminal_source(source_root: tuple[Path, dict[str, Path]], tmp_path: Path) -> None:
    root, paths = source_root
    (root / "summary.json").unlink()
    with pytest.raises(ValueError, match="complete source summary/final/provenance"):
        salvage.salvage_run(source_root=root, output_root=tmp_path / "missing", source_paths=paths, expected_query_count=4)


def test_salvage_rejects_source_summary_hash_drift(source_root: tuple[Path, dict[str, Path]], tmp_path: Path) -> None:
    root, paths = source_root
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["final_sha256"] = "0" * 64
    _write(summary_path, summary)
    with pytest.raises(ValueError, match="source summary final hash"):
        salvage.salvage_run(source_root=root, output_root=tmp_path / "summary-drift", source_paths=paths, expected_query_count=4)


def test_salvage_rejects_source_provenance_hash_drift(source_root: tuple[Path, dict[str, Path]], tmp_path: Path) -> None:
    root, paths = source_root
    provenance_path = root / "final-provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance[0]["status"] = "tampered"
    _write(provenance_path, provenance)
    with pytest.raises(ValueError, match="source summary provenance hash"):
        salvage.salvage_run(source_root=root, output_root=tmp_path / "provenance-drift", source_paths=paths, expected_query_count=4)


def test_salvage_request_drift_falls_back_the_affected_batch(
    source_root: tuple[Path, dict[str, Path]], tmp_path: Path
) -> None:
    root, paths = source_root
    request_path = root / "batch-0001" / "attempt-0001" / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["queries"][0]["question"] = "tampered"
    _write(request_path, request)
    result = salvage.salvage_run(source_root=root, output_root=tmp_path / "request-drift", source_paths=paths, expected_query_count=4)
    assert result["salvaged_query_count"] == 1
    assert result["fallback_query_count"] == 3


def test_salvage_accepted_marker_drift_falls_back_entire_batch(
    source_root: tuple[Path, dict[str, Path]], tmp_path: Path
) -> None:
    root, paths = source_root
    marker_path = root / "batch-0001" / "accepted.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["queries"][0]["answers"][0]["answer"] = "tampered"
    _write(marker_path, marker)
    result = salvage.salvage_run(source_root=root, output_root=tmp_path / "accepted-drift", source_paths=paths, expected_query_count=4)
    assert result["salvaged_query_count"] == 1
    assert result["fallback_query_count"] == 3


def test_salvage_accepted_source_final_binding_drift_falls_back_entire_batch(
    source_root: tuple[Path, dict[str, Path]], tmp_path: Path
) -> None:
    root, paths = source_root
    final_path = root / "final-organizer.json"
    final_rows = json.loads(final_path.read_text(encoding="utf-8"))
    final_rows[0]["Answers"][0]["Answer"] = "tampered"
    _write(final_path, final_rows)
    provenance_path = root / "final-provenance.json"
    _write(provenance_path, runner._build_provenance(runner.load_rows(**paths, expected_query_count=4), final_rows))
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["final_sha256"] = runner._sha_file(final_path)
    summary["provenance_sha256"] = runner._sha_file(provenance_path)
    _write(summary_path, summary)
    result = salvage.salvage_run(source_root=root, output_root=tmp_path / "final-drift", source_paths=paths, expected_query_count=4)
    assert result["salvaged_query_count"] == 1
    assert result["fallback_query_count"] == 3


def test_salvage_rejects_existing_output_symlink(
    source_root: tuple[Path, dict[str, Path]], tmp_path: Path
) -> None:
    root, paths = source_root
    output = tmp_path / "existing-link"
    output.symlink_to(tmp_path / "target", target_is_directory=True)
    with pytest.raises(FileExistsError, match="output root must be absent"):
        salvage.salvage_run(source_root=root, output_root=output, source_paths=paths, expected_query_count=4)
