#!/usr/bin/env python3
"""Authenticate and score the bounded multi-pass counting canary."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from trec_video_2026.vqa_counting_evidence import (
    parse_prediction_count,
    parse_reference_count,
)
from trec_video_2026.vqa_bounded_multipass_counting import (
    CandidateView,
    CountSpec,
    UnsupportedCountSpec,
    count_result_to_dict,
    merge_refinement,
    parse_count_spec,
    parse_coverage_review,
    parse_evidence,
    reduce_count,
)
from trec_video_2026.vqa_submission import canonical_sha256


def load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_self_hash(payload: Mapping[str, object], *, name: str) -> str:
    claimed = payload.get("complete_sha256")
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise ValueError(f"{name} lacks complete_sha256")
    value = dict(payload)
    value.pop("complete_sha256")
    if canonical_sha256(value) != claimed:
        raise ValueError(f"{name} self-hash mismatch")
    return claimed


def build_decision(
    *,
    rows: Sequence[Mapping[str, object]],
    fixed_cost_usd: float,
    refinement_total_cost_usd: float,
) -> dict[str, object]:
    failures = [row for row in rows if row.get("panel_role") == "confirmed_failure"]
    controls = [row for row in rows if row.get("panel_role") == "correct_control"]
    if len(failures) != 8 or len(controls) != 4 or len(rows) != 12:
        raise ValueError("decision requires exactly 8 failures and 4 controls")
    fixed_rescued = sum(bool(row.get("fixed_correct")) for row in failures)
    fixed_controls = sum(bool(row.get("fixed_correct")) for row in controls)
    fixed_total = sum(bool(row.get("fixed_correct")) for row in rows)
    baseline_total = sum(bool(row.get("baseline_correct")) for row in rows)
    fixed_gain = fixed_total - baseline_total
    fixed_passed = fixed_rescued >= 5 and fixed_controls == 4 and fixed_gain >= 3
    added = sum(
        bool(row.get("refinement_correct")) and not bool(row.get("fixed_correct"))
        for row in rows
    )
    regressions = sum(
        bool(row.get("fixed_correct")) and not bool(row.get("refinement_correct"))
        for row in rows
    )
    refinement_controls = sum(bool(row.get("refinement_correct")) for row in controls)
    refinement_passed = (
        fixed_passed
        and added >= 2
        and regressions == 0
        and refinement_controls == 4
        and refinement_total_cost_usd <= 2 * fixed_cost_usd + 1e-12
    )
    return {
        "fixed": {
            "rescued": fixed_rescued,
            "failure_total": 8,
            "controls_preserved": fixed_controls,
            "control_total": 4,
            "gain_over_baseline": fixed_gain,
            "gate_passed": fixed_passed,
        },
        "refinement": {
            "added_over_fixed": added,
            "regressions_from_fixed": regressions,
            "controls_preserved": refinement_controls,
            "control_total": 4,
            "fixed_cost_usd": fixed_cost_usd,
            "total_cost_usd": refinement_total_cost_usd,
            "gate_passed": refinement_passed,
        },
        "selected_arm": (
            "refinement" if refinement_passed else "fixed" if fixed_passed else "none"
        ),
    }


def _verify_preparation(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    preparation = load_json(root / "preparation-complete.json")
    verify_self_hash(preparation, name="preparation terminal")
    if preparation.get("schema_version") == "bounded-multipass-preparation-complete-v1":
        _verify_source_bundle(preparation, name="preparation terminal")
        input_files = preparation.get("input_files")
        if not isinstance(input_files, Mapping):
            raise ValueError("preparation input files are missing")
        for name, binding in input_files.items():
            if not isinstance(binding, Mapping):
                raise ValueError(f"preparation input binding is invalid: {name}")
            path = Path(str(binding.get("path")))
            if file_sha256(path) != binding.get("sha256"):
                raise ValueError(f"preparation input changed: {name}")
    panel = load_json(root / "panel.freeze.json")
    panel_hash = verify_self_hash(panel, name="panel freeze")
    requests = load_json(root / "requests.label-blind.json")
    requests_hash = verify_self_hash(requests, name="requests freeze")
    if panel_hash != preparation.get("panel_complete_sha256"):
        raise ValueError("preparation does not bind panel")
    if requests_hash != preparation.get("requests_complete_sha256"):
        raise ValueError("preparation does not bind requests")
    panel_rows = panel.get("rows")
    request_rows = requests.get("rows")
    if not isinstance(panel_rows, list) or not isinstance(request_rows, list):
        raise ValueError("preparation row arrays are missing")
    panel_ids = [row.get("video_id") for row in panel_rows]
    request_ids = [row.get("video_id") for row in request_rows]
    if (
        len(panel_ids) != 12
        or len(set(panel_ids)) != 12
        or set(panel_ids) != set(request_ids)
    ):
        raise ValueError("preparation identities are not an exact twelve-row set")
    return panel, requests


def _verify_upload(
    *, root: Path, video_id: str, request_row: Mapping[str, object]
) -> tuple[dict[str, Any], str]:
    row_root = root / "rows" / video_id
    claim = load_json(row_root / "upload-claim.json")
    claim_hash = verify_self_hash(claim, name=f"{video_id} upload claim")
    upload = load_json(row_root / "upload.json")
    upload_hash = verify_self_hash(upload, name=f"{video_id} upload")
    if upload.get("upload_claim_complete_sha256") != claim_hash:
        raise ValueError(f"{video_id} upload does not bind claim")
    for key in ("delivery_path", "delivery_sha256", "delivery_mime_type"):
        if upload.get(key) != request_row.get(key):
            raise ValueError(f"{video_id} upload {key} changed")
    delivery_path = Path(str(request_row["delivery_path"]))
    if file_sha256(delivery_path) != request_row.get("delivery_sha256"):
        raise ValueError(f"{video_id} delivery bytes changed")
    source_path = Path(str(request_row["source_path"]))
    if file_sha256(source_path) != request_row.get("source_sha256"):
        raise ValueError(f"{video_id} source bytes changed")
    return upload, upload_hash


def _verify_stage(
    *,
    stage_root: Path,
    expected_terminal_hash: object,
    upload: Mapping[str, object],
    upload_hash: str,
    predecessor_hash: str | None,
) -> str:
    terminal = load_json(stage_root / "complete.json")
    terminal_hash = verify_self_hash(terminal, name=f"{stage_root} terminal")
    if terminal_hash != expected_terminal_hash:
        raise ValueError(f"{stage_root} terminal binding changed")
    if terminal.get("outcome") != "complete":
        raise ValueError(f"{stage_root} is not complete")
    request = load_json(stage_root / "request.json")
    request_hash = verify_self_hash(request, name=f"{stage_root} request")
    if request_hash != terminal.get("request_sha256"):
        raise ValueError(f"{stage_root} terminal does not bind request")
    if request.get("predecessor_sha256") != predecessor_hash:
        raise ValueError(f"{stage_root} predecessor binding changed")
    _verify_source_bundle(request, name=str(stage_root))
    prompt_path = stage_root / "prompt.txt"
    if hashlib.sha256(prompt_path.read_bytes()).hexdigest() != request.get(
        "prompt_sha256"
    ):
        raise ValueError(f"{stage_root} prompt changed")
    schema = load_json(stage_root / "response-schema.json")
    verify_self_hash(schema, name=f"{stage_root} response schema")
    if canonical_sha256(schema.get("payload")) != request.get("response_schema_sha256"):
        raise ValueError(f"{stage_root} response schema changed")
    media = request.get("media_binding")
    if not isinstance(media, Mapping):
        raise ValueError(f"{stage_root} media binding is missing")
    if media.get("upload_complete_sha256") != upload_hash:
        raise ValueError(f"{stage_root} request does not bind upload")
    if media.get("delivery_sha256") != upload.get("delivery_sha256"):
        raise ValueError(f"{stage_root} request does not bind delivery")
    if media.get("file_uri") != upload.get("file_uri"):
        raise ValueError(f"{stage_root} request file URI changed")
    artifacts = media.get("static_artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(f"{stage_root} static artifact list is missing")
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ValueError(f"{stage_root} static artifact is invalid")
        path = Path(str(artifact.get("path")))
        if path.stat().st_size != artifact.get("size_bytes"):
            raise ValueError(f"{stage_root} static artifact size changed")
        if file_sha256(path) != artifact.get("sha256"):
            raise ValueError(f"{stage_root} static artifact bytes changed")
    claim = load_json(stage_root / "paid-call-claim.json")
    claim_hash = verify_self_hash(claim, name=f"{stage_root} paid claim")
    if claim_hash != terminal.get("paid_call_claim_sha256"):
        raise ValueError(f"{stage_root} terminal does not bind paid claim")
    identity = claim.get("identity")
    if (
        not isinstance(identity, Mapping)
        or identity.get("request_sha256") != request_hash
    ):
        raise ValueError(f"{stage_root} claim does not bind request")
    metadata = load_json(stage_root / "provider-metadata.json")
    metadata_hash = verify_self_hash(metadata, name=f"{stage_root} provider metadata")
    if metadata_hash != terminal.get("provider_metadata_sha256"):
        raise ValueError(f"{stage_root} terminal does not bind provider metadata")
    expected_model = request.get("model")
    if expected_model != "gemini-3.1-pro-preview":
        raise ValueError(f"{stage_root} request model is not the approved model")
    reported_model_version = metadata.get("model_version")
    if reported_model_version is not None:
        if reported_model_version != expected_model:
            raise ValueError(f"{stage_root} provider model_version changed")
    elif metadata.get("model") != expected_model:
        raise ValueError(f"{stage_root} legacy provider model changed")
    raw_path = stage_root / "raw-response.txt"
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != metadata.get(
        "raw_response_sha256"
    ):
        raise ValueError(f"{stage_root} raw response changed")
    parsed = load_json(stage_root / "parsed.json")
    parsed_hash = verify_self_hash(parsed, name=f"{stage_root} parsed")
    if parsed_hash != terminal.get("parsed_sha256"):
        raise ValueError(f"{stage_root} terminal does not bind parsed response")
    if json.loads(raw_path.read_text()) != parsed.get("payload"):
        raise ValueError(f"{stage_root} parsed response differs from raw response")
    validated = load_json(stage_root / "validated.json")
    validated_hash = verify_self_hash(validated, name=f"{stage_root} validated")
    if validated_hash != terminal.get("validated_sha256"):
        raise ValueError(f"{stage_root} terminal does not bind validated response")
    if (
        parsed.get("request_sha256") != request_hash
        or validated.get("request_sha256") != request_hash
    ):
        raise ValueError(f"{stage_root} normalized artifacts do not bind request")
    return terminal_hash


def _validated_stage_payload(stage_root: Path) -> dict[str, Any]:
    validated = load_json(stage_root / "validated.json")
    verify_self_hash(validated, name=f"{stage_root} validated")
    payload = validated.get("payload")
    if not isinstance(payload, dict):
        raise ValueError(f"{stage_root} validated payload is not an object")
    return payload


def _stage_terminal(stage_root: Path) -> dict[str, Any]:
    terminal = load_json(stage_root / "complete.json")
    verify_self_hash(terminal, name=f"{stage_root} terminal")
    return terminal


def _assert_result_matches(
    *, result: Mapping[str, object], recomputed: Mapping[str, object], name: str
) -> None:
    for field in ("outcome", "value", "reasons", "selected_record_ids"):
        if result.get(field) != recomputed.get(field):
            raise ValueError(f"{name} {field} does not match replay")


def _parse_fixed_chain(
    *,
    root: Path,
    video_id: str,
    request_row: Mapping[str, object],
) -> tuple[CountSpec | UnsupportedCountSpec, dict[str, object], dict[str, object]]:
    row_root = root / "rows" / video_id / "fixed"
    spec_payload = _validated_stage_payload(row_root / "count-spec")
    question = str(request_row["question"])
    duration = float(request_row["duration_seconds"])
    if "unsupported_reason" in spec_payload:
        return (
            UnsupportedCountSpec(
                unit=str(spec_payload.get("unit")),
                reason=str(spec_payload["unsupported_reason"]),
            ),
            spec_payload,
            {},
        )
    parsed_spec = parse_count_spec(
        spec_payload,
        question=question,
        duration_seconds=duration,
    )
    if not isinstance(parsed_spec, CountSpec):
        raise ValueError(f"{video_id} fixed CountSpec unexpectedly unsupported")
    evidence_payload = _validated_stage_payload(row_root / "evidence")
    ledger = parse_evidence(
        evidence_payload,
        spec=parsed_spec,
        requested_views=parsed_spec.candidate_views,
    )
    coverage_payload = _validated_stage_payload(row_root / "coverage")
    review = parse_coverage_review(
        coverage_payload,
        spec=parsed_spec,
        ledger=ledger,
    )
    recomputed = reduce_count(parsed_spec, ledger, review)
    return (
        parsed_spec,
        {
            "ledger": ledger,
            "review": review,
            "recomputed": recomputed,
        },
        {
            "spec_payload": spec_payload,
            "evidence_payload": evidence_payload,
            "coverage_payload": coverage_payload,
        },
    )


def _replay_refinement_chain(
    *,
    root: Path,
    video_id: str,
    request_row: Mapping[str, object],
    result: Mapping[str, object],
    stage_hashes: Mapping[str, object],
    stage_terminals: Mapping[str, Mapping[str, object]],
) -> None:
    fixed_root = root / "rows" / video_id / "fixed"
    fixed_result = load_json(fixed_root / "result.json")
    fixed_result_hash = verify_self_hash(fixed_result, name=f"{video_id} fixed result")
    if result.get("fixed_result_sha256") != fixed_result_hash:
        raise ValueError(f"{video_id} refinement fixed-result provenance changed")
    parsed_spec, chain, _payloads = _parse_fixed_chain(
        root=root, video_id=video_id, request_row=request_row
    )
    if isinstance(parsed_spec, UnsupportedCountSpec):
        if stage_hashes:
            raise ValueError(f"{video_id} unsupported refinement has paid stages")
        _assert_result_matches(
            result=result,
            recomputed=fixed_result,
            name=f"{video_id} no-op refinement result",
        )
        return
    initial_review = chain["review"]
    request_value = initial_review.refinement_request
    if not stage_hashes:
        if request_value is not None:
            raise ValueError(f"{video_id} refinement omitted a requested refinement")
        _assert_result_matches(
            result=result,
            recomputed=fixed_result,
            name=f"{video_id} no-op refinement result",
        )
        return
    if set(stage_hashes) != {"refinement_evidence", "final_coverage"}:
        raise ValueError(f"{video_id} refinement stage set is incomplete")
    if request_value is None:
        raise ValueError(f"{video_id} refinement ran without a requested refinement")
    if request_value.kind == "window":
        refinement_view = CandidateView(
            "R0",
            "window",
            start_seconds=request_value.start_seconds,
            end_seconds=request_value.end_seconds,
        )
    elif request_value.kind == "shot_frame":
        original = next(
            (
                view
                for view in parsed_spec.candidate_views
                if view.view_id == request_value.view_id
            ),
            None,
        )
        if original is None or original.timestamp_seconds is None:
            raise ValueError(f"{video_id} refinement shot frame is unbound")
        refinement_view = CandidateView(
            "R0", "shot_frame", timestamp_seconds=original.timestamp_seconds
        )
    else:
        raise ValueError(f"{video_id} crop refinement cannot be authenticated")
    refinement_payload = _validated_stage_payload(
        root / "rows" / video_id / "refinement" / "refinement-evidence"
    )
    refinement_ledger = parse_evidence(
        refinement_payload,
        spec=parsed_spec,
        requested_views=(refinement_view,),
    )
    merged = merge_refinement(chain["ledger"], refinement_ledger)
    final_payload = _validated_stage_payload(
        root / "rows" / video_id / "refinement" / "final-coverage"
    )
    final_review = parse_coverage_review(
        final_payload,
        spec=parsed_spec,
        ledger=merged,
        requested_views=(*parsed_spec.candidate_views, refinement_view),
    )
    recomputed = count_result_to_dict(reduce_count(parsed_spec, merged, final_review))
    _assert_result_matches(
        result=result, recomputed=recomputed, name=f"{video_id} refinement result"
    )
    for field, stage in (
        ("refinement_evidence_sha256", "refinement_evidence"),
        ("final_coverage_sha256", "final_coverage"),
    ):
        if result.get(field) != stage_terminals[stage].get("validated_sha256"):
            raise ValueError(f"{video_id} refinement {field} provenance changed")


def _verify_source_bundle(payload: Mapping[str, object], *, name: str) -> None:
    revision = payload.get("source_revision")
    sources = payload.get("implementation_sources")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or not isinstance(sources, Mapping)
    ):
        raise ValueError(f"{name} source provenance is missing")
    repository = Path(__file__).resolve().parents[1]
    for relative, claimed in sources.items():
        if not isinstance(relative, str) or not isinstance(claimed, str):
            raise ValueError(f"{name} source provenance is invalid")
        current_path = repository / relative
        current_hash = file_sha256(current_path) if current_path.is_file() else None
        historical = subprocess.run(
            ["git", "show", f"{revision}:{relative}"],
            cwd=repository,
            capture_output=True,
        )
        historical_hash = (
            hashlib.sha256(historical.stdout).hexdigest()
            if historical.returncode == 0
            else None
        )
        if claimed not in {current_hash, historical_hash}:
            raise ValueError(f"{name} source hash changed: {relative}")


def authenticate_arm(*, root: Path, arm: str) -> dict[str, dict[str, Any]]:
    if arm not in {"fixed", "refinement"}:
        raise ValueError("arm must be fixed or refinement")
    if arm == "refinement":
        authenticate_arm(root=root, arm="fixed")
    _panel, requests = _verify_preparation(root)
    request_by_id = {str(row["video_id"]): row for row in requests["rows"]}
    arm_terminal = load_json(root / f"{arm}-arm-complete.json")
    arm_terminal_hash = verify_self_hash(arm_terminal, name=f"{arm} arm terminal")
    freeze = load_json(root / f"{arm}-predictions-freeze.json")
    verify_self_hash(freeze, name=f"{arm} predictions freeze")
    if freeze.get("arm_complete_sha256") != arm_terminal_hash:
        raise ValueError(f"{arm} freeze does not bind arm terminal")
    freeze_rows = freeze.get("rows")
    if not isinstance(freeze_rows, list):
        raise ValueError(f"{arm} freeze rows are missing")
    freeze_by_id = {str(row["video_id"]): row for row in freeze_rows}
    if set(freeze_by_id) != set(request_by_id) or len(freeze_rows) != 12:
        raise ValueError(f"{arm} freeze identity set differs from preparation")
    arm_rows = arm_terminal.get("row_complete_sha256")
    if not isinstance(arm_rows, Mapping) or set(arm_rows) != set(request_by_id):
        raise ValueError(f"{arm} arm terminal identity set differs from preparation")
    results: dict[str, dict[str, Any]] = {}
    for video_id, request_row in request_by_id.items():
        upload, upload_hash = _verify_upload(
            root=root, video_id=video_id, request_row=request_row
        )
        row_root = root / "rows" / video_id / arm
        terminal = load_json(row_root / "complete.json")
        terminal_hash = verify_self_hash(terminal, name=f"{video_id} {arm} terminal")
        if terminal_hash != arm_rows.get(video_id):
            raise ValueError(f"{video_id} {arm} arm binding changed")
        if terminal.get("outcome") != "complete":
            raise ValueError(f"{video_id} {arm} row is not complete")
        freeze_row = freeze_by_id[video_id]
        if freeze_row.get("row_complete_sha256") != terminal_hash:
            raise ValueError(f"{video_id} {arm} freeze row binding changed")
        stage_hashes = terminal.get("stage_complete_sha256")
        if not isinstance(stage_hashes, Mapping):
            raise ValueError(f"{video_id} {arm} stage bindings are missing")
        if arm == "fixed":
            stage_dirs = {
                "count_spec": "count-spec",
                "evidence": "evidence",
                "coverage": "coverage",
            }
            predecessor = None
        else:
            stage_dirs = {
                "refinement_evidence": "refinement-evidence",
                "final_coverage": "final-coverage",
            }
            fixed_terminal = load_json(root / "rows" / video_id / "fixed/complete.json")
            predecessor = verify_self_hash(
                fixed_terminal, name=f"{video_id} fixed predecessor"
            )
        allowed_stage_sets = (
            ({"count_spec"}, set(stage_dirs))
            if arm == "fixed"
            else (set(), set(stage_dirs))
        )
        if set(stage_hashes) not in allowed_stage_sets:
            raise ValueError(f"{video_id} {arm} has an invalid required-stage set")
        stage_terminals: dict[str, dict[str, Any]] = {}
        for key, directory in stage_dirs.items():
            if key not in stage_hashes:
                continue
            predecessor = _verify_stage(
                stage_root=row_root / directory,
                expected_terminal_hash=stage_hashes[key],
                upload=upload,
                upload_hash=upload_hash,
                predecessor_hash=predecessor,
            )
            stage_terminals[key] = _stage_terminal(row_root / directory)
        result = load_json(row_root / "result.json")
        result_hash = verify_self_hash(result, name=f"{video_id} {arm} result")
        if result_hash != terminal.get(
            "result_sha256"
        ) or result_hash != freeze_row.get("result_sha256"):
            raise ValueError(f"{video_id} {arm} result binding changed")
        for key in ("outcome", "value", "reasons", "selected_record_ids"):
            if result.get(key) != freeze_row.get(key):
                raise ValueError(f"{video_id} {arm} frozen {key} changed")
        if arm == "fixed":
            parsed_spec, chain, payloads = _parse_fixed_chain(
                root=root, video_id=video_id, request_row=request_row
            )
            if isinstance(parsed_spec, UnsupportedCountSpec):
                if set(stage_hashes) != {"count_spec"}:
                    raise ValueError(
                        f"{video_id} unsupported CountSpec has extra stages"
                    )
                if (
                    result.get("outcome") != "abstain"
                    or result.get("value") is not None
                ):
                    raise ValueError(
                        f"{video_id} unsupported CountSpec result is not abstention"
                    )
                if result.get("reasons") != [parsed_spec.reason]:
                    raise ValueError(f"{video_id} unsupported CountSpec reason changed")
                if result.get("count_spec_sha256") != stage_terminals["count_spec"].get(
                    "validated_sha256"
                ):
                    raise ValueError(f"{video_id} unsupported CountSpec proof changed")
            else:
                recomputed = count_result_to_dict(chain["recomputed"])
                _assert_result_matches(
                    result=result,
                    recomputed=recomputed,
                    name=f"{video_id} fixed result",
                )
                for field, stage in (
                    ("count_spec_sha256", "count_spec"),
                    ("evidence_sha256", "evidence"),
                    ("coverage_sha256", "coverage"),
                ):
                    if result.get(field) != stage_terminals[stage].get(
                        "validated_sha256"
                    ):
                        raise ValueError(f"{video_id} fixed {field} provenance changed")
        else:
            _replay_refinement_chain(
                root=root,
                video_id=video_id,
                request_row=request_row,
                result=result,
                stage_hashes=stage_hashes,
                stage_terminals=stage_terminals,
            )
        results[video_id] = result
    return results


def load_baseline_answers(
    *, path: Path, expected_questions: Mapping[str, str]
) -> dict[str, str]:
    freeze = load_json(path)
    conditions = freeze.get("conditions")
    if not isinstance(conditions, Mapping):
        raise ValueError("baseline conditions are missing")
    rows = conditions.get("direct-answer")
    if not isinstance(rows, list):
        raise ValueError("baseline direct-answer condition is missing")
    result = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("baseline row is invalid")
        video_id = str(row.get("video_id"))
        if video_id not in expected_questions:
            continue
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError(f"baseline text is missing for {video_id}")
        if hashlib.sha256(text.encode()).hexdigest() != row.get("response_sha256"):
            raise ValueError(f"baseline response hash changed for {video_id}")
        if row.get("question") != expected_questions[video_id]:
            raise ValueError(f"baseline question changed for {video_id}")
        if row.get("model_version") != "gemini-3.1-pro-preview":
            raise ValueError(f"baseline model changed for {video_id}")
        snapshot = Path(str(row.get("terminal_snapshot")))
        if file_sha256(snapshot) != row.get("terminal_snapshot_sha256"):
            raise ValueError(f"baseline terminal snapshot changed for {video_id}")
        result[video_id] = text
    if set(result) != set(expected_questions):
        raise ValueError("baseline identity set differs from current panel")
    return result


def _arm_incremental_cost(root: Path, arm: str) -> float:
    directories = (
        ("count-spec", "evidence", "coverage")
        if arm == "fixed"
        else ("refinement-evidence", "final-coverage")
    )
    total = 0.0
    for row_root in (root / "rows").iterdir():
        for directory in directories:
            path = row_root / arm / directory / "provider-metadata.json"
            if not path.exists():
                continue
            metadata = load_json(path)
            verify_self_hash(metadata, name=f"{arm} provider metadata")
            total += float(metadata["observed_cost_usd"])
    return total


def _metric_module() -> Any:
    path = Path(__file__).with_name("score_gemini_counting_recovery.py")
    spec = importlib.util.spec_from_file_location("counting_recovery_metrics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load established metric implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metric_mean(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    fields = ("meteor", "bertscore_f1", "sts")
    result = {
        field: statistics.fmean(float(row[field]) for row in rows) for field in fields
    }
    result["semantic_composite"] = statistics.fmean(result.values())
    return result


def build_score(
    *,
    root: Path,
    baseline_freeze: Path,
    bert_model: Path,
    sts_cache_path: Path,
) -> dict[str, object]:
    panel, _requests = _verify_preparation(root)
    fixed = authenticate_arm(root=root, arm="fixed")
    refinement = authenticate_arm(root=root, arm="refinement")
    panel_by_id = {str(row["video_id"]): row for row in panel["rows"]}
    baseline = load_baseline_answers(
        path=baseline_freeze,
        expected_questions={
            video_id: str(row["question"]) for video_id, row in panel_by_id.items()
        },
    )
    metric_impl = _metric_module()
    references = {
        video_id: str(row["reference_answer"]) for video_id, row in panel_by_id.items()
    }
    prediction_sets = {
        "baseline": baseline,
        "fixed": {
            video_id: str(result["value"]) if result.get("outcome") == "exact" else ""
            for video_id, result in fixed.items()
        },
        "refinement": {
            video_id: str(result["value"]) if result.get("outcome") == "exact" else ""
            for video_id, result in refinement.items()
        },
    }
    sts_cache = load_json(sts_cache_path)
    metric_sets = {}
    environment = None
    for arm, predictions in prediction_sets.items():
        metric_rows, current_environment = metric_impl.compute_metric_rows(
            predictions=predictions,
            references=references,
            bert_model=bert_model,
            sts_cache=sts_cache,
        )
        metric_sets[arm] = metric_rows
        environment = current_environment
    rows = []
    for video_id, panel_row in panel_by_id.items():
        reference_count = panel_row.get("reference_count")
        if not isinstance(reference_count, int):
            parsed = parse_reference_count(str(panel_row["reference_answer"]))
            reference_count = parsed.value
        if reference_count is None:
            raise ValueError(f"reference count is unavailable for {video_id}")
        role = (
            "confirmed_failure"
            if str(panel_row["stratum"]).startswith("failure")
            else "correct_control"
        )
        baseline_count = parse_prediction_count(baseline[video_id]).value
        fixed_correct = (
            fixed[video_id].get("outcome") == "exact"
            and fixed[video_id].get("value") == reference_count
        )
        refinement_correct = (
            refinement[video_id].get("outcome") == "exact"
            and refinement[video_id].get("value") == reference_count
        )
        rows.append(
            {
                "video_id": video_id,
                "stratum": panel_row["stratum"],
                "panel_role": role,
                "reference_answer": panel_row["reference_answer"],
                "reference_count": reference_count,
                "baseline_prediction": baseline[video_id],
                "baseline_count": baseline_count,
                "baseline_correct": baseline_count == reference_count,
                "fixed_outcome": fixed[video_id]["outcome"],
                "fixed_value": fixed[video_id]["value"],
                "fixed_correct": fixed_correct,
                "refinement_outcome": refinement[video_id]["outcome"],
                "refinement_value": refinement[video_id]["value"],
                "refinement_correct": refinement_correct,
                "metrics": {arm: metric_sets[arm][video_id] for arm in metric_sets},
            }
        )
    fixed_cost = _arm_incremental_cost(root, "fixed")
    refinement_incremental = _arm_incremental_cost(root, "refinement")
    decision = build_decision(
        rows=rows,
        fixed_cost_usd=fixed_cost,
        refinement_total_cost_usd=fixed_cost + refinement_incremental,
    )
    aggregates = {
        arm: _metric_mean([metric_sets[arm][video_id] for video_id in panel_by_id])
        for arm in metric_sets
    }
    score = {
        "schema_version": "bounded-multipass-score-v1",
        "completion": {"completed_rows": len(rows), "planned_rows": 12},
        "cost": {
            "fixed_usd": fixed_cost,
            "refinement_incremental_usd": refinement_incremental,
            "total_usd": fixed_cost + refinement_incremental,
        },
        "decision": decision,
        "semantic_metrics": aggregates,
        "metric_environment": environment,
        "inputs": {
            "baseline_freeze_sha256": file_sha256(baseline_freeze),
            "bert_model_config_sha256": file_sha256(bert_model / "config.json"),
            "sts_cache_sha256": file_sha256(sts_cache_path),
        },
        "rows": rows,
    }
    score["complete_sha256"] = canonical_sha256(score)
    return score


def sanitize_score(score: Mapping[str, object]) -> dict[str, object]:
    decision = score["decision"]
    metrics = score["semantic_metrics"]
    return {
        "schema_version": "bounded-multipass-sanitized-summary-v1",
        "completion": score["completion"],
        "cost": score["cost"],
        "decision": decision,
        "semantic_metrics": metrics,
        "evidence_validity": "Diagnostic",
        "decision_disposition": (
            "Promoted for held-out replication"
            if decision["selected_arm"] != "none"
            else "Not promoted under the tested condition"
        ),
        "private_score_complete_sha256": score["complete_sha256"],
    }


def _write_once(path: Path, payload: Mapping[str, object]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text() != rendered:
            raise FileExistsError(f"refusing to overwrite conflicting artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-freeze", type=Path, required=True)
    parser.add_argument("--bert-model", type=Path, required=True)
    parser.add_argument("--sts-cache", type=Path, required=True)
    parser.add_argument("--sanitized-output", type=Path, required=True)
    args = parser.parse_args()
    score = build_score(
        root=args.root,
        baseline_freeze=args.baseline_freeze,
        bert_model=args.bert_model,
        sts_cache_path=args.sts_cache,
    )
    _write_once(args.root / "score.json", score)
    sanitized = sanitize_score(score)
    sanitized["complete_sha256"] = canonical_sha256(sanitized)
    _write_once(args.sanitized_output, sanitized)
    print(json.dumps(sanitized, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
