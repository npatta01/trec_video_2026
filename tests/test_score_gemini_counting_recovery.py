from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from trec_video_2026.vqa_submission import canonical_sha256

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "score_gemini_counting_recovery.py"
)
SPEC = importlib.util.spec_from_file_location("score_gemini_counting_recovery", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def self_hashed(payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result.pop("complete_sha256", None)
    result["complete_sha256"] = canonical_sha256(result)
    return result


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_fixture(
    root: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    rows = [
        ("failure-a", "confirmed_failure", "four", 4),
        ("failure-b", "confirmed_failure", "four", 4),
        ("control", "correct_control", "one", 1),
    ]
    manifest_rows = [
        {
            "video_id": video_id,
            "question": "How many events occur?",
            "duration_seconds": 8.0,
            "source_path": f"/source/{video_id}.mkv",
            "source_sha256": f"source-{video_id}",
            "delivery_path": f"/delivery/{video_id}.webm",
            "delivery_sha256": f"delivery-{video_id}",
            "delivery_mime_type": "video/webm",
        }
        for video_id, _, _, _ in rows
    ]
    manifest = self_hashed(
        {
            "schema_version": 1,
            "condition": "gemini-native-counting-recovery-v1",
            "rows": manifest_rows,
        }
    )
    manifest_sha256 = manifest["complete_sha256"]
    panel_rows = []
    freeze_rows = []
    for video_id, role, reference, value in rows:
        panel_rows.append(
            {
                "video_id": video_id,
                "panel_role": role,
                "reference_answer": reference,
            }
        )
        row_root = root / "rows" / video_id
        manifest_row = next(row for row in manifest_rows if row["video_id"] == video_id)
        upload_claim = self_hashed(
            {
                "schema_version": 1,
                "video_id": video_id,
                "delivery_sha256": manifest_row["delivery_sha256"],
                "delivery_mime_type": manifest_row["delivery_mime_type"],
            }
        )
        write_json(row_root / "upload-claim.json", upload_claim)
        upload = self_hashed(
            {
                "schema_version": 1,
                "video_id": video_id,
                "delivery_sha256": manifest_row["delivery_sha256"],
                "delivery_mime_type": manifest_row["delivery_mime_type"],
                "upload_claim_complete_sha256": upload_claim["complete_sha256"],
                "file_name": f"files/{video_id}",
                "file_uri": f"https://example.test/files/{video_id}",
                "file_state": "ACTIVE",
                "expiration_time": "fixture",
            }
        )
        write_json(row_root / "upload.json", upload)
        prior = None
        stage_hashes = {}
        stages = ["locate", "verify", "challenge"]
        if video_id == "failure-a":
            stages.insert(1, "locate-correction-v2")
        for stage in stages:
            stage_root = row_root / stage
            if stage == "locate-correction-v2":
                write_json(stage_root / "validated.json", {"stage": stage})
                terminal = self_hashed(
                    {
                        "schema_version": 1,
                        "stage": stage,
                        "outcome": "success",
                        "prior_complete_sha256": prior,
                        "estimated_cost_usd": 0.0,
                        "invalid_raw_response_sha256": file_sha256(
                            row_root / "locate" / "raw-response.txt"
                        ),
                        "invalid_parsed_sha256": file_sha256(
                            row_root / "locate" / "parsed.json"
                        ),
                        "validated_sha256": file_sha256(stage_root / "validated.json"),
                    }
                )
            else:
                request = self_hashed(
                    {
                        "schema_version": 1,
                        "stage": stage,
                        "model": "gemini-3.1-pro-preview",
                        "request_manifest_sha256": manifest_sha256,
                        "upload_complete_sha256": upload["complete_sha256"],
                        "delivery_sha256": manifest_row["delivery_sha256"],
                        "file_uri": upload["file_uri"],
                        "mime_type": manifest_row["delivery_mime_type"],
                        "prior_complete_sha256": prior,
                        "fps": 4.0 if stage == "locate" else 12.0,
                        "start_offset": "0.000s" if stage == "verify" else None,
                        "end_offset": "8.000s" if stage == "verify" else None,
                        "response_schema_sha256": {
                            "locate": MODULE._SCHEMA_SHA256["locate"],
                            "verify": MODULE._SCHEMA_SHA256["verify"],
                            "challenge": MODULE._SCHEMA_SHA256["challenge"],
                        }[stage],
                        "temperature": 0.0,
                        "seed": 42,
                        "max_output_tokens": 2048,
                        "response_mime_type": "application/json",
                        "thinking_level": "LOW",
                        "media_resolution": "HIGH",
                    }
                )
                write_json(stage_root / "request.json", request)
                claim = self_hashed(
                    {
                        "schema_version": 1,
                        "stage": stage,
                        "request_complete_sha256": request["complete_sha256"],
                    }
                )
                write_json(stage_root / "paid-call-claim.json", claim)
                write_text(
                    stage_root / "raw-response.txt", f"raw response for {stage}\n"
                )
                write_json(stage_root / "parsed.json", {"stage": stage})
                if not (stage == "locate" and video_id == "failure-a"):
                    write_json(stage_root / "validated.json", {"stage": stage})
                write_json(
                    stage_root / "provider-metadata.json",
                    {"stage": stage, "response_id": f"response-{stage}"},
                )
                terminal = self_hashed(
                    {
                        "schema_version": 1,
                        "stage": stage,
                        "outcome": "invalid_response"
                        if stage == "locate" and video_id == "failure-a"
                        else "success",
                        "prior_complete_sha256": prior,
                        "estimated_cost_usd": 0.01,
                        "request_sha256": request["complete_sha256"],
                        "paid_call_claim_sha256": claim["complete_sha256"],
                        "raw_response_sha256": file_sha256(
                            stage_root / "raw-response.txt"
                        ),
                        "parsed_sha256": file_sha256(stage_root / "parsed.json"),
                        "provider_metadata_sha256": file_sha256(
                            stage_root / "provider-metadata.json"
                        ),
                    }
                )
                if stage != "locate" or video_id != "failure-a":
                    terminal["validated_sha256"] = file_sha256(
                        stage_root / "validated.json"
                    )
                    terminal = self_hashed(terminal)
            if stage == "locate-correction-v2":
                terminal["invalid_raw_response_sha256"] = file_sha256(
                    row_root / "locate" / "raw-response.txt"
                )
                terminal["invalid_parsed_sha256"] = file_sha256(
                    row_root / "locate" / "parsed.json"
                )
                terminal = self_hashed(terminal)
            write_json(row_root / stage / "complete.json", terminal)
            prior = terminal["complete_sha256"]
            stage_hashes[stage] = prior
        result = self_hashed(
            {
                "outcome": "exact",
                "value": value,
                "accepted_instance_ids": ["I1"],
                "reason_codes": [],
            }
        )
        write_json(row_root / "result.json", result)
        complete = self_hashed(
            {
                "schema_version": 1,
                "outcome": "success",
                "video_id": video_id,
                "request_manifest_sha256": manifest_sha256,
                "upload_complete_sha256": upload["complete_sha256"],
                "delivery_sha256": manifest_row["delivery_sha256"],
                "stage_complete_sha256": stage_hashes,
                "result_sha256": file_sha256(row_root / "result.json"),
                "estimated_cost_usd": 0.03,
            }
        )
        write_json(row_root / "complete.json", complete)
        freeze_rows.append(
            {
                "video_id": video_id,
                "row_complete_sha256": complete["complete_sha256"],
                "estimated_cost_usd": 0.03,
                "result": result,
            }
        )
    freeze = self_hashed(
        {
            "schema_version": 1,
            "request_manifest_sha256": manifest_sha256,
            "completion": {"completed": 3, "total": 3},
            "rows": freeze_rows,
            "total_estimated_cost_usd": 0.09,
        }
    )
    panel = self_hashed({"schema_version": 1, "rows": panel_rows})
    return manifest, freeze, panel


def rebind_row(
    root: Path,
    freeze: dict[str, object],
    video_id: str,
    mutate_complete: Callable[[dict[str, object]], None],
) -> None:
    row_root = root / "rows" / video_id
    complete_path = row_root / "complete.json"
    complete = json.loads(complete_path.read_text())
    mutate_complete(complete)
    complete = self_hashed(complete)
    write_json(complete_path, complete)
    for row in freeze["rows"]:
        if row["video_id"] == video_id:
            row["row_complete_sha256"] = complete["complete_sha256"]
            break
    freeze_hash = freeze.pop("complete_sha256")
    del freeze_hash
    freeze["complete_sha256"] = canonical_sha256(
        {key: value for key, value in freeze.items() if key != "complete_sha256"}
    )


def rebind_stage_request(
    root: Path,
    freeze: dict[str, object],
    video_id: str,
    stage: str,
    mutate_request: Callable[[dict[str, object]], None],
) -> None:
    row_root = root / "rows" / video_id
    request_path = row_root / stage / "request.json"
    request = json.loads(request_path.read_text())
    mutate_request(request)
    request = self_hashed(request)
    write_json(request_path, request)
    terminal_path = row_root / stage / "complete.json"
    terminal = json.loads(terminal_path.read_text())
    terminal["request_sha256"] = request["complete_sha256"]
    terminal = self_hashed(terminal)
    write_json(terminal_path, terminal)

    def rebind_terminal(complete: dict[str, object]) -> None:
        complete["stage_complete_sha256"][stage] = terminal["complete_sha256"]

    rebind_row(root, freeze, video_id, rebind_terminal)


def metric_rows() -> dict[str, dict[str, object]]:
    return {
        video_id: {
            "meteor": 1.0,
            "bertscore_f1": 1.0,
            "sts": 1.0,
            "sts_provenance": "fixture",
        }
        for video_id in ("failure-a", "failure-b", "control")
    }


def baseline_rows() -> dict[str, dict[str, object]]:
    return {
        video_id: {
            "meteor": 0.5,
            "bertscore_f1": 0.5,
            "sts": 0.5,
            "semantic_composite": 0.5,
        }
        for video_id in ("failure-a", "failure-b", "control")
    }


def test_score_authenticates_chain_and_passes_two_of_two_rescue_gate(
    tmp_path: Path,
) -> None:
    manifest, freeze, panel = build_fixture(tmp_path)

    score = MODULE.build_score(
        root=tmp_path,
        manifest=manifest,
        freeze=freeze,
        panel=panel,
        metric_rows=metric_rows(),
        baseline_rows=baseline_rows(),
    )

    assert score["completion"] == {"completed": 3, "total": 3}
    assert score["rescue"] == {"rescued": 2, "total": 2, "rate": 1.0}
    assert score["control"] == {"preserved": 1, "total": 1, "rate": 1.0}
    assert score["semantic_metrics"] == {
        "bertscore_f1": 1.0,
        "meteor": 1.0,
        "semantic_composite": 1.0,
        "sts": 1.0,
    }
    assert score["baseline_semantic_metrics"] == {
        "bertscore_f1": 0.5,
        "meteor": 0.5,
        "semantic_composite": 0.5,
        "sts": 0.5,
    }
    assert score["deltas_treatment_minus_baseline"]["semantic_composite"] == 0.5
    assert score["gate_passed"] is True


def test_score_authenticates_frozen_one_row_prefix_without_evaluating_gate(
    tmp_path: Path,
) -> None:
    manifest, freeze, panel = build_fixture(tmp_path)
    freeze["rows"] = freeze["rows"][:1]
    freeze["completion"] = {"completed": 1, "total": 1}
    freeze["manifest_row_count"] = 3
    freeze["total_estimated_cost_usd"] = 0.03
    freeze = self_hashed(freeze)

    score = MODULE.build_score(
        root=tmp_path,
        manifest=manifest,
        freeze=freeze,
        panel=panel,
        metric_rows={"failure-a": metric_rows()["failure-a"]},
        baseline_rows={"failure-a": baseline_rows()["failure-a"]},
    )

    assert score["completion"] == {"completed": 1, "total": 1}
    assert score["rescue"] == {"rescued": 1, "total": 1, "rate": 1.0}
    assert score["control"] == {"preserved": 0, "total": 0, "rate": None}
    assert score["gate_evaluable"] is False
    assert score["gate_passed"] is False


def test_score_rejects_post_freeze_result_mutation(tmp_path: Path) -> None:
    manifest, freeze, panel = build_fixture(tmp_path)
    result_path = tmp_path / "rows" / "failure-a" / "result.json"
    result = json.loads(result_path.read_text())
    result["value"] = 99
    write_json(result_path, result)

    with pytest.raises(ValueError, match="result hash changed"):
        MODULE.build_score(
            root=tmp_path,
            manifest=manifest,
            freeze=freeze,
            panel=panel,
            metric_rows=metric_rows(),
            baseline_rows=baseline_rows(),
        )


def test_score_rejects_panel_mutation_after_selection_freeze(tmp_path: Path) -> None:
    manifest, freeze, panel = build_fixture(tmp_path)
    panel["rows"][0]["reference_answer"] = "three"

    with pytest.raises(ValueError, match="panel self-hash changed"):
        MODULE.build_score(
            root=tmp_path,
            manifest=manifest,
            freeze=freeze,
            panel=panel,
            metric_rows=metric_rows(),
            baseline_rows=baseline_rows(),
        )


def test_score_rejects_missing_mandatory_stage(tmp_path: Path) -> None:
    manifest, freeze, panel = build_fixture(tmp_path)

    def remove_challenge(complete: dict[str, object]) -> None:
        del complete["stage_complete_sha256"]["challenge"]

    rebind_row(tmp_path, freeze, "failure-a", remove_challenge)

    with pytest.raises(ValueError, match="mandatory stage"):
        MODULE.build_score(
            root=tmp_path,
            manifest=manifest,
            freeze=freeze,
            panel=panel,
            metric_rows=metric_rows(),
            baseline_rows=baseline_rows(),
        )


def test_score_rejects_row_complete_identity_or_outcome_mutation(
    tmp_path: Path,
) -> None:
    manifest, freeze, panel = build_fixture(tmp_path)

    def mutate_row_complete(complete: dict[str, object]) -> None:
        complete["video_id"] = "wrong-video-id"
        complete["outcome"] = "provider_error"

    rebind_row(tmp_path, freeze, "failure-a", mutate_row_complete)

    with pytest.raises(ValueError, match="row failure-a video_id changed"):
        MODULE.build_score(
            root=tmp_path,
            manifest=manifest,
            freeze=freeze,
            panel=panel,
            metric_rows=metric_rows(),
            baseline_rows=baseline_rows(),
        )


def test_score_rejects_row_complete_non_success_outcome(tmp_path: Path) -> None:
    manifest, freeze, panel = build_fixture(tmp_path)

    def mutate_outcome(complete: dict[str, object]) -> None:
        complete["outcome"] = "provider_error"

    rebind_row(tmp_path, freeze, "failure-a", mutate_outcome)

    with pytest.raises(ValueError, match="row failure-a complete outcome"):
        MODULE.build_score(
            root=tmp_path,
            manifest=manifest,
            freeze=freeze,
            panel=panel,
            metric_rows=metric_rows(),
            baseline_rows=baseline_rows(),
        )


def test_score_rejects_terminal_referenced_file_mutation(tmp_path: Path) -> None:
    manifest, freeze, panel = build_fixture(tmp_path)
    raw_path = tmp_path / "rows" / "failure-a" / "verify" / "raw-response.txt"
    write_text(raw_path, "tampered response\n")

    with pytest.raises(ValueError, match="verify raw_response file hash changed"):
        MODULE.build_score(
            root=tmp_path,
            manifest=manifest,
            freeze=freeze,
            panel=panel,
            metric_rows=metric_rows(),
            baseline_rows=baseline_rows(),
        )


def test_score_rejects_missing_terminal_referenced_file(tmp_path: Path) -> None:
    manifest, freeze, panel = build_fixture(tmp_path)
    metadata_path = (
        tmp_path / "rows" / "failure-a" / "challenge" / "provider-metadata.json"
    )
    metadata_path.unlink()

    with pytest.raises(ValueError, match="challenge provider_metadata file missing"):
        MODULE.build_score(
            root=tmp_path,
            manifest=manifest,
            freeze=freeze,
            panel=panel,
            metric_rows=metric_rows(),
            baseline_rows=baseline_rows(),
        )


def test_score_rejects_upload_claim_self_hash_mutation(tmp_path: Path) -> None:
    manifest, freeze, panel = build_fixture(tmp_path)
    claim_path = tmp_path / "rows" / "failure-a" / "upload-claim.json"
    claim = json.loads(claim_path.read_text())
    claim["delivery_sha256"] = "tampered-delivery"
    write_json(claim_path, claim)

    with pytest.raises(
        ValueError, match="row failure-a upload claim self-hash changed"
    ):
        MODULE.build_score(
            root=tmp_path,
            manifest=manifest,
            freeze=freeze,
            panel=panel,
            metric_rows=metric_rows(),
            baseline_rows=baseline_rows(),
        )


def test_score_rejects_row_complete_upload_binding_mutation(tmp_path: Path) -> None:
    manifest, freeze, panel = build_fixture(tmp_path)

    def mutate_upload_binding(complete: dict[str, object]) -> None:
        complete["upload_complete_sha256"] = "tampered-upload"

    rebind_row(tmp_path, freeze, "failure-a", mutate_upload_binding)

    with pytest.raises(ValueError, match="row failure-a upload binding changed"):
        MODULE.build_score(
            root=tmp_path,
            manifest=manifest,
            freeze=freeze,
            panel=panel,
            metric_rows=metric_rows(),
            baseline_rows=baseline_rows(),
        )


@pytest.mark.parametrize(
    ("field", "value", "label"),
    [
        ("request_manifest_sha256", "wrong-manifest", "request manifest"),
        ("upload_complete_sha256", "wrong-upload", "upload complete"),
        ("delivery_sha256", "wrong-delivery", "delivery"),
        ("file_uri", "wrong-uri", "file uri"),
        ("mime_type", "video/mp4", "mime type"),
    ],
)
def test_score_rejects_request_media_binding_mutations(
    tmp_path: Path,
    field: str,
    value: str,
    label: str,
) -> None:
    manifest, freeze, panel = build_fixture(tmp_path)

    def mutate_request(request: dict[str, object]) -> None:
        request[field] = value

    rebind_stage_request(
        tmp_path,
        freeze,
        "failure-a",
        "verify",
        mutate_request,
    )

    with pytest.raises(
        ValueError, match=f"row failure-a verify {label} binding changed"
    ):
        MODULE.build_score(
            root=tmp_path,
            manifest=manifest,
            freeze=freeze,
            panel=panel,
            metric_rows=metric_rows(),
            baseline_rows=baseline_rows(),
        )


def test_score_rejects_request_configuration_mutation(tmp_path: Path) -> None:
    manifest, freeze, panel = build_fixture(tmp_path)

    rebind_stage_request(
        tmp_path,
        freeze,
        "failure-a",
        "verify",
        lambda request: request.update({"fps": 4.0}),
    )

    with pytest.raises(ValueError, match="verify request configuration changed"):
        MODULE.build_score(
            root=tmp_path,
            manifest=manifest,
            freeze=freeze,
            panel=panel,
            metric_rows=metric_rows(),
            baseline_rows=baseline_rows(),
        )


def test_score_rejects_optional_correction_without_invalid_locator(
    tmp_path: Path,
) -> None:
    manifest, freeze, panel = build_fixture(tmp_path)

    def remove_correction(complete: dict[str, object]) -> None:
        del complete["stage_complete_sha256"]["locate-correction-v2"]

    rebind_row(tmp_path, freeze, "failure-a", remove_correction)

    with pytest.raises(ValueError, match="locator must succeed without correction"):
        MODULE.build_score(
            root=tmp_path,
            manifest=manifest,
            freeze=freeze,
            panel=panel,
            metric_rows=metric_rows(),
            baseline_rows=baseline_rows(),
        )


def test_score_rejects_broken_stage_prior_chain(tmp_path: Path) -> None:
    manifest, freeze, panel = build_fixture(tmp_path)
    terminal_path = tmp_path / "rows" / "failure-a" / "challenge" / "complete.json"
    terminal = json.loads(terminal_path.read_text())
    terminal["prior_complete_sha256"] = "broken-prior"
    terminal = self_hashed(terminal)
    write_json(terminal_path, terminal)

    def rebind_challenge(complete: dict[str, object]) -> None:
        complete["stage_complete_sha256"]["challenge"] = terminal["complete_sha256"]

    rebind_row(tmp_path, freeze, "failure-a", rebind_challenge)

    with pytest.raises(ValueError, match="challenge prior binding changed"):
        MODULE.build_score(
            root=tmp_path,
            manifest=manifest,
            freeze=freeze,
            panel=panel,
            metric_rows=metric_rows(),
            baseline_rows=baseline_rows(),
        )


def test_sanitized_projection_removes_private_rows_and_answers(tmp_path: Path) -> None:
    manifest, freeze, panel = build_fixture(tmp_path)
    score = MODULE.build_score(
        root=tmp_path,
        manifest=manifest,
        freeze=freeze,
        panel=panel,
        metric_rows=metric_rows(),
        baseline_rows=baseline_rows(),
    )

    sanitized = MODULE.sanitize_score(score)

    rendered = json.dumps(sanitized)
    assert "rows" not in sanitized
    assert "failure-a" not in rendered
    assert "reference_answer" not in rendered
    assert sanitized["rescue"]["rate"] == 1.0
