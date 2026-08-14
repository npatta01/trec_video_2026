from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.genai import types

from trec_video_2026.vqa_submission_gemini import (
    GenerationResult,
    TokenUsage,
    UploadedFileRecord,
)

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_gemini_counting_evidence.py"
)
SPEC = importlib.util.spec_from_file_location("gemini_counting_evidence", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def request_row() -> dict[str, object]:
    return {
        "video_id": "v1",
        "question": "How many passes occur?",
        "duration_seconds": 8.0,
        "source_path": "/private/v1.mkv",
        "source_sha256": "source-sha",
        "frozen_map": "SCENE | 0-8 | two passes",
        "file_name": "files/v1",
        "file_uri": "https://example.invalid/files/v1",
        "file_state": "ACTIVE",
        "expiration_time": "2099-01-01T00:00:00+00:00",
        "delivery_mime_type": "video/webm",
        "delivery_sha256": "delivery-sha",
    }


def evidence_text() -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "count_unit": "repeated_actions_transitions",
            "target": "passes",
            "scope": "complete video",
            "records": [
                {
                    "record_id": "R1",
                    "start_seconds": 1.0,
                    "end_seconds": 1.5,
                    "scene": "first pass",
                    "rationale": "distinct completed pass",
                },
                {
                    "record_id": "R2",
                    "start_seconds": 5.0,
                    "end_seconds": 5.5,
                    "scene": "second pass",
                    "rationale": "distinct completed pass",
                },
            ],
            "exclusions": [],
            "uncertainty": "none",
        }
    )


class FakeClient:
    def __init__(self, text: str) -> None:
        self.calls = []
        self.files = SimpleNamespace(get=self.get_file)
        self.models = SimpleNamespace(generate_content=self.generate_content)
        self.text = text

    def get_file(self, *, name: str):
        assert name == "files/v1"
        return SimpleNamespace(
            name=name,
            uri="https://example.invalid/files/v1",
            state=SimpleNamespace(name="ACTIVE"),
        )

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            text=self.text,
            usage_metadata=SimpleNamespace(
                prompt_token_count=10_000,
                candidates_token_count=500,
                thoughts_token_count=1_000,
                cached_content_token_count=0,
                total_token_count=11_500,
            ),
            candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
            model_version="gemini-fixture",
            response_id="response-fixture",
        )


class FakeOpenRouterTransport:
    provider_name = "openrouter"

    def __init__(self) -> None:
        self.parts = None
        self.calls = 0

    def upload(self, path: Path, *, mime_type: str) -> UploadedFileRecord:
        return UploadedFileRecord(
            file_id="file-1", mime_type=mime_type, sha256=MODULE._file_sha256(path)
        )

    def generate(self, *, model: str, parts, schema) -> GenerationResult:
        self.calls += 1
        self.parts = parts
        return GenerationResult(
            payload=json.loads(evidence_text()),
            raw_response=evidence_text(),
            usage=TokenUsage(input_tokens=10_000, output_tokens=1_500),
            provider_metadata={
                "provider": "Google",
                "model": model,
                "request_id": "or-fixture",
                "reported_cost_usd": 0.038,
            },
        )

    def delete(self, file_id: str) -> None:
        return None


def pricing() -> dict[str, object]:
    return {
        "schema_version": 1,
        "model": "gemini-3.1-pro-preview",
        "tier": "standard_under_200k_input_tokens",
        "input_usd_per_million_tokens": 2.0,
        "output_and_thinking_usd_per_million_tokens": 12.0,
        "high_context_threshold_tokens": 200_000,
        "high_context_input_usd_per_million_tokens": 4.0,
        "high_context_output_and_thinking_usd_per_million_tokens": 18.0,
        "source_url": "https://ai.google.dev/gemini-api/docs/pricing",
        "accessed_at": "2026-08-14T14:25:00Z",
    }


def test_cost_estimate_charges_thinking_at_output_rate() -> None:
    assert MODULE.estimate_cost_usd(
        prompt_tokens=10_000,
        output_tokens=500,
        thinking_tokens=1_000,
        pricing=pricing(),
    ) == pytest.approx(0.038)


def test_response_schema_is_accepted_by_the_pinned_sdk_boundary() -> None:
    schema = types.Schema.model_validate(MODULE.RESPONSE_SCHEMA)
    assert schema.properties["schema_version"].type.value == "INTEGER"


def test_provider_schema_omits_fields_rejected_by_generate_content_endpoint() -> None:
    def walk(value):
        if isinstance(value, dict):
            assert "additionalProperties" not in value
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(MODULE.RESPONSE_SCHEMA)


def test_cost_estimate_uses_high_context_rates_at_threshold() -> None:
    assert MODULE.estimate_cost_usd(
        prompt_tokens=200_000,
        output_tokens=1_000,
        thinking_tokens=1_000,
        pricing=pricing(),
    ) == pytest.approx(0.836)


def test_batch_cost_uses_half_price_rates() -> None:
    batch_pricing = {
        **pricing(),
        "batch_input_usd_per_million_tokens": 1.0,
        "batch_output_and_thinking_usd_per_million_tokens": 6.0,
        "batch_high_context_input_usd_per_million_tokens": 2.0,
        "batch_high_context_output_and_thinking_usd_per_million_tokens": 9.0,
    }
    assert MODULE.estimate_cost_usd(
        prompt_tokens=10_000,
        output_tokens=500,
        thinking_tokens=1_000,
        pricing=batch_pricing,
        mode="batch",
    ) == pytest.approx(0.019)


def test_batch_request_keeps_native_video_question_and_atomic_schema_together() -> None:
    request = MODULE.build_batch_request(
        row=request_row(),
        request_manifest_sha256="manifest-sha",
        pricing_sha256="pricing-sha",
    )
    assert request.metadata == {
        "condition": "atomic-counting-evidence-v1",
        "pricing_sha256": "pricing-sha",
        "request_manifest_sha256": "manifest-sha",
        "video_id": "v1",
    }
    assert request.contents.parts[0].file_data.file_uri == request_row()["file_uri"]
    assert "How many passes occur?" in request.contents.parts[1].text
    assert request.config.response_mime_type == "application/json"
    assert request.config.temperature == 0.0
    assert request.config.seed == 42
    assert request.config.thinking_config.thinking_level.value == "HIGH"


def test_batch_terminal_payload_materializes_atomic_result_and_batch_cost(
    tmp_path: Path,
) -> None:
    batch_pricing = {
        **pricing(),
        "batch_input_usd_per_million_tokens": 1.0,
        "batch_output_and_thinking_usd_per_million_tokens": 6.0,
        "batch_high_context_input_usd_per_million_tokens": 2.0,
        "batch_high_context_output_and_thinking_usd_per_million_tokens": 9.0,
    }
    payload = {
        "name": "batches/fixture",
        "dest": {
            "inlinedResponses": [
                {
                    "metadata": {
                        "condition": "atomic-counting-evidence-v1",
                        "pricing_sha256": "pricing-sha",
                        "request_manifest_sha256": "manifest-sha",
                        "video_id": "v1",
                    },
                    "response": {
                        "candidates": [
                            {"content": {"parts": [{"text": evidence_text()}]}}
                        ],
                        "usageMetadata": {
                            "promptTokenCount": 10_000,
                            "candidatesTokenCount": 500,
                            "thoughtsTokenCount": 1_000,
                            "totalTokenCount": 11_500,
                        },
                        "modelVersion": "gemini-fixture",
                        "responseId": "response-fixture",
                    },
                }
            ]
        },
    }
    results = MODULE.materialize_batch_payload(
        root=tmp_path,
        payload=payload,
        rows_by_id={"v1": request_row()},
        expected_video_ids=["v1"],
        request_manifest_sha256="manifest-sha",
        pricing=batch_pricing,
    )
    assert results[0]["status"] == "passed"
    assert results[0]["derived_count"] == 2
    assert results[0]["estimated_cost_usd"] == pytest.approx(0.019)
    assert (tmp_path / "rows" / "v1" / "complete.json").is_file()


def test_budget_guard_stops_before_projected_request_crosses_ceiling() -> None:
    with pytest.raises(ValueError, match="USD 3.00 ceiling"):
        MODULE.assert_budget_allows(spent_usd=2.96, projected_request_usd=0.05)


def test_atomic_attempt_claim_prevents_duplicate_paid_call(tmp_path: Path) -> None:
    row_root = tmp_path / "rows" / "v1"
    MODULE.reserve_attempt(
        row_root=row_root,
        attempt={"video_id": "v1", "provider": "fixture"},
    )
    with pytest.raises(FileExistsError, match="paid attempt already claimed"):
        MODULE.reserve_attempt(
            row_root=row_root,
            attempt={"video_id": "v1", "provider": "fixture"},
        )


def test_shared_budget_reservations_include_nested_run_spend(tmp_path: Path) -> None:
    complete_path = tmp_path / "run-a" / "rows" / "v1" / "complete.json"
    complete_path.parent.mkdir(parents=True)
    complete_path.write_text(json.dumps({"estimated_cost_usd": 2.6}))
    with pytest.raises(ValueError, match="USD 3.00 ceiling"):
        MODULE.reserve_issue_budget(
            budget_root=tmp_path,
            reservation_id="run-b:v2",
            reserved_usd=0.5,
        )


def test_shared_budget_reservation_is_exclusive_across_rows(tmp_path: Path) -> None:
    MODULE.reserve_issue_budget(
        budget_root=tmp_path,
        reservation_id="run-a:v1",
        reserved_usd=1.6,
    )
    with pytest.raises(ValueError, match="USD 3.00 ceiling"):
        MODULE.reserve_issue_budget(
            budget_root=tmp_path,
            reservation_id="run-b:v2",
            reserved_usd=1.5,
        )


def test_request_reservation_rejects_media_outside_frozen_duration_bound() -> None:
    row = request_row()
    row["duration_seconds"] = 30.001
    with pytest.raises(ValueError, match="30-second frozen-panel bound"):
        MODULE.assert_request_reservation_bound(row)


def test_phase_stops_after_first_nonpassing_terminal() -> None:
    MODULE.assert_terminal_allows_continuation({"status": "passed", "video_id": "v1"})
    with pytest.raises(RuntimeError, match="v2.*invalid_schema"):
        MODULE.assert_terminal_allows_continuation(
            {"status": "invalid_schema", "video_id": "v2"}
        )


def test_one_provider_call_preserves_raw_response_and_derives_count(
    tmp_path: Path,
) -> None:
    client = FakeClient(evidence_text())
    result = MODULE.run_one(
        root=tmp_path,
        row=request_row(),
        request_manifest_sha256="manifest-sha",
        pricing=pricing(),
        client=client,
        spent_usd=0.0,
        projected_request_usd=0.07,
    )

    assert len(client.calls) == 1
    assert result["status"] == "passed"
    assert result["derived_count"] == 2
    assert result["estimated_cost_usd"] == pytest.approx(0.038)
    row_root = tmp_path / "rows" / "v1"
    assert (row_root / "raw-response.txt").read_text() == evidence_text()
    assert json.loads((row_root / "result.json").read_text())["derived_count"] == 2
    request = json.loads((row_root / "request.json").read_text())
    assert request["request_manifest_sha256"] == "manifest-sha"
    assert "reference" not in json.dumps(request).lower()
    assert "prior_answer" not in json.dumps(request).lower()
    call = client.calls[0]
    assert call["model"] == "gemini-3.1-pro-preview"
    assert call["config"].temperature == 0.0
    assert call["config"].seed == 42
    assert call["config"].thinking_config.include_thoughts is False


def test_invalid_json_is_frozen_terminally_without_retry(tmp_path: Path) -> None:
    client = FakeClient("not json")
    result = MODULE.run_one(
        root=tmp_path,
        row=request_row(),
        request_manifest_sha256="manifest-sha",
        pricing=pricing(),
        client=client,
        spent_usd=0.0,
        projected_request_usd=0.07,
    )
    assert result["status"] == "invalid_json"
    assert len(client.calls) == 1
    assert (tmp_path / "rows" / "v1" / "complete.json").is_file()


def test_openrouter_one_call_uses_original_native_video_and_reported_cost(
    tmp_path: Path,
) -> None:
    source = tmp_path / "v1.mkv"
    source.write_bytes(b"fixture native video bytes")
    row = request_row()
    row["source_path"] = str(source)
    row["source_sha256"] = MODULE._file_sha256(source)
    transport = FakeOpenRouterTransport()

    result = MODULE.run_openrouter_one(
        root=tmp_path / "run",
        row=row,
        request_manifest_sha256="manifest-sha",
        pricing=pricing(),
        transport=transport,
        spent_usd=0.0,
        projected_request_usd=0.07,
    )

    assert transport.calls == 1
    assert result["status"] == "passed"
    assert result["derived_count"] == 2
    assert result["estimated_cost_usd"] == pytest.approx(0.038)
    assert transport.parts[0]["mime_type"] == "video/x-matroska"
    assert "How many passes occur?" in transport.parts[1]


def test_freeze_binds_every_terminal_row_and_refuses_incomplete_panel(
    tmp_path: Path,
) -> None:
    client = FakeClient(evidence_text())
    MODULE.run_one(
        root=tmp_path,
        row=request_row(),
        request_manifest_sha256="manifest-sha",
        pricing=pricing(),
        client=client,
        spent_usd=0.0,
        projected_request_usd=0.07,
    )
    freeze = MODULE.freeze_predictions(
        root=tmp_path,
        video_ids=["v1"],
        request_manifest_sha256="manifest-sha",
        filename="canary-predictions-freeze.json",
    )
    assert freeze["row_count"] == 1
    assert freeze["terminal_status_counts"] == {"passed": 1}
    with pytest.raises(ValueError, match="incomplete row"):
        MODULE.freeze_predictions(
            root=tmp_path,
            video_ids=["v1", "v2"],
            request_manifest_sha256="manifest-sha",
            filename="predictions-freeze.json",
        )
