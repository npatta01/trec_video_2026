from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from google.genai import types

from trec_video_2026.vqa_gemini_counting_recovery import TimeRange
from trec_video_2026.vqa_submission import canonical_sha256

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_gemini_counting_recovery.py"
)
SPEC = importlib.util.spec_from_file_location("run_gemini_counting_recovery", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_pass_video_parts_use_frozen_fps_and_offsets() -> None:
    locator = MODULE.build_video_part(
        file_uri="files/v1", mime_type="video/webm", fps=4.0, time_range=None
    )
    verifier = MODULE.build_video_part(
        file_uri="files/v1",
        mime_type="video/webm",
        fps=12.0,
        time_range=TimeRange(1.5, 6.25),
    )

    assert locator.video_metadata.fps == 4.0
    assert locator.video_metadata.start_offset is None
    assert verifier.video_metadata.fps == 12.0
    assert verifier.video_metadata.start_offset == "1.500s"
    assert verifier.video_metadata.end_offset == "6.250s"
    assert verifier.media_resolution.level.value == "MEDIA_RESOLUTION_HIGH"


def test_all_response_schemas_are_accepted_by_pinned_sdk() -> None:
    for schema in (
        MODULE.LOCATOR_RESPONSE_SCHEMA,
        MODULE.LEDGER_RESPONSE_SCHEMA,
        MODULE.CHALLENGE_RESPONSE_SCHEMA,
    ):
        accepted = types.Schema.model_validate(schema)
        assert accepted.type.value == "OBJECT"
    assert (
        "generic_occurrence"
        in MODULE.LOCATOR_RESPONSE_SCHEMA["properties"]["count_unit"]["enum"]
    )


def test_paid_stage_claim_is_exclusive(tmp_path: Path) -> None:
    stage_root = tmp_path / "locate"
    identity = MODULE._self_hashed({"schema_version": 1, "stage": "locate"})

    claim = MODULE.reserve_paid_stage(stage_root, identity)

    assert claim.is_file()
    payload = json.loads(claim.read_text())
    claimed = payload.pop("complete_sha256")
    assert canonical_sha256(payload) == claimed
    assert payload["request_complete_sha256"] == identity["complete_sha256"]
    with pytest.raises(FileExistsError, match="paid stage already claimed"):
        MODULE.reserve_paid_stage(stage_root, identity)


def test_budget_refuses_reservation_that_could_cross_one_dollar(tmp_path: Path) -> None:
    MODULE.reserve_budget(
        budget_root=tmp_path,
        reservation_id="row-1-locate",
        reserved_usd=0.34,
    )
    MODULE.settle_budget(
        budget_root=tmp_path,
        reservation_id="row-1-locate",
        observed_usd=0.20,
    )
    MODULE.reserve_budget(
        budget_root=tmp_path,
        reservation_id="row-2-locate",
        reserved_usd=0.60,
    )

    with pytest.raises(RuntimeError, match="budget ceiling"):
        MODULE.reserve_budget(
            budget_root=tmp_path,
            reservation_id="row-3-locate",
            reserved_usd=0.21,
        )


def test_settlement_cannot_exceed_reserved_amount(tmp_path: Path) -> None:
    MODULE.reserve_budget(
        budget_root=tmp_path,
        reservation_id="row-1-locate",
        reserved_usd=0.10,
    )

    with pytest.raises(RuntimeError, match="exceeds reservation"):
        MODULE.settle_budget(
            budget_root=tmp_path,
            reservation_id="row-1-locate",
            observed_usd=0.11,
        )


def test_prepare_manifest_projects_only_label_blind_fields(tmp_path: Path) -> None:
    source = tmp_path / "v1.mkv"
    source.write_bytes(b"source")
    delivery = tmp_path / "deliveries" / "v1" / "delivery-media" / "v1.webm"
    delivery.parent.mkdir(parents=True)
    delivery.write_bytes(b"delivery")
    rows = [
        {
            "video_id": "v1",
            "question": "How many passes occur?",
            "duration_seconds": 8.0,
            "source_path": str(source),
            "source_sha256": hashlib.sha256(b"source").hexdigest(),
            "delivery_sha256": hashlib.sha256(b"delivery").hexdigest(),
            "delivery_mime_type": "video/webm",
            "frozen_map": "contains an old count",
            "file_uri": "expired-provider-file",
            "panel_role": "confirmed_failure",
            "reference_answer": "two",
        }
    ]

    manifest = MODULE.prepare_request_manifest(
        source_rows=rows,
        ordered_video_ids=("v1",),
        delivery_root=tmp_path / "deliveries",
        condition="gemini-native-counting-recovery-hardened-v2",
    )

    assert manifest["condition"] == "gemini-native-counting-recovery-hardened-v2"
    assert manifest["rows"] == [
        {
            "video_id": "v1",
            "question": "How many passes occur?",
            "duration_seconds": 8.0,
            "source_path": str(source),
            "source_sha256": hashlib.sha256(b"source").hexdigest(),
            "delivery_path": str(delivery),
            "delivery_sha256": hashlib.sha256(b"delivery").hexdigest(),
            "delivery_mime_type": "video/webm",
        }
    ]
    rendered = json.dumps(manifest)
    assert "frozen_map" not in rendered
    assert "reference_answer" not in rendered
    assert "panel_role" not in rendered
    assert "file_uri" not in rendered


def test_upload_is_reused_for_all_row_stages(tmp_path: Path) -> None:
    media = tmp_path / "v1.webm"
    media.write_bytes(b"delivery")

    class Files:
        def __init__(self) -> None:
            self.upload_calls = 0

        def upload(self, **kwargs):
            self.upload_calls += 1
            return SimpleNamespace(name="files/v1")

        def get(self, *, name):
            assert name == "files/v1"
            return SimpleNamespace(
                name=name,
                uri="https://example.invalid/files/v1",
                state=SimpleNamespace(name="ACTIVE"),
                expiration_time="2099-01-01T00:00:00Z",
            )

    files = Files()
    client = SimpleNamespace(files=files)
    row = {
        "video_id": "v1",
        "delivery_path": str(media),
        "delivery_sha256": hashlib.sha256(b"delivery").hexdigest(),
        "delivery_mime_type": "video/webm",
    }

    first = MODULE.ensure_upload(client=client, row=row, row_root=tmp_path / "row")
    second = MODULE.ensure_upload(client=client, row=row, row_root=tmp_path / "row")

    assert files.upload_calls == 1
    assert first == second
    assert first["file_uri"] == "https://example.invalid/files/v1"
    claim = json.loads((tmp_path / "row" / "upload-claim.json").read_text())
    claimed = claim.pop("complete_sha256")
    assert canonical_sha256(claim) == claimed
    assert first["upload_claim_complete_sha256"] == claimed


def test_upload_claim_prevents_concurrent_duplicate_upload(tmp_path: Path) -> None:
    media = tmp_path / "v1.webm"
    media.write_bytes(b"delivery")
    row = {
        "video_id": "v1",
        "delivery_path": str(media),
        "delivery_sha256": hashlib.sha256(b"delivery").hexdigest(),
        "delivery_mime_type": "video/webm",
    }
    row_root = tmp_path / "row"
    MODULE.reserve_upload(row_root=row_root, row=row)
    files = SimpleNamespace(
        upload=lambda **kwargs: pytest.fail("concurrent upload must not start")
    )

    with pytest.raises(FileExistsError, match="upload already claimed"):
        MODULE.ensure_upload(
            client=SimpleNamespace(files=files), row=row, row_root=row_root
        )


def test_run_row_rejects_unbound_uploaded_media_before_any_paid_call(
    tmp_path: Path,
) -> None:
    row = {
        "video_id": "v1",
        "question": "How many passes occur?",
        "duration_seconds": 8.0,
        "delivery_sha256": "delivery-sha",
        "delivery_mime_type": "video/webm",
    }
    row_root = tmp_path / "rows" / "v1"
    claim = json.loads(MODULE.reserve_upload(row_root=row_root, row=row).read_text())
    upload = MODULE._self_hashed(
        {
            "schema_version": 1,
            "video_id": "v1",
            "delivery_sha256": "delivery-sha",
            "delivery_mime_type": "video/webm",
            "upload_claim_complete_sha256": claim["complete_sha256"],
            "file_uri": "files/intended",
            "file_state": "ACTIVE",
        }
    )
    MODULE._write_once_json(row_root / "upload.json", upload)
    client = SequenceClient([])

    with pytest.raises(ValueError, match="uploaded media binding changed"):
        MODULE.run_row(
            client=client,
            row=row,
            file_uri="files/unrelated",
            upload_complete_sha256=upload["complete_sha256"],
            row_root=row_root,
            budget_root=tmp_path / "budget",
            request_manifest_sha256="manifest-sha",
            pricing=MODULE.fixture_pricing(),
            cumulative_limit_usd=0.34,
        )

    assert client.count_calls == []
    assert client.calls == []


class FailingClient:
    def __init__(self) -> None:
        self.calls = 0
        self.models = SimpleNamespace(generate_content=self.generate_content)

    def generate_content(self, **kwargs):
        self.calls += 1
        raise RuntimeError("provider exploded")


class ResponseClient:
    def __init__(self, text: str) -> None:
        self.calls = 0
        self.text = text
        self.models = SimpleNamespace(generate_content=self.generate_content)

    def generate_content(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            text=self.text,
            model_version="gemini-fixture",
            response_id="response-fixture",
            usage_metadata=SimpleNamespace(
                prompt_token_count=1_000,
                candidates_token_count=100,
                thoughts_token_count=100,
            ),
        )


class SequenceClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls = []
        self.count_calls = []
        self.models = SimpleNamespace(
            count_tokens=self.count_tokens,
            generate_content=self.generate_content,
        )

    def count_tokens(self, **kwargs):
        self.count_calls.append(kwargs)
        return SimpleNamespace(total_tokens=1_000)

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            text=next(self.responses),
            model_version="gemini-fixture",
            response_id=f"response-{len(self.calls)}",
            usage_metadata=SimpleNamespace(
                prompt_token_count=1_000,
                candidates_token_count=100,
                thoughts_token_count=100,
            ),
        )


def test_provider_failure_is_terminal_and_never_retried(tmp_path: Path) -> None:
    client = FailingClient()
    stage_root = tmp_path / "locate"

    with pytest.raises(RuntimeError, match="provider exploded"):
        MODULE.execute_paid_stage(
            client=client,
            stage_root=stage_root,
            stage="locate",
            video_part=MODULE.build_video_part(
                file_uri="files/v1",
                mime_type="video/webm",
                fps=4.0,
                time_range=None,
            ),
            prompt="locate",
            response_schema={"type": "object"},
            prior_complete_sha256=None,
            request_manifest_sha256="manifest-sha",
            upload_complete_sha256="upload-sha",
            delivery_sha256="delivery-sha",
            reservation={"reservation_id": "row-1-locate", "reserved_usd": 0.10},
            budget_root=tmp_path / "budget",
            pricing=MODULE.fixture_pricing(),
        )

    assert client.calls == 1
    terminal = json.loads((stage_root / "complete.json").read_text())
    assert terminal["outcome"] == "provider_error"
    assert terminal["attempted_provider_calls"] == 1
    with pytest.raises(FileExistsError, match="paid stage already claimed"):
        MODULE.execute_paid_stage(
            client=client,
            stage_root=stage_root,
            stage="locate",
            video_part=MODULE.build_video_part(
                file_uri="files/v1",
                mime_type="video/webm",
                fps=4.0,
                time_range=None,
            ),
            prompt="locate",
            response_schema={"type": "object"},
            prior_complete_sha256=None,
            request_manifest_sha256="manifest-sha",
            upload_complete_sha256="upload-sha",
            delivery_sha256="delivery-sha",
            reservation={"reservation_id": "row-1-locate-2", "reserved_usd": 0.10},
            budget_root=tmp_path / "budget",
            pricing=MODULE.fixture_pricing(),
        )


def test_invalid_paid_response_is_frozen_before_validation_error(
    tmp_path: Path,
) -> None:
    client = ResponseClient('{"schema_version": 99}')

    def reject(payload):
        raise ValueError(f"wrong schema: {payload['schema_version']}")

    with pytest.raises(ValueError, match="wrong schema"):
        MODULE.execute_paid_stage(
            client=client,
            stage_root=tmp_path / "locate",
            stage="locate",
            video_part=MODULE.build_video_part(
                file_uri="files/v1",
                mime_type="video/webm",
                fps=4.0,
                time_range=None,
            ),
            prompt="locate",
            response_schema={"type": "object"},
            prior_complete_sha256=None,
            request_manifest_sha256="manifest-sha",
            upload_complete_sha256="upload-sha",
            delivery_sha256="delivery-sha",
            reservation={"reservation_id": "row-1-locate", "reserved_usd": 0.10},
            budget_root=tmp_path / "budget",
            pricing=MODULE.fixture_pricing(),
            validate_response=reject,
        )

    assert client.calls == 1
    terminal = json.loads((tmp_path / "locate" / "complete.json").read_text())
    assert terminal["outcome"] == "invalid_response"
    assert terminal["attempted_provider_calls"] == 1
    assert (tmp_path / "locate" / "raw-response.txt").read_text() == client.text
    assert json.loads((tmp_path / "locate" / "parsed.json").read_text()) == {
        "schema_version": 99
    }
    provider = json.loads((tmp_path / "locate" / "provider-metadata.json").read_text())
    assert provider["model_version"] == "gemini-fixture"
    assert provider["response_id"] == "response-fixture"
    assert provider["prompt_tokens"] == 1_000
    assert provider["output_tokens"] == 100
    assert provider["thinking_tokens"] == 100
    assert terminal["provider_metadata_sha256"] == MODULE._file_sha256(
        tmp_path / "locate" / "provider-metadata.json"
    )


def test_locator_alias_correction_is_additive_and_binds_invalid_paid_terminal(
    tmp_path: Path,
) -> None:
    locate = tmp_path / "rows" / "v1" / "locate"
    parsed = {
        "schema_version": 1,
        "count_unit": "times",
        "target": "events",
        "candidates": [],
        "reviewed_spans": [{"start_seconds": 0.0, "end_seconds": 8.0}],
        "unresolved_spans": [],
    }
    raw = json.dumps(parsed)
    locate.mkdir(parents=True)
    (locate / "raw-response.txt").write_text(raw)
    (locate / "parsed.json").write_text(json.dumps(parsed))
    terminal = {
        "schema_version": 1,
        "stage": "locate",
        "outcome": "invalid_response",
        "prior_complete_sha256": None,
        "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "error_message": "unsupported count_unit: times",
    }
    terminal["complete_sha256"] = canonical_sha256(terminal)
    (locate / "complete.json").write_text(json.dumps(terminal))
    original = (locate / "complete.json").read_bytes()

    correction = MODULE.apply_locator_alias_correction(
        row_root=tmp_path / "rows" / "v1",
        duration_seconds=8.0,
    )

    assert (locate / "complete.json").read_bytes() == original
    corrected = json.loads(
        (
            tmp_path / "rows" / "v1" / "locate-correction-v2" / "validated.json"
        ).read_text()
    )
    assert corrected["count_unit"] == "generic_occurrence"
    assert correction["prior_complete_sha256"] == terminal["complete_sha256"]
    assert correction["correction_reason"] == "label_independent_count_unit_alias"


def test_resume_after_alias_correction_skips_paid_locator_call(tmp_path: Path) -> None:
    row_root = tmp_path / "rows" / "v1"
    locate = row_root / "locate"
    locator = {
        "schema_version": 1,
        "count_unit": "times",
        "target": "passes",
        "candidates": [
            {
                "candidate_id": "C1",
                "start_seconds": 2.0,
                "end_seconds": 3.0,
                "reason": "possible pass",
            }
        ],
        "reviewed_spans": [{"start_seconds": 0.0, "end_seconds": 8.0}],
        "unresolved_spans": [],
    }
    raw = json.dumps(locator)
    locate.mkdir(parents=True)
    (locate / "raw-response.txt").write_text(raw)
    (locate / "parsed.json").write_text(json.dumps(locator))
    invalid = {
        "schema_version": 1,
        "stage": "locate",
        "outcome": "invalid_response",
        "prior_complete_sha256": None,
        "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "error_message": "unsupported count_unit: times",
        "estimated_cost_usd": 0.01,
    }
    invalid["complete_sha256"] = canonical_sha256(invalid)
    (locate / "complete.json").write_text(json.dumps(invalid))
    MODULE.apply_locator_alias_correction(row_root=row_root, duration_seconds=8.0)
    MODULE.reserve_budget(
        budget_root=tmp_path / "budget",
        reservation_id="v1-locate",
        reserved_usd=0.34,
    )
    MODULE.settle_budget(
        budget_root=tmp_path / "budget",
        reservation_id="v1-locate",
        observed_usd=0.01,
    )
    ledger = {
        "schema_version": 1,
        "instances": [
            {
                "instance_id": "I1",
                "start_seconds": 2.1,
                "peak_seconds": 2.5,
                "end_seconds": 2.9,
                "before_state": "player A controls ball",
                "during_state": "ball travels",
                "after_state": "player B controls ball",
                "separation_evidence": "completed transfer",
                "predicate_supported": True,
                "ambiguous": False,
            }
        ],
        "unresolved_observations": [],
    }
    challenge = {
        "schema_version": 1,
        "instance_dispositions": [
            {"instance_id": "I1", "disposition": "accept", "reason": "distinct"}
        ],
        "missed_instances": [],
        "requested_ranges": [],
        "unresolved_reasons": [],
    }
    client = SequenceClient([json.dumps(ledger), json.dumps(challenge)])
    row = {
        "video_id": "v1",
        "question": "How many passes occur?",
        "duration_seconds": 8.0,
        "delivery_mime_type": "video/webm",
        "delivery_sha256": "delivery-sha",
    }
    upload_claim = json.loads(
        MODULE.reserve_upload(row_root=row_root, row=row).read_text()
    )
    upload = MODULE._self_hashed(
        {
            "schema_version": 1,
            "video_id": "v1",
            "delivery_sha256": "delivery-sha",
            "delivery_mime_type": "video/webm",
            "upload_claim_complete_sha256": upload_claim["complete_sha256"],
            "file_uri": "files/v1",
            "file_state": "ACTIVE",
        }
    )
    MODULE._write_once_json(row_root / "upload.json", upload)

    complete = MODULE.resume_row_after_locator_correction(
        client=client,
        row=row,
        file_uri="files/v1",
        upload_complete_sha256=upload["complete_sha256"],
        row_root=row_root,
        budget_root=tmp_path / "budget",
        request_manifest_sha256="manifest-sha",
        pricing=MODULE.fixture_pricing(),
        cumulative_limit_usd=0.34,
    )

    assert len(client.calls) == 2
    assert [call["contents"][0].video_metadata.fps for call in client.calls] == [
        12.0,
        12.0,
    ]
    assert complete["stage_complete_sha256"]["locate"] == invalid["complete_sha256"]
    assert "locate-correction-v2" in complete["stage_complete_sha256"]


def test_run_row_executes_distinct_three_passes_and_freezes_exact_result(
    tmp_path: Path,
) -> None:
    locator = {
        "schema_version": 1,
        "count_unit": "event_transition",
        "target": "pass",
        "candidates": [
            {
                "candidate_id": "C1",
                "start_seconds": 2.0,
                "end_seconds": 3.0,
                "reason": "possible pass",
            }
        ],
        "reviewed_spans": [{"start_seconds": 0.0, "end_seconds": 8.0}],
        "unresolved_spans": [],
    }
    ledger = {
        "schema_version": 1,
        "instances": [
            {
                "instance_id": "I1",
                "start_seconds": 2.1,
                "peak_seconds": 2.5,
                "end_seconds": 2.9,
                "before_state": "ball controlled by player A",
                "during_state": "ball travels between players",
                "after_state": "ball controlled by player B",
                "separation_evidence": "one completed transfer",
                "predicate_supported": True,
                "ambiguous": False,
            }
        ],
        "unresolved_observations": [],
    }
    challenge = {
        "schema_version": 1,
        "instance_dispositions": [
            {"instance_id": "I1", "disposition": "accept", "reason": "distinct"}
        ],
        "missed_instances": [],
        "requested_ranges": [],
        "unresolved_reasons": [],
    }
    client = SequenceClient(
        [json.dumps(locator), json.dumps(ledger), json.dumps(challenge)]
    )
    row = {
        "video_id": "v1",
        "question": "How many passes occur?",
        "duration_seconds": 8.0,
        "delivery_mime_type": "video/webm",
        "delivery_sha256": "delivery-sha",
    }
    row_root = tmp_path / "rows" / "v1"
    upload_claim = json.loads(
        MODULE.reserve_upload(row_root=row_root, row=row).read_text()
    )
    upload = MODULE._self_hashed(
        {
            "schema_version": 1,
            "video_id": "v1",
            "delivery_sha256": "delivery-sha",
            "delivery_mime_type": "video/webm",
            "upload_claim_complete_sha256": upload_claim["complete_sha256"],
            "file_name": "files/v1",
            "file_uri": "files/v1",
            "file_state": "ACTIVE",
            "expiration_time": "fixture",
        }
    )
    MODULE._write_once_json(row_root / "upload.json", upload)

    complete = MODULE.run_row(
        client=client,
        row=row,
        file_uri="files/v1",
        upload_complete_sha256=upload["complete_sha256"],
        row_root=row_root,
        budget_root=tmp_path / "budget",
        request_manifest_sha256="manifest-sha",
        pricing=MODULE.fixture_pricing(),
        cumulative_limit_usd=0.34,
    )

    assert len(client.calls) == 3
    assert len(client.count_calls) == 3
    assert all(
        call["config"].thinking_config.thinking_level.value == "LOW"
        for call in client.calls
    )
    parts = [call["contents"][0] for call in client.calls]
    assert [part.video_metadata.fps for part in parts] == [4.0, 12.0, 12.0]
    assert parts[1].video_metadata.start_offset == "0.500s"
    assert parts[1].video_metadata.end_offset == "4.500s"
    assert parts[2].video_metadata.start_offset is None
    locate_request = json.loads(
        (tmp_path / "rows" / "v1" / "locate" / "request.json").read_text()
    )
    assert locate_request["file_uri"] == "files/v1"
    assert locate_request["mime_type"] == "video/webm"
    assert locate_request["upload_complete_sha256"] == upload["complete_sha256"]
    assert locate_request["delivery_sha256"] == "delivery-sha"
    assert locate_request["response_schema_sha256"] == canonical_sha256(
        MODULE.LOCATOR_RESPONSE_SCHEMA
    )
    assert locate_request["thinking_level"] == "LOW"
    result = json.loads((tmp_path / "rows" / "v1" / "result.json").read_text())
    assert result["outcome"] == "exact"
    assert result["value"] == 1
    assert complete["outcome"] == "success"
    assert complete["stage_complete_sha256"] == {
        "challenge": json.loads(
            (tmp_path / "rows" / "v1" / "challenge" / "complete.json").read_text()
        )["complete_sha256"],
        "locate": json.loads(
            (tmp_path / "rows" / "v1" / "locate" / "complete.json").read_text()
        )["complete_sha256"],
        "verify": json.loads(
            (tmp_path / "rows" / "v1" / "verify" / "complete.json").read_text()
        )["complete_sha256"],
    }
    manifest = {
        "schema_version": 1,
        "condition": "gemini-native-counting-recovery-v1",
        "rows": [row],
        "complete_sha256": "manifest-sha",
    }
    freeze = MODULE.freeze_predictions(root=tmp_path, manifest=manifest)
    assert freeze["completion"] == {"completed": 1, "total": 1}
    assert freeze["rows"][0]["result"]["value"] == 1

    challenge_raw = tmp_path / "rows" / "v1" / "challenge" / "raw-response.txt"
    challenge_raw.write_text("mutated")
    with pytest.raises(ValueError, match="raw response hash changed"):
        MODULE.freeze_predictions(root=tmp_path, manifest=manifest)


def test_freeze_requires_all_mandatory_stages(tmp_path: Path) -> None:
    row_root = tmp_path / "rows" / "v1"
    row_root.mkdir(parents=True)
    row = {
        "video_id": "v1",
        "delivery_sha256": "delivery-sha",
        "delivery_mime_type": "video/webm",
    }
    upload_claim = json.loads(
        MODULE.reserve_upload(row_root=row_root, row=row).read_text()
    )
    upload = MODULE._self_hashed(
        {
            "schema_version": 1,
            "video_id": "v1",
            "delivery_sha256": "delivery-sha",
            "delivery_mime_type": "video/webm",
            "upload_claim_complete_sha256": upload_claim["complete_sha256"],
            "file_uri": "files/v1",
        }
    )
    MODULE._write_once_json(row_root / "upload.json", upload)
    result = MODULE._self_hashed(
        {
            "schema_version": 1,
            "outcome": "exact",
            "value": 1,
            "accepted_instance_ids": ["I1"],
            "reason_codes": [],
        }
    )
    MODULE._write_once_json(row_root / "result.json", result)
    complete = MODULE._self_hashed(
        {
            "schema_version": 1,
            "outcome": "success",
            "video_id": "v1",
            "request_manifest_sha256": "manifest-sha",
            "upload_complete_sha256": upload["complete_sha256"],
            "delivery_sha256": "delivery-sha",
            "stage_complete_sha256": {},
            "result_sha256": MODULE._file_sha256(row_root / "result.json"),
            "estimated_cost_usd": 0.0,
        }
    )
    MODULE._write_once_json(row_root / "complete.json", complete)
    manifest = {
        "schema_version": 1,
        "rows": [row],
        "complete_sha256": "manifest-sha",
    }

    with pytest.raises(ValueError, match="mandatory stages"):
        MODULE.freeze_predictions(root=tmp_path, manifest=manifest)


def test_run_canary_max_rows_one_cannot_continue_to_later_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        {
            "video_id": video_id,
            "delivery_sha256": f"delivery-{video_id}",
            "delivery_mime_type": "video/webm",
        }
        for video_id in MODULE.ORDERED_VIDEO_IDS
    ]
    manifest = MODULE._self_hashed(
        {"schema_version": 1, "rows": rows, "condition": "fixture"}
    )
    pricing = MODULE._self_hashed({"schema_version": 1, **MODULE.fixture_pricing()})
    MODULE._write_once_json(tmp_path / "requests.label-blind.json", manifest)
    MODULE._write_once_json(tmp_path / "pricing.json", pricing)
    attempted: list[str] = []
    frozen_limits: list[int | None] = []

    monkeypatch.setenv("GEMINI_API_KEY", "fixture")
    monkeypatch.setattr(MODULE.genai, "Client", lambda **kwargs: object())
    monkeypatch.setattr(
        MODULE,
        "ensure_upload",
        lambda *, row, **kwargs: {
            "file_uri": f"files/{row['video_id']}",
            "complete_sha256": f"upload-{row['video_id']}",
        },
    )
    monkeypatch.setattr(
        MODULE,
        "run_row",
        lambda *, row, **kwargs: attempted.append(str(row["video_id"])),
    )
    monkeypatch.setattr(MODULE, "budget_exposure", lambda root: 0.0)

    def freeze(*, row_limit=None, **kwargs):
        frozen_limits.append(row_limit)
        return {"completion": {"completed": row_limit, "total": row_limit}}

    monkeypatch.setattr(MODULE, "freeze_predictions", freeze)

    result = MODULE.run_canary(
        root=tmp_path, pricing_path=tmp_path / "pricing.json", max_rows=1
    )

    assert attempted == [MODULE.ORDERED_VIDEO_IDS[0]]
    assert frozen_limits == [1]
    assert result["completion"] == {"completed": 1, "total": 1}
