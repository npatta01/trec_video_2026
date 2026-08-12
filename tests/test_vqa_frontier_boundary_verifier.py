from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from trec_video_2026 import vqa_frontier_boundary_verifier as verifier_module
from trec_video_2026.vqa_frontier_boundary_verifier import (
    BoundaryVerifierError,
    PROVIDER_VERIFIER_SCHEMA,
    VERIFIER_POLICY,
    PreparedBoundaryVerifier,
    build_verifier_prompt,
    freeze_verifier,
    preflight_verifier,
    prepare_verifier,
    run_verifier,
    status_verifier,
    validate_verifier_output,
)
from trec_video_2026.vqa_experiments.temporal_bottleneck import freeze_json_artifact
from trec_video_2026.vqa_frontier_map_media import DeliveryMedia
from trec_video_2026.vqa_frontier_phase0_contract import validate_neutral_map
from trec_video_2026.vqa_submission_openrouter import OpenRouterTransport


def neutral_map() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "perceived_duration_seconds": 4.0,
        "global_description": "A person changes position.",
        "entities": [],
        "events": [
            {
                "event_id": "E1",
                "start_seconds": 0.0,
                "end_seconds": 2.0,
                "keyframe_seconds": [1.0],
                "description": "The first state is visible.",
                "entity_ids": [],
                "actions": ["A state remains visible"],
                "relations": [],
                "state_before": "The clip starts.",
                "state_change": "The first state appears.",
                "state_after": "The first state remains.",
                "visual_observations": ["A stable view."],
                "speech_observations": [],
                "sound_observations": [],
                "visible_text": [],
                "confidence": 0.8,
                "uncertainties": ["The onset is approximate."],
            },
            {
                "event_id": "E2",
                "start_seconds": 2.0,
                "end_seconds": 4.0,
                "keyframe_seconds": [3.0],
                "description": "The second state is visible.",
                "entity_ids": [],
                "actions": ["The state changes"],
                "relations": [],
                "state_before": "The first state remains.",
                "state_change": "The view changes.",
                "state_after": "The second state remains.",
                "visual_observations": ["A changed view."],
                "speech_observations": [],
                "sound_observations": [],
                "visible_text": [],
                "confidence": 0.8,
                "uncertainties": [],
            },
        ],
        "global_uncertainties": [],
    }


def verifier_output() -> dict[str, Any]:
    draft = validate_neutral_map(neutral_map(), 4.0)
    return {
        "schema_version": 1,
        "draft_map_sha256": draft.sha256,
        "verdict": "corrected",
        "event_reviews": [
            {
                "draft_event_ids": ["E1"],
                "action": "adjusted",
                "observed_start_seconds": 0.0,
                "observed_end_seconds": 2.2,
                "evidence_timestamps": [0.5, 2.1],
                "rationale": "The transition occurs after two seconds.",
                "uncertainty": "The boundary is uncertain by 0.1 seconds.",
            },
            {
                "draft_event_ids": ["E2"],
                "action": "adjusted",
                "observed_start_seconds": 2.2,
                "observed_end_seconds": 4.0,
                "evidence_timestamps": [2.3, 3.8],
                "rationale": "The second state begins after the transition.",
                "uncertainty": "none",
            },
        ],
        "corrected_map": neutral_map(),
        "global_notes": ["The draft boundary was coarse."],
    }


def test_prompt_is_question_blind_and_binds_untrusted_draft() -> None:
    prompt = build_verifier_prompt(neutral_map(), 4.0)

    assert "question-blind" in prompt.lower()
    assert "untrusted" in prompt.lower()
    assert "complete native video and audio" in prompt.lower()
    assert "add omitted observable states" in prompt.lower()
    assert validate_neutral_map(neutral_map(), 4.0).sha256 in prompt
    assert "Question:" not in prompt


def test_validate_verifier_output_requires_exact_bound_evidence() -> None:
    normalized = validate_verifier_output(
        verifier_output(), draft_map=neutral_map(), duration_seconds=4.0
    )

    assert normalized["verdict"] == "corrected"
    assert normalized["event_reviews"][0]["observed_end_seconds"] == 2.2


@pytest.mark.parametrize("mutation", ["hash", "extra", "missing_event", "timestamp"])
def test_validate_verifier_output_rejects_unbound_or_incomplete_data(
    mutation: str,
) -> None:
    payload = verifier_output()
    if mutation == "hash":
        payload["draft_map_sha256"] = "0" * 64
    elif mutation == "extra":
        payload["question"] = "must not enter verifier output"
    elif mutation == "missing_event":
        payload["event_reviews"] = payload["event_reviews"][:1]
    else:
        payload["event_reviews"][0]["evidence_timestamps"] = [5.0]

    with pytest.raises((TypeError, ValueError)):
        validate_verifier_output(payload, draft_map=neutral_map(), duration_seconds=4.0)


def test_provider_schema_projection_omits_gemini_unsupported_keywords() -> None:
    rendered = json.dumps(PROVIDER_VERIFIER_SCHEMA, sort_keys=True)

    for keyword in (
        '"additionalProperties"',
        '"const"',
        '"minimum"',
        '"minItems"',
        '"minLength"',
        '"maxLength"',
    ):
        assert keyword not in rendered


def test_prepare_uses_frozen_aggregate_not_mutable_result_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_run = tmp_path / "source-run"
    (source_run / "results").mkdir(parents=True)
    frozen_map = neutral_map()
    mutable_map = neutral_map()
    mutable_map["global_description"] = "A different mutable result."
    pilot_results = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "results": [
            {
                "phase": "pilot",
                "video_id": "pilot-video",
                "condition_id": "neutral_map",
                "status": "succeeded",
                "validated_output": frozen_map,
            }
        ],
    }
    pilot_results_path = source_run / "pilot-results.json"
    pilot_results_path.write_text(json.dumps(pilot_results), encoding="utf-8")
    pilot_freeze_path = source_run / "pilot-freeze.json"
    pilot_freeze_path.write_text(
        json.dumps(
            freeze_json_artifact(
                pilot_results_path, bindings={"run_id": "fixture-run"}
            )
        ),
        encoding="utf-8",
    )
    (source_run / "results" / "neutral_map.jsonl").write_text(
        json.dumps(
            {
                "phase": "pilot",
                "video_id": "pilot-video",
                "condition_id": "neutral_map",
                "status": "succeeded",
                "validated_output": mutable_map,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    audit_path = audit_dir / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "fixture-run",
                "panel_sha256": "a" * 64,
                "phase": "pilot",
                "decision": "revise_contract",
                "prediction_artifact_sha256": hashlib.sha256(
                    pilot_results_path.read_bytes()
                ).hexdigest(),
                "prediction_freeze_sha256": hashlib.sha256(
                    pilot_freeze_path.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    audit_freeze_path = audit_dir / "freeze.json"
    audit_freeze_path.write_text(
        json.dumps(
            freeze_json_artifact(
                audit_path,
                bindings={
                    "run_id": "fixture-run",
                    "panel_sha256": "a" * 64,
                    "phase": "pilot",
                    "decision": "revise_contract",
                },
            )
        ),
        encoding="utf-8",
    )
    media_root = tmp_path / "canonical"
    media = media_root / "fixture" / "pilot-video.webm"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"native-video")
    row = SimpleNamespace(
        video_id="pilot-video",
        duration_seconds=4.0,
        media_relative_path="fixture/pilot-video.webm",
        media_sha256=hashlib.sha256(media.read_bytes()).hexdigest(),
        size_bytes=media.stat().st_size,
    )
    monkeypatch.setattr(verifier_module, "CANONICAL_DATA_ROOT", media_root)
    monkeypatch.setattr(
        verifier_module,
        "load_prepared",
        lambda _path: SimpleNamespace(
            pilot_row=row, run_id="fixture-run", panel_sha256="a" * 64
        ),
    )

    prepared = prepare_verifier(
        source_run, audit_freeze_path, "pilot-video", tmp_path / "verifier"
    )

    assert prepared.draft_map.sha256 == validate_neutral_map(frozen_map, 4.0).sha256
    assert prepared.draft_map.sha256 != validate_neutral_map(mutable_map, 4.0).sha256


class FakeResponse:
    status = 200

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def read(self, _limit: int = -1) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def close(self) -> None:
        pass


class FakeOpener:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.requests: list[Any] = []

    def open(self, request: Any, *, timeout: float) -> FakeResponse:
        self.requests.append(request)
        return self.response


def prepared_fixture(tmp_path: Path) -> PreparedBoundaryVerifier:
    root = tmp_path / "verifier"
    root.mkdir()
    (root / "attempts.jsonl").touch()
    media = root / "clip.webm"
    media.write_bytes(b"fixture-video")
    digest = hashlib.sha256(media.read_bytes()).hexdigest()
    provenance = root / "delivery.json"
    provenance.write_text("{}", encoding="utf-8")
    draft = validate_neutral_map(neutral_map(), 4.0)
    source_run = root / "source-run"
    source_run.mkdir()
    pilot_results_path = source_run / "pilot-results.json"
    pilot_results_path.write_text(
        json.dumps({"schema_version": 1, "run_id": "fixture-run"}),
        encoding="utf-8",
    )
    pilot_freeze_path = source_run / "pilot-freeze.json"
    pilot_freeze_path.write_text(
        json.dumps(
            freeze_json_artifact(
                pilot_results_path, bindings={"run_id": "fixture-run"}
            )
        ),
        encoding="utf-8",
    )
    audit = root / "source-audit"
    audit.mkdir()
    audit_path = audit / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "fixture-run",
                "prediction_artifact_sha256": hashlib.sha256(
                    pilot_results_path.read_bytes()
                ).hexdigest(),
                "prediction_freeze_sha256": hashlib.sha256(
                    pilot_freeze_path.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    (audit / "freeze.json").write_text(
        json.dumps(
            freeze_json_artifact(audit_path, bindings={"run_id": "fixture-run"})
        ),
        encoding="utf-8",
    )
    manifest = {
        "video_id": "pilot-video",
        "duration_seconds": 4.0,
        "source_sha256": digest,
        "source_run": str(source_run),
        "source_audit_freeze": str(audit / "freeze.json"),
        "projected_call_cost_usd": 0.5,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "draft-map.json").write_text(json.dumps(draft.to_dict()), encoding="utf-8")
    delivery = DeliveryMedia(
        path=media,
        provenance_path=provenance,
        mime_type="video/webm",
        sha256=digest,
        size_bytes=media.stat().st_size,
        source_sha256=digest,
        source_size_bytes=media.stat().st_size,
        policy_version="fixture",
        derived=False,
    )
    return PreparedBoundaryVerifier(root, manifest, draft, media, delivery)


def test_one_call_runner_preserves_raw_freezes_and_refuses_retry(tmp_path: Path) -> None:
    prepared = prepared_fixture(tmp_path)
    response = FakeResponse(
        {
            "model": "google/gemini-3.1-pro-preview",
            "provider": "Google AI Studio",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(verifier_output()),
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300,
                "cost": 0.05,
            },
        }
    )
    opener = FakeOpener(response)
    transport = OpenRouterTransport("fixture-key", opener=opener, policy=VERIFIER_POLICY)

    result = run_verifier(prepared, transport)

    assert result["status"] == "succeeded"
    assert result["cost_usd"] == 0.05
    assert len(opener.requests) == 1
    assert (prepared.path / "request.json").is_file()
    assert (prepared.path / "raw-response.json").is_file()
    assert len((prepared.path / "attempts.jsonl").read_text().splitlines()) == 2
    freeze = freeze_verifier(prepared)
    assert freeze["status"] == "frozen"
    assert status_verifier(prepared)["terminal_attempts"] == 1
    with pytest.raises(FileExistsError):
        run_verifier(prepared, transport)


@pytest.mark.parametrize(
    ("model", "provider", "cost", "failure_code"),
    [
        ("google/gemini-3.1-pro-preview", None, 0.05, "provider_failure"),
        (None, "Google AI Studio", 0.05, "provider_failure"),
        (
            "google/gemini-3.1-pro-preview",
            "Google AI Studio",
            None,
            "provider_evidence",
        ),
        (
            "google/gemini-3.1-pro-preview",
            "Google AI Studio",
            30.0,
            "cost_cap_exceeded",
        ),
    ],
)
def test_terminal_provider_evidence_and_cap_failures_cannot_freeze(
    tmp_path: Path,
    model: str | None,
    provider: str | None,
    cost: float | None,
    failure_code: str,
) -> None:
    prepared = prepared_fixture(tmp_path)
    response_payload: dict[str, Any] = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(verifier_output())},
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "total_tokens": 300,
        },
    }
    if model is not None:
        response_payload["model"] = model
    if provider is not None:
        response_payload["provider"] = provider
    if cost is not None:
        response_payload["usage"]["cost"] = cost
    opener = FakeOpener(FakeResponse(response_payload))
    transport = OpenRouterTransport("fixture-key", opener=opener, policy=VERIFIER_POLICY)

    result = run_verifier(prepared, transport)

    assert result["status"] == "failed"
    assert result["failure_code"] == failure_code
    assert len(opener.requests) == 1
    with pytest.raises(BoundaryVerifierError):
        freeze_verifier(prepared)


def test_terminal_schema_failure_is_preserved_without_retry(tmp_path: Path) -> None:
    prepared = prepared_fixture(tmp_path)
    opener = FakeOpener(
        FakeResponse(
            {
                "model": "google/gemini-3.1-pro-preview",
                "provider": "Google AI Studio",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"schema_version": 1}),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                    "cost": 0.01,
                },
            }
        )
    )
    transport = OpenRouterTransport("fixture-key", opener=opener, policy=VERIFIER_POLICY)

    result = run_verifier(prepared, transport)

    assert result["status"] == "failed"
    assert result["failure_code"] == "schema_validation"
    assert (prepared.path / "raw-response.json").is_file()
    assert len(opener.requests) == 1
    with pytest.raises(FileExistsError):
        run_verifier(prepared, transport)


def test_preflight_rejects_audit_prediction_freeze_mismatch(tmp_path: Path) -> None:
    prepared = prepared_fixture(tmp_path)
    audit_path = prepared.path / "source-audit" / "audit.json"
    audit = json.loads(audit_path.read_text())
    audit["prediction_freeze_sha256"] = "0" * 64
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    audit_freeze_path = prepared.path / "source-audit" / "freeze.json"
    audit_freeze_path.write_text(
        json.dumps(
            freeze_json_artifact(audit_path, bindings={"run_id": "fixture-run"})
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prediction-freeze relationship"):
        preflight_verifier(prepared)


def test_first_freeze_revalidates_upstream_before_writing(tmp_path: Path) -> None:
    prepared = prepared_fixture(tmp_path)
    opener = FakeOpener(
        FakeResponse(
            {
                "model": "google/gemini-3.1-pro-preview",
                "provider": "Google AI Studio",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(verifier_output()),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 200,
                    "total_tokens": 300,
                    "cost": 0.05,
                },
            }
        )
    )
    run_verifier(
        prepared,
        OpenRouterTransport("fixture-key", opener=opener, policy=VERIFIER_POLICY),
    )
    upstream = prepared.path / "source-run" / "pilot-results.json"
    upstream.write_bytes(upstream.read_bytes() + b" ")

    with pytest.raises(ValueError, match="artifact_sha256"):
        freeze_verifier(prepared)
    assert not (prepared.path / "freeze.json").exists()


def test_unfrozen_status_revalidates_upstream_chain(tmp_path: Path) -> None:
    prepared = prepared_fixture(tmp_path)
    upstream = prepared.path / "source-run" / "pilot-results.json"
    upstream.write_bytes(upstream.read_bytes() + b" ")

    with pytest.raises(ValueError, match="artifact_sha256"):
        status_verifier(prepared)


@pytest.mark.parametrize(
    "target",
    [
        "manifest.json",
        "request.json",
        "attempts.jsonl",
        "raw-response.json",
        "clip.webm",
        "source-run/pilot-results.json",
    ],
)
def test_freeze_detects_complete_chain_tampering(
    tmp_path: Path, target: str
) -> None:
    prepared = prepared_fixture(tmp_path)
    opener = FakeOpener(
        FakeResponse(
            {
                "model": "google/gemini-3.1-pro-preview",
                "provider": "Google AI Studio",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(verifier_output()),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 200,
                    "total_tokens": 300,
                    "cost": 0.05,
                },
            }
        )
    )
    transport = OpenRouterTransport("fixture-key", opener=opener, policy=VERIFIER_POLICY)
    run_verifier(prepared, transport)
    freeze_verifier(prepared)
    path = prepared.path / target
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="artifact|freeze"):
        freeze_verifier(prepared)
    with pytest.raises(ValueError, match="artifact|freeze"):
        status_verifier(prepared)
