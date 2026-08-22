from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from trec_video_2026.vqa_experiments.vqa_mc_four_option_agent import (
    OBSERVER_SCHEMA,
    append_round_stage,
    initialize_notebook,
    parse_initializer,
    parse_observer,
    parse_planner,
)
from trec_video_2026.vqa_experiments.vqa_mc_common import sha256_file


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "vqa_mc_four_option_runner",
    ROOT / "scripts/run_vqa_mc_gemini_four_option_agent_poc20.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _row(video_id: str = "video-a") -> dict[str, Any]:
    return {
        "query_id": video_id,
        "video_id": video_id,
        "stratum": "counting",
        "question": "How many times does the person jump?",
        "duration_seconds": 6.0,
        "options": [
            {"option_id": option_id, "text": text}
            for option_id, text in zip(
                "ABCD",
                ("once", "twice", "three times", "four times"),
                strict=True,
            )
        ],
    }


def _upload() -> dict[str, str]:
    return {
        "file_uri": "https://provider.invalid/video-a",
        "mime_type": "video/webm",
    }


def _initializer_payload() -> dict[str, Any]:
    return {
        "provisional_ranking": list("BACD"),
        "ledger": [
            {
                "option_id": option_id,
                "observable_claim": f"Claim for {option_id}.",
                "supporting_evidence_ids": [],
                "contradicting_evidence_ids": [],
                "untested_claims": ["A new observation is needed."],
                "status": "viable",
                "interpretation": f"Option {option_id} remains possible.",
            }
            for option_id in "ABCD"
        ],
        "literal_observations": [
            {
                "timestamp_seconds": 2.0,
                "literal_observation": "The person leaves the floor.",
                "interpretation": "This may be a jump.",
            }
        ],
        "unresolved_distinctions": ["Whether another jump occurs."],
    }


def _action_payload() -> dict[str, Any]:
    return {
        "kind": "inspect_interval",
        "start_seconds": 1.0,
        "end_seconds": 4.0,
        "fps": 8.0,
        "resolution": "high",
        "neutral_request": "Describe every departure from and return to the ground.",
        "tested_option_ids": list("ABCD"),
        "expected_outcomes": [
            {"option_id": option_id, "literal_outcome": f"Count for {option_id}."}
            for option_id in "ABCD"
        ],
    }


def _planner_payload() -> dict[str, Any]:
    return {
        "provisional_ranking": list("BACD"),
        "unresolved_distinction": "The complete action count.",
        "action": _action_payload(),
        "why_discriminating": "A gapless count distinguishes all choices.",
    }


def _observer_payload() -> dict[str, Any]:
    return {
        "observations": [
            {
                "timestamp_seconds": 2.5,
                "literal_observation": "One departure and landing are visible.",
                "interpretation": "One cycle is complete.",
            }
        ],
        "exact_transcription": "No transcription was requested.",
        "focus_resolution": "present",
        "view_limitations": "The first and final second are outside this view.",
    }


def _notebook() -> dict[str, Any]:
    initializer = parse_initializer(
        _initializer_payload(),
        duration_seconds=6.0,
        protected_ranking=list("BACD"),
    )
    notebook = initialize_notebook("video-a", list("BACD"), initializer)
    planner = parse_planner(
        _planner_payload(), duration_seconds=6.0, round_index=1
    )
    notebook = append_round_stage(
        notebook, round_index=1, stage="planner", record=planner
    )
    observer = parse_observer(
        _observer_payload(),
        action=planner["action"],
        duration_seconds=6.0,
        view_id="view-1",
    )
    return append_round_stage(
        notebook, round_index=1, stage="observer", record=observer
    )


def _capabilities() -> Any:
    return runner.ModelCapabilities(
        output_token_limit=65_536,
        input_token_limit=1_000_000,
        model_version="gemini-3.7-flash-20260813",
        supported_actions=("generateContent",),
    )


def _request_payload(tmp_path: Path, *, row_count: int = 1) -> Path:
    rows = [_row(f"video-{index}") for index in range(row_count)]
    path = tmp_path / "requests.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": "test-four-option-agent",
                "rows": rows,
                "variants": [
                    {
                        "variant_id": f"{row['video_id']}:canonical",
                        "video_id": row["video_id"],
                        "variant_kind": "canonical",
                        "display_order": list("ABCD"),
                    }
                    for row in rows
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _control_path(tmp_path: Path, *, row_count: int = 1) -> Path:
    path = tmp_path / "control.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rows": [
                    {
                        "video_id": f"video-{index}",
                        "variant_kind": "canonical",
                        "status": "passed",
                        "ranking": list("BACD"),
                    }
                    for index in range(row_count)
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _historical_root(tmp_path: Path, *, row_count: int = 1) -> Path:
    root = tmp_path / "historical"
    for index in range(row_count):
        video_id = f"video-{index}"
        delivery = root / "deliveries" / video_id / "delivery-media" / f"{video_id}.webm"
        delivery.parent.mkdir(parents=True)
        delivery.write_bytes(b"fake video")
    return root


class _MetadataModels:
    def __init__(self, output_token_limit: int) -> None:
        self.output_token_limit = output_token_limit
        self.requested_model: str | None = None

    def get(self, *, model: str) -> Any:
        self.requested_model = model
        return SimpleNamespace(
            name=model,
            version="gemini-3.7-flash-20260813",
            input_token_limit=1_000_000,
            output_token_limit=self.output_token_limit,
            supported_actions=["generateContent"],
        )


class _MetadataClient:
    def __init__(self, output_token_limit: int) -> None:
        self.models = _MetadataModels(output_token_limit)


def test_live_capabilities_use_model_output_token_limit() -> None:
    client = _MetadataClient(output_token_limit=65_536)

    capabilities = runner.get_model_capabilities(client)

    assert capabilities.output_token_limit == 65_536
    assert capabilities.input_token_limit == 1_000_000
    assert capabilities.model_version == "gemini-3.7-flash-20260813"
    assert client.models.requested_model == runner.MODEL


def test_every_role_uses_live_provider_output_maximum_and_high_thinking() -> None:
    capabilities = _capabilities()
    action = parse_planner(
        _planner_payload(), duration_seconds=6.0, round_index=1
    )["action"]
    requests = [
        runner.build_initializer_request(
            row=_row(),
            upload=_upload(),
            protected_ranking=list("BACD"),
            capabilities=capabilities,
        ),
        runner.build_planner_request(
            row=_row(), notebook=_notebook(), round_index=2, capabilities=capabilities
        ),
        runner.build_observer_request(
            upload=_upload(), action=action, capabilities=capabilities
        ),
        runner.build_updater_request(
            row=_row(), notebook=_notebook(), capabilities=capabilities
        ),
        runner.build_critic_request(
            row=_row(), notebook=_notebook(), capabilities=capabilities
        ),
        runner.build_judge_request(
            row=_row(),
            upload=_upload(),
            notebook=_notebook(),
            capabilities=capabilities,
        ),
    ]

    assert {request.config.max_output_tokens for request in requests} == {65_536}
    assert all(
        request.config.thinking_config.thinking_level.name.endswith("HIGH")
        for request in requests
    )


def test_observer_request_passes_host_information_barrier() -> None:
    action = parse_planner(
        _planner_payload(), duration_seconds=6.0, round_index=1
    )["action"]
    request = runner.build_observer_request(
        upload=_upload(), action=action, capabilities=_capabilities()
    )

    audit = runner.audit_observer_request(
        request,
        prohibited_values=[
            _row()["question"],
            *[option["text"] for option in _row()["options"]],
            *list("ABCD"),
            "BACD",
        ],
    )

    assert audit == {"passed": True, "matched_value_hashes": []}


def test_observer_schema_has_no_answer_or_ranking_fields() -> None:
    serialized = json.dumps(OBSERVER_SCHEMA).casefold()
    assert "ranking" not in serialized
    assert "option_id" not in serialized


def test_observer_uses_requested_interval_fps_and_resolution() -> None:
    action = parse_planner(
        _planner_payload(), duration_seconds=6.0, round_index=1
    )["action"]
    request = runner.build_observer_request(
        upload=_upload(), action=action, capabilities=_capabilities()
    )
    video = request.contents[0].parts[0]

    assert video.video_metadata.start_offset == "1.000s"
    assert video.video_metadata.end_offset == "4.000s"
    assert video.video_metadata.fps == 8.0
    assert video.media_resolution.level.name.endswith("HIGH")


def test_formal_cost_prices_every_recovery_at_provider_maximum(tmp_path: Path) -> None:
    request_path = _request_payload(tmp_path, row_count=20)
    payload = runner.load_requests(
        request_path, expected_rows=20, expected_variants=20
    )

    estimate = runner.estimate_formal_cost(payload, capabilities=_capabilities())

    assert estimate["provider_call_ceiling"] == 20 * 28
    assert estimate["output_token_ceiling"] == 20 * 28 * 65_536
    assert estimate["formal_cost_usd"] > runner.MAX_COST_USD


def test_preflight_blocks_before_delivery_probe_when_formal_bound_exceeds_cap(
    tmp_path: Path,
) -> None:
    probes: list[Path] = []

    with pytest.raises(ValueError, match="formal worst-case"):
        runner.preflight(
            requests_path=_request_payload(tmp_path),
            protected_control_path=_control_path(tmp_path),
            historical_root=_historical_root(tmp_path),
            capabilities=_capabilities(),
            max_cost_usd=0.000001,
            expected_rows=1,
            expected_variants=1,
            delivery_probe=lambda path: probes.append(path),
        )

    assert probes == []


def test_requests_reject_label_fields_and_control_identity_drift(tmp_path: Path) -> None:
    request_path = _request_payload(tmp_path)
    payload = json.loads(request_path.read_text())
    payload["rows"][0]["correct_option_id"] = "A"
    request_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="label"):
        runner.load_requests(request_path, expected_rows=1, expected_variants=1)

    clean = _request_payload(tmp_path)
    control = _control_path(tmp_path)
    control_payload = json.loads(control.read_text())
    control_payload["rows"][0]["video_id"] = "wrong-video"
    control.write_text(json.dumps(control_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        runner.preflight(
            requests_path=clean,
            protected_control_path=control,
            historical_root=_historical_root(tmp_path),
            capabilities=runner.ModelCapabilities(output_token_limit=1),
            max_cost_usd=10.0,
            expected_rows=1,
            expected_variants=1,
        )


def test_requests_validate_but_do_not_execute_order_diagnostic_variants(
    tmp_path: Path,
) -> None:
    path = _request_payload(tmp_path)
    payload = json.loads(path.read_text())
    payload["variants"].extend(
        [
            {
                "variant_id": "video-0:reversed",
                "video_id": "video-0",
                "variant_kind": "reversed",
                "display_order": list("DCBA"),
            },
            {
                "variant_id": "video-0:repeat",
                "video_id": "video-0",
                "variant_kind": "repeat",
                "display_order": list("ABCD"),
            },
        ]
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = runner.load_requests(path, expected_rows=1, expected_variants=None)

    assert len(loaded["rows"]) == 1
    assert len(loaded["variants"]) == 3


def _update_payload(ranking: str = "BACD", observation_id: str = "round-1-o1") -> dict[str, Any]:
    ledger = _initializer_payload()["ledger"]
    ledger[0]["supporting_evidence_ids"] = [observation_id]
    return {
        "provisional_ranking": list(ranking),
        "ledger": ledger,
        "mapped_observation_ids": [observation_id],
        "unresolved_distinctions": [],
    }


def _critic_payload(decision: str = "accept_stop") -> dict[str, Any]:
    return {
        "decision": decision,
        "tested_viable_option_ids": list("ABCD"),
        "leader_evidence_ids": ["round-1-o1"],
        "unexplained_counterevidence_ids": [],
        "next_unresolved_distinction": (
            "No feasible observation remains."
            if decision == "accept_stop"
            else "Inspect another view for contradictory evidence."
        ),
        "direct_replacement_supported": False,
        "reason": "Literal evidence has been considered for all viable options.",
    }


def _judge_payload(
    ranking: str = "BACD", *, stop_status: str = "evidence_certified"
) -> dict[str, Any]:
    return {
        "ranking": list(ranking),
        "decisive_evidence_ids": ["round-1-o1"],
        "strongest_counterevidence": "Part of the clip may contain another event.",
        "counterevidence_resolution": "The complete judge view resolves the concern.",
        "protected_answer_action": "retained",
        "direct_contradiction_evidence_ids": [],
        "replacement_support_evidence_ids": [],
        "stop_status": stop_status,
        "rationale": "The cited literal evidence supports the protected answer.",
    }


def _provider_response(
    value: dict[str, Any], *, finish_reason: str = "STOP"
) -> dict[str, Any]:
    return {
        "text": json.dumps(value),
        "finishReason": finish_reason,
        "modelVersion": "gemini-3.7-flash-20260813",
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 10,
            "totalTokenCount": 110,
        },
    }


class _Models:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.request_count = 0

    def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        if model != runner.MODEL or not contents or config is None:
            raise AssertionError("runner passed an invalid provider request")
        self.request_count += 1
        return self.responses.pop(0)


class _Client:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.models = _Models(responses)


def _one_observation_stop_path() -> list[dict[str, Any]]:
    return [
        _provider_response(_initializer_payload()),
        _provider_response(_planner_payload()),
        _provider_response(_observer_payload()),
        _provider_response(_update_payload()),
        _provider_response(_critic_payload("accept_stop")),
        _provider_response(_judge_payload()),
    ]


def _three_observation_path() -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = [_provider_response(_initializer_payload())]
    for round_index in range(1, 4):
        planner = _planner_payload()
        observation_id = f"round-{round_index}-o1"
        responses.extend(
            [
                _provider_response(planner),
                _provider_response(_observer_payload()),
                _provider_response(_update_payload(observation_id=observation_id)),
                _provider_response(_critic_payload("continue")),
            ]
        )
    responses.append(_provider_response(_judge_payload(stop_status="forced_stop")))
    return responses


def _run_fake(
    tmp_path: Path,
    *,
    responses: list[dict[str, Any]],
    output_name: str = "run",
) -> tuple[dict[str, Any], Path, _Client]:
    output = tmp_path / output_name
    client = _Client(responses)
    result = runner.run_phase(
        requests_path=_request_payload(tmp_path),
        protected_control_path=_control_path(tmp_path),
        historical_root=_historical_root(tmp_path),
        output_root=output,
        capabilities=_capabilities(),
        client=client,
        upload_file=lambda _client, _path: _upload(),
        max_cost_usd=100.0,
        expected_rows=1,
        expected_variants=1,
    )
    return result, output, client


def _final_row(output: Path) -> dict[str, Any]:
    final = json.loads((output / "final-predictions-freeze.json").read_text())
    return final["rows"][0]


def test_max_tokens_is_recovered_and_never_becomes_an_answer(tmp_path: Path) -> None:
    responses = [
        _provider_response(_initializer_payload(), finish_reason="MAX_TOKENS"),
        *_one_observation_stop_path(),
    ]

    result, output, client = _run_fake(tmp_path, responses=responses)

    assert result["recovery_call_count"] == 1
    assert result["operational_invalid_count"] == 0
    assert client.models.request_count == 7
    first = json.loads(
        (output / "rows/00/initializer/attempt-1.json").read_text()
    )
    assert first["finish_reason"] == "MAX_TOKENS"


def test_second_invalid_initializer_uses_unknown_ledger_fallback(
    tmp_path: Path,
) -> None:
    responses = [
        _provider_response({"broken": 1}),
        _provider_response({"still_broken": 1}),
        *_one_observation_stop_path()[1:],
    ]

    result, output, _client = _run_fake(tmp_path, responses=responses)

    assert result["operational_invalid_count"] == 1
    assert _final_row(output)["initializer_status"] == "fallback_unknown_ledger"


def test_one_observation_path_cannot_stop_before_observer(tmp_path: Path) -> None:
    result, output, _client = _run_fake(
        tmp_path, responses=_one_observation_stop_path()
    )
    row = _final_row(output)

    assert row["observer_action_count"] == 1
    assert row["stop_status"] == "evidence_certified"
    assert result["provider_call_counts_by_role"] == {
        "critic": 1,
        "initializer": 1,
        "judge": 1,
        "observer": 1,
        "planner": 1,
        "updater": 1,
    }


def test_three_continue_decisions_force_stop_and_still_judge(tmp_path: Path) -> None:
    result, output, _client = _run_fake(
        tmp_path, responses=_three_observation_path()
    )
    row = _final_row(output)

    assert row["observer_action_count"] == 3
    assert row["stop_status"] == "forced_stop"
    assert row["judge_status"] == "passed"
    assert result["provider_request_count"] == 14


def test_invalid_final_judge_falls_back_to_protected_ranking_but_is_invalid(
    tmp_path: Path,
) -> None:
    responses = _one_observation_stop_path()[:-1] + [
        _provider_response({"broken": 1}),
        _provider_response({"still_broken": 1}),
    ]

    result, output, _client = _run_fake(tmp_path, responses=responses)
    row = _final_row(output)

    assert row["ranking"] == list("BACD")
    assert row["final_source"] == "protected_direct_fallback"
    assert row["status"] == "operationally_invalid"
    assert result["operational_invalid_count"] == 1


def test_verify_output_rejects_tampered_observer_audit(tmp_path: Path) -> None:
    _result, output, _client = _run_fake(
        tmp_path, responses=_one_observation_stop_path()
    )
    audit_path = output / "rows/00/round-1/observer/request-audit.json"
    audit_path.write_text(json.dumps({"passed": False}), encoding="utf-8")

    with pytest.raises(ValueError, match="observer audit"):
        runner.verify_output(output)


def test_full_phase_reuses_canary_rows_only_when_protocol_hashes_match(
    tmp_path: Path,
) -> None:
    _result, canary, _client = _run_fake(
        tmp_path, responses=_one_observation_stop_path(), output_name="canary"
    )
    gate_path = tmp_path / "canary-gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "continue_to_full": True,
                "canary_manifest_sha256": sha256_file(
                    canary / "run-manifest.json"
                ),
                "canary_final_sha256": sha256_file(
                    canary / "final-predictions-freeze.json"
                ),
            }
        ),
        encoding="utf-8",
    )
    full_requests = _request_payload(tmp_path)

    result = runner.verify_canary_reuse(
        canary_root=canary,
        full_requests_path=full_requests,
        canary_gate_path=gate_path,
    )

    assert result["reusable_row_count"] == 1
    assert result["reusable_video_ids"] == ["video-0"]

    manifest_path = canary / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["prompt_contract_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol identity"):
        runner.verify_canary_reuse(
            canary_root=canary,
            full_requests_path=full_requests,
            canary_gate_path=gate_path,
        )


def test_experiment_preflight_blocks_formal_full_cost_without_video_probe(
    tmp_path: Path,
) -> None:
    full_path = _request_payload(tmp_path, row_count=20)
    full = json.loads(full_path.read_text())
    canary_path = tmp_path / "canary-requests.json"
    canary_path.write_text(
        json.dumps(
            {
                **full,
                "rows": full["rows"][:5],
                "variants": full["variants"][:5],
            }
        ),
        encoding="utf-8",
    )
    probes: list[Path] = []

    result = runner.build_experiment_preflight(
        canary_requests_path=canary_path,
        full_requests_path=full_path,
        protected_control_path=_control_path(tmp_path, row_count=20),
        historical_root=_historical_root(tmp_path, row_count=20),
        capabilities=_capabilities(),
        delivery_probe=lambda path: probes.append(path),
    )

    assert result["status"] == "preflight_blocked"
    assert result["paid_run_allowed"] is False
    assert result["cost"]["provider_call_ceiling"] == 20 * 28
    assert result["cost"]["formal_cost_usd"] > 1.5
    assert probes == []


def test_preflight_cli_writes_blocked_artifact_without_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    full_path = _request_payload(tmp_path, row_count=20)
    full = json.loads(full_path.read_text())
    canary_path = tmp_path / "canary-requests.json"
    canary_path.write_text(
        json.dumps(
            {
                **full,
                "rows": full["rows"][:5],
                "variants": full["variants"][:5],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "preflight.json"
    client = _MetadataClient(output_token_limit=65_536)
    monkeypatch.setattr(runner, "_api_key", lambda: "test-key")
    monkeypatch.setattr(runner.genai, "Client", lambda *, api_key: client)

    exit_code = runner.main(
        [
            "--preflight",
            "--canary-requests",
            str(canary_path),
            "--full-requests",
            str(full_path),
            "--protected-control",
            str(_control_path(tmp_path, row_count=20)),
            "--historical-root",
            str(_historical_root(tmp_path, row_count=20)),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    artifact = json.loads(output.read_text())
    assert artifact["status"] == "preflight_blocked"
    assert artifact["paid_run_allowed"] is False
    assert not hasattr(client.models, "generate_content")
