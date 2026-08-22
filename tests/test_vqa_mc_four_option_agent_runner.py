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
