from __future__ import annotations

import importlib.util
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from trec_video_2026.vqa_bounded_multipass_counting import CandidateView


SCRIPT = Path("scripts/run_bounded_multipass_counting.py")
SPEC = importlib.util.spec_from_file_location("run_bounded", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_build_temporal_parts_bind_every_window_and_sampling_parameter() -> None:
    parts = runner.build_temporal_parts(
        file_uri="files/v1",
        mime_type="video/webm",
        views=(
            CandidateView("W0", "window", start_seconds=0.0, end_seconds=2.5),
            CandidateView("W1", "window", start_seconds=1.5, end_seconds=4.0),
        ),
        fps=7.5,
    )
    assert len(parts) == 2
    assert [part.video_metadata.fps for part in parts] == [7.5, 7.5]
    assert [part.video_metadata.start_offset for part in parts] == ["0.000s", "1.500s"]
    assert [part.video_metadata.end_offset for part in parts] == ["2.500s", "4.000s"]
    assert all(part.file_data.file_uri == "files/v1" for part in parts)


def test_build_stage_request_binds_media_schema_and_not_labels() -> None:
    request = runner.build_stage_request(
        stage="count_spec",
        arm="fixed",
        video_id="v1",
        model=runner.MODEL,
        prompt="question-only structured task",
        response_schema={"type": "object"},
        media_binding={
            "file_uri": "files/v1",
            "mime_type": "video/webm",
            "delivery_sha256": "a" * 64,
            "upload_complete_sha256": "b" * 64,
            "fps": 4.0,
            "views": [{"view_id": "full", "start_seconds": 0.0, "end_seconds": 4.0}],
        },
        predecessor_sha256=None,
    )
    assert request["model"] == "gemini-3.1-pro-preview"
    assert request["thinking_level"] == "LOW"
    assert request["max_output_tokens"] == 8192
    assert len(request["source_revision"]) == 40
    assert set(request["implementation_sources"]) == {
        "pyproject.toml",
        "scripts/run_bounded_multipass_counting.py",
        "src/trec_video_2026/vqa_bounded_multipass_counting.py",
        "src/trec_video_2026/vqa_counting_evidence.py",
        "src/trec_video_2026/vqa_gate.py",
        "src/trec_video_2026/vqa_submission.py",
        "uv.lock",
    }
    assert request["media_binding"]["file_uri"] == "files/v1"
    rendered = json.dumps(request).lower()
    assert "reference" not in rendered
    assert "baseline" not in rendered
    assert "expected" not in rendered


def test_exclusive_claim_allows_one_concurrent_owner(tmp_path: Path) -> None:
    identity = {"stage": "evidence", "request_sha256": "a" * 64}

    def reserve() -> str:
        try:
            runner.reserve_claim(tmp_path / "stage", identity)
        except FileExistsError:
            return "refused"
        return "owned"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = sorted(pool.map(lambda _: reserve(), range(2)))
    assert outcomes == ["owned", "refused"]


def test_budget_refuses_reservation_that_can_cross_hard_ceiling(tmp_path: Path) -> None:
    runner.reserve_budget(
        budget_root=tmp_path,
        reservation_id="call-1",
        reserved_usd=2.5,
        hard_ceiling_usd=3.0,
    )
    with pytest.raises(RuntimeError, match="budget ceiling"):
        runner.reserve_budget(
            budget_root=tmp_path,
            reservation_id="call-2",
            reserved_usd=0.6,
            hard_ceiling_usd=3.0,
        )
    ledger = json.loads((tmp_path / "budget-ledger.json").read_text())
    assert set(ledger["reservations"]) == {"call-1"}


def test_settle_budget_records_observed_cost_without_reopening_reservation(
    tmp_path: Path,
) -> None:
    runner.reserve_budget(
        budget_root=tmp_path,
        reservation_id="call-1",
        reserved_usd=2.5,
        hard_ceiling_usd=3.0,
    )
    runner.settle_budget(
        budget_root=tmp_path, reservation_id="call-1", observed_usd=0.4
    )
    with pytest.raises(RuntimeError, match="already settled"):
        runner.settle_budget(
            budget_root=tmp_path, reservation_id="call-1", observed_usd=0.4
        )
    ledger = json.loads((tmp_path / "budget-ledger.json").read_text())
    assert ledger["reservations"]["call-1"]["observed_usd"] == 0.4


def test_write_once_stage_terminal_refuses_mutation(tmp_path: Path) -> None:
    path = tmp_path / "complete.json"
    first = runner.write_self_hashed_once(
        path, {"schema_version": "stage-complete-v1", "outcome": "exact"}
    )
    second = runner.write_self_hashed_once(
        path, {"schema_version": "stage-complete-v1", "outcome": "exact"}
    )
    assert first == second
    with pytest.raises(FileExistsError, match="conflicting"):
        runner.write_self_hashed_once(
            path, {"schema_version": "stage-complete-v1", "outcome": "abstain"}
        )


def test_client_from_env_uses_established_gemini_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}
    monkeypatch.setenv("GEMINI_API_KEY", "private-test-key")
    monkeypatch.setattr(
        runner.genai,
        "Client",
        lambda *, api_key: captured.setdefault("client", SimpleNamespace(key=api_key)),
    )
    client = runner.client_from_env()
    assert client.key == "private-test-key"
    assert captured["client"] is client


class FakeFiles:
    def upload(self, *, file: Path):
        assert file.is_file()
        return SimpleNamespace(
            name="files/v1",
            uri="https://example.invalid/files/v1",
            state=SimpleNamespace(name="ACTIVE"),
            expiration_time="future",
        )


class FakeModels:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = list(payloads)

    def generate_content(self, **_kwargs):
        payload = self.payloads.pop(0)
        return SimpleNamespace(
            text=json.dumps(payload),
            usage_metadata=SimpleNamespace(
                prompt_token_count=100,
                candidates_token_count=50,
                thoughts_token_count=0,
            ),
            candidates=[SimpleNamespace(finish_reason="STOP")],
        )


class FakeClient:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.files = FakeFiles()
        self.models = FakeModels(payloads)


def test_run_fixed_row_freezes_exact_result_from_three_structured_stages(
    tmp_path: Path,
) -> None:
    delivery = tmp_path / "v1.webm"
    delivery.write_bytes(b"bound delivery")
    question = "How many times does the person jump?"
    spec_payload = {
        "schema_version": "bounded-multipass-count-spec-v1",
        "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
        "target": "the person's completed jump",
        "unit": "action_repetition",
        "route": "temporal",
        "scope": {"start_seconds": 0.0, "end_seconds": 4.0},
        "inclusions": ["completed jump cycles"],
        "exclusions": ["preparatory crouches"],
        "identity_policy": "observable_episode",
        "candidate_views": [
            {
                "view_id": "W0",
                "kind": "window",
                "start_seconds": 0.0,
                "end_seconds": 4.0,
            }
        ],
        "uncertainty": "none",
    }
    evidence_payload = {
        "schema_version": "bounded-multipass-evidence-v1",
        "reviewed_view_ids": ["W0"],
        "records": [
            {
                "record_id": "E0",
                "view_ids": ["W0"],
                "start_seconds": 0.5,
                "peak_seconds": 0.8,
                "end_seconds": 1.1,
                "target_supported": True,
                "ambiguous": False,
                "descriptor": "first jump",
            },
            {
                "record_id": "E1",
                "view_ids": ["W0"],
                "start_seconds": 2.5,
                "peak_seconds": 2.8,
                "end_seconds": 3.1,
                "target_supported": True,
                "ambiguous": False,
                "descriptor": "second jump",
            },
        ],
    }
    coverage_payload = {
        "schema_version": "bounded-multipass-coverage-v1",
        "complete": True,
        "reviewed_view_ids": ["W0"],
        "accepted_record_ids": ["E0", "E1"],
        "rejected_record_ids": [],
        "missing_view_ids": [],
        "disputes": [],
        "refinement_request": None,
        "notes": "complete",
    }
    row = {
        "video_id": "v1",
        "question": question,
        "duration_seconds": 4.0,
        "delivery_path": str(delivery),
        "delivery_sha256": hashlib.sha256(b"bound delivery").hexdigest(),
        "delivery_mime_type": "video/webm",
    }
    terminal = runner.run_fixed_row(
        client=FakeClient([spec_payload, evidence_payload, coverage_payload]),
        row=row,
        root=tmp_path / "run",
        hard_ceiling_usd=3.0,
    )
    assert terminal["outcome"] == "complete"
    result = json.loads((tmp_path / "run/rows/v1/fixed/result.json").read_text())
    assert result["outcome"] == "exact"
    assert result["value"] == 2
    assert set(terminal["stage_complete_sha256"]) == {
        "count_spec",
        "evidence",
        "coverage",
    }


def test_run_refinement_row_adds_one_targeted_window_then_freezes(
    tmp_path: Path,
) -> None:
    delivery = tmp_path / "v1.webm"
    delivery.write_bytes(b"bound delivery")
    question = "How many times does the person jump?"
    spec_payload = {
        "schema_version": "bounded-multipass-count-spec-v1",
        "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
        "target": "the person's completed jump",
        "unit": "action_repetition",
        "route": "temporal",
        "scope": {"start_seconds": 0.0, "end_seconds": 4.0},
        "inclusions": ["completed jump cycles"],
        "exclusions": ["preparatory crouches"],
        "identity_policy": "observable_episode",
        "candidate_views": [
            {
                "view_id": "W0",
                "kind": "window",
                "start_seconds": 0.0,
                "end_seconds": 4.0,
            }
        ],
        "uncertainty": "possible second jump",
    }
    evidence_payload = {
        "schema_version": "bounded-multipass-evidence-v1",
        "reviewed_view_ids": ["W0"],
        "records": [
            {
                "record_id": "E0",
                "view_ids": ["W0"],
                "start_seconds": 0.5,
                "peak_seconds": 0.8,
                "end_seconds": 1.1,
                "target_supported": True,
                "ambiguous": False,
                "descriptor": "first jump",
            }
        ],
    }
    incomplete_coverage = {
        "schema_version": "bounded-multipass-coverage-v1",
        "complete": False,
        "reviewed_view_ids": ["W0"],
        "accepted_record_ids": ["E0"],
        "rejected_record_ids": [],
        "missing_view_ids": [],
        "disputes": [{"record_id": "E0", "reason": "possible later omission"}],
        "refinement_request": {
            "kind": "window",
            "start_seconds": 2.0,
            "end_seconds": 3.5,
            "view_id": None,
            "reason": "inspect later motion",
        },
        "notes": "one targeted window needed",
    }
    refinement_evidence = {
        "schema_version": "bounded-multipass-evidence-v1",
        "reviewed_view_ids": ["R0"],
        "records": [
            {
                "record_id": "E1",
                "view_ids": ["R0"],
                "start_seconds": 2.5,
                "peak_seconds": 2.8,
                "end_seconds": 3.1,
                "target_supported": True,
                "ambiguous": False,
                "descriptor": "second jump",
            }
        ],
    }
    final_coverage = {
        "schema_version": "bounded-multipass-coverage-v1",
        "complete": True,
        "reviewed_view_ids": ["W0"],
        "accepted_record_ids": ["E0", "E1"],
        "rejected_record_ids": [],
        "missing_view_ids": [],
        "disputes": [],
        "refinement_request": None,
        "notes": "resolved",
    }
    row = {
        "video_id": "v1",
        "question": question,
        "duration_seconds": 4.0,
        "delivery_path": str(delivery),
        "delivery_sha256": hashlib.sha256(b"bound delivery").hexdigest(),
        "delivery_mime_type": "video/webm",
    }
    root = tmp_path / "run"
    client = FakeClient(
        [
            spec_payload,
            evidence_payload,
            incomplete_coverage,
            refinement_evidence,
            final_coverage,
        ]
    )
    fixed = runner.run_fixed_row(
        client=client, row=row, root=root, hard_ceiling_usd=3.0
    )
    assert fixed["outcome"] == "complete"
    assert (
        json.loads((root / "rows/v1/fixed/result.json").read_text())["outcome"]
        == "abstain"
    )

    refined = runner.run_refinement_row(
        client=client, row=row, root=root, hard_ceiling_usd=3.0
    )
    assert refined["outcome"] == "complete"
    result = json.loads((root / "rows/v1/refinement/result.json").read_text())
    assert result["outcome"] == "exact"
    assert result["value"] == 2
    assert set(refined["stage_complete_sha256"]) == {
        "refinement_evidence",
        "final_coverage",
    }
