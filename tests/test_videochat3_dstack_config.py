from pathlib import Path
import re
import subprocess

import yaml


RETIRED_A2_CANARY = Path("dstack/videochat3-runner-canary-20260730-a2.retired.yml")
RETIRED_A3_CANARY = Path("dstack/videochat3-runner-canary-20260730-a3.retired.yml")
RUNNABLE_A3_CANARY = Path("dstack/videochat3-runner-canary-20260730-a3.dstack.yml")
RUNNABLE_A4_CANARY = Path("dstack/videochat3-runner-canary-20260730-a4.dstack.yml")
RETIRED_A4_CANARY = Path("dstack/videochat3-runner-canary-20260730-a4.retired.yml")
RUNNABLE_A5_CANARY = Path("dstack/videochat3-runner-canary-20260730-a5.dstack.yml")
RETIRED_A5_CANARY = Path("dstack/videochat3-runner-canary-20260730-a5.retired.yml")
DEBUG_ENV = Path("dstack/videochat3-debug-20260731-a1.dstack.yml")
A6_CANARY = Path("dstack/videochat3-runner-canary-20260731-a6.dstack.yml")
RETIRED_DEBUG_ENV = Path("dstack/videochat3-debug-20260731-a1.retired.yml")
RETIRED_A6_CANARY = Path("dstack/videochat3-runner-canary-20260731-a6.retired.yml")
OLD_RUNNABLE_CANARY = Path("dstack/videochat3-runner-canary.dstack.yml")
PILOT = Path("dstack/videochat3-pilot.dstack.yml")
GATE_20260801 = Path("dstack/videochat3-gate-20260801-a1.dstack.yml")
PILOT_20260801 = Path("dstack/videochat3-pilot-20260801-a1.dstack.yml")
RETIRED_GATE_20260801 = Path(
    "dstack/videochat3-gate-20260801-a1.retired.yml"
)
RETIRED_PILOT_20260801 = Path(
    "dstack/videochat3-pilot-20260801-a1.retired.yml"
)
RETIRED_PILOT = Path("dstack/videochat3-pilot-20260730-a1.retired.yml")
WRAPPER = Path("scripts/run-videochat3-pilot-dstack.sh")
DEBUG_PREP = Path("scripts/prepare-videochat3-debug-dstack.sh")
REQUIREMENTS = Path("configs/vqa/videochat3_cuda_requirements.txt")
RUNNABLE_RUNTIME_CERT_A3 = Path(
    "dstack/videochat3-runtime-cert-20260731-a3.dstack.yml"
)
RUNNABLE_RUNTIME_CERT_A4 = Path(
    "dstack/videochat3-runtime-cert-20260731-a4.dstack.yml"
)
RETIRED_RUNTIME_CERT_A4 = Path(
    "dstack/videochat3-runtime-cert-20260731-a4.retired.yml"
)
RETIRED_RUNTIME_CERT_A3 = Path(
    "dstack/videochat3-runtime-cert-20260731-a3.retired.yml"
)
RETIRED_RUNTIME_CERT_A1 = Path(
    "dstack/videochat3-runtime-cert-20260731-a1.retired.yml"
)
RETIRED_RUNTIME_CERT_A2 = Path(
    "dstack/videochat3-runtime-cert-20260731-a2.retired.yml"
)
RUNTIME_CERT_PREP = Path("scripts/prepare-videochat3-runtime-cert-dstack.sh")
RUNTIME_CERT_RUNNER = Path("scripts/run-videochat3-runtime-cert-dstack.sh")
VIDEOCHAT3_EVIDENCE = Path("plans/videochat3-4b-pilot.md")
RETIRED_TIMESTAMP_DEV_A1 = Path(
    "dstack/videochat3-timestamp-dev-20260801-a1.retired.yml"
)
RETIRED_TIMESTAMP_DEV_A2 = Path(
    "dstack/videochat3-timestamp-dev-20260801-a2.retired.yml"
)
A3_TIMESTAMP_DEV = Path("dstack/videochat3-timestamp-dev-20260801-a3.dstack.yml")
A3_TIMESTAMP_DEV_RETIRED = Path(
    "dstack/videochat3-timestamp-dev-20260801-a3.retired.yml"
)
A3_TIMESTAMP_DEV_RUNNER = Path(
    "scripts/run-videochat3-timestamp-dev-dstack.sh"
)
VIDEOCHAT3_24FRAME_DEV = Path(
    "dstack/videochat3-24frame-dev-20260801-a1.dstack.yml"
)
VIDEOCHAT3_24FRAME_RUNNER = Path(
    "scripts/run-videochat3-24frame-dev-dstack.sh"
)
VIDEOCHAT3_24FRAME_RETIRED = Path(
    "dstack/videochat3-24frame-dev-20260801-a1.retired.yml"
)
CONSUMED_CLOUD_SCRIPTS = (
    Path("scripts/prepare-videochat3-debug-dstack.sh"),
    Path("scripts/prepare-videochat3-runtime-cert-dstack.sh"),
    Path("scripts/run-videochat3-input-flow-canary-dstack.sh"),
    Path("scripts/run-videochat3-pilot-dstack.sh"),
    Path("scripts/run-videochat3-runtime-cert-dstack.sh"),
)

IMAGE = (
    "docker.io/pytorch/pytorch@"
    "sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca"
)
GATE_20260801_SHA256 = (
    "1f476fb3a0f7f5cac3bd3a790348bc6a3eed91c16311e79909f48f8c9760a772"
)
PILOT_20260801_SHA256 = (
    "8b5cf944553ac000795b837090799f07041c8b86cc3ed9a997dc52a9ed4941c4"
)
PILOT_24FRAME_SHA256 = (
    "d13a1135184d92988af28df0fcda47a6d0debef268a8861cfd4ced3898a9518b"
)
GATE_20260801_IDS = ("BDKkVYmWqdM", "O60MtMuSpf8", "Fk_j6re_Uiw")
PILOT_20260801_IDS = (
    "CH8Dp7EgW4I",
    "EtNOkUJd5w8",
    "Gp4LVTn3YbY",
    "KgOvCPOX-Ms",
    "EMI_anpHII4",
    "LusRurRrmf0",
)
PRIVATE_STAGE_LOCATOR = re.compile(
    r"videochat3-(?:timestamp|24frame)-"
    r"(?:input|canary(?:-recheck)?|gate|pilot)-20\d{6}-a\d"
)


def test_completed_24frame_dev_is_retired_and_cannot_be_resubmitted() -> None:
    assert not VIDEOCHAT3_24FRAME_DEV.exists()
    assert not VIDEOCHAT3_24FRAME_RUNNER.exists()
    record = yaml.safe_load(VIDEOCHAT3_24FRAME_RETIRED.read_text(encoding="utf-8"))

    assert record["run_name"] == "videochat3-24frame-dev-20260801-a1"
    assert record["run_id"] == "d5cc3983-76e5-4468-a7b2-8c29beea27c9"
    assert record["status"] == "retired-consumed-valid-negative"
    assert record["submission_allowed"] is False
    assert record["submitted"] is True
    assert record["source_config_sha256"] == (
        "0f19ef1275250fc53f970896578f7387990c50e165f11c67bc14dc9c035cf413"
    )
    assert record["stage_runner_sha256"] == (
        "fa38c0a4b0efba41dd226b0551681dce2186256383f44cb2e316f6fbd70e9f6d"
    )
    assert record["input"]["pilot_manifest_sha256"] == PILOT_24FRAME_SHA256
    assert record["completion"] == {
        "synthetic_canary": "1/1",
        "matched_gate_videochat3": "2/2",
        "matched_gate_qwen": "2/2",
        "pilot_videochat3": "6/6",
        "pilot_qwen": "6/6",
        "blind_grades": "12/12",
        "disagreement_adjudications": "0/0",
    }
    assert record["quality"]["target_improvements"] == "1/4"
    assert record["quality"]["target_regressions"] == "3/4"
    assert record["quality"]["control_regressions"] == "2/2"
    assert record["quality"]["passed"] is False
    assert record["actual_cost_usd"] == 0.0763
    assert record["cumulative_videochat3_cost_usd"] == 0.7335
    assert record["evidence_validity"] == "Verified"
    assert record["decision_disposition"] == "Not promoted under the tested condition"
    assert record["security_incident"]["credential_rotation_required"] is True
    assert record["artifacts"]["pilot"]["checksum_verified"] is True
    assert record["artifacts"]["grading"]["checksum_verified"] is True

    text = VIDEOCHAT3_24FRAME_RETIRED.read_text(encoding="utf-8")
    assert "hf://" not in text
    for forbidden in (
        "HF_TOKEN",
        "secrets.",
        "\ntype:",
        "\nenv:",
        "\ncommands:",
        "\nfiles:",
    ):
        assert forbidden not in text


def test_all_consumed_videochat3_cloud_scripts_fail_closed() -> None:
    """Catches a historical cloud entrypoint accepting any consumed identity."""

    for script in CONSUMED_CLOUD_SCRIPTS:
        completed = subprocess.run(
            ["bash", str(script)],
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 64, script
        assert completed.stdout == ""
        assert completed.stderr == (
            "retired: VideoChat3 cloud execution is permanently disabled; "
            "no run identity is authorized\n"
        )


def test_tracked_videochat3_provenance_excludes_private_locators_and_row_grades() -> None:
    """Catches private operational coordinates entering publication history."""

    paths = (
        *sorted(Path("dstack").glob("videochat3*.retired.yml")),
        Path("plans/videochat3-4b-pilot.md"),
        Path("docs/project-status/experiment-ledger.qmd"),
        Path("docs/project-status/experiment-ledger.md"),
        Path("docs/project-status/research-models.qmd"),
        Path("docs/project-status/research-models.md"),
        Path("reports/project-status/experiment-ledger.html"),
        Path("reports/project-status/research-models.html"),
        *CONSUMED_CLOUD_SCRIPTS,
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "hf://buckets/" not in text, path
        assert PRIVATE_STAGE_LOCATOR.search(text) is None, path
        for video_id in (*GATE_20260801_IDS, *PILOT_20260801_IDS):
            assert video_id not in text, path

    def locator_keys(value: object) -> set[str]:
        if isinstance(value, dict):
            direct = {
                str(key)
                for key in value
                if str(key).lower()
                in {
                    "artifact_name",
                    "file",
                    "files",
                    "path",
                    "prefix",
                    "stage_names",
                    "stages",
                    "uri",
                }
                or str(key).lower().endswith(
                    ("_uri", "_path", "_prefix", "_file", "_artifact_name", "_stage_names")
                )
            }
            return direct | set().union(
                *(locator_keys(nested) for nested in value.values())
            )
        if isinstance(value, list):
            return set().union(*(locator_keys(nested) for nested in value))
        return set()

    for path in sorted(Path("dstack").glob("videochat3*.retired.yml")):
        record = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "grades" not in str(record.get("blind_grading", {})), path
        assert locator_keys(record) == set(), path


def test_failed_timestamp_dev_a1_is_retired_and_cannot_be_resubmitted() -> None:
    """Catches accidental reuse of the consumed typo-bearing dev identity."""

    assert not Path(
        "dstack/videochat3-timestamp-dev-20260801-a1.dstack.yml"
    ).exists()
    text = RETIRED_TIMESTAMP_DEV_A1.read_text(encoding="utf-8")
    record = yaml.safe_load(text)

    assert record["run_name"] == "videochat3-timestamp-dev-20260801-a1"
    assert record["run_id"] == "65785a53-a647-42a7-9ad0-54a039b12c21"
    assert record["status"] == "retired-consumed-runtime-failure"
    assert record["submission_allowed"] is False
    assert record["submitted"] is True
    assert record["source_config_sha256"] == (
        "4dfee796d186c5300f698b19644b4c9fa1eb10ccdf761f75ff1f5efb28849d5a"
    )
    assert record["actual_cost_usd"] == 0.0369
    assert record["cumulative_videochat3_cost_usd"] == 0.4915
    assert record["completion"] == {
        "dev_initialization": "0/1",
        "synthetic_canary": "0/1",
        "gate": "0/3",
        "videochat3_pilot": "0/6",
        "qwen_pilot": "0/6",
    }
    assert record["remote_artifact_state"] == "not-created"
    assert "missing_hex_character" in record["root_cause"]
    assert "type: dev-environment" not in text
    assert "init:" not in text
    assert "HF_TOKEN" not in text
    assert "secrets." not in text


def test_completed_timestamp_dev_a2_is_retired_after_persistent_stop() -> None:
    """Catches a third submission after the consumed corrective dev identity."""

    for runnable in (
        Path("dstack/videochat3-timestamp-dev-20260801-a1.dstack.yml"),
        Path("dstack/videochat3-timestamp-dev-20260801-a2.dstack.yml"),
    ):
        assert not runnable.exists()

    text = RETIRED_TIMESTAMP_DEV_A2.read_text(encoding="utf-8")
    record = yaml.safe_load(text)
    assert record["run_name"] == "videochat3-timestamp-dev-20260801-a2"
    assert record["run_id"] == "da69c2a2-f102-4f98-b98f-dd0fa1b12e4b"
    assert record["status"] == "retired-consumed-premature-completion"
    assert record["submission_allowed"] is False
    assert record["submitted"] is True
    assert record["source_config_sha256"] == (
        "0c408d64dfa3327d0a3bb6e4176c8e47c0d9a2aeb13d5b7f865f0bafc6ce57d9"
    )
    assert record["actual_cost_usd"] == 0.0237
    assert record["cumulative_videochat3_cost_usd"] == 0.5152
    assert record["completion"] == {
        "dev_initialization": "1/1",
        "synthetic_canary": "0/1",
        "gate": "0/3",
        "videochat3_pilot": "0/6",
        "qwen_pilot": "0/6",
    }
    assert record["executed_dependency_snapshot"][
        "requirements_env_sha256"
    ] == (
        "9d4cb01013132aac5304911ad7844b03fb3ed3b685fa6248da8ab969a502dde6"
    )
    assert record["planned_stage_count"] == 3
    assert record["remote_artifact_state"] == "not-created"
    assert record["evidence_validity"] == "Incomplete"
    assert record["decision_disposition"] == "Persistent technical stop"
    assert record["root_cause"]["classification"] == "confirmed_config_defect"
    assert "type: dev-environment" not in text
    assert "init:" not in text
    assert "HF_TOKEN" not in text
    assert "secrets." not in text


def test_a3_timestamp_dev_is_retired_with_verified_negative_evidence() -> None:
    """Prevents reuse and locks the condition-bound comparison result."""

    assert not A3_TIMESTAMP_DEV.exists()
    assert not A3_TIMESTAMP_DEV_RUNNER.exists()
    text = A3_TIMESTAMP_DEV_RETIRED.read_text(encoding="utf-8")
    record = yaml.safe_load(text)

    assert record["run_name"] == "videochat3-timestamp-dev-20260801-a3"
    assert record["run_id"] == "9bd81210-af7e-4b1a-a9bd-5fa20dae0302"
    assert record["status"] == "retired-consumed-valid-negative"
    assert record["submission_allowed"] is False
    assert record["submitted_snapshot"]["source_config_sha256"] == (
        "83cbbe023c05123e39a0a54787ce4f68612a02093ee978d7934ecc35430020e8"
    )
    assert record["effective_snapshot"]["source_snapshot_sha256"] == (
        "ac64eb2976601014680a848f21ab0080bd6a0f984c8a969aa6a1394d2f9e1c18"
    )
    assert record["execution"]["actual_cost_usd"] == 0.142
    assert record["cumulative_videochat3_cost_usd"] == 0.6572
    assert record["completion"] == {
        "dev_initialization": "1/1",
        "synthetic_canary_recheck": "1/1",
        "gate": "3/3",
        "videochat3_pilot": "6/6",
        "qwen_pilot": "6/6",
        "blind_graders": "2/2",
        "disagreement_adjudication": "1/1",
    }
    assert record["quality"] == {
        "target_improvements": "1/4",
        "target_ties": "2/4",
        "target_regressions": "1/4",
        "control_regressions": "2/2",
        "passed": False,
    }
    assert record["evidence_validity"] == "Verified"
    assert record["decision_disposition"] == (
        "Not promoted under the tested condition"
    )
    assert record["artifacts"]["matched_pilot"][
        "listed_downloaded_and_checksum_verified"
    ] is True
    assert record["artifacts"]["matched_pilot"]["predictions_sha256"] == (
        "6705f5e0c65e1344317edf89400356bf7a485da36bd781b1d0db4427d03dfb59"
    )
    assert "type: dev-environment" not in text
    assert "init:" not in text
    assert "HF_TOKEN" not in text
    assert "secrets." not in text


def test_consumed_20260801_gate_and_pilot_are_retired_with_verified_provenance(
) -> None:
    assert not GATE_20260801.exists()
    assert not PILOT_20260801.exists()

    gate = yaml.safe_load(RETIRED_GATE_20260801.read_text(encoding="utf-8"))
    pilot = yaml.safe_load(RETIRED_PILOT_20260801.read_text(encoding="utf-8"))

    assert gate["run_name"] == "videochat3-gate-20260801-a1"
    assert gate["run_id"] == "a14b9052-0a3d-418b-9f1a-b1212b47a154"
    assert gate["source_config_sha256"] == (
        "37b2e1b844946bff1aeb13ee3eeee9c1d7cbb96991486ceed520d0adf9e84ff6"
    )
    assert gate["actual_cost_usd"] == 0.0292
    assert gate["completion"] == "3/3"
    assert gate["evidence_validity"] == "Invalid"
    assert gate["decision_disposition"] == "Open hypothesis"
    assert gate["artifact"]["sha256s_sha256"] == (
        "273bc721ad0d4f5c3f8fa54594f7bb8806bc066af6d013d2baa9673080e95206"
    )

    assert pilot["run_name"] == "videochat3-pilot-20260801-a1"
    assert pilot["run_id"] == "b7eb62b3-67ab-4a89-ab27-ef95e3136070"
    assert pilot["source_config_sha256"] == (
        "702e2a870cd5ad2f859e912654c58e41289c5a3fa7b7bd30bc16b4c0b8f01804"
    )
    assert pilot["actual_cost_usd"] == 0.0459
    assert pilot["completion"] == {"videochat3": "6/6", "qwen": "6/6"}
    assert pilot["quality_passed"] is False
    assert pilot["improved_target_count"] == "1/4"
    assert pilot["regressed_control_count"] == "2/2"
    assert pilot["evidence_validity"] == "Invalid"
    assert pilot["decision_disposition"] == "Open hypothesis"
    assert pilot["artifact"]["sha256s_sha256"] == (
        "ef26e5671d7fcf558edb56ea4b020e459e57280508c4eb94be66d36ccb2a24a1"
    )
    assert pilot["artifact"]["grading_sha256s_sha256"] == (
        "fbb7190485831e7ba870923390b11e3c3de38396e6f4cf66c0710b7cf371c292"
    )
    assert pilot["artifact"]["scored_comparison_sha256"] == (
        "42ce8737cb0c440455f70cd27f21b9083f3a99f1a6d2b93e822f65ddcef61529"
    )
    assert pilot["blind_grading"]["context_isolated_third_adjudicator"] is True
    assert (
        pilot["blind_grading"][
            "superseded_context_inherited_adjudication_preserved"
        ]
        is True
    )
    assert pilot["cumulative_videochat3_cost_usd"] == 0.4546


def test_persistent_input_flow_preview_is_retired_and_cannot_be_submitted() -> None:
    runnable = Path(
        "dstack/videochat3-input-flow-canary-20260801-a1.dstack.yml"
    )
    retired = Path(
        "dstack/videochat3-input-flow-canary-20260801-a1.retired.yml"
    )

    assert not runnable.exists()
    record = yaml.safe_load(retired.read_text(encoding="utf-8"))
    assert record["run_name"] == "videochat3-input-flow-canary-20260801-a1"
    assert record["status"] == "retired-unsubmitted-persistent-preview-error"
    assert record["submission_allowed"] is False
    assert record["submitted"] is False
    assert record["source_config_sha256"] == (
        "d13aa4c9aafa833c96f50787996d6f3c5923479c2d983dfacb99ee41c012cf7f"
    )
    assert record["actual_cost_usd"] == 0.0
    assert record["completion"] == "0/1"
    assert record["evidence_validity"] == "Incomplete"
    assert record["decision_disposition"] == "Diagnostic signal"
    assert record["previews"]["initial"]["sha256"] == (
        "750825a0c44e142249d8194e6b0a4cbe0144fb654f2fbe25175f00bd1dde44e1"
    )
    assert record["previews"]["corrected"]["sha256"] == (
        "95f938b5cc2e879e910d1ee54c1682060174a34453c3d7c8c67e976639a25162"
    )


def test_retired_configs_are_nonrunnable_and_secret_free() -> None:
    paths = (
        RETIRED_GATE_20260801,
        RETIRED_PILOT_20260801,
        Path("dstack/videochat3-input-flow-canary-20260801-a1.retired.yml"),
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        record = yaml.safe_load(text)
        assert record["submission_allowed"] is False
        assert "type: task" not in text
        assert "commands:" not in text
        assert "HF_TOKEN" not in text
        assert "secrets." not in text


def test_retired_gate_provenance_preserves_the_declared_input_boundary() -> None:
    record = yaml.safe_load(RETIRED_GATE_20260801.read_text(encoding="utf-8"))
    snapshot = record["submitted_snapshot"]

    assert snapshot["image"] == IMAGE
    assert snapshot["gate_requests_sha256"] == GATE_20260801_SHA256
    assert snapshot["ordered_row_set_sha256"] == GATE_20260801_SHA256
    assert "qwen_model_id" not in snapshot


def test_retired_pilot_provenance_preserves_the_matched_input_boundary() -> None:
    record = yaml.safe_load(RETIRED_PILOT_20260801.read_text(encoding="utf-8"))
    snapshot = record["submitted_snapshot"]

    assert snapshot["image"] == IMAGE
    assert snapshot["pilot_requests_sha256"] == PILOT_20260801_SHA256
    assert snapshot["ordered_row_set_sha256"] == PILOT_20260801_SHA256
    assert snapshot["qwen_model_id"] == "Qwen/Qwen3.5-4B"
    assert snapshot["qwen_revision"] == (
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    )


def test_consumed_a2_canary_is_retired_and_not_a_runnable_dstack_config() -> None:
    text = RETIRED_A2_CANARY.read_text(encoding="utf-8")

    assert not OLD_RUNNABLE_CANARY.exists()
    assert "run_name: videochat3-runner-canary-20260730-a2" in text
    assert f"image: {IMAGE}" in text
    assert "status: retired-consumed-run" in text
    assert "submission_allowed: false" in text
    assert "type: task" not in text
    assert "commands:" not in text
    assert "HF_TOKEN" not in text


def test_consumed_a3_canary_is_retired_and_cannot_be_resubmitted() -> None:
    text = RETIRED_A3_CANARY.read_text(encoding="utf-8")

    assert not RUNNABLE_A3_CANARY.exists()
    assert "run_name: videochat3-runner-canary-20260730-a3" in text
    assert "run_id: 90fb6f5d-a909-4cfd-814d-a004d69392c7" in text
    assert "status: retired-consumed-run" in text
    assert "submission_allowed: false" in text
    assert f"image: {IMAGE}" in text
    assert "gpu_selector: A40:1:24GB.." in text
    assert "max_price_usd_per_hour: 1.00" in text
    assert "max_duration: 30m" in text
    assert "model_id: MCG-NJU/VideoChat3-4B" in text
    assert "model_revision: 37fa901ec5913f84bc31108ebc1e60ad1903634c" in text
    assert (
        "requirements_sha256: 959a2aac3e8847aff94c8f302cd135d5c76b3e1e3dcd0c71c1921f5aa58569fd"
        in text
    )
    assert "actual_cost_usd: 0.0283" in text
    assert "failure_phase: bootstrap-runtime" in text
    assert "click_declared: 8.3.1" in text
    assert "huggingface_hub: 1.24.0" in text
    assert "requires click>=8.4.2" in text
    assert "type: task" not in text
    assert "commands:" not in text
    assert "HF_TOKEN" not in text
    assert "secrets." not in text


def test_unsubmitted_a4_canary_is_retired_after_persistent_capacity_stop() -> None:
    text = RETIRED_A4_CANARY.read_text(encoding="utf-8")

    assert not RUNNABLE_A4_CANARY.exists()
    assert "run_name: videochat3-runner-canary-20260730-a4" in text
    assert "status: retired-unsubmitted-capacity-stop" in text
    assert "submission_allowed: false" in text
    assert "submitted: false" in text
    assert (
        "source_config_sha256: 64e54e924133a2ca94c058320e285e8d8dd5f9e58946981eac7faf37b0f5d3f4"
        in text
    )
    assert f"image: {IMAGE}" in text
    assert "gpu_selector: A40:1:24GB.." in text
    assert "spot_policy: on-demand" in text
    assert "max_price_usd_per_hour: 1.00" in text
    assert "max_duration: 30m" in text
    assert "preview_attempts: 2" in text
    assert text.count("preview_result: no-matching-instance-offers") == 2
    assert "compatible_scalar_offers_observed: 2" in text
    assert "incremental_cost_usd: 0.0000" in text
    assert "cumulative_videochat3_cost_usd: 0.0514" in text
    assert "type: task" not in text
    assert "commands:" not in text
    assert "HF_TOKEN" not in text
    assert "secrets." not in text


def test_consumed_a5_canary_is_retired_after_verified_workload_failure() -> None:
    text = RETIRED_A5_CANARY.read_text(encoding="utf-8")

    assert not RUNNABLE_A5_CANARY.exists()
    assert "run_name: videochat3-runner-canary-20260730-a5" in text
    assert "run_id: 02b9d381-15a4-4092-9728-5329f759513c" in text
    assert "status: retired-consumed-run" in text
    assert "submission_allowed: false" in text
    assert f"image: {IMAGE}" in text
    assert "gpu_selector: A40:1:24GB.." in text
    assert (
        "source_config_sha256: 8ade2881e16a70722beefe6b2734a35339bdfffecf9489c6a67a98e1dcf7f64f"
        in text
    )
    assert "model_revision: 37fa901ec5913f84bc31108ebc1e60ad1903634c" in text
    assert (
        "requirements_sha256: 9d4cb01013132aac5304911ad7844b03fb3ed3b685fa6248da8ab969a502dde6"
        in text
    )
    assert "max_price_usd_per_hour: 1.00" in text
    assert "max_duration: 30m" in text
    assert "actual_cost_usd: 0.0283" in text
    assert "cumulative_videochat3_cost_usd: 0.0797" in text
    assert "failure_phase: workload" in text
    assert "metadata_object: ExactTimestampMetadata" in text
    assert "accepted_metadata_types: VideoMetadata-or-dict" in text
    assert (
        "bundle_manifest_sha256: 4a1765a9ce2414c2186660ad4499969b7ec9155774ee574f99206a9b626fc3b7"
        in text
    )
    assert "type: task" not in text
    assert "commands:" not in text
    assert "HF_TOKEN" not in text
    assert "secrets." not in text


def test_interactive_debug_environment_is_retired_after_persistent_preview_stop() -> (
    None
):
    config = yaml.safe_load(RETIRED_DEBUG_ENV.read_text(encoding="utf-8"))

    assert not DEBUG_ENV.exists()
    assert config["status"] == "retired-unsubmitted-control-plane-stop"
    assert config["submission_allowed"] is False
    assert config["submitted"] is False
    assert config["run_name"] == "videochat3-debug-20260731-a1"
    assert config["source_config_sha256"] == (
        "c76304e7997b8c93fc021383dcdc44481a0d57c31688fb469de10655c9d3ad5e"
    )
    assert config["debug_prep_script_sha256"] == (
        "ddde9da05ac8eb8481ff33383d7345570347c3f6bb88cfcb1784b047618e8558"
    )
    assert config["image"] == IMAGE
    assert config["model_revision"] == ("37fa901ec5913f84bc31108ebc1e60ad1903634c")
    assert config["preview_attempts"] == 2
    assert config["preview_result"] == "server-connection-failed"
    assert config["preview_sha256"] == (
        "8ec63b7b920423b7a978cd8cdb26b045bd646f13c4411599eef78a97fe9a48ad"
    )
    assert config["actual_cost_usd"] == 0.0
    assert "type" not in config
    assert "env" not in config
    assert "init" not in config
    assert "files" not in config


def test_a6_canary_is_retired_unpreviewed_when_interactive_gate_did_not_pass() -> None:
    config = yaml.safe_load(RETIRED_A6_CANARY.read_text(encoding="utf-8"))

    assert not A6_CANARY.exists()
    assert config["status"] == "retired-unsubmitted-prerequisite-stop"
    assert config["submission_allowed"] is False
    assert config["submitted"] is False
    assert config["previewed"] is False
    assert config["run_name"] == "videochat3-runner-canary-20260731-a6"
    assert config["source_config_sha256"] == (
        "e69aba8632801fd075f992b65bba56b73756ece7e886dcdc6209f7e0c4176b52"
    )
    assert config["blocked_by"] == "videochat3-debug-20260731-a1"
    assert config["actual_cost_usd"] == 0.0
    assert "type" not in config
    assert "env" not in config
    assert "commands" not in config
    assert "files" not in config


def test_unsubmitted_pilot_is_retired_after_runtime_prerequisite_stop() -> None:
    """Catches the private pilot becoming runnable after certification stopped."""

    assert not PILOT.exists()
    retired = yaml.safe_load(RETIRED_PILOT.read_text(encoding="utf-8"))

    assert retired["status"] == "retired-unsubmitted-prerequisite-stop"
    assert retired["submission_allowed"] is False
    assert retired["submitted"] is False
    assert retired["run_name"] == "videochat3-pilot-20260730-a1"
    assert retired["blocked_by"] == "videochat3-runtime-cert-20260731-a2"
    assert retired["source_config_sha256"] == (
        "b383f82eeaaad7f63c3c1eb2abb92ba92ce454df3e0f8831f30188279e9e078c"
    )
    assert retired["image"] == IMAGE
    assert retired["treatment_revision"] == (
        "37fa901ec5913f84bc31108ebc1e60ad1903634c"
    )
    assert retired["control_revision"] == (
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    )
    assert retired["gate_requests_sha256"] == (
        "7160e3c4e3bd7b2641004b4bc1331841828b4d1b2e9e89b0a0dee718512c6ebf"
    )
    assert retired["pilot_requests_sha256"] == (
        "3ab0fc1dc36373914b42c44efdb8b305cb0a1536706995377e5c4513467ce403"
    )
    assert retired["actual_cost_usd"] == 0.0
    for forbidden in ("type", "env", "init", "files", "commands"):
        assert forbidden not in retired
    text = RETIRED_PILOT.read_text(encoding="utf-8")
    assert "HF_TOKEN" not in text
    assert "secrets." not in text
    assert "data/" not in text


def test_reauthorized_plan_stays_within_total_budget() -> None:
    a2_actual_cost = 0.0231
    a3_actual_cost = 0.0283
    a5_actual_cost = 0.0283
    runtime_cert_a1_actual_cost = 0.0304
    runtime_cert_a2_actual_cost = 0.0269
    cumulative_actual_cost = sum(
        (
            a2_actual_cost,
            a3_actual_cost,
            a5_actual_cost,
            runtime_cert_a1_actual_cost,
            runtime_cert_a2_actual_cost,
        )
    )

    assert cumulative_actual_cost == 0.137
    assert cumulative_actual_cost <= 10.00


def test_videochat3_cuda_requirements_remain_exactly_pinned() -> None:
    requirements = REQUIREMENTS.read_text(encoding="utf-8")

    assert (
        "transformers @ https://github.com/huggingface/transformers/archive/"
        in requirements
    )
    assert "cbf4d720ec734edb77d25452b2790c7e4be2f8d7" in requirements
    assert "accelerate==1.12.0" in requirements
    assert "huggingface-hub==1.24.0" in requirements
    assert "click==8.4.2" in requirements


def test_consumed_a4_dev_environment_is_retired_and_cannot_be_resubmitted() -> None:
    assert not RUNNABLE_RUNTIME_CERT_A4.exists()
    retired = yaml.safe_load(RETIRED_RUNTIME_CERT_A4.read_text(encoding="utf-8"))

    assert retired["status"] == "retired-consumed-runtime-result"
    assert retired["submission_allowed"] is False
    assert retired["run_name"] == "videochat3-runtime-cert-20260731-a4"
    assert retired["run_id"] == "8dbb7b72-4332-4690-b451-c67879268a0d"
    assert retired["submitted_snapshot"]["git_commit"] == (
        "486f18681b9ceb828ea80dfba37afae5edbc76f1"
    )
    assert retired["submitted_snapshot"]["source_config_sha256"] == (
        "a28c35685ad9ba0a2021131bf5090bd432f3103f4a7eccd09d4e1abbbebc3c1a"
    )
    assert retired["submitted_snapshot"]["image"] == IMAGE
    assert retired["observed_execution"]["output_head_tie"] == {
        "was_already_tied": False,
        "is_tied": True,
        "shape": [151936, 2560],
        "dtype": "bfloat16",
        "device": "cuda:0",
    }
    assert retired["observed_execution"]["output_disposition"] == (
        "coherent short sentence on synthetic input"
    )
    assert retired["actual_cost_usd"] == 0.026
    assert retired["cumulative_videochat3_cost_usd"] == 0.3795
    assert retired["completion"] == "1/1"
    assert retired["evidence_validity"] == "Diagnostic"
    assert retired["decision_disposition"] == "Diagnostic signal"
    assert retired["artifact"]["checksum_verified"] is True
    assert retired["artifact"]["prediction_sha256"] == (
        "df7b04e33a9eb7435c0614295fddb34ef90c9c67b287b0f1f8747250201c8baa"
    )
    assert retired["artifact"]["certificate_sha256"] == (
        "f558eb3c48f5ec428e271bb62b6708f2abf15a347dbd5b5178a63add920a1fe6"
    )
    for forbidden in ("type", "env", "init", "files", "commands"):
        assert forbidden not in retired
    text = RETIRED_RUNTIME_CERT_A4.read_text(encoding="utf-8")
    assert "HF_TOKEN" not in text
    assert "secrets." not in text


def test_a4_evidence_records_diagnosis_without_quality_promotion() -> None:
    evidence = VIDEOCHAT3_EVIDENCE.read_text(encoding="utf-8")

    assert "## A4 output-head diagnostic result" in evidence
    assert "**Evidence validity:** **Diagnostic**" in evidence
    assert "**Decision disposition:** **Diagnostic signal**" in evidence
    assert "A person's face." in evidence
    assert "A4 actual cost was `$0.0260`" in evidence
    assert "VideoChat3 cost is `$0.3795`" in evidence
    assert "df7b04e33a9eb7435c0614295fddb34ef90c9c67b287b0f1f8747250201c8baa" in evidence
    assert "f558eb3c48f5ec428e271bb62b6708f2abf15a347dbd5b5178a63add920a1fe6" in evidence
    assert "does not establish VideoChat3 answer quality" in evidence
    assert "Competition compatibility remains `0/3`" in evidence


def test_runtime_cert_identities_are_retired_after_persistent_interactive_stop() -> (
    None
):
    """Catches either consumed runtime-cert identity becoming runnable again."""

    assert not Path("dstack/videochat3-runtime-cert-20260731-a1.dstack.yml").exists()
    assert not Path("dstack/videochat3-runtime-cert-20260731-a2.dstack.yml").exists()

    expected = {
        RETIRED_RUNTIME_CERT_A1: {
            "run_name": "videochat3-runtime-cert-20260731-a1",
            "run_id": "e7d2c5aa-dc15-4f09-aa18-ec81970781eb",
            "actual_cost_usd": 0.0304,
        },
        RETIRED_RUNTIME_CERT_A2: {
            "run_name": "videochat3-runtime-cert-20260731-a2",
            "run_id": "1724f5fa-a758-4c0d-a74a-1c8b756c3d0f",
            "actual_cost_usd": 0.0269,
        },
    }
    for path, fields in expected.items():
        retired = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert retired["status"] == "retired-consumed-runtime-stop"
        assert retired["submission_allowed"] is False
        assert retired["completion"] == "0/1"
        assert retired["evidence_validity"] == "Incomplete"
        assert retired["decision_disposition"] == (
            "Not promoted under the tested condition"
        )
        for key, value in fields.items():
            assert retired[key] == value
        for forbidden in ("type", "env", "init", "files", "commands"):
            assert forbidden not in retired
        text = path.read_text(encoding="utf-8")
        assert "HF_TOKEN" not in text
        assert "secrets." not in text


def test_consumed_a3_dev_environment_is_retired_with_verified_result() -> None:
    """Catches the completed interactive identity becoming runnable again."""

    assert not RUNNABLE_RUNTIME_CERT_A3.exists()
    retired = yaml.safe_load(RETIRED_RUNTIME_CERT_A3.read_text(encoding="utf-8"))

    assert retired["status"] == "retired-consumed-runtime-result"
    assert retired["submission_allowed"] is False
    assert retired["run_name"] == "videochat3-runtime-cert-20260731-a3"
    assert retired["run_id"] == "a4661ded-80a9-4f4f-a210-8aee2ad19d2f"
    assert retired["submitted_snapshot"]["source_config_sha256"] == (
        "4191cfc0bf872081932e08b3906853b0e2537feeeb49924b9a323445f0483356"
    )
    assert retired["submitted_snapshot"]["image"] == IMAGE
    assert retired["observed_execution"]["selected_offer"] == {
        "backend": "runpod",
        "region": "CA-MTL-1",
        "gpu": "NVIDIA A40 48GB",
        "price_usd_per_hour": 0.44,
    }
    assert retired["observed_execution"]["synthetic_generation_completed"] is True
    assert retired["actual_cost_usd"] == 0.2165
    assert retired["cumulative_videochat3_cost_usd"] == 0.3535
    assert retired["completion"] == "1/1"
    assert retired["evidence_validity"] == "Compatibility-only"
    assert retired["decision_disposition"] == (
        "Not promoted under the tested condition"
    )
    assert retired["artifact"]["checksum_verified"] is True
    assert retired["artifact"]["prediction_sha256"] == (
        "b7abd5c772def49fd97221fa5b65630520b97a892efd3a0f8e3fe4683106b9bd"
    )
    assert retired["artifact"]["certificate_sha256"] == (
        "d67e66fdd1a4b61272f1c6e61592f2d16ce6431e84074a64f07ff004c6006174"
    )
    for forbidden in ("type", "env", "init", "files", "commands"):
        assert forbidden not in retired
    text = RETIRED_RUNTIME_CERT_A3.read_text(encoding="utf-8")
    assert "HF_TOKEN" not in text
    assert "secrets." not in text


def test_runtime_cert_evidence_records_compatibility_only_negative_result() -> None:
    """Catches the synthetic result being promoted or losing its boundary."""

    evidence = VIDEOCHAT3_EVIDENCE.read_text(encoding="utf-8")

    assert "Runtime certification completion was `1/1`" in evidence
    assert "**Evidence validity:** **Compatibility-only**" in evidence
    assert "**Decision disposition:** **Not promoted under the tested condition**" in evidence
    assert "a3 actual cost was `$0.2165`" in evidence
    assert "cumulative VideoChat3 cost is `$0.3535`" in evidence
    assert "prediction, certificate, and checksum-manifest payloads" in evidence
    assert "b7abd5c772def49fd97221fa5b65630520b97a892efd3a0f8e3fe4683106b9bd" in evidence
    assert "lm_head.weight" in evidence
    assert "incoherent multilingual token sequence" in evidence
    assert "compatibility remained `0/3`" in evidence
    assert "matched pilot completion remained VideoChat3 `0/6`" in evidence
    assert "## A3 dev-only reopening frozen before execution" in evidence
    assert "videochat3-runtime-cert-20260731-a3` at most once" in evidence
    assert "No a4, private pilot" in evidence
