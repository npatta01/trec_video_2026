#!/usr/bin/env python3
"""Freeze a zero-network blind structured frame-counting dry run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import trec_video_2026.vqa_blind_frame_counting as blind_frame_module
import trec_video_2026.vqa_count_frame_presence as frame_presence_module
from trec_video_2026.vqa_blind_frame_counting import (
    UnsupportedCountSpec,
    build_frame_decision_request,
    certify_frame_presence_result,
    frame_decision_request_to_dict,
    frame_decision_set_to_dict,
    frame_presence_certificate_to_dict,
    frame_presence_ledger_to_dict,
    parse_frame_decisions,
    parse_visible_episode_count_spec,
    presence_count_result_to_dict,
    reconcile_frame_decisions,
    select_challenge_frames,
    visible_episode_count_spec_to_dict,
)
from trec_video_2026.vqa_count_frame_presence import reduce_presence_episodes
from trec_video_2026.vqa_submission import canonical_sha256


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _mapping(value: Any, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _presentations(value: Any) -> Sequence[Mapping[str, object]]:
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        raise TypeError("presentations must be an array of JSON objects")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_revision(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _self_hashed(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    result.pop("complete_sha256", None)
    result["complete_sha256"] = canonical_sha256(result)
    return result


def _render_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _write_chain(output_dir: Path, payloads: Mapping[str, object]) -> None:
    rendered = {name: _render_json(payload) for name, payload in payloads.items()}
    for name, content in rendered.items():
        destination = output_dir / name
        if destination.exists() and destination.read_text(encoding="utf-8") != content:
            raise FileExistsError(
                f"refusing to overwrite conflicting freeze: {destination}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in rendered.items():
        destination = output_dir / name
        if destination.exists():
            continue
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, destination)


def prepare_dry_run(
    *,
    spec_payload: Mapping[str, object],
    inventory: Mapping[str, object],
    presentations: Sequence[Mapping[str, object]],
    primary_payload: Mapping[str, object],
    challenge_payload: Mapping[str, object],
    repository: Path,
) -> dict[str, object]:
    """Build one complete authenticated chain without inference or references."""

    spec = parse_visible_episode_count_spec(spec_payload)
    if isinstance(spec, UnsupportedCountSpec):
        raise ValueError("dry run requires a supported visible-episode CountSpec")
    request = build_frame_decision_request(spec, inventory, presentations)
    primary = parse_frame_decisions(request, primary_payload)
    selected_challenge_ids = select_challenge_frames(request, primary)
    challenge = parse_frame_decisions(
        request,
        challenge_payload,
        required_frame_ids=selected_challenge_ids,
    )
    ledger = reconcile_frame_decisions(request, primary, challenge)
    result = reduce_presence_episodes(ledger)
    interface_path = Path(blind_frame_module.__file__)
    reducer_path = Path(frame_presence_module.__file__)
    certificate = certify_frame_presence_result(
        spec=spec,
        request=request,
        primary=primary,
        challenge=challenge,
        ledger=ledger,
        result=result,
        interface_source_sha256=_file_sha256(interface_path),
        reducer_source_sha256=_file_sha256(reducer_path),
        source_revision=_source_revision(repository),
    )
    return {
        "spec.freeze.json": visible_episode_count_spec_to_dict(spec),
        "request.freeze.json": frame_decision_request_to_dict(request),
        "primary.freeze.json": frame_decision_set_to_dict(primary),
        "challenge.freeze.json": frame_decision_set_to_dict(challenge),
        "ledger.freeze.json": _self_hashed(frame_presence_ledger_to_dict(ledger)),
        "result.freeze.json": _self_hashed(presence_count_result_to_dict(result)),
        "certificate.freeze.json": frame_presence_certificate_to_dict(certificate),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--presentations", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repository = Path(__file__).resolve().parents[1]
    payloads = prepare_dry_run(
        spec_payload=_mapping(_load_json(args.spec), name="spec"),
        inventory=_mapping(_load_json(args.inventory), name="inventory"),
        presentations=_presentations(_load_json(args.presentations)),
        primary_payload=_mapping(_load_json(args.primary), name="primary"),
        challenge_payload=_mapping(_load_json(args.challenge), name="challenge"),
        repository=repository,
    )
    _write_chain(args.output_dir, payloads)
    certificate = payloads["certificate.freeze.json"]
    assert isinstance(certificate, Mapping)
    print(
        json.dumps(
            {
                "outcome": certificate["outcome"],
                "value": certificate["value"],
                "complete_sha256": certificate["complete_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
