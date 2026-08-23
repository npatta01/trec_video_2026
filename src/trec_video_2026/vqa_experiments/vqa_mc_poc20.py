"""Frozen, label-isolated contract and scoring primitives for the VQA MC POC."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = 1
EXPERIMENT_ID = "vqa-mc-poc20-20260821"
SHUFFLE_SEED = "trecvid-vqa-mc-poc20-20260821-v1"
OPTION_IDS = tuple("ABCD")
STRATA = ("temporal", "causal", "ocr_detail", "counting", "speech")
VIDEO_IDS_BY_STRATUM = {
    "temporal": ("BDKkVYmWqdM", "LAa5O3ToD2c", "JpTsHQHzptE", "LxZuar32o1o"),
    "causal": ("Bvv6lMUTlEw", "RIbj7xukchQ", "D_HTq7IiSPo", "LziNpkXGoDE"),
    "ocr_detail": ("DC4FE3hI67U", "JkfvwpRl488", "Nvg0mgGelUU", "Gb9otyfOuqA"),
    "counting": ("G5AgzHKJINU", "JJ8sKvg55sM", "OVAy9Y-zh1E", "KoQD9x6VrSk"),
    "speech": ("G6zu9D0j8hk", "LKiaCf1QZFw", "LSz_xBDBPNc", "O7lBcBruXHM"),
}
VIDEO_IDS = tuple(
    video_id for stratum in STRATA for video_id in VIDEO_IDS_BY_STRATUM[stratum]
)
CANARY_VIDEO_IDS = frozenset(video_ids[0] for video_ids in VIDEO_IDS_BY_STRATUM.values())
REPEAT_VIDEO_IDS = frozenset(("BDKkVYmWqdM", "G6zu9D0j8hk"))
CONDITIONS = ("video-plus-transcript", "native-video")
CORRECT_POSITION_IDS = tuple("BDACCADBDBCAACBDBADC")

DEFAULT_PROTOCOL = Path("vqa/benchmarks/direct-gemini-batch-475-v1/protocol.json")
DEFAULT_ANNOTATIONS = Path("vqa/2025/training/annotations")
DEFAULT_WHISPER = Path(
    "vqa/benchmarks/vqa-factorial-475-20260809-f2/conditions/4b-current/"
    "payload/evidence/requests.accepted-whisper.json"
)
DEFAULT_OUTPUT = Path("vqa/benchmarks/vqa-mc-poc20-20260821/inputs")


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_data_path(data_root: Path, relative: Path) -> Path:
    lexical = PurePosixPath(relative.as_posix())
    if lexical.is_absolute() or ".." in lexical.parts or lexical.parts[0] != "vqa":
        raise ValueError("input path must be a safe data-root-relative VQA path")
    root = data_root.resolve(strict=True)
    resolved = (root / relative).resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("input path escapes data root") from error
    return resolved


def _normalize_source_path(data_root: Path, value: str) -> tuple[str, Path]:
    candidate = Path(value)
    if candidate.is_absolute():
        try:
            relative = candidate.resolve(strict=True).relative_to(data_root)
        except ValueError as error:
            raise ValueError("source path escapes data root") from error
    else:
        relative = candidate
    resolved = _safe_data_path(data_root, relative)
    return relative.as_posix(), resolved


def _read_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected an object: {path.name}")
    return value


def _protocol_rows(path: Path) -> dict[str, Mapping[str, Any]]:
    payload = _read_object(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("protocol must contain rows")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("video_id"), str):
            raise ValueError("protocol row is invalid")
        indexed[row["video_id"]] = row
    return indexed


def _transcripts(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        raise ValueError("Whisper input must contain rows")
    result: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Whisper row is invalid")
        video_id = row.get("video_id")
        transcript = row.get("transcript") or row.get("text") or row.get("timestamped_text")
        if isinstance(video_id, str) and isinstance(transcript, str) and transcript.strip():
            result[video_id] = transcript.strip()
    return result


def _ordered_options(annotation: Mapping[str, Any], video_id: str, correct_position: str) -> tuple[list[dict[str, str]], str]:
    correct = annotation.get("correct_answer")
    incorrect = annotation.get("incorrect_answers")
    if not isinstance(correct, str) or not correct.strip():
        raise ValueError(f"correct answer is invalid for {video_id}")
    if (
        not isinstance(incorrect, list)
        or len(incorrect) != 3
        or not all(isinstance(value, str) and value.strip() for value in incorrect)
    ):
        raise ValueError(f"incorrect answers are invalid for {video_id}")
    answers = [correct.strip(), *(value.strip() for value in incorrect)]
    if len(set(answers)) != 4:
        raise ValueError(f"answer options are not unique for {video_id}")
    distractors = sorted(
        answers[1:],
        key=lambda value: hashlib.sha256(
            f"{SHUFFLE_SEED}:{video_id}:{value}".encode("utf-8")
        ).hexdigest(),
    )
    correct_index = OPTION_IDS.index(correct_position)
    ordered = list(distractors)
    ordered.insert(correct_index, answers[0])
    return [
        {"option_id": option_id, "text": text}
        for option_id, text in zip(OPTION_IDS, ordered, strict=True)
    ], correct_position


def _variants(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for row in rows:
        video_id = str(row["video_id"])
        variants.append(
            {
                "variant_id": f"{video_id}:canonical",
                "video_id": video_id,
                "variant_kind": "canonical",
                "display_order": list(OPTION_IDS),
            }
        )
    for video_id in VIDEO_IDS:
        if video_id in CANARY_VIDEO_IDS:
            variants.append(
                {
                    "variant_id": f"{video_id}:reversed",
                    "video_id": video_id,
                    "variant_kind": "reversed",
                    "display_order": list(reversed(OPTION_IDS)),
                }
            )
    for video_id in VIDEO_IDS:
        if video_id in REPEAT_VIDEO_IDS:
            variants.append(
                {
                    "variant_id": f"{video_id}:repeat",
                    "video_id": video_id,
                    "variant_kind": "repeat",
                    "display_order": list(OPTION_IDS),
                }
            )
    return variants


def build_bundle(data_root: Path) -> dict[str, Any]:
    """Build the private request/label split from frozen 2025 sources."""

    data_root = data_root.resolve(strict=True)
    protocol_path = _safe_data_path(data_root, DEFAULT_PROTOCOL)
    annotation_root = _safe_data_path(data_root, DEFAULT_ANNOTATIONS)
    whisper_path = _safe_data_path(data_root, DEFAULT_WHISPER)
    protocol = _protocol_rows(protocol_path)
    transcripts = _transcripts(whisper_path)
    request_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, str]] = []
    for index, video_id in enumerate(VIDEO_IDS):
        stratum = next(
            name for name, video_ids in VIDEO_IDS_BY_STRATUM.items() if video_id in video_ids
        )
        source = protocol.get(video_id)
        if source is None or source.get("question_stratum") != stratum:
            raise ValueError(f"protocol stratum mismatch for {video_id}")
        annotation_path = annotation_root / f"{video_id}.json"
        annotation = _read_object(annotation_path)
        if annotation.get("question") != source.get("question"):
            raise ValueError(f"question mismatch for {video_id}")
        transcript = transcripts.get(video_id)
        if not transcript:
            raise ValueError(f"accepted transcript is unavailable for {video_id}")
        source_path = source.get("source_path")
        if not isinstance(source_path, str):
            raise ValueError(f"source path is invalid for {video_id}")
        relative_source_path, media_path = _normalize_source_path(data_root, source_path)
        if sha256_file(media_path) != source.get("source_sha256"):
            raise ValueError(f"source hash mismatch for {video_id}")
        options, correct_option_id = _ordered_options(
            annotation, video_id, CORRECT_POSITION_IDS[index]
        )
        request_rows.append(
            {
                "query_id": video_id,
                "video_id": video_id,
                "stratum": stratum,
                "question": source["question"],
                "options": options,
                "duration_seconds": source["duration_seconds"],
                "source_path": relative_source_path,
                "source_sha256": source["source_sha256"],
                "transcript": transcript,
                "transcript_sha256": sha256_bytes(transcript.encode("utf-8")),
            }
        )
        label_rows.append(
            {"query_id": video_id, "correct_option_id": correct_option_id}
        )
    if Counter(row["stratum"] for row in request_rows) != Counter({name: 4 for name in STRATA}):
        raise ValueError("panel is not balanced across strata")
    if Counter(row["correct_option_id"] for row in label_rows) != Counter({key: 5 for key in OPTION_IDS}):
        raise ValueError("correct positions are not balanced")
    requests = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "label_policy": "options_visible_correctness_separate",
        "conditions": list(CONDITIONS),
        "rows": request_rows,
        "variants": _variants(request_rows),
    }
    labels = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "rows": label_rows,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "selection_policy": "five_canary_plus_three_shortest_transcript_available_per_stratum",
        "shuffle_seed": SHUFFLE_SEED,
        "row_count": len(request_rows),
        "variant_count": len(requests["variants"]),
        "qwen_condition_count": len(CONDITIONS),
        "request_sha256": sha256_bytes(canonical_json(requests)),
        "label_sha256": sha256_bytes(canonical_json(labels)),
        "source_sha256s": {
            DEFAULT_PROTOCOL.as_posix(): sha256_file(protocol_path),
            DEFAULT_WHISPER.as_posix(): sha256_file(whisper_path),
        },
    }
    return {"requests": requests, "labels": labels, "manifest": manifest}


def parse_ranking(raw: object, *, option_ids: Sequence[str] = OPTION_IDS) -> list[str]:
    """Parse only a canonical JSON object containing one complete permutation."""

    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("ranking is not canonical JSON") from error
    else:
        payload = raw
    if not isinstance(payload, Mapping) or set(payload) != {"ranking"}:
        raise ValueError("ranking object must contain only the ranking field")
    ranking = payload.get("ranking")
    expected = list(option_ids)
    if not isinstance(ranking, list) or len(ranking) != len(expected):
        raise ValueError("ranking must contain every option exactly once")
    if not all(isinstance(value, str) for value in ranking) or sorted(ranking) != sorted(expected):
        raise ValueError("ranking must be a permutation of the option IDs")
    return list(ranking)


def fuse_borda(rankings: Sequence[Sequence[str]]) -> list[str]:
    """Fuse complete rankings with equal vote weight and stable ID tie-breaks."""

    if not rankings:
        raise ValueError("at least one ranking is required")
    option_ids = sorted(rankings[0])
    if any(sorted(ranking) != option_ids for ranking in rankings):
        raise ValueError("all rankings must cover the same options")
    scores = {option_id: 0 for option_id in option_ids}
    count = len(option_ids)
    for ranking in rankings:
        for index, option_id in enumerate(ranking):
            scores[option_id] += count - index - 1
    return sorted(option_ids, key=lambda option_id: (-scores[option_id], option_id))


def score_rankings(
    predictions: Mapping[str, Sequence[str] | None], labels: Mapping[str, str]
) -> dict[str, int | float]:
    """Score rankings against frozen option IDs, penalizing invalid rows as zero."""

    valid = 0
    top1 = 0
    reciprocal_rank = 0.0
    for query_id, correct_option_id in labels.items():
        ranking = predictions.get(query_id)
        if ranking is None:
            continue
        parsed = parse_ranking({"ranking": list(ranking)}, option_ids=OPTION_IDS)
        valid += 1
        if parsed[0] == correct_option_id:
            top1 += 1
        reciprocal_rank += 1.0 / (parsed.index(correct_option_id) + 1)
    row_count = len(labels)
    return {
        "row_count": row_count,
        "valid_count": valid,
        "invalid_count": row_count - valid,
        "top1_correct": top1,
        "top1_accuracy": top1 / row_count if row_count else 0.0,
        "mrr": reciprocal_rank / row_count if row_count else 0.0,
    }


def write_bundle(data_root: Path, output_root: Path) -> dict[str, Any]:
    bundle = build_bundle(data_root)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"output already exists: {output_root}")
    output_root.mkdir(parents=True)
    for name in ("requests", "labels", "manifest"):
        (output_root / f"{name}.json").write_bytes(canonical_json(bundle[name]))
    checksum_lines = [
        f"{sha256_file(path)}  {path.name}"
        for path in sorted(output_root.glob("*.json"))
    ]
    (output_root / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return bundle["manifest"]


def main() -> int:
    parser = argparse.ArgumentParser(prog="vqa_mc_poc20")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.data_root / DEFAULT_OUTPUT
    print(json.dumps(write_bundle(args.data_root, output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
