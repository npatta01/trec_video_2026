"""Prepare and verify the private weather media and two transcript excerpts."""

import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
VQA = DATA / "vqa"
OUT = ROOT / "docs/paper/private-frames"
TEST_VIDEOS = VQA / "2026/testing/videos"
SHARDS = VQA / "benchmarks/deadline-submission-full-20260816/qwen36/shards"
BASELINE = VQA / "submissions/qwen35-safety-20260810-a1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_transcript(path: Path, query_id: int, shard: int, video_id: str) -> str:
    shard_dir = SHARDS / str(shard)
    request_name = f"requests/{query_id}_video-plus-transcript.json"
    checksums = dict(
        (name, digest)
        for digest, name in (line.split(maxsplit=1) for line in (shard_dir / "SHA256SUMS").read_text().splitlines())
    )
    if sha256(path) != checksums[request_name]:
        raise SystemExit(f"Q{query_id} request does not match the archived file checksum.")

    request = json.loads(path.read_text())["request"]
    manifest = json.loads((shard_dir / "run-manifest.json").read_text())["requests"]
    saved = next(
        row for row in manifest
        if row["query_id"] == query_id and row["condition_id"] == "video-plus-transcript"
    )
    if request["query_id"] != query_id or request["video_id"] != video_id:
        raise SystemExit(f"Q{query_id} request identifies a different clip.")
    if sha256(TEST_VIDEOS / video_id) != request["video_sha256"]:
        raise SystemExit(f"Q{query_id} video does not match its saved request.")
    transcript = request["transcript"]
    if hashlib.sha256(transcript.encode()).hexdigest() != saved["transcript_sha256"]:
        raise SystemExit(f"Q{query_id} transcript does not match the saved run manifest.")
    return transcript


def verify_weather_frames(clip: Path) -> list[Path]:
    requests = json.loads((BASELINE / "requests.json").read_text())
    baseline = next(row for row in requests if row["query_id"] == 1835)
    frame_dir = BASELINE / "evidence/frames" / clip.name
    frames = [frame_dir / Path(path).name for path in baseline["frame_paths"]]
    evidence = json.loads((frame_dir / "evidence.json").read_text())

    if not sha256(clip) == evidence["source_sha256"] == baseline["source_sha256"]:
        raise SystemExit("The saved frame inputs disagree on the source video hash.")
    if len(frames) != 12 or baseline["frame_timestamps"] != [
        frame["timestamp_seconds"] for frame in evidence["frames"]
    ]:
        raise SystemExit("The baseline request does not match the saved frame schedule.")
    for path, frame in zip(frames, evidence["frames"], strict=True):
        if path.name != frame["path"] or sha256(path) != frame["sha256"]:
            raise SystemExit(f"Frame identity check failed for {path.name}.")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speech-request", required=True, type=Path, help="Archived Q448 request JSON")
    args = parser.parse_args()
    if not DATA.is_dir():
        raise SystemExit("Link the shared dataset before preparing deck media.")

    clip = TEST_VIDEOS / "8083.webm"
    frame_paths = verify_weather_frames(clip)
    weather = read_transcript(
        SHARDS / "2/requests/1835_video-plus-transcript.json", 1835, 2, clip.name
    )
    speech = read_transcript(args.speech_request, 448, 3, "6950.webm")

    OUT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=OUT) as temporary:
        staging = Path(temporary)
        video = staging / "weather-clip-av.mp4"
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-v", "error", "-i", str(clip),
                "-map", "0:v:0", "-map", "0:a:0",
                "-c:v", "libx264", "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(video),
            ],
            check=True,
        )
        for index, frame in enumerate(frame_paths, 1):
            subprocess.run(
                ["convert", str(frame), "-resize", "360x", "-strip", "-quality", "88",
                 str(staging / f"weather-frame-{index:02d}.jpg")],
                check=True,
            )

        (staging / "weather-transcript.html").write_text(escape(weather.partition(" ")[2]))
        speech_lines = []
        for line in speech.splitlines():
            timestamp, spoken = line.split(" ", 1)
            speech_lines.append(
                f'<span class="speech-time">{escape(timestamp)}</span><span>{escape(spoken)}</span>'
            )
        (staging / "speech-transcript.html").write_text("\n".join(speech_lines))
        for artifact in staging.iterdir():
            artifact.chmod(0o600)
            artifact.replace(OUT / artifact.name)

    print(f"Verified the weather frames and both transcript excerpts. Deck media is in {OUT}.")


if __name__ == "__main__":
    main()
