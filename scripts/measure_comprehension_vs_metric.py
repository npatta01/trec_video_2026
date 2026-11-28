"""Measure whether a frozen VQA run understood the video, independent of phrasing.

The official AG metrics score how closely our wording matches the reference wording. They
cannot separate "understood the video" from "wrote a plausible sentence". The 2025 training
annotations ship three human-written wrong answers per row, which gives an instrument that
can: is our frozen answer closer to the correct answer than to any wrong one? Chance is 25%.

Two controls guard against the wrong answers simply being silly:

  REAL      our answer for this video       -> understanding + plausibility
  SHUFFLED  our answer for a different one  -> plausibility only
  QUESTION  the question text, no answer    -> plausibility only

Then the same rows are split by that result and scored the official METEOR way, which shows
how much of the metric's range separates understanding from confabulation.

Read-only. No provider calls. Nothing is written under data/.

Usage:
    uv run --with sentence-transformers --with nltk --with numpy \
      python scripts/measure_comprehension_vs_metric.py
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = REPO_ROOT / "data/vqa/2025/training/annotations"
DEFAULT_FREEZE = (
    REPO_ROOT / "data/vqa/benchmarks/direct-gemini-batch-475-v1/predictions-freeze.json"
)
DEFAULT_CONDITION = "direct-answer"
JUDGE_MODEL = "all-MiniLM-L6-v2"
CHANCE = 0.25


def normalize(text: object) -> str:
    """Match the official scorer's normalization."""
    return " ".join(str(text).strip().lower().split())


def load_annotations(directory: Path) -> dict[str, dict]:
    """Return {video_id: annotation} for rows carrying a reference and wrong answers."""
    out: dict[str, dict] = {}
    for path in glob.glob(str(directory / "*.json")):
        try:
            record = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("correct_answer") and record.get("incorrect_answers"):
            out[Path(path).stem] = record
    return out


def load_predictions(freeze_path: Path, condition: str) -> list[dict]:
    freeze = json.loads(freeze_path.read_text())
    if condition not in freeze["conditions"]:
        raise SystemExit(
            f"condition {condition!r} not in freeze; available: "
            f"{sorted(freeze['conditions'])}"
        )
    return freeze["conditions"][condition]


def derangement(size: int, seed: int) -> list[int]:
    """A shuffle where no row keeps its own index."""
    rng = random.Random(seed)
    order = list(range(size))
    while True:
        rng.shuffle(order)
        if all(order[i] != i for i in range(size)):
            return order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--condition", default=DEFAULT_CONDITION)
    parser.add_argument("--shuffle-seeds", type=int, default=5)
    args = parser.parse_args()

    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.translate.meteor_score import meteor_score
    from sentence_transformers import SentenceTransformer

    for resource in ("wordnet", "punkt", "punkt_tab", "omw-1.4"):
        try:
            nltk.download(resource, quiet=True)
        except Exception as error:  # noqa: BLE001 - a cached corpus is fine offline
            print(f"  note: could not fetch nltk {resource!r} ({error}); "
                  "using any cached copy", file=sys.stderr)

    annotations = load_annotations(args.annotations)
    rows = [
        (r["video_id"], r["text"], r.get("question", ""), r.get("question_stratum", "unknown"))
        for r in load_predictions(args.freeze, args.condition)
        if r.get("video_id") in annotations and r.get("text")
    ]
    if not rows:
        raise SystemExit("no rows matched annotations; check the data symlink")

    judge = SentenceTransformer(JUDGE_MODEL)

    def embed(texts: list[str]) -> np.ndarray:
        return judge.encode(
            [normalize(t) for t in texts],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=128,
        )

    # Candidate order is always [correct, *wrong]; index 0 is the target.
    candidates = [
        embed(
            [annotations[vid]["correct_answer"]]
            + [w for w in annotations[vid]["incorrect_answers"] if w]
        )
        for vid, _, _, _ in rows
    ]
    strata = [s for _, _, _, s in rows]
    n = len(rows)

    def discriminate(probes: list[str]) -> tuple[float, list[bool]]:
        embedded = embed(probes)
        hits = [bool(candidates[i][0] @ embedded[i] > (candidates[i][1:] @ embedded[i]).max())
                for i in range(n)]
        return sum(hits) / n, hits

    answers = [r[1] for r in rows]
    questions = [r[2] or "what happens in this video" for r in rows]

    acc_real, hits_real = discriminate(answers)
    acc_question, _ = discriminate(questions)
    acc_shuffled = float(
        np.mean([
            discriminate([answers[j] for j in derangement(n, seed)])[0]
            for seed in range(args.shuffle_seeds)
        ])
    )

    half_ci = 1.96 * (acc_real * (1 - acc_real) / n) ** 0.5

    print("=" * 74)
    print(f"COMPREHENSION vs METRIC — condition {args.condition!r}, n={n}")
    print("=" * 74)
    print(f"  REAL      answer for this video     : {acc_real:6.1%}  (+/- {half_ci:.1%})")
    print(f"  SHUFFLED  answer for another video  : {acc_shuffled:6.1%}"
          f"  ({args.shuffle_seeds} derangements)")
    print(f"  QUESTION  question text only        : {acc_question:6.1%}")
    print(f"  chance                              : {CHANCE:6.1%}")
    print()
    print(f"  video-specific signal (REAL - SHUFFLED) : {acc_real - acc_shuffled:+.1%}")
    print(f"  plausibility floor    (SHUFFLED - 25%)  : {acc_shuffled - CHANCE:+.1%}")

    by_stratum: dict[str, list[bool]] = defaultdict(list)
    for stratum, hit in zip(strata, hits_real):
        by_stratum[stratum].append(hit)
    print("\n  by question stratum:")
    for stratum, hits in sorted(by_stratum.items(), key=lambda kv: -len(kv[1])):
        print(f"    {stratum:<16} n={len(hits):>4}   {sum(hits)/len(hits):>6.1%}")

    # How the official metric scores the same split.
    def meteor(reference: str, hypothesis: str) -> float:
        return meteor_score(
            [word_tokenize(normalize(reference))], word_tokenize(normalize(hypothesis))
        )

    understood, missed, our_len, ref_len = [], [], [], []
    for (vid, text, _, _), hit in zip(rows, hits_real):
        reference = annotations[vid]["correct_answer"]
        score = meteor(reference, text)
        if hit:
            understood.append(score)
            our_len.append(len(text.split()))
            ref_len.append(len(reference.split()))
        else:
            missed.append(score)

    wrong_floor = st.mean([
        meteor(record["correct_answer"], wrong)
        for record in annotations.values()
        for wrong in record["incorrect_answers"]
        if wrong
    ])

    print("\n" + "=" * 74)
    print("WHERE THE SCORE GOES  (official METEOR on the same rows)")
    print("=" * 74)
    print(f"  understood the video   n={len(understood):>4} ({len(understood)/n:>4.0%})"
          f"   METEOR {st.mean(understood):.4f}")
    print(f"  missed the video       n={len(missed):>4} ({len(missed)/n:>4.0%})"
          f"   METEOR {st.mean(missed):.4f}")
    print(f"  confidently wrong (human wrong answers)     METEOR {wrong_floor:.4f}")
    print(f"  overall                                     METEOR "
          f"{st.mean(understood + missed):.4f}")
    print()
    print(f"  understanding is worth {st.mean(understood) - wrong_floor:+.4f} METEOR "
          f"over guessing plausibly")
    print(f"  our answer length {st.mean(our_len):.1f} words vs reference "
          f"{st.mean(ref_len):.1f} words")


if __name__ == "__main__":
    main()
