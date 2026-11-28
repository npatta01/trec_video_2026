"""Measure what the AG scorer's ten-answer averaging costs a diverse answer set.

The official scorer takes the mean of METEOR, BERTScore and STS across every answer
submitted for a query, and the validator requires exactly ten distinct answers. Under a mean
with a distinctness rule, ten rewordings of one good answer beat one good answer plus nine
different guesses, because at most one guess can be right and the rest drag the mean toward
the wrong-answer floor.

This script measures three things:

  1. the floor  - what a plausible-but-wrong answer earns, from the 1,443 human-written
                  wrong answers in the 2025 training annotations
  2. the policies - a frozen run scored as ten rewordings vs. as one best answer plus nine
                  alternatives, plus the NDCG each shape produces
  3. our run    - whether the submitted answer sets are rewordings or genuinely different

Read-only. No provider calls. Nothing is written under data/.

Usage:
    uv run --with nltk --with numpy --with scikit-learn \
      python scripts/measure_ag_answer_set_averaging.py
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import statistics as st
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = REPO_ROOT / "data/vqa/2025/training/annotations"
DEFAULT_FREEZE = (
    REPO_ROOT / "data/vqa/benchmarks/direct-gemini-batch-475-v1/predictions-freeze.json"
)
DEFAULT_SUBMISSION = REPO_ROOT / "submissions/trecvid-2026/vqa-ag/q35_4b_u12/submission.json"
ANSWERS_PER_QUERY = 10


def normalize(text: object) -> str:
    return " ".join(str(text).strip().lower().split())


def ndcg_from_gains(gains: list[float], k: int | None = None) -> float:
    """The official scorer's NDCG: gains judged against the submitted rank order."""
    from sklearn.metrics import ndcg_score

    values = np.asarray(gains, dtype=float)
    if values.size == 0:
        return 0.0
    if values.size == 1:
        return 1.0
    ranked = np.arange(values.size, 0, -1, dtype=float)
    return float(ndcg_score(values.reshape(1, -1), ranked.reshape(1, -1), k=k))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--condition", default="direct-answer")
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--sample", type=int, default=120,
                        help="queries sampled from the submission for structure checks")
    args = parser.parse_args()

    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.translate.meteor_score import meteor_score

    for resource in ("wordnet", "punkt", "punkt_tab", "omw-1.4"):
        try:
            nltk.download(resource, quiet=True)
        except Exception as error:  # noqa: BLE001 - a cached corpus is fine offline
            print(f"  note: could not fetch nltk {resource!r} ({error}); "
                  "using any cached copy", file=sys.stderr)

    def meteor(reference: str, hypothesis: str) -> float:
        return meteor_score(
            [word_tokenize(normalize(reference))], word_tokenize(normalize(hypothesis))
        )

    # 1. the floor a plausible-but-wrong answer earns
    annotations: dict[str, dict] = {}
    for path in glob.glob(str(args.annotations / "*.json")):
        try:
            record = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("correct_answer"):
            annotations[Path(path).stem] = record

    floor_scores = [
        meteor(record["correct_answer"], wrong)
        for record in annotations.values()
        for wrong in record.get("incorrect_answers") or []
        if wrong
    ]
    floor = st.mean(floor_scores)

    print("=" * 74)
    print("WRONG-ANSWER FLOOR")
    print("=" * 74)
    print(f"  METEOR(human wrong answer, reference)  n={len(floor_scores)}"
          f"   mean {floor:.4f}   median {st.median(floor_scores):.4f}")

    # 2. what each answer-set shape scores
    freeze = json.loads(args.freeze.read_text())
    predictions = {
        r["video_id"]: r["text"]
        for r in freeze["conditions"][args.condition]
        if r.get("video_id") and r.get("text")
    }
    scored = [
        meteor(annotations[vid]["correct_answer"], text)
        for vid, text in predictions.items()
        if vid in annotations
    ]
    best = st.mean(scored)
    diverse = (best + (ANSWERS_PER_QUERY - 1) * floor) / ANSWERS_PER_QUERY

    print("\n" + "=" * 74)
    print(f"ANSWER-SET POLICIES  (condition {args.condition!r}, n={len(scored)})")
    print("=" * 74)
    print(f"  best answer on its own                       METEOR {best:.4f}")
    print(f"  10 rewordings of the best answer             METEOR {best:.4f}   (ceiling)")
    print(f"  best + 9 different alternatives              METEOR {diverse:.4f}")
    print(f"  ---> the diverse shape keeps {100 * diverse / best:.1f}% of what we earned")

    rng = random.Random(0)
    paraphrase_ndcg = ndcg_from_gains([best + rng.gauss(0, 0.01) for _ in range(ANSWERS_PER_QUERY)])
    diverse_ndcg = float(np.mean([
        ndcg_from_gains(_diverse_gains(best, floor, rng))
        for _ in range(2000)
    ]))
    print(f"\n  NDCG, 10 rewordings              : {paraphrase_ndcg:.4f}")
    print(f"  NDCG, 10 alternatives            : {diverse_ndcg:.4f}"
          f"   (best answer at an arbitrary rank)")

    # 3. which shape did we actually submit?
    if args.submission.exists():
        submission = json.loads(args.submission.read_text())
        sample = rng.sample(submission, min(args.sample, len(submission)))
        pairwise, to_rank1 = [], []
        for query in sample:
            answers = [a["Answer"] for a in sorted(query["Answers"], key=lambda a: a["Rank"])]
            for other in answers[1:]:
                to_rank1.append(meteor(answers[0], other))
            for i in range(len(answers)):
                for j in range(i + 1, len(answers)):
                    pairwise.append(meteor(answers[i], answers[j]))

        print("\n" + "=" * 74)
        print(f"SUBMITTED RUN {args.submission.parent.name}  ({len(sample)}-query sample)")
        print("=" * 74)
        print(f"  mean pairwise METEOR among the 10 answers : {st.mean(pairwise):.4f}")
        print(f"  mean METEOR(rank-1, each tail answer)     : {st.mean(to_rank1):.4f}")
        print(f"  wrong-answer floor for comparison         : {floor:.4f}")
        print("  near 1.0 => rewordings, score-preserving;"
              " near the floor => different guesses, score-diluting")
    else:
        print(f"\n  (no submission at {args.submission}; skipping structure check)")


def _diverse_gains(best: float, floor: float, rng: random.Random) -> list[float]:
    """One good answer at an arbitrary rank, the rest scattered around the wrong-answer floor."""
    gains = [max(0.0, floor + rng.gauss(0, 0.05)) for _ in range(ANSWERS_PER_QUERY)]
    gains[rng.randrange(ANSWERS_PER_QUERY)] = best
    return gains


if __name__ == "__main__":
    main()
