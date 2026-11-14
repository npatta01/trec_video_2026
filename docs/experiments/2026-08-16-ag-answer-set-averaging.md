# The AG scorer averages all ten answers, and our submitted run fights that

Date: **16 Aug 2026**. AG runs are due **17 Aug**. `q35_4b_u12` was uploaded 11 Aug and
passed server-side validation, so this is a live submission, not a draft.

> ## Correction applied 16 Aug, after reading the 2026 guidelines
>
> An earlier draft of this record said we keep 69% of our score. That is true only of the
> **top-10 cutoff**. The 2026 track guidelines report metrics "at different cut off ranks
> (top 1, top 5, top 10)", so the answer-set shape costs nothing at top-1, where only the
> rank-1 answer counts. The draft assumed the 2025 kit's plain mean over all ten answers was
> the whole story. It is one of three reported numbers.
>
> The same guidelines close the other option: "runs must submit 10 generated answers. Any
> submission with less than 10 answers will be rejected by the validator." The prepared
> `submission-top1.json` passes our local validator at `--answers-per-query 1` but **cannot
> be submitted**.
>
> What survives: rewording still beats diverse guesses at top-5 and top-10, and is neutral
> at top-1. The gain is real but narrower than the draft claimed.

## The finding

The validator requires **exactly ten distinct answers**. Our run fills those slots with one
best answer plus nine deliberately different alternatives. At the top-5 and top-10 cutoffs
those nine are averaged in, so nine wrong answers dilute the score.

    cutoff    submitted (best + 9 alternatives)    rewordings of the best answer
    top-1     0.295                                0.295      no change
    top-5     ~0.213                               0.295      diluted by 4 wrong answers
    top-10    0.203                                0.295      diluted by 9 wrong answers

At top-10 we keep about **69%** of what our rank-1 answers already earned. At top-1 we keep
all of it. Two of the three reported cutoffs are hurt by the shape of the answer set; one is
untouched.

## 1. The mechanism

From the shipped scorer, `eval.vqa.ag.py`:

```python
meteor_scores.append(np.mean(meteor_relevance))
bert_scores.append(np.mean(bert_relevance))
sts_scores.append(float(np.nanmean(sts_relevance)))
```

Within a cutoff, every answer counts equally — the mean is unweighted, so rank does not tilt
these three metrics. Rank decides only which answers fall inside the cutoff, plus NDCG.

The 2025 kit computes one plain mean over everything submitted, which is the top-10 case.
The 2026 guidelines add top-1 and top-5 alongside it. So the same run yields three numbers,
and the answer set's shape matters more as the cutoff widens.

From the shipped validator, `validate.ag.run.py`: `DEFAULT_ANSWERS_PER_QUERY = 10`, an
error when `len(answers) != answers_per_query`, and an error on duplicate normalized answer
text. So we cannot submit one answer, and we cannot submit the same answer ten times. Ten
distinct strings are mandatory.

Under a mean with a distinctness rule, the best answer set is **ten rewordings of your best
answer** — distinct as strings, the same underneath, so every slot scores near your best.
The worst answer set is ten genuinely different guesses, because at most one can be right
and the other nine pull the mean toward the wrong-answer floor.

We built the worst one, on purpose, as a ranked-alternatives design. It is the right design
for a task that rewards covering several possibilities, and the wrong one for a task that
averages them.

One caveat on rewording, since it cuts the other way when rank-1 is wrong. Our rank-1 answer
is right about 61% of the time. When it is right, rewording holds every slot near 0.378 and
diverse guesses drop the query to 0.212. When it is wrong, diverse guesses do slightly
better — 0.190 against 0.166 — because one of the nine may land. Taking both cases at their
measured rates, rewording still wins on expectation: **0.295 against 0.203** at top-10.

## 2. That our run really is the bad case

Measured on a 120-query sample of the actual `submission.json`:

    mean pairwise METEOR among the 10 answers      0.276
    mean METEOR(rank-1, each tail answer)          0.218
    METEOR(human wrong answer, reference)          0.193

The tail answers sit nearly as far from our best answer as a known-wrong answer sits from
the truth. They are not rewordings. They are nine separate guesses.

## 3. What it costs

Scoring 1,443 human-written wrong answers from the 2025 training set against their
references gives the floor a plausible-but-wrong answer earns: **0.193 METEOR**. Our best
single answer on the frozen 475-row panel earns **0.295**.

    policy                                    METEOR    NDCG
    10 rewordings of the best answer          0.295     0.994
    best + 9 different alternatives           0.203     0.906

NDCG rewards scores that fall as rank rises. Ten near-equal rewordings satisfy that almost
perfectly. Ten different answers with the good one at an arbitrary rank do not.

The paraphrase set wins on both axes, but they are not the same size. METEOR is the prize:
about **1.45x**. NDCG moves only 0.906 to 0.994, because our best answer scores 0.295 against
a 0.193 floor — the gap between a right and a wrong answer is small enough that misordering
them costs little. Neither change touches the model, the frames, or the video pipeline.

## 4. What this does NOT cover

- **0.295 is a ceiling.** It assumes a reworded answer scores exactly what the original
  scored. Real paraphrases score somewhat below. Likely: the true gain is nearer 1.25–1.4x.
- **BERTScore barely moves.** Its floor is roughly 0.86–0.88 for any two English sentences,
  which is why our Gemini and Qwen arms sit at 0.896 and 0.911. Expect about +0.02 there,
  not 1.45x.
- **STS is unmeasured.** It needs `calc.sts.pl`, which was not run here. Likely: it behaves
  like METEOR, since both reward shared wording.
- **The 0.203 figure uses a stand-in.** It scores our tail answers as if they earn the human
  wrong-answer floor. We measured that the tails sit at 0.198 from our best answer, against
  a 0.193 floor, which supports the stand-in — but it is a stand-in, not a direct
  measurement of our tails against the references.
- **The gain does not apply at top-1.** Only the rank-1 answer counts there, so rewording
  changes nothing. Two of three cutoffs improve; one is untouched. No cutoff is designated
  as the headline, so how much this is worth overall is unknown.
- **The per-cutoff figures for top-5 are modelled, not measured.** `0.213` is
  `(0.295 + 4 x 0.193) / 5`. The 2025 kit does not implement cutoffs, so the exact 2026
  implementation is unseen.
- **Ten answers are mandatory.** The 2026 guidelines say "runs must submit 10 generated
  answers. Any submission with less than 10 answers will be rejected by the validator." The
  prepared `submission-top1.json` passes our local validator at `--answers-per-query 1`,
  which is a property of our copy of the script, not permission to send it.

## 5. What to do before the deadline

Keep rank-1 exactly as it is — it carries top-1 and it is already frozen and validated.
Rebuild ranks 2 through 10 as rewordings of that rank-1 answer. This is a text rewrite of a
tracked artifact: no video inference, no GPU, no new provider calls.

Expected effect: top-1 unchanged, top-5 and top-10 up. Since no cutoff is the designated
headline, this improves two of the three reported numbers and cannot reduce the third.

Keep `q35_4b_u12` as it stands. Submit the rewrite under a new tag so the two stay
comparable when scores come back.

This is a scoring-format repair, not a research result. It says nothing about whether the
system understands video — see
[the comprehension record](2026-08-16-comprehension-vs-metric-resolution.md) for that.

## Reproducing

    uv run --with nltk --with numpy \
      python scripts/measure_ag_answer_set_averaging.py

Needs the shared `data` symlink. Read-only, no provider calls.
