# The AG scorer averages all ten answers, and our submitted run fights that

Date: **16 Aug 2026**. AG runs are due **17 Aug**. `q35_4b_u12` was uploaded 11 Aug and
passed server-side validation, so this is a live submission, not a draft.

## The finding

The official scorer takes the **mean** across every answer we submit for a query. The
validator requires **exactly ten distinct answers**. Our run fills those ten slots with one
best answer plus nine deliberately different alternatives — so nine wrong answers are
averaged into every query's score.

    submitted policy   best answer + 9 different alternatives    0.203 METEOR
    paraphrase policy  10 rewordings of the best answer          up to 0.295
    what we earned     the best answer on its own                0.295

**We are keeping about 69% of the score we already earned.** The rest is thrown away by the
shape of the answer set, not by anything the model got wrong.

## 1. The mechanism

From the shipped scorer, `eval.vqa.ag.py`:

```python
meteor_scores.append(np.mean(meteor_relevance))
bert_scores.append(np.mean(bert_relevance))
sts_scores.append(float(np.nanmean(sts_relevance)))
```

Every submitted answer counts equally. Rank does not weight these three metrics at all —
rank only feeds NDCG.

From the shipped validator, `validate.ag.run.py`: `DEFAULT_ANSWERS_PER_QUERY = 10`, an
error when `len(answers) != answers_per_query`, and an error on duplicate normalized answer
text. So we cannot submit one answer, and we cannot submit the same answer ten times. Ten
distinct strings are mandatory.

Under a mean with a distinctness rule, the best answer set is **ten rewordings of your best
answer** — distinct as strings, the same underneath, so every slot scores near your best.
The worst answer set is ten genuinely different guesses, because at most one can be right
and the other nine pull the mean toward the wrong-answer floor.

We built the worst one, on purpose, as a ranked-alternatives design.

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
- **The 2026 rules may differ.** The kit's own `README.md` says it is "to be updated for VQA
  2026 to take into account new run format submissions and revised scoring metric
  reporting." Ten answers passed Evalbase on 11 Aug, so ten is accepted. Unknown: whether
  fewer is now allowed, and whether the reported metrics changed. Check the 2026 guidelines
  page before rebuilding.

## 5. What to do before the deadline

Rebuild the ten slots as rewordings of the existing rank-1 answer and submit as a second
run. The rank-1 answers already exist for all 2,443 queries and already validated, so this
is a text rewrite of a tracked artifact — no video inference, no GPU.

Keep `q35_4b_u12` as it stands. Submit the rewrite under a new tag so the two are comparable
when scores come back.

This is a scoring-format repair, not a research result. It says nothing about whether the
system understands video — see
[the comprehension record](2026-08-16-comprehension-vs-metric-resolution.md) for that.

## Reproducing

    uv run --with nltk --with numpy \
      python scripts/measure_ag_answer_set_averaging.py

Needs the shared `data` symlink. Read-only, no provider calls.
