# The system understands the video. The metric can't see it.

Date: **16 Aug 2026**. Panel: the frozen 475-row Gemini 3.1 Pro native-AV run
(`direct-gemini-batch-475-v1`, frozen 13 Aug). No new provider calls were made for this
analysis — it re-reads predictions that were already frozen, and grades them a second way.

## The finding

Gemini picked the correct answer out of four **60.8%** of the time. Guessing is 25%.

That number never appears in any of our reports, because our reports score how closely our
words match the official answer's words, not whether we were right.

    what we measure today   how close our wording is to the reference   0.295 METEOR
    what the system does    picks the right answer out of four          60.8%

Every "not promoted" and "interval crosses zero" in the ledger since July is explained by
this gap. We were not failing to improve the system. We were measuring it with something
that cannot show improvement.

## 1. The test

The 2025 training annotations ship three human-written wrong answers next to each correct
one. We have never used them — the code excludes them from prompts as scoring-private, and
that is correct for generating answers. It also meant we threw away the only instrument in
the dataset that measures whether the system understood anything.

For each of 475 rows: take the frozen answer, and check whether it sits closer to the
correct answer than to any of the three wrong ones. Closeness is sentence-embedding cosine
(`all-MiniLM-L6-v2`). Chance is 25%.

The predictions were frozen on 13 Aug, three days before this scoring existed, and the
wrong answers never entered a prompt. So this is label-blind by construction, not by
protocol promise.

## 2. The controls

The obvious objection: the wrong answers might be silly on their face, so any sensible
sentence beats them. Two controls, both run:

    REAL       our answer for THIS video           60.8%   (+/- 4.4%, n=475)
    SHUFFLED   our answer for a DIFFERENT video    27.6%   (mean of 5 derangements)
    QUESTION   the question text, no answer        34.5%
    chance                                         25.0%

**Video-specific signal: +33.2 points.** The plausibility floor is +2.6 points. Feed the
system's own words back against the wrong video and it collapses to guessing. The
discrimination comes from having watched that specific video, not from writing plausible
English.

By question type:

    ocr_detail        n= 25    84.0%
    temporal          n=177    61.6%
    other             n=160    60.0%
    counting          n= 57    59.6%
    causal            n= 29    58.6%
    sound             n= 19    52.6%
    speech            n=  8    25.0%

## 3. What the metric does with that

Split the same 475 rows by whether the system demonstrably understood the video, then score
each half the official way:

    understood the video    n=289 (61%)    METEOR 0.378
    missed the video        n=186 (39%)    METEOR 0.166
    confidently wrong, plausible                  0.193   (the 1,443 human wrong answers)
    overall                                       0.295

**Getting it right is worth +0.185 METEOR over guessing plausibly.** That is the entire
distance between understanding and confabulation, on the scale we have been reading.

So an intervention that improves understanding by three points moves the reported score by
about 0.005. On a 475-row panel that is invisible. Our Gemini-versus-Qwen result — +0.0083,
interval crossing zero — was never a test that could resolve anything.

Counting shows it most sharply. Counting sits at 59.6% discrimination, statistically
indistinguishable from every other category. The official metric reports counting as our
worst stratum, STS 0.082. Counting answers are one or two words, and word-overlap scoring
against a one-word reference is close to binary. The system knows the count. The metric
reports catastrophe.

## 4. Where the points actually go

On rows we got right, our answers run **10.1 words** against a **7.4-word** reference. Right
content, wrong shape.

Estimate, not a measurement: if those 289 understood rows were phrased in the reference's
register, overall METEOR lands near 0.41–0.50 against 0.295 today. The arithmetic is
`(289 × target + 186 × 0.193) / 475` at target 0.55 and 0.70. Treat the target as an
assumption — nobody has measured what a well-phrased correct answer scores here.

That is still the largest single block of points on the table, and it is not a trick.
The task is named Answer Generation. Saying the thing you understood, in the form the task
expects, is the task.

## 5. What genuinely looks weak

Audio. `sound` at 52.6% (n=19) and `speech` at 25.0% (n=8). Both samples are far too small
to conclude from, but they are the only categories that do not clear the floor cleanly, and
they line up with CASE-002 in the failure library, where ASR missed a non-lexical vocal
event. If there is a real capability gap left, likely: it is ears, not eyes.

## 6. What this does NOT cover

- **The judge is weak.** `all-MiniLM-L6-v2` is a small embedding model. 60.8% is a noisy
  lower bound; some correct answers are misjudged. A stronger judge would move it, probably up.
- **Picking is not producing.** Landing nearer the correct answer than three wrong ones shows
  the information got in. It does not show the system can state it unprompted. A direct
  multiple-choice prompt would measure this properly. NIST ships `eval.vqa.mc.py` and runs an
  MC task; the instrument exists and we have not run it.
- **The wrong answers were written for multiple choice.** They are not adversarial against
  our system specifically.
- **n=8 for speech.** That supports nothing on its own.
- **One model, one panel.** Gemini native-AV only. The matched Qwen arm was not re-scored
  this way. Unknown: whether the 4B baseline shows the same understand-but-can't-say split,
  and it should be checked before any conclusion about model choice is revised.
- **This does not revise any model ranking.** It says the comparisons we ran lacked the
  resolution to rank anything. That is a statement about the instrument, not a promotion.

## 7. What follows

1. Re-score the matched Qwen arm the same way. If it also lands near 60%, then every model
   comparison in the ledger was measuring wording, and the "no candidate displaces
   Qwen3.5-4B" line needs to be marked as unresolved rather than settled.
2. Run the direct multiple-choice prompt. It is cheap and it turns this lower bound into a
   real number.
3. Move effort to phrasing: length and register matched to the reference distribution, fit on
   the 481 labelled training rows. That is where the measurable points are.
4. Stop using single-answer semantic composite as a promotion gate. It cannot separate a
   system that understands from one that guesses well.

## Reproducing

    uv run --with sentence-transformers --with nltk --with numpy \
      python scripts/measure_comprehension_vs_metric.py

Needs the shared `data` symlink. Read-only: it does not write to `data/`, and makes no
provider calls.
