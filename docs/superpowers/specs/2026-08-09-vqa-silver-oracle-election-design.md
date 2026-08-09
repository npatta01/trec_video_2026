# Full-panel VQA silver-oracle and answer-election design

## Approval and objective

The user explicitly asked to review the architecture with an advisor panel and
then test it, prefers larger confidence-producing bets, and does not want a
manual full-panel evaluation. This design incorporates the unanimous panel
direction and remains inside that approved scope.

The objective is to determine, on every annotated VQA row with local media,
whether the main recoverable headroom comes from selecting the right visual or
speech moment, selecting among complementary generated answers, or combining
both.

## Evidence already established

- The canonical annotation set contains 481 VQA rows; 475 have local media and
  form the complete development panel. Six missing-media IDs remain explicit
  exclusions. VTT files are captions, not VQA answer labels.
- The verified four-condition factorial used all 475 rows. Its best condition,
  4B-current, scored METEOR `0.233622`, BERTScore F1 `0.911542`, STS
  `0.348610`, and composite `C=0.497924`.
- A blanket 8B switch reduced `C` by `0.008303` on average; a blanket
  question-aware prompt added only `0.003320`, with both paired intervals
  crossing zero.
- Accepted Whisper improved `C` by `0.02518`, with paired 95% interval
  `[0.01443, 0.03715]`.
- A reference-aware per-row maximum over the four frozen answers reaches
  `C=0.550749`, a `+0.052825` ceiling. This proves candidate complementarity,
  not a deployable election gain.
- Four full answer arms required `0.801` aggregate GPU-hours and `13.6` minutes
  of parallel answer wall time. Full-panel answer inference is not the current
  bottleneck.

## Panel and boundaries

- Primary panel: the 475 sorted VQA 2025 training IDs with both annotations and
  local media, digest
  `95a623c0add252541b9c81ff4910e949dc81db17482add2c467b761237b22b0f`.
- Excluded: the six annotated rows without local media, recorded by ID and
  reason in the private protocol.
- Untouched confirmation set: all 1,883 labeled 2025 test rows. No selector,
  prompt, judge, gate, or routing choice may inspect them before architecture
  freeze.
- Raw media, annotations, transcripts, frames, predictions, judge rankings,
  and logs remain in ignored local storage and private Hugging Face Bucket
  prefixes. Only sanitized aggregates and hashes may enter Git.
- No manual row-quality grading is required. Automatic semantic metrics and a
  reference-blind model judge are primary; human inspection is allowed only
  after the frozen result as a descriptive failure audit.

## Silver-oracle selector

Use
`Qwen/Qwen3-Omni-30B-A3B-Thinking@2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b`
in its pinned digest runtime. Each selector call sees exactly:

- the complete source video with native audio;
- video duration (all panel videos are 5–30 seconds);
- the question;
- the reference answer;
- the frozen timestamped Whisper transcript, including segment indices, or
  explicit null.

It returns only this terminal object:

```json
{
  "status": "localized",
  "intervals": [{"start_seconds": 3.2, "end_seconds": 8.7}],
  "whisper_segment_ids": [1, 2]
}
```

`status` is `localized` or `non_localizable`; `intervals` contains zero to two
non-overlapping in-range intervals; and `whisper_segment_ids` contains zero to
three valid unique indices. Combined interval duration cannot exceed
`min(video_duration, max(12, min(30, 0.2 * video_duration)))` seconds. A
non-localizable record has empty arrays. Captions, rationales, confidence prose,
answers, hints, rewritten transcript text, and arbitrary keys are rejected.

The runtime stores only the validated object, a raw-output hash, timing, and
model/runtime identity. It does not persist model reasoning. One predeclared
schema retry is allowed with the same evidence, a fixed repair instruction, and
seed 43 after the primary seed-42 call. The answer evidence builder has no API
that accepts reference text or raw selector output.

This is a reference-assisted **silver-oracle** selector. It measures achievable
headroom under a strong automatic locator, not a true human ideal and not a
deployable competition component.

## Evidence factorial

The answerer is frozen Qwen3.5-4B/current. The nine cells cross three 12-frame
views with three transcript views:

| Frame view | No Whisper (`N`) | Full accepted Whisper (`F`) | Selected span (`S`) |
|---|---:|---:|---:|
| Existing global uniform-12 (`G`) | G/N | G/F | G/S |
| Six global + six silver-local (`H`) | H/N | H/F | H/S |
| Twelve silver-local (`L`) | L/N | L/F | L/S |

`G` reuses the exact verified decoded frames only after checksum identity is
proven. `H` and `L` use the existing actual-PTS nearest-unused allocator. One
interval receives all local slots; two intervals split slots by deterministic
Hamilton allocation. Timestamp collisions are filled by the nearest unused
decoded frame inside the same interval, then by the nearest unused global frame
only under a recorded construction fallback.

`N` contains zero transcript bytes. `F` contains the exact accepted Whisper
bytes or null from the verified evidence bundle. `S` contains only the exact
verbatim timestamped lines named by `whisper_segment_ids`, or null. Selector
intervals, status, reference, rationale, model identity, and confidence never
enter the answer prompt.

All cells use the same model revision, current prompt, greedy decoding, token
limit, image order, and `65,536..262,144` pixels per frame. Run all nine cells
for matched provenance; reuse is allowed only when the complete model-input
digest is byte-identical.

Primary evidence contrast: `H/S - G/F`. Secondary mechanism contrasts are
`H/F - G/F` and `L/F - G/F` for visual timing, and `G/S - G/F`, `H/S - H/F`,
and `L/S - L/F` for transcript timing. `N` cells quantify transcript presence.

## Reference-blind election

Use the same pinned Qwen3-Omni revision as a separate judge invocation. The
judge never receives a reference, metric score, selector output, condition ID,
or producer model ID. Candidate order is deterministically randomized per row.
It returns a complete ranking and may not rewrite candidates.

- Pool A: the four frozen factorial candidates with the common global-12 plus
  full-Whisper evidence. This is the clean deployable-election test available
  today.
- Pool B: the nine new same-answerer evidence candidates. The judge sees the
  deduplicated union of G/H/L frame packets and full plus selected transcript
  bytes.
- Pool C: the deduplicated union of Pools A and B under the same union evidence.
  This estimates the combined silver-locator-plus-election ceiling.

Normalize whitespace/case/punctuation before exact deduplication. Preserve the
first randomized candidate identity for byte-identical answers. Report
pairwise BERTScore similarity and the count of near-duplicate pairs as a
diagnostic; do not silently drop non-identical semantic paraphrases in this
first judge test.

## Metrics and gates

For every answer and elected top-1 output, capture normalized exact match,
token F1, METEOR, BERTScore F1, STS, and
`C=(METEOR+BERTScore_F1+STS)/3`. Exact match is diagnostic only. Use a fixed
seed and 10,000 paired row-bootstrap resamples for contrasts.

For each complete judge ranking, compute NDCG@`K`, where
`K=min(10, unique_candidate_count)`, and offline graded relevance is the frozen
candidate `C`. This NDCG is a local election diagnostic, not a claim of official
test performance.

- Evidence gate: `H/S - G/F >= +0.015 C`, paired 95% lower bound above zero,
  at least 95% localized construction coverage, and non-negative point deltas
  for METEOR, BERTScore F1, and STS.
- Pool-A election gate: elected top-1 minus 4B-current `>= +0.010 C`, paired
  lower bound above zero, at least 30% of Pool A's reference-aware oracle gain,
  and no component regression greater than `0.002`.
- If both pass, build locator → multipass answerer → election.
- If only evidence passes, prioritize a reference-blind locator and retain
  election as the ranked-output shell.
- If only election passes, scale deterministic evidence diversity and election
  without a complex locator first.
- If neither passes, stop multiplying weak passes and test fold-based
  fine-tuning or a materially stronger/native-audio answerer.

No efficacy early stop is allowed. Operationally stop only when the canary
fails, full selector schema failure exceeds 5% after one retry, any answer cell
is incomplete, provenance checks fail, or the predeclared spend/wall cap is
reached.

## Canary, budgets, and artifacts

The ten-row canary is stratified by the frozen question-only rules: temporal,
counting, audio/text, causal, OCR/detail, and other. It checks only structured
output, interval ranges, transcript indices, frame extraction, request leakage,
model-input identity, and end-to-end completion. It may not compute or reveal
treatment metrics. All ten rows must validate after at most one schema retry.

Caps for the complete diagnostic are 500 primary selector calls, 500 selector
retry calls only on malformed rows, 1,500 judge calls across pools, 12 aggregate
GPU-hours, USD 30 provider cost, and 24 hours wall time after the canary starts.

Each row retains selected intervals, frame requested timestamps and actual PTS,
frame hashes and extraction identity, contact sheets, transcript source and
selected line indices, exact request hashes, answers, ranking, runtime identity,
and completion status. A safe tracked artifact contains only aggregate metrics,
confidence intervals, counts, model/runtime identities, panel/evidence digests,
and private evidence URIs/checksums.

## Verification

Pure contracts are test-driven. Before launch, verify the shared data symlink,
panel digest, source/runtime pins, recursive label rejection in answer and judge
requests, output schemas, retry bounds, frame uniqueness/order, transcript
verbatim identity, candidate randomization, no overwrite, and private-path
ignore rules. Every remote payload is checksum-verified after retrieval. The
scorer independently recomputes row means, paired effects, bootstrap intervals,
gate booleans, and NDCG arithmetic.
