# Advisory question

Given the complete 475-row annotated-development factorial below, what is the
highest-confidence next experiment and near-term competition architecture?
Specifically, should the project prioritize an automated reference-assisted
oracle evidence/timing ceiling, multiple evidence passes plus answer election,
or another multistage video-understanding design? Critique the proposed
120-row oracle-plus-election test and make it concrete enough to execute now.

## Desired outcome

Choose one decisive, bounded experiment that can start immediately, identifies
whether the remaining gap is evidence selection versus answer selection, and
directly informs an August 17 AG submission architecture. State the exact
panel, arms, evidence budgets, selector/answerer roles, metrics, leakage
boundary, stopping rule, and what should run on the full 475 rows afterward.
Prefer a larger, confidence-producing bet over another tiny anecdotal probe.

## User constraints and corrections

- Only VQA rows with question/answer annotations are relevant to answer-quality
  evaluation. The official 2025 training collection has 481 labels and local
  media for 475; those 475 are the current development panel. VTT files are
  captions, not VQA answer labels, and must not be counted as labeled VQA.
- The locally available 1,883 labeled 2025 test rows are reserved for one-shot
  confirmation after the architecture freezes, not iterative tuning.
- Evaluate with METEOR, BERTScore F1, STS, and ranked NDCG semantics. Exact
  match is only a diagnostic; it previously made 435 rows look categorically
  wrong when many were paraphrases or formatting variants.
- The user wants outputs both with and without Whisper, and selected frame
  timestamps/hashes/contact sheets/audio provenance to remain an artifact.
- The user explicitly wants to know the answer ceiling under ideal frames and
  ideal audio timing. A selector may use a stronger or paid model if needed.
  Manual evaluation is strongly disfavored.
- Evidence selection and final answering must be separated. For a diagnostic
  oracle arm, references may be used by the offline selector only if the final
  answerer receives frames/verbatim transcript spans and never receives the
  reference, selector rationale, generated caption, or answer hint.
- The user wants multiple passes and a practical multistage architecture, not
  only a collection of tiny disconnected tests.
- Remote GPU work should run in parallel. A prior five-GPU-hour serial
  projection was rejected as too slow.
- Competition urgency is high: current date 2026-08-09; tentative AG deadline
  is 2026-08-17. Review is not the terminal deliverable: a bounded test should
  be implemented and run after the decision.
- Keep data, raw predictions, media, transcripts, and logs private/ignored.
  Do not publish or change sharing permissions.

## Current project state

- Repository and authoritative worktree:
  `/home/npatta01/.codex/worktrees/5f31/trec_video_2026`
- Branch and HEAD: `codex/vqa-multipass-panel` at `2dd8975`.
- Dirty state: pre-existing user QMD/generated-report changes and untracked
  historical experiment files coexist with one current plan edit and the new
  sanitized factorial artifact. Selective staging is required.
- Active workstreams: the 475-row factorial is complete and scored; the next
  oracle/multipass experiment has not yet been implemented or launched.

## Evidence

### E1 — Annotated panel identity

- Status: Verified
- Source: `docs/experiments/artifacts/vqa-factorial-475-20260809-f2.json`
  (`panel`, SHA-256
  `614134334b5c9947c177549ad158f6a8c1e6032c13a5163cf73b700fe00081c5`)
- Relevance: establishes the only current quality panel and denominator.
- Excerpt or compact result: 481 annotation files; 475 complete media-backed
  rows in every condition; six declared missing-media IDs; canonical ID digest
  `95a623c0add252541b9c81ff4910e949dc81db17482add2c467b761237b22b0f`.

### E2 — Full with/without-Whisper baseline

- Status: Verified
- Source:
  `docs/experiments/artifacts/qwen35-4b-12f-whisper-semantic-20260809-d1.json`
- Relevance: establishes that transcript evidence helps under the same 475-row
  panel and semantic scorer.
- Excerpt or compact result:

  | Condition | METEOR | BERTScore F1 | STS | Composite C |
  |---|---:|---:|---:|---:|
  | Qwen3.5-4B, uniform 12 frames | 0.1866 | 0.9029 | 0.3236 | 0.4711 |
  | Same plus accepted Whisper | 0.2311 | 0.9110 | 0.3467 | 0.4962 |

  Whisper delta C was +0.0252 with paired bootstrap 95% interval
  `[+0.0144, +0.0371]`. METEOR, BERTScore, and STS all improved. Single-answer
  NDCG is 1.0 by definition and non-informative.

### E3 — Complete matched 2x2 factorial

- Status: Verified
- Source:
  `docs/experiments/artifacts/vqa-factorial-475-20260809-f2.json`
- Relevance: isolates model family and answer-policy effects while holding all
  12 frames, timestamps, accepted Whisper bytes/nulls, pixel bounds, decoding,
  and panel rows fixed.
- Excerpt or compact result:

  | Model / policy | METEOR | BERTScore F1 | STS | C | Mean tokens |
  |---|---:|---:|---:|---:|---:|
  | Qwen3.5-4B / current | 0.2336 | 0.9115 | 0.3486 | **0.4979** | 4.47 |
  | Qwen3.5-4B / question-aware | **0.2405** | 0.9073 | 0.3459 | 0.4979 | 5.05 |
  | Qwen3-VL-8B / current | 0.2152 | 0.9045 | 0.3392 | 0.4863 | 4.55 |
  | Qwen3-VL-8B / question-aware | 0.2286 | 0.9026 | 0.3476 | 0.4929 | 6.29 |

  All four arms completed 475/475 and share evidence checksum
  `b7610cbdf5716269b06b7d66fdf81376873cb8c98f24a9aba25a502f9176926c`.
  No treatment passed the preregistered promotion gate. The 4B current arm is
  the frozen answerer for the next experiment.

### E4 — Factorial mechanism effects

- Status: Verified
- Source: same artifact, `factorial_effects` and `decision`.
- Relevance: distinguishes a weak-model hypothesis from prompt/evidence gaps.
- Excerpt or compact result: averaged over policy, changing 4B to 8B produced
  delta C `-0.0083` (95% interval `[-0.0200, +0.0021]`), METEOR `-0.0152`, and
  BERTScore `-0.0059`. Averaged over model, the question-aware policy produced
  delta C `+0.0033` (95% interval `[-0.0020, +0.0085]`), METEOR `+0.0101`, and
  BERTScore `-0.0030`. The interaction was `+0.0067` C with an interval crossing
  zero. This rejects a blanket “just use the larger model” or “just use the
  type prompt” answer under fixed evidence.

### E5 — Strong candidate complementarity / answer-election ceiling

- Status: Verified diagnostic; not deployable because selection uses references
- Source: deterministic recomputation from the four per-row numeric arrays in
  `docs/experiments/artifacts/vqa-factorial-475-20260809-f2.json`.
- Relevance: measures whether multiple saved candidates contain enough
  complementary correct content to justify election/reranking.
- Excerpt or compact result: choosing the highest-C candidate per row among
  the four frozen answers yields METEOR `0.2973`, BERTScore `0.9193`, STS
  `0.4356`, and C `0.5507`; delta C versus 4B current is `+0.0528`. Selection
  counts are 4B-current 261, 4B-question-aware 45, 8B-current 109, and
  8B-question-aware 60. Even the two-candidate oracle between 4B-current and
  8B-question-aware reaches C `0.5358` (`+0.0378`). This is an upper bound on a
  reference-blind election stage, not evidence that a deployable judge already
  achieves it.

### E6 — Coarse question-type routing is modest

- Status: Inference / post-hoc exploratory
- Source: same per-row arrays, selecting the aggregate-best frozen condition
  separately for audio/text, causal, counting, other, and temporal strata.
- Relevance: tests whether a simple question-only router captures the oracle
  complementarity.
- Excerpt or compact result: post-hoc stratum routing gives METEOR `0.2484`,
  BERTScore `0.9099`, STS `0.3511`, and C `0.5031`, only `+0.0052` C versus
  4B-current. Coarse question type captures little of the per-row +0.0528
  ceiling; evidence-aware election or finer confidence signals appear needed.

### E7 — Corrected interpretation of the 48-row evidence review

- Status: Verified for a selected diagnostic sample; not prevalence evidence
- Source:
  `data/vqa/benchmarks/qwen35-whisper-frame-audit-20260803-c2/manual-review.json`
  joined to the semantic artifact in E2.
- Relevance: corrects the earlier claim that “evidence present in 41/48” means
  41 genuine answerer failures.
- Excerpt or compact result: reviewers marked answer-relevant evidence present
  in 41/48 selected transition rows, but only 3 of those 41 had both answers
  below C 0.40; 22 had at least one answer at or above C 0.60. Fifteen of 17
  exact-match-both-wrong rows had evidence present, showing that exact match
  inflated failure counts. The sample still contains three clear low-semantic-
  score evidence-present cases, but it does not establish their population
  frequency.

### E8 — Resolution evidence is currently insufficient

- Status: Verified canary result plus unknown population effect
- Source: `docs/superpowers/plans/2026-08-09-vqa-factorial-pilot.md`
  (Task 4 completion record).
- Relevance: the user asked whether evidence-present failures are resolution,
  prompt, or model weakness.
- Excerpt or compact result: one high-resolution OCR canary remained correct
  and unchanged; the temporal canary failed despite relevant global frames and
  transcript evidence. The full 2x2 held resolution fixed, so it rules against
  the tested blanket model/prompt changes but does not isolate a population
  resolution effect. Resolution/detail must be a routed arm or separate axis,
  not inferred from the factorial.

### E9 — Operational timing and utilization

- Status: Verified
- Source: factorial artifact `operational_evidence`; dstack terminal records.
- Relevance: the next test must fit the deadline and avoid the rejected serial
  five-hour projection.
- Excerpt or compact result: answer-pass wall time was 562 s (4B current),
  816 s (4B question-aware), 752 s (8B current), and 754 s (8B question-aware).
  With four GPUs, answer inference was bounded by 13.6 minutes rather than the
  sum of four arms. Total measured answer GPU time was 0.801 GPU-hours. The
  verified evidence/Whisper build is reusable. Provisioning, model loading,
  and Hub transfer dominated end-to-end wall time; local bundles now hardlink
  the identical evidence subtree and retain full checksum verification.

### E10 — Relevant competitive mechanisms

- Status: Verified primary-source synthesis; 2026 opponent systems unknown
- Source:
  `reports/trecvid_vqa_2026/competitive_gap_primary_sources_2026-08-09.md`.
- Relevance: identifies what the current one-pass fixed-view system lacks.
- Excerpt or compact result: the strongest 2025 AG system, CERTH-ITI, fine-
  tuned Qwen2.5-Omni-7B on the 500 TREC examples and generated ten greedy
  predictions from different random frame subsets, then ranked by vote. NII_UIT
  used 12 frames, spatial tiling, question-type prompts, and timestamped
  Whisper. HLTCOE generated multiple candidates and learned a listwise
  reranker. The current project has accepted Whisper and 12-frame artifacts but
  no local coarse-to-fine locator, no ten-view test-time scaling, and no
  validated answer election stage. The 2025 winner's reported METEOR was about
  0.260; current best full-panel METEOR is 0.2405 under the question-aware arm
  and 0.2336 under the best composite arm.

### E11 — Prior manual temporal-oracle attempt is not reusable as a result

- Status: Verified stopped experiment
- Source:
  `data/vqa/benchmarks/temporal-required-moment-oracle-10/agreement-stop-decision.json`.
- Relevance: explains why there is still no oracle evidence result despite
  substantial existing interval/frame-planning code.
- Excerpt or compact result: a ten-row manual required-moment study stopped
  before frame extraction/inference because two reviewers' median temporal IoU
  was 0.4792 versus a preregistered 0.50 threshold. It therefore supplies no
  oracle answer ceiling. Repeating a labor-intensive manual labeling protocol
  conflicts with the user's request; a new automated reference-assisted
  diagnostic must be clearly labeled as an oracle/silver-oracle ceiling.

## Candidate choices or proposed plan

- **Choice A — 120-row reference-assisted oracle evidence 2x3.** Freeze a
  stratified 120-row sample from the 475. A strong selector sees full video,
  question, and reference and returns only one or two sufficient time
  intervals plus verbatim Whisper segment IDs. Construct three 12-frame views:
  global uniform; 6-global/6-oracle-local hybrid; and 12-oracle-local. Pair
  each with full accepted Whisper versus oracle-selected verbatim transcript
  span/null. Run only the frozen 4B-current answerer without references.
  Measure selector construction coverage and answer metrics. This directly
  estimates evidence/timing headroom but does not test a deployable selector.
- **Choice B — Candidate election first.** Use the four already frozen answers
  plus evidence packets on all 475 rows. Ask a stronger reference-blind judge
  to rank semantically deduplicated candidates based on question/evidence,
  then compare its top choice to 4B-current and to the reference-aware oracle.
  This is fast and attacks the measured +0.0528 ceiling, but may not diagnose
  missing evidence because all candidates share the same global evidence.
- **Choice C — Full ten-view multipass now.** Generate ten deterministic
  frame-subset answers per row with the 4B-current answerer, then vote or use an
  evidence-aware reranker. This most closely mirrors the 2025 winner, but a
  blind launch risks scaling ten weak global views if temporal evidence is the
  true bottleneck.
- **Proposed combined test for criticism:** run Choice A on 120 rows, but
  preserve every arm's generated answer as a candidate. Report (1) global
  one-pass, (2) oracle-evidence one-pass, (3) reference-blind election across
  the six evidence candidates, and (4) reference-aware candidate oracle. If
  oracle evidence adds at least +0.015 C with positive lower bootstrap bound,
  build a deployable question-conditioned locator. If evidence adds little but
  reference-blind election captures at least 30% of the candidate-oracle gain
  without metric regressions, scale multipass/election to 475. If neither
  passes, move to a stronger/native-audio answerer or fine-tuning instead of
  multiplying passes.

## Known unknowns

- No local experiment yet measures selector recall, oracle-evidence answer
  quality, and end-to-end predicted-selector quality under one protocol.
- The ideal-resolution effect is unknown beyond one already-correct OCR
  canary. The fixed 262,144-pixel ceiling may still miss small text/detail.
- Accepted Whisper is text, not native audio. Music, sound events, prosody,
  speaker identity, and impact sounds do not have an ideal-audio ceiling.
- A reference-aware candidate oracle demonstrates complementarity, not a
  deployable reranker. The attainable fraction under a reference-blind judge
  is unknown.
- The 120-row sample size, exact stratification, strong selector/judge model,
  cost cap, and interval schema need to be frozen before implementation.
- The current coarse lexical strata overlap and are not an evaluated router.
- The 2025 labeled test set remains untouched for confirmation, so all current
  decisions may overfit the 475 training rows.
- The official AG submission needs ten ranked answers; one-answer local NDCG
  cannot validate final list ranking.

## Scope boundary

- Read-only review. No implementation or external actions.
