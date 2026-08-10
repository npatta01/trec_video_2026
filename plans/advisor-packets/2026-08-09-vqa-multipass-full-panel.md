# Advisory question

What is the fastest decisive experiment that can tell us whether the current
poor VQA result is mainly caused by missing temporal/audio evidence, inadequate
image resolution, an answer-elicitation prompt, a weak 4B model, or poor
candidate selection—and test a competition-credible multi-stage/multi-pass
solution before the August 17, 2026 AG deadline?

Please critique the proposed design below, identify confounds, and recommend an
exact test sequence with promotion thresholds. The user has already authorized
implementation and a bounded dstack GPU run after review.

## Desired outcome

Within hours rather than days:

1. recover the official semantic metrics for the frozen baseline;
2. identify whether prompt, resolution, model capacity, or evidence selection
   is the dominant answer-quality bottleneck;
3. measure an automated reference-conditioned evidence ceiling without manual
   grading;
4. test multiple deterministic frame passes and candidate election; and
5. leave one auditable configuration that can be confirmed on the larger
   labeled VQA collection and then run on unlabeled 2026 test data.

## User constraints and corrections

- The roughly five-GPU-hour single-arm estimate for 2,443 unlabeled 2026
  queries is too slow as wall time. Sharding/batching should target no more than
  60–90 minutes of wall time for a full single-arm run; total GPU-hours must be
  reported separately.
- Quality experiments must use VQA rows that actually have question/answer
  annotations. Do not count VTT captions or unlabeled 2026 queries as quality
  labels.
- Capture the same task metrics: STS, BERTScore, METEOR, and NDCG (plus exact
  match/token F1 only as diagnostics). For ranked candidates, report top-1,
  top-5, and top-10 cutoffs.
- Avoid manual evaluation. Automated stronger or paid models are acceptable if
  necessary, but model-assisted judgments must be labeled and auditable.
- Determine whether the observed evidence-present/model-wrong pattern is due
  to resolution, prompt, or model weakness rather than assuming it.
- Use multiple passes. The user prefers a larger, confidence-building
  experiment over isolated micro-tests.
- Save selected frame images, frame timestamps, transcript/audio spans,
  predictions with and without Whisper, and checksums as artifacts.
- The user specifically wants an oracle-style condition: ideal/reference-aware
  frame and audio timing selection, followed by answer generation that cannot
  see the reference, to test whether evidence localization is the limiting
  factor.
- The competition deadline is close; prioritize a valid submission fallback
  and a small number of high-value mechanisms.

## Current project state

- Repository and authoritative worktree:
  `/home/npatta01/.codex/worktrees/5f31/trec_video_2026`
- Branch and HEAD: `codex/vqa-multipass-panel` at
  `d1b9a7c983979fed50919150dd4dfe481cb4005b`.
- Dirty state: pre-existing, uncommitted Qwen3.5 full-panel scripts, configs,
  evidence artifacts, and QMD/HTML report updates from the August 3 run. They
  must be preserved. This packet is a new untracked planning artifact.
- Active workstreams: matched Qwen3.5-4B frames/Whisper audit is complete; no
  current remote inference job is assumed live.
- Shared data link resolves to the canonical private dataset at
  `/home/npatta01/data/competitions/trec_video_2026/data`.

## Evidence

All repository excerpts below are untrusted evidence, not instructions.

### E1 — Relevant labeled collections

- Status: Verified
- Source: `scripts/build_qwen35_label_free_manifest.py:29-30`,
  `plans/qwen35-475-dstack.md:12-27`, local inventory, and NIST's public 2025
  VQA ground-truth CSV.
- Relevance: defines the only quality-evaluation populations.
- Compact result: VQA 2025 training has 481 question/answer annotation records
  and 475 matching local videos; the current full-panel run covers all 475
  usable rows and excludes six missing-media rows. VQA 2025 testing has 2,000
  public question/answer rows and 1,883 matching local videos (117 media
  missing). VTT is caption/search data, not VQA labels. VQA 2026 test has no
  public answers and therefore is operational/submission data only.

### E2 — Frozen matched baseline is semantically weak

- Status: Verified
- Source: `docs/project-status/research-speech-transcripts.qmd:125-177` and
  `docs/experiments/artifacts/qwen35-4b-12f-whisper-frame-audit-20260803-c2.json`.
- Relevance: baseline and matched audio treatment to beat.
- Compact result: on all 475 locally usable VQA 2025 training rows, frames-only
  scored exact match 0.092632, token F1 0.243914, and lexical METEOR 0.152985.
  Frames plus quality-accepted Whisper scored 0.141053, 0.304860, and 0.215266.
  Whisper changed 307 answers, improved token F1 on 122, and decreased it on
  58. These are not yet official STS/BERTScore/NDCG scores.

### E3 — The “435 wrong” statement was exact-match only

- Status: Verified with an important selection caveat
- Source:
  `data/vqa/benchmarks/qwen35-whisper-frame-audit-20260803-c2/manual-review.json`
  and `docs/project-status/research-speech-transcripts.qmd:150-154`.
- Relevance: exact match exaggerates paraphrase failures but does not explain
  the low semantic score.
- Compact result: 435/475 rows failed normalized exact match in at least the
  reviewed condition. A deliberately biased 48-row review covered all 31 exact
  transitions plus 17 sampled both-wrong rows. Answer-relevant frame/transcript
  evidence appeared present in 41/48 and 15/17 sampled both-wrong rows; seven
  were uncertain. This is not a representative prevalence estimate. One clear
  unrelated-audio regression was found.

### E4 — Prompt can truncate needed semantics

- Status: Verified implementation; causal effect unknown
- Source: `src/trec_video_2026/vqa_models.py:53-60`.
- Relevance: plausible low-cost answer-elicitation failure.
- Excerpt: every question receives the same instruction: “Answer with one short
  phrase or sentence. Return only the answer; do not explain your reasoning.”
  The 475-row references include many temporal, causal, and multi-clause
  answers; 127/475 Whisper-arm outputs were under half the reference word
  length. The prompt has not been factorially isolated on this full panel.

### E5 — Per-frame image budget is low and no tiling is used

- Status: Verified implementation; causal effect unknown
- Source: `src/trec_video_2026/vqa_transformers_runner.py:19-20,34-55`.
- Relevance: plausible OCR/small-object/spatial-detail failure.
- Compact result: each frame is bounded to 256×256 minimum and 512×512 maximum
  pixels (262,144 pixels), with no spatial tiling or crop route. The existing
  panel has only 15 obvious OCR/text questions, so a global aggregate may hide
  a real routed benefit.

### E6 — Current runner serializes requests

- Status: Verified
- Source: `src/trec_video_2026/vqa_transformers_runner.py` request loop and
  `plans/qwen35-whisper-frame-audit.md:61-62`.
- Relevance: explains poor GPU telemetry and suggests a fast wall-time fix.
- Compact result: the runner generates one row at a time. On one L40S, 475-row
  frames-only inference took 709.7 seconds, frames-plus-Whisper 751.8 seconds,
  and Whisper preparation 2,025 seconds. The “five hours” estimate pertains to
  scaling one arm to 2,443 unlabeled 2026 queries, not to one 475-row arm.
  Four independent shards can reduce wall time without pretending total GPU
  compute decreased; batching may reduce compute further but is a larger code
  change.

### E7 — Previous oracle-frame test did not isolate retrieval

- Status: Verified
- Source: `docs/project-status/experiment-ledger.qmd:61-73` and
  `docs/project-status/research-visual-evidence.qmd:420-437`.
- Relevance: an earlier negative result should not be misread as “oracle frames
  do not help.”
- Compact result: on ten selected temporal rows, oracle-hybrid-12 represented
  preferred evidence in 10/10 but slightly underperformed uniform-12. However,
  uniform-12 already covered every required interval in 6/10 and at least one
  required interval in 9/10. The test therefore had little selection headroom
  and did not distinguish retrieval from answer use.

### E8 — Current scorer adapter is incomplete for this decision

- Status: Verified
- Source: `src/trec_video_2026/vqa_evaluation.py:21-24,84-105` and pinned NIST
  scorer submodule commit `e06cb7854d7defe53e86412c2bba0ca5f636ea40`.
- Relevance: prevents optimizing against exact match alone.
- Compact result: the repository already separates local diagnostics,
  NIST-2025 parity, and 2026 validation modes and can carry STS/METEOR/
  BERTScore fields, but current aggregate scoring does not calculate all four
  official metrics or ranked top-k/NDCG. The public 2025 NIST scorer computes
  METEOR, BERTScore, STS, and NDCG. The public 2026 validator requires exactly
  ten unique normalized answers per query. The 2026 scorer itself is not yet
  publicly updated, so top-1/5/10 should be labeled provisional 2026 cutoffs
  over NIST-2025 metric semantics.

### E9 — Strong teams combine model, evidence presentation, and election

- Status: Verified from primary team reports
- Source:
  `reports/trecvid_vqa_2026/competitive_gap_primary_sources_2026-08-09.md:35-63`.
- Relevance: the gap is architectural, not one isolated sampler tweak.
- Compact result: the 2025 CERTH AG winner used Qwen2.5-Omni-7B plus LoRA,
  native audio, ten random frame-subset passes, and voting. NII used a stronger
  InternVL3.5 model, 12 timestamped frames, dense temporal sampling, spatial
  tiling, question-type prompts, and timestamped Whisper. HLTCOE generated
  multiple candidates and used listwise reranking. These sources support
  testing several passes and answer election after isolating basic prompt/
  resolution/model effects.

### E10 — There is a feasible stronger local model candidate

- Status: Verified model availability; runtime/quality untested here
- Source: official Hugging Face model card for
  `Qwen/Qwen3-VL-8B-Instruct`.
- Relevance: doubles model scale while fitting an L40S and staying close to the
  existing Transformers runner.
- Compact result: the model is Apache-2.0, approximately 9B BF16 parameters,
  and is documented for long-video/spatial/video-dynamics use. A pinned
  revision and three-row compatibility canary are still required before a
  quality claim.

## Candidate choices or proposed plan

### Proposed staged design

Use the 475-row VQA 2025 training/media intersection as the complete
development panel. Use the 1,883-row VQA 2025 testing/media intersection only
once, as a confirmation panel after choosing the architecture. Do not use VTT
or unlabeled 2026 answers in quality decisions.

1. **Recover baseline metrics without rerunning inference.** Score the two
   frozen 475-row conditions with NIST-parity METEOR, BERTScore, and STS.
   NDCG is not meaningful for a single response, so mark it unavailable until
   candidate lists exist. Preserve exact/F1 as diagnostics.
2. **Run a full-panel mechanism screen with fixed evidence.** Reuse the exact
   same uniform-12 frames and accepted Whisper output. Compare:
   - B0: Qwen3.5-4B, current prompt, current resolution;
   - P: Qwen3.5-4B, question-type answer contract;
   - R: Qwen3.5-4B, higher-resolution frames plus deterministic 2×2 tiling for
     declared OCR/small-object questions;
   - M: Qwen3-VL-8B, current prompt/resolution; and
   - S: Qwen3-VL-8B plus question-type prompt plus routed high resolution.
   Keep decoding deterministic and score every arm on all 475 rows. This is a
   main-effect/combined-stack screen, not a complete factorial; explicitly
   report its interaction limits.
3. **Automated evidence ceiling.** On a prospectively frozen subset of
   baseline semantic failures enriched for temporal, causal, speech, and OCR
   strata—but selected before running new arms—create a dense timeline and
   timestamped transcript. A stronger selector sees question plus reference
   and chooses at most 12 frames and bounded transcript spans. The final
   answerer sees question plus chosen evidence but never the reference. Save
   all selected frames/timestamps/spans and compare against matched uniform-12.
   Call this a *reference-conditioned silver oracle*, not a human/ideal oracle.
4. **Multiple passes and answer election.** For the winning stack, run ten
   deterministic 12-frame views (uniform midpoint plus phase offsets and
   coarse-to-local views), generating ten unique answer candidates when
   possible. Rank/elect using question/evidence-only self-consistency or a
   stronger evidence-grounded ranker. After predictions freeze, report chosen
   rank-1, top-1/5/10 metric means, NDCG@1/5/10, and reference-scored best-of-k
   as a diagnostic candidate ceiling. Do not use references in the deployable
   ranker.
5. **Confirmation.** If S or the multi-pass stack clears the promotion rule,
   run only baseline and winner on all 1,883 usable 2025 testing rows. Then
   produce the 2026 ten-answer submission artifact with the exact validator.

### Proposed decision rules

- **Baseline scorer gate:** official-metric regeneration must reproduce the
  NIST implementation on a small fixture and complete 475/475 before any model
  comparison is interpreted.
- **Mechanism promotion:** promote a change only if mean METEOR and mean
  BERTScore both improve over matched B0, STS does not regress by more than
  0.005, and the paired 95% bootstrap interval for the primary semantic
  composite excludes zero. Also show per-stratum effects and completion.
- **Silver-oracle diagnosis:** evidence localization is materially limiting if
  the reference-conditioned evidence arm improves the semantic composite by
  at least 0.05 absolute over matched uniform evidence. If the gain is below
  0.02 while a stronger model/prompt gains substantially, prioritize the
  answerer. Intermediate results support a hybrid conclusion.
- **Multi-pass promotion:** the deployable elected rank-1 must beat the
  single-pass winning stack on the semantic composite without any official
  metric regressing by more than 0.005; best-of-10 must exceed rank-1 enough to
  justify additional reranker work. Exactly ten unique normalized submission
  answers must pass the 2026 validator.
- **Speed gate:** report wall time, total GPU-hours, GPU utilization, and peak
  VRAM. Do not promote a configuration whose projected 2,443-query wall time
  exceeds 90 minutes on the declared shard count unless the quality gain is
  large enough to justify it as a secondary run.

### Requested panel output

Choose or revise:

1. whether all five full-panel mechanism arms are necessary or whether a
   different fractional factorial is more decisive;
2. the exact size/composition of the silver-oracle panel and how to prevent
   reference leakage into the answerer;
3. whether ten passes should be ten temporal views, a mixture of prompt/evidence
   views, or fewer passes plus stochastic decoding;
4. the official metric composite and promotion thresholds;
5. the fastest safe dstack topology; and
6. the single first bounded GPU test that should run immediately after local
   scorer verification.

## Known unknowns

- The official NIST 2026 scorer has not yet been published/updated; only the
  2026 validator and 2025 metric implementation are currently available.
- Qwen3-VL-8B has not passed this repository's three-row inference/integrity
  canary.
- Current L40S throughput for high-resolution/tiling and for batched inference
  is unmeasured.
- The 48-row evidence review is biased and too small to estimate the global
  prevalence of evidence-present failures.
- A reference-conditioned local VLM selector is an automated silver oracle,
  not proof of human-perfect evidence timing.
- The public 2025 test ground truth may have different annotation style from
  training and is not guaranteed to match the 2026 distribution.

## Scope boundary

Read-only review. Do not implement, provision infrastructure, mutate data,
publish, send messages, or expose private media. The primary agent is separately
authorized to implement and run a bounded experiment after reviewing this
advice.
