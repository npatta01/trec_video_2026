# Qwen ASR plus Forced-Aligner Quality Diagnostics


Date: 2026-08-16

Primary theme: **Speech and Transcript Evidence**

Historical Stage 0.5d disposition: **Not ready for comparison**

Stage 0.5e mechanism disposition: **Ready for a fresh aligned-English
quality panel**

Stage 0.5f quality disposition: **Not promoted under the tested
condition**

## Plain-English result

The original Stage 0.5d run was one fresh eight-video English speech
test of Qwen3-ASR plus the Qwen forced aligner. The container itself
worked, but Qwen returned no transcript for two videos. Six videos
produced valid aligned output. Because a fair comparison required all
eight rows, that test remains recorded as `not_ready`.

This does **not** show that Qwen is better or worse than Whisper. The
paired quality score was intentionally not created when two rows failed.

The completed Stage 0.5f follow-up forced English on 12 new
question-selected speech rows. It passed the operational gate with
`12/12` completed and `12/12` valid aligned outputs, so the frozen
quality inputs were opened. The predeclared exact official-answer-phrase
metric was `0/12` for Qwen and `0/12` for Whisper, with all 12 rows in
the paired neither-present cell. The forced-English condition completed
without the observed empty-output problem on this panel, but it produced
no directional quality signal and does not advance Qwen.

## Frozen condition

- Eight fresh English exact-speech WAV rows from the preserved oracle
  panel.
- Prior Qwen rows were excluded before preparation.
- Requests contained only run/condition/video identity, an absolute WAV
  path, and `mode=transcription`.
- ASR revision: `7278e1e70fe206f11671096ffdd38061171dd6e5`.
- Forced-aligner revision: `c7cbfc2048c462b0d63a45797104fc9db3ad62b7`.
- Digest-pinned offline ROCm container, read-only mounts, and one
  attempt.
- Paid cost: `$0.00`.

## Observed result

| Measure                  |              Result |
|--------------------------|--------------------:|
| Required rows            |                   8 |
| Completed rows           |                   6 |
| Valid aligned rows       |                   6 |
| Empty-output failures    |                   2 |
| Container exit code      |                   0 |
| Maximum allocated memory | 6,161,528,320 bytes |
| Container elapsed time   |      64.105 seconds |
| Frozen decision          |         `not_ready` |

The two failures were deterministic `Qwen3-ASR returned empty text`
runner errors. The raw output, terminal record, validation, and
prediction freeze were preserved without retry or relabeling.

## Interpretation boundary

This is a zero-cost incomplete-output diagnostic. It establishes that
the corrected harness can process multiple rows, but not that the model
has better transcript fidelity, better timestamps, downstream VQA value,
or replacement value relative to Whisper. A future attempt would need a
separately frozen panel and an explicit resolution for empty Qwen
output.

## Evidence

- Private run root:
  `data/vqa/benchmarks/model-verifier-canaries-20260814/stage05d-qwen-aligner-quality-v1/`
- Sanitized aggregate:
  [`qwen-aligner-quality-20260816/`](artifacts/qwen-aligner-quality-20260816/)
- Harness:
  [`vqa_qwen_aligner_canary.py`](../../src/trec_video_2026/vqa_qwen_aligner_canary.py)
- Plan:
  [`2026-08-16-qwen-aligner-quality-test.md`](../superpowers/plans/2026-08-16-qwen-aligner-quality-test.md)

## Stage 0.5e empty-output mechanism diagnostic

Date: 2026-08-16

### Question and decision rule

The follow-up tested whether the two preserved empty outputs were
reproduced by the aligned automatic-language path and whether forcing
`language="English"` removed them under the same pinned Qwen models and
offline runtime. The predeclared decision rule was: if the aligned
automatic-language baseline reproduces both failures and the
aligned-English treatment completes all four rows with valid alignment,
advance only a fresh quality panel using that English-language
condition. The four diagnostic rows are never scored.

### Frozen method and panel

- Four rows selected from the frozen Stage 0.5d results in request
  order: the two preserved empty-output failures and the first two
  preserved passing controls.
- Four process-isolated arms ran once, in order, with read-only mounts,
  `--network none`, pinned model revisions, and `$0.00` paid compute.
- Qwen3-ASR revision: `7278e1e70fe206f11671096ffdd38061171dd6e5`;
  forced-aligner revision: `c7cbfc2048c462b0d63a45797104fc9db3ad62b7`.
- The diagnostic froze before any labels, annotations, Whisper text, or
  scoring inputs were opened. Row-level Qwen outputs remained private
  and were never manually inspected or scored, but deterministic
  validation and the original host selector did programmatically read
  transcript/alignment fields as disclosed below.

### Post-run protocol correction

Independent review found a confirmed protocol deviation in the original
host selector: for rows already marked `passed`, it redundantly read and
stripped the private `text`, `language`, and `chunks` fields. The
predeclared design allowed selection from status, error, order,
identity, duration, and hash metadata only. The selector did not copy
those values into the diagnostic, and labels, annotations, Whisper text,
and scoring inputs remained unopened.

This deviation did not change the frozen panel or the aggregate
mechanism result. Stage 0.5d’s separately frozen validation records six
completed rows and six valid alignments, so every one of its six
`passed` rows had already met the transcript/alignment contract; the
first two passed statuses in frozen order are therefore the same two
controls. Neither frozen experiment was rerun. The host harness now
cross-checks only passed/failed statuses against those frozen
completion/alignment counts, and a regression test fails if selection
accesses any private transcript field.

### Observed result

| Arm | Condition | Successes | Failures | Valid aligned | Validation | Wall seconds |
|----|----|---:|---:|---:|----|---:|
| A `aligned_auto` | aligner, automatic language, timestamps on | 2/4 | 2 | 2/4 | `not_ready` | 46.722 |
| B `base_auto` | no aligner, automatic language, timestamps off | 2/4 | 2 | 0/4 | `not_ready` | 46.691 |
| C `aligner_plain_auto` | aligner, automatic language, timestamps off | 2/4 | 2 | 0/4 | `not_ready` | 46.294 |
| D `aligned_english` | aligner, English, timestamps on | 4/4 | 0 | 4/4 | `ready` | 47.787 |

All four arm terminals exited `0` and were contract-valid. Maximum peak
memory was 6,175,102,976 bytes for A, 4,331,441,664 for B, 6,175,102,464
for C, and 6,174,173,184 for D. The frozen status is
`diagnostic_frozen`; baseline reproducibility is `true`; the advancing
arm is `aligned_english`; and the decision is
`ready_for_fresh_aligned_quality_panel`.

### Interpretation boundary

On this four-row frozen mechanism panel, forcing `language="English"`
removed the two preserved empty-output failures. Each automatic-language
explanatory arm remained at `2` successes and `2` failures, and neither
restored all four rows. The bounded evidence therefore points to the
automatic-language path association rather than merely loading the
aligner or requesting timestamps; it is not an internal causal bug
claim. This is condition-bound mechanism evidence, not a
transcript-quality result, Whisper comparison, or downstream VQA result.
The four rows remain unscored, and the original Stage 0.5d summary
remains preserved in the evidence bundle.

### Authorized next action, now completed

Run a new label-blind aligned-English Qwen quality panel using no reused
Qwen rows. Hold the comparator, media, model revisions, prompts,
decoding, and all-row readiness gate fixed; load labels only after every
prediction freezes. At the Stage 0.5e freeze, this was the authorized
next action rather than a completed run. It was subsequently completed
as Stage 0.5f and is recorded below; the historical Stage 0.5e mechanism
decision remains unchanged.

### Evidence and immutable bindings

- Private diagnostic root:
  `data/vqa/benchmarks/model-verifier-canaries-20260814/stage05e-qwen-empty-output-diagnostic-v1/`
- Sanitized aggregate:
  [`qwen-aligner-quality-20260816/`](artifacts/qwen-aligner-quality-20260816/)
- New aggregate record:
  [`diagnostic-summary.json`](artifacts/qwen-aligner-quality-20260816/diagnostic-summary.json)
- Input-freeze SHA-256:
  `aed463e5c6f206e346c29b41b56b7199f07874981ffadc908d41974cf0466083`
- Diagnostic-freeze payload SHA-256:
  `d157e6dc2187ea14d7ead4f42f8ee2d8c14dddc8707acd314d1e7c4a02fe57ec`

## Stage 0.5f fresh aligned-English quality test

Date: 2026-08-16

### Question, hypothesis, and predeclared rule

The completed follow-up asked whether the pinned Qwen3-ASR plus
forced-aligner path would finish a fresh question-indicated speech panel
when every request fixed `language="English"`, and whether its
transcript contained the task’s complete official answer phrase more
often than the already frozen Whisper comparator. Any operational
failure would have frozen `not_ready` without opening score inputs.
After a `12/12` ready freeze, only a strictly higher Qwen containment
count would have authorized a larger confirmatory or downstream test; a
tie or lower count would stop this Qwen quality route.

### Frozen method and integrity boundary

- The selector used only the ordered 475-row question/identity manifest,
  WAV metadata, and a complete private Qwen3-ASR exposure census. It
  selected the first eligible fresh rows in fixed `6` reported-speech,
  `3` conversation, and `3` vocalization strata, restored to manifest
  order. It did not open annotations, answers, Whisper content or
  status, prior Qwen output content, maps, grades, or scores.
- The exposure census classified all `57` exact marker-bearing private
  JSON/JSONL files: `52` genuine exposure sources and five
  marker-bearing files within two fully inventoried preparation-only
  roots that had never begun an inference attempt. The ledger bound all
  six files in those roots, zero attempt artifacts, the preserved
  selection fingerprint, and `16` excluded prior/oracle identities. The
  fresh selector replayed the same 12-row private fingerprint before
  execution.
- The one attempt bound the Qwen3-ASR and forced-aligner revisions
  above, the digest-pinned offline ROCm image, BF16/SDPA, forced
  English, timestamps on, read-only mounts, `--network none`, source
  audio hashes/durations, committed controller revision
  `d9e43c94c2e441f5bc85508ed1b41d738e33289f`, and `$0.00` paid compute.
- Predictions, terminal evidence, validation, and their freeze were
  written before the scorer opened the canonical annotations and
  checksum-bound frozen Whisper artifact. Row identities, transcripts,
  answers, and row-level scores remain private.

### Observed result

| Measure                                  |                      Result |
|------------------------------------------|----------------------------:|
| Required rows                            |                          12 |
| Completed rows                           |                          12 |
| Valid aligned rows                       |                          12 |
| Failures                                 |                           0 |
| Container exit code                      |                           0 |
| Container wall time                      |             110.416 seconds |
| Summed row inference time                |             102.241 seconds |
| Maximum allocated memory                 |         6,162,299,392 bytes |
| Qwen complete official phrase present    |                        0/12 |
| Whisper complete official phrase present |                        0/12 |
| Both present                             |                           0 |
| Qwen only present                        |                           0 |
| Whisper only present                     |                           0 |
| Neither present                          |                          12 |
| Operational decision                     |    `ready_for_quality_test` |
| Quality decision                         | `no_directional_difference` |

### Interpretation, validity, and decision

- **Evidence validity:** **Valid but negative.** The sole attempt and
  all-row gate completed, and scoring occurred only after the verified
  prediction freeze.
- **Decision disposition:** **Not promoted under the tested condition.**
  The forced-English condition completed this fresh panel without an
  empty-output failure, but Qwen did not exceed Whisper on the
  predeclared containment metric.
- **What this rules out now:** the need for another same-condition
  completion diagnostic, and a claim that this exact test supplies a
  Qwen quality advantage.
- **What it does not establish:** WER, transcript fidelity, timestamp
  accuracy, general English robustness, Whisper replacement value,
  downstream VQA improvement, or model ranking. A `0`/`0` containment
  tie is non-discriminating and does not imply that both transcripts
  were equally accurate or equally inaccurate.
- **Uncertainty:** official VQA answers are not transcript references.
  Exact answer-phrase containment can be useful for a decision-bound
  retrieval signal, but all-neither output supplies no
  transcript-fidelity measurement.
- **Reconsider when:** a separately designed experiment has
  independently grounded transcript references or a frozen downstream
  VQA decision rule. Do not scale or rerun this same containment
  comparison.
- **Supersession:** this result completes the action authorized by Stage
  0.5e; it does not supersede the valid Stage 0.5d incomplete-output
  observation or the Stage 0.5e mechanism result.

### Evidence and immutable bindings

- Private run root:
  `data/vqa/benchmarks/model-verifier-canaries-20260814/stage05f-qwen-aligned-english-quality-v1/`
- Sanitized aggregate:
  [`aligned-english-summary.json`](artifacts/qwen-aligner-quality-20260816/aligned-english-summary.json)
- Exposure-index SHA-256:
  `5f76a1abb06d5055355abd4350d1dc68b13cf687770dc613400ecead81fb0bf3`
- Input-freeze SHA-256:
  `797812494b4f139fd4f57a3ed271de163e92b934745f1977bfd6d59489623475`
- Prediction-freeze SHA-256:
  `0851bb9e0e26e22f69e80548b6f0857e033183c8a149fafc9cad062fea3a8c1e`
- Post-freeze-score SHA-256:
  `5dfad614628ce57e236793067dfa3a9f2e529bfcd1998304ba29901cffab1a94`

## Separate proposed Gemini 3.7 Flash native-AV smoke

No Gemini call was issued by this diagnostic. Google documents stable
`gemini-3.7-flash` as a native-multimodal model with text, image, video,
audio, and PDF inputs, text output, structured outputs, a
1,048,576-token input limit, and a 65,536-token output limit in the
[official model
page](https://ai.google.dev/gemini-api/docs/models/gemini-3.7-flash).
The [official
changelog](https://ai.google.dev/gemini-api/docs/changelog) records GA
release on 2026-08-13. The [official pricing
page](https://ai.google.dev/gemini-api/docs/pricing) lists introductory
standard paid pricing through 2026-12-31 of `$0.75/M` input and
`$3.75/M` output and Batch/Flex pricing of `$0.375/M` input and
`$1.875/M` output.

This is a separate proposed compatibility smoke, not an arm of the Qwen
mechanism test and not a superiority claim. If authorized, compare it
against the pinned Gemini 3.1 Pro native-AV condition with the same
media bytes and media resolution, API route and request surface,
timestamp and FPS settings, structured-output schema, post-freeze
evaluation, and declared panel. The [video-understanding
guide](https://ai.google.dev/gemini-api/docs/video-understanding)
documents timestamp prompts and default FPS behavior. Google’s
[latest-model migration
guidance](https://ai.google.dev/gemini-api/docs/latest-model) says
sampling controls including `temperature`, `top_p`, and `top_k` are
deprecated for Gemini 3.6 and later, so the existing 3.1 Pro request
cannot be copied unchanged. The smallest valid next action is therefore
a separate two-video same-media Batch compatibility smoke, direct-answer
first, with the 3.7 arm omitting deprecated sampling controls. The
current repository harness is hard-pinned to Gemini 3.1 Pro and exactly
475 rows, so that smoke requires a new model/panel surface before any
paid call. No Gemini call was issued here.
