# `g37_grouped_unique`: Grouped Grounded Rewrites

## Current state

- **Task:** TRECVID 2026 Video Question Answering, Answer Generation
- **Run tag:** `g37_grouped_unique`
- **State:** Compact release candidate `g37_grouped_unique-rc2` uploaded to
  Evalbase on 2026-08-17; server-side validation succeeded after the truncated
  RC1 upload was replaced
- **Submission rows:** 2,443 of 2,443 official test queries
- **Answers per query:** 10
- **Local validator:** Passed with zero errors and zero warnings

## Release candidate designation

`g37_grouped_unique-rc2` is the selected organizer-upload candidate as of
2026-08-17. It is a compact, single-line serialization of the exact RC1 JSON
value: all 2,443 query objects and 24,430 answer objects compare equal after
parsing. The designation binds the tracked upload file to SHA-256
`d6e3731f19e4dca575815afcd4ed1b163126936b8bb1bbacfd8c675e4473741d`.
It passed the official local validator with 2,443 of 2,443 queries, ten answers
per query, zero errors, and zero warnings. All rank-one answers are unchanged
from the validated fusion source, 1,833 previously accepted grouped rows remain
byte-exact as parsed objects, and the repair introduced no new answer strings.

RC2 is now the submitted and organizer-validated artifact. Any later content
change must receive a new release-candidate identifier and repeat the local and
server-side validator gates.

### RC1 validation failure and RC2 correction

The file selected for the RC1 Evalbase upload was not the complete tracked RC1
artifact. The submitted copy had SHA-256
`f0381ffb4c95ce4dc5e64d1194a7f6244bfb083cf32662c09b83318228dcd24e`,
size 3,080,192 bytes, and ended mid-object at line 132,557. The intended RC1
artifact had SHA-256
`e60c05933290f4f6644e8f574bd5f8e42bbb01935e893038c678fdfcdc25160b`
and size 3,180,520 bytes. Evalbase therefore correctly reported invalid JSON.

RC2 changes only JSON whitespace and is 2,034,752 bytes. It replaced the failed
RC1 run under the same run tag and Evalbase reported validation success. The
selected local upload file had SHA-256
`d6e3731f19e4dca575815afcd4ed1b163126936b8bb1bbacfd8c675e4473741d`;
the interface did not expose a server-side stored-file hash.

## Method summary

This is a text-only answer-consolidation treatment over frozen model outputs.
For each question, Gemini 3.7 Flash received the question and the available
rank-one answers from Gemini native audio/video, Qwen3.6
transcript-conditioned video, Qwen3.6 native video, and the Qwen3.5 control.
It grouped answers with the same meaning, preserved conflicting meanings as
separate groups, retained the existing fusion rank-one answer byte-for-byte,
and generated ten grounded wording variants. It was instructed not to add
facts and every semantic group had to retain an exact source answer.

Strict local checks rejected any question with invalid grouping, duplicate
answers, missing source coverage, a changed rank-one answer, or an unsupported
conflict merge. The original freeze retained 1,833 grouped rows and used the
validated `g37_q36_fusion` row for 610 questions. A label-free repair then
recovered 609 of those 610: 319 by deterministic cleanup, 185 from accepted
fresh responses, 75 from independently valid query objects inside rejected
batches, and 30 by removing duplicates and filling only with existing fusion
answers. One question (`Q_ID` 673) still uses its original fusion row because
the union of available answers contains only nine punctuation-insensitive
unique formulations; no new wording was invented to force a tenth.

The final file preserves all 2,443 rank-one answers exactly. It has 585
redundant answer slots across 442 rows under the stricter
case-and-punctuation-insensitive token audit, improving on the original
grouped file's 620 slots across 477 rows. The official organizer normalization
reports ten distinct answers for every query. No second LLM verifier call was
used.

No reference answers, evaluation labels, annotations, video, audio, or
transcripts entered the rewrite calls. This method summary describes
construction and does not establish answer quality; organizer validation is
recorded separately in the current state and upload sections.

## Upload artifact

- **Tracked upload file:**
  [Download `submission.json`](../../submissions/trecvid-2026/vqa-ag/g37_grouped_unique/submission.json)
- **Private source path:**
  `data/vqa/benchmarks/grouped-rewrite-fallback-repair-20260817/final-v2/final-organizer.json`
- **Size:** 2,034,752 bytes
- **SHA-256:**
  `d6e3731f19e4dca575815afcd4ed1b163126936b8bb1bbacfd8c675e4473741d`
- **Provenance SHA-256:**
  `5638a0993af18eb4ca85a911532701951c9e3d10b201cf81e5c5ad42dda07c7f`
- **Repair manifest SHA-256:**
  `d16a9da0f216040abeba9f9abeb6bb21986242cf0eca74bb93e15b2e927267fd`
- **Validator report SHA-256:**
  `b9754a26b559b2edfdaa86ec9eef15d5e31d2794c8bc9c8bb154617fcdadd2a6`
- **Total accounted API cost:** $12.50898375

Evalbase reported server-side validation success for run tag
`g37_grouped_unique` on 2026-08-17. The exact submission time and server-side
stored-file hash were not exposed in the available confirmation view.

## Evalbase copy-paste answers

### Runtag

`g37_grouped_unique`

### What base model(s) did you use?

> Gemini 3.7 Flash generated grouped, grounded wording variants from frozen
> rank-one answers produced by Gemini 3.7 Flash native audio/video,
> Qwen/Qwen3.6-27B with an automated speech transcript, Qwen/Qwen3.6-27B with
> native video, and the Qwen3.5-4B control. The rewrite model saw only the
> question and these candidate answers. It grouped semantically equivalent
> answers, kept conflicts separate, preserved the existing fusion rank-one
> answer exactly, and produced ten distinct answers. Questions that failed
> strict validation used the validated Gemini–Qwen fusion row unchanged.

### Fine-tuned or off-the-shelf?

Select **Off-the-shelf**. No model was fine-tuned.

### What training data did you use?

> Not applicable. We did not train or fine-tune any model. The rewrite stage
> used only frozen model-generated candidate answers and released test
> questions; it did not use reference answers, evaluation labels, or test
> annotations.

### How long did development take?

> Approximately two days for the final Qwen3.6, Gemini, fusion, grouped
> rewriting, recovery, audit, and validation workflow, building on earlier
> baseline experiments.

### Compute resources

> Candidate generation used four H100 80 GB Qwen3.6 shards, a separate GPU
> speech-transcription task, and the hosted Gemini 3.7 Flash API. The grouped
> rewrite stage used the hosted Gemini API with one text-only call per batch of
> five questions and no second verifier call. The final organizer file contains
> 2,443 queries with 10 answers each and passed the official validator with zero
> errors and zero warnings.
