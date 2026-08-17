# `g37_grouped_unique`: Grouped Grounded Rewrites

## Current state

- **Task:** TRECVID 2026 Video Question Answering, Answer Generation
- **Run tag:** `g37_grouped_unique`
- **State:** Upload artifact tracked and locally validated; not uploaded
- **Submission rows:** 2,443 of 2,443 official test queries
- **Answers per query:** 10
- **Local validator:** Passed with zero errors and zero warnings

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
conflict merge. The final offline salvage accepted independently valid
rewrites for 1,833 questions and copied the already validated
`g37_q36_fusion` row unchanged for the remaining 610 questions. No second LLM
verifier call was used.

No reference answers, evaluation labels, annotations, video, audio, or
transcripts entered the rewrite calls. This record establishes preparation and
format validity, not answer quality or organizer acceptance.

## Upload artifact

- **Tracked upload file:**
  [Download `submission.json`](../../submissions/trecvid-2026/vqa-ag/g37_grouped_unique/submission.json)
- **Private source path:**
  `data/vqa/benchmarks/grouped-rewrite-full-20260817-grouped_unique_generator_only-a3-salvage-final/final-organizer.json`
- **Size:** 3,225,489 bytes
- **SHA-256:**
  `fe35de8c45806f97382bc93fa6be1c73521a7f596cd9221517483c44d1d8e711`
- **Provenance SHA-256:**
  `201f4225127ae3e1c294b5024482a2da62494f4e73bcb396e04cc6cdc08756b0`
- **Salvage manifest SHA-256:**
  `fc0be6932aa0cae6cd19675583bb34b49a2153be87aae220169682cd524b2c98`
- **Validator report SHA-256:**
  `b9754a26b559b2edfdaa86ec9eef15d5e31d2794c8bc9c8bb154617fcdadd2a6`
- **Total conservative accounted API cost:** $11.27937675

After Evalbase accepts the run, append the submission timestamp, run URL or
confirmation identifier, server-side validation result, and uploaded-file
hash. Do not change this state based on local validation alone.

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
