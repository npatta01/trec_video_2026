# `q36_dualview`: Qwen3.6 Transcript–Native Dual View

## Current state

- **Task:** TRECVID 2026 Video Question Answering, Answer Generation
- **Run tag:** `q36_dualview`
- **State:** Upload artifact tracked and locally validated; not uploaded
- **Submission rows:** 2,443 of 2,443 official test queries
- **Answers per query:** 10
- **Local validator:** Passed with zero errors and zero warnings

## Method summary

This is the open-weight dual-view submission. It combines Qwen3.6 answers
conditioned on the supplied Whisper transcript with Qwen3.6 answers generated
from native video evidence. The transcript-conditioned view is ranked first;
native-video and frozen control alternatives provide visual and failure-mode
coverage below it after normalized deduplication.

No reference answers, evaluation labels, or annotations entered inference or
fusion. This record establishes preparation and format validity, not answer
quality or organizer acceptance.

## Upload artifact

- **Tracked upload file:**
  [Download `submission.json`](../../submissions/trecvid-2026/vqa-ag/q36_dualview/submission.json)
- **Private source path:**
  `data/vqa/submissions/deadline-20260817-ready/01-qwen36-transcript-first-top10.json`
- **Size:** 3,192,428 bytes
- **SHA-256:**
  `d8ccd445dd1cb3dd7cbd54454c012184f9fe1a21d3187fb7b068e09c90fc6a1a`
- **Provenance SHA-256:**
  `765985620995167382c94f5bf39e7cc40ec2c96efeba5d87d5278c5d131ee285`
- **Validator report SHA-256:**
  `b9754a26b559b2edfdaa86ec9eef15d5e31d2794c8bc9c8bb154617fcdadd2a6`

After Evalbase accepts the run, append the submission timestamp, run URL or
confirmation identifier, server-side validation result, and uploaded-file
hash. Do not change this state based on local validation alone.

## Evalbase copy-paste answers

### Runtag

`q36_dualview`

### What base model(s) did you use?

> The off-the-shelf Qwen/Qwen3.6-27B multimodal model. For each question, one
> answer was generated from native video evidence and a second answer was
> generated with an automated speech transcript supplied alongside the video.
> The transcript-conditioned answer was ranked first, followed by the
> native-video answer and frozen alternatives after deduplication.

### Fine-tuned or off-the-shelf?

Select **Off-the-shelf open-source model**. No model was fine-tuned.

### What training data did you use?

> Not applicable. We did not train or fine-tune the model. We used the
> released Qwen3.6-27B weights without reference answers, evaluation labels,
> or test annotations.

### How long did development take?

> Approximately one day for the final dual-view inference, assembly, audit,
> and validation workflow, building on earlier baseline experiments.

### Compute resources

> Qwen3.6-27B inference used four H100 80 GB GPU shards with vLLM. Automated
> speech transcription used a separate GPU task. The final organizer file
> contains 2,443 queries with 10 answers each and passed the official validator
> with zero errors and zero warnings.
