# Qwen aligner quality diagnostic — sanitized evidence index

This index records the aggregate result of the fresh eight-row English
Qwen3-ASR plus forced-aligner diagnostic. It contains no row IDs, transcripts,
questions, answers, or private media.

- Private root:
  `data/vqa/benchmarks/model-verifier-canaries-20260814/stage05d-qwen-aligner-quality-v1/`
- Experiment ID: `qwen-aligner-stage05d-20260816-quality-v1`
- Decision: `not_ready`
- Completion: `6/8` rows with valid alignment
- Failure boundary: two deterministic `Qwen3-ASR returned empty text` outputs
- Paid cost: `$0.00`

The complete terminal evidence remains private under the canonical dataset
root. This diagnostic was not scored against Whisper because the predeclared
8/8 gate failed.

See [`summary.json`](summary.json) for the hash-bound aggregate record.
