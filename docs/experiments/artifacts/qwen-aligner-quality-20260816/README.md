# Qwen aligner quality diagnostics — sanitized evidence index

This index preserves the original Stage 0.5d quality summary and adds the
Stage 0.5e empty-output mechanism diagnostic. It contains no row IDs,
transcripts, questions, answers, or private media.

## Preserved Stage 0.5d quality summary

- Private root:
  `data/vqa/benchmarks/model-verifier-canaries-20260814/stage05d-qwen-aligner-quality-v1/`
- Experiment ID: `qwen-aligner-stage05d-20260816-quality-v1`
- Decision: `not_ready`
- Completion: `6/8` rows with valid alignment
- Failure boundary: two deterministic `Qwen3-ASR returned empty text` outputs
- Paid cost: `$0.00`

The complete terminal evidence remains private under the canonical dataset
root. The Stage 0.5d diagnostic was not scored against Whisper because the
predeclared 8/8 gate failed. See [`summary.json`](summary.json) for its
hash-bound aggregate record.

## Added Stage 0.5e mechanism diagnostic

- Private root:
  `data/vqa/benchmarks/model-verifier-canaries-20260814/stage05e-qwen-empty-output-diagnostic-v1/`
- Experiment ID: `qwen-aligner-stage05e-20260816-empty-output-v1`
- Decision: `ready_for_fresh_aligned_quality_panel`
- Panel: four rows selected from the frozen Stage 0.5d terminal status
- Baseline reproduction: `true`
- Advancing arm: `aligned_english` (`4/4` valid aligned successes)
- Paid cost: `$0.00`

The four-row mechanism diagnostic is never scored. Its condition-bound result
is that forcing `language="English"` removed the two preserved empty-output
failures while each automatic-language explanatory arm remained at 2 successes
and 2 failures and neither restored all four rows. See
[`diagnostic-summary.json`](diagnostic-summary.json) for the sanitized,
hash-bound aggregate record. A fresh label-blind aligned-English quality panel
using no reused Qwen rows is authorized as the next Qwen action; it has not
run here.

## Post-run protocol correction

Independent review found that the original Stage 0.5e host selector
programmatically read private `text`, `language`, and `chunks` fields on rows
already marked passed, contrary to the metadata-only selection design. It
copied no row-level values into this bundle, and no labels, annotations,
Whisper text, or scoring inputs were opened. Stage 0.5d's frozen aggregate had
already certified all six passed rows as completed and alignment-valid, so a
corrected metadata-only selection yields the same first two controls. Neither
frozen run was repeated. The corrected harness cross-checks only statuses and
permitted metadata against the frozen completion/alignment counts, and its
regression test traps private transcript-field access.
