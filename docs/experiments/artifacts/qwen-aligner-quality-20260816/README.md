# Qwen aligner quality diagnostics — sanitized evidence index

This index preserves the original Stage 0.5d quality summary, the Stage 0.5e
empty-output mechanism diagnostic, and the completed Stage 0.5f fresh
aligned-English quality test. It contains no row IDs, transcripts, questions,
answers, row-level scores, or private media.

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
using no reused Qwen rows was authorized as the next Qwen action and is now
recorded below.

## Added Stage 0.5f fresh aligned-English quality test

- Private root:
  `data/vqa/benchmarks/model-verifier-canaries-20260814/stage05f-qwen-aligned-english-quality-v1/`
- Experiment ID: `qwen-aligner-stage05f-20260816-aligned-english-12-v1`
- Evidence validity: **Operationally valid; construct-invalid for ASR quality**
- Decision disposition: **Not promoted under the tested condition**
- Operational gate: `12/12` completed and `12/12` valid aligned rows
- Quality result: Qwen `0/12`, frozen Whisper `0/12`, with all `12` rows in
  the paired neither-present cell
- Decision: `no_directional_difference`
- Paid cost: `$0.00`

The sole attempt completed with exit code `0`. The `12/12` prediction freeze
was verified before annotations or frozen Whisper text were opened. The
forced-English condition completed this fresh panel without an empty-output
failure. A later private read-only audit reproduced all source bindings and
stored booleans, but found that only one task directly requested a recoverable
literal phrase or name and ten tasks did not support exact transcript
containment. The `0`/`0` score is therefore preserved as frozen history but is
construct-invalid for comparing ASR quality. It does not promote Qwen, measure
WER or timestamp accuracy, or establish transcript fidelity or downstream VQA
value. See [`aligned-english-summary.json`](aligned-english-summary.json) for
the original sanitized aggregate and
[`aligned-english-metric-audit.json`](aligned-english-metric-audit.json) for
the aggregate correction.

The post-hoc audit also records, without row-level material, that all 12 Qwen
outputs were non-empty; the quality gates accepted `8/12` Qwen and `10/12`
Whisper outputs; raw Whisper had one exact hit that the frozen timestamp gate
excluded; and mean unordered official-answer token recall was `0.035` Qwen
and `0.212` Whisper. The token-recall check is descriptive and
non-promotional because official VQA answers are not transcript references.

## Added Stage 0.5g transcript-to-answer diagnostic

- Private root:
  `data/vqa/benchmarks/model-verifier-canaries-20260814/stage05g-qwen-transcript-answer-v1/`
- Experiment ID: `qwen-transcript-answer-stage05g-v1`
- Frozen input: the same 12 Stage 0.5f Qwen and Whisper transcripts; no ASR
  rerun
- Answer stage: pinned `Qwen/Qwen3.5-4B`, transcript plus question, no frames
- Completion: `24/24` answers in one local attempt
- Paid cost: `$0.00`
- Decision: `formulation_bridge_observed`

After all answers froze, one arm-blinded semantic review found `1/12` correct,
one ambiguous, and `10/12` incorrect in the Qwen-transcript arm, versus `2/12`
correct and `10/12` incorrect in the Whisper-transcript arm. The paired result
was one both-correct row, one Whisper-only correct row, no Qwen-only correct
row, and ten neither-correct rows.

This establishes that a normal answer step can bridge some answer-wording gaps,
so the historical `0/12` containment result is not evidence of twelve bad
transcriptions. This fixed answer stage did not repair the same ten rows in
either arm, but the diagnostic does not identify whether that came from
transcript omission, evidence available only in the video, or prompt/model
answer-stage error. The tiny single-reviewer result does not rank Qwen ASR
against Whisper. See
[`transcript-answer-summary.json`](transcript-answer-summary.json) for the
sanitized aggregate and immutable private hashes.

Independent review found that the original scorer did not rederive the private
arm key before unblinding. The corrected verifier accepted the preserved key
and reproduced the judgments, score, and score-freeze artifacts byte for byte
in a copied root; the aggregate result did not change. A key-swap regression
now fails closed.

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
