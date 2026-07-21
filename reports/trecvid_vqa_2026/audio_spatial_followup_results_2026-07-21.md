# Audio and spatial follow-up results

Date: 2026-07-21

## Decision summary

Neither tested hypothesis recovered an additional correct answer on the two
failed examples.

- Full Whisper large-v3 produced English for the speech clip and changed Qwen's
  answer from `End of the trend` to `Where the money is`. This proves Qwen used
  the transcript, but the answer is still wrong and the translation is
  semantically disputed. Full large-v3 should not advance on this evidence.
- Three question-guided 680×590 detail crops changed neither the color answer nor
  its metrics: Qwen returned `pink` before and after. Higher spatial resolution
  did not solve this example.

This is a targeted diagnostic over two failures, not an accuracy estimate. The
100-video run was not started.

## Frozen comparisons

References were absent from every model request and joined only here during
scoring.

| Probe | Evidence | Prediction | Official reference | EM | Token F1 | METEOR |
|---|---|---|---|---:|---:|---:|
| Speech | 12 frames, no accepted transcript | End of the trend | The expected contents of the bag. | 0.000 | 0.286 | 0.280 |
| Speech | 12 frames + full-large-v3 translation | Where the money is | The expected contents of the bag. | 0.000 | 0.000 | 0.075 |
| Color | 12 frames | pink | Pale orange. | 0.000 | 0.000 | 0.000 |
| Color | 12 frames + 3 question-guided detail crops | pink | Pale orange. | 0.000 | 0.000 | 0.000 |

BERTScore was not rerun because its pinned RoBERTa encoder was not present in the
current cache; downloading another large model only for this two-answer
diagnostic was not justified. Historical results already showed that BERTScore
can remain high for clearly wrong answers. STS remains disabled because the
historical implementation uses an external plaintext service, and single-answer
NDCG is not meaningful.

## Full Whisper large-v3

The pinned `openai/whisper-large-v3` revision
`06f233fe06e710322aca913c1bc4249a0d71fce1` ran in FP16 on the Radeon 8060S
through ROCm.

| Video | Mechanical QC | Semantic review | Output summary | Time | Peak allocation |
|---|---|---|---|---:|---:|
| `BDKkVYmWqdM` | accepted | consistent with audible English | He's riding a bike across a tennis net. | 9.27 s | 4.24 GiB |
| `O60MtMuSpf8` | accepted | unverified non-speech description | Music Outro | 9.33 s | 4.24 GiB |
| `Fk_j6re_Uiw` | accepted | rejected: semantic conflict | argument about Pakistan and money | 15.73 s | 4.30 GiB |

The speech output is more usable-looking than Turbo's rejected Hindi-script
text, but English fluency is not fidelity. The earlier native transcript contains
`कट्टा` (`katta`, gun/pistol) where full large-v3 repeatedly says `money`.
Because of that conflict, the downstream Qwen request is exploratory rather than
an accepted ASR condition. Its answer following the transcript's `money` wording
confirms propagation of the disputed translation, not improved VQA accuracy.

## Question-guided spatial crops

The herpes image is already present in the first three frozen 1080×1920 samples.
Without consulting the reference color, the visible embedded image was cropped
using `x=165`, `y=1075`, `width=680`, `height=590`. The three crops were appended
to the original 12 frames and explicitly described as detail views at timestamps
0.610, 1.831, and 3.052 seconds.

Every source and crop SHA-256 verifies, and a manual inspection confirms the
crops contain the question-requested image without label-derived annotations.
The enhanced crop manifest records the source-evidence directory and the
retained three-crop contact sheet, making that inspection reproducible.
Qwen still answered `pink`. Therefore this failure is not explained by the image
being too small in the original request. The image itself contains pale
orange/peach skin together with strong red/pink lesions, so the official
single-color reference is visually ambiguous; this is an inference from the
evidence, not a claim that the label is invalid.

Because region selection was manual and question-guided, this condition would
not have been scale-ready even if it had succeeded.

## Runtime and provenance

| Qwen condition | Result | Time | Peak allocation |
|---|---|---:|---:|
| Full-large-v3 transcript | Where the money is | 338.16 s | 10.77 GiB |
| Question-guided crops | pink | 413.18 s | 11.42 GiB |

Both requests shared one loaded Qwen 3.5 4B process (2.90-second model load) and
reported `cuda:0 (ROCm)`, Qwen revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, and runtime image digest
`510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5`.
The unexpectedly long per-request times are recorded observations; their cause
was not instrumented and is not attributed to JIT or crop count.

The Qwen batch request was retained before execution and its hash verifies. An
initial result array was reconstructed from captured output. To remove that
audit limitation, the identical retained request was then rerun after the runner
was changed to embed `run_id`, `condition_id`, and `video_id` in every result.
That rerun independently reproduced both answers; stdout and stderr were
retained byte-for-byte with SHA-256 hashes in `execution.json`. The ASR request
and result files remain reconstructed records, and are explicitly identified as
such rather than claimed as original byte streams.

Generated requests, evidence, attempts, and results remain ignored beneath
`data/vqa/benchmarks/followups/audio-spatial-2026-07-21/`. The shared append-only
index contains five unique run/condition/video identities and seven append-only
attempt records, including the two exact-capture Qwen reruns; no dataset media
or generated output is committed.

An independent re-review found no critical, important, or minor issues. It
recomputed the request, stdout, stderr, crop, source, manifest, and contact-sheet
hashes; confirmed the three crops are exact unannotated pixel crops; matched
every output identity to its request and attempt record; and reran all 216 tests
plus the three-video preflight successfully.

## Recommendation

Keep 12 frames as the simple baseline for now. On these examples:

- additional translation capacity changed the answer but did not improve its
  correctness;
- additional spatial pixels did not change the answer;
- the temporal bike question was already answered correctly with 12 frames.

If audio deserves another diagnostic, use a different speech-translation family
such as SeamlessM4T rather than treating full Whisper as solved. If frame count
is tested later, run a clean 12-versus-24-frame comparison over a larger labeled
sample containing genuinely brief events; this color example does not support
the need for more temporal samples.

Canonical design: `docs/superpowers/specs/2026-07-21-audio-spatial-followup-design.md`

Canonical execution plan: `docs/superpowers/plans/2026-07-21-audio-spatial-followup.md`
