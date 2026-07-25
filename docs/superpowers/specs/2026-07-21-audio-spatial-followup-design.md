# Three-video audio and spatial follow-up design

Date: 2026-07-21

## Objective

Test the two evidence hypotheses raised by the completed three-video gate without
starting the 100-video benchmark:

1. a faithful speech-to-English translation may recover the speech-heavy answer;
2. a question-relevant high-resolution crop may recover the spatial/color answer.

The follow-up is diagnostic. It does not estimate competition accuracy and it
does not replace the frozen three-video gate.

## Alternatives considered

### Targeted causal diagnostics (selected)

Run full Whisper large-v3 on the three existing cached WAV files, then rerun
Qwen 3.5 4B only where the new accepted transcript differs materially. For the
color failure, add high-resolution crops of the visible herpes image selected
from the question and pixels alone. This is the smallest test that isolates the
two suspected causes.

### Uniform tiling across every frame

Apply a 2x2 tiling policy to all 12 frames of every video. This is more reusable,
but it expands each request substantially, changes more than the failed visual
example needs, and may dilute rather than clarify the relevant evidence.

### Increase temporal sampling to 24 frames

This is scale-ready but poorly matched to the observed failures: the bike action
is already fully represented, the herpes image is already present in three
samples, and the argument answer is carried by audio.

## Frozen inputs and isolation

- Reuse the same three 2025 training examples, questions, cached 16 kHz mono
  audio, and 12 bin-midpoint frames from `three-video-gate`.
- Never include reference answers or label-derived color words in an inference
  request, crop-selection instruction, filename, or prompt.
- Preserve all existing attempts and transcripts. Write follow-up evidence and
  attempts under a new ignored run directory and use append-only records.
- Pin every new model revision and runtime image. Record elapsed time, peak ROCm
  allocation, transcript QC, and evidence fingerprints.

## Audio diagnostic

Use `openai/whisper-large-v3` with FP16 ROCm inference and
`task="translate"`. Generate one timestamped result for each cached audio file.
Apply the existing hallucination-rate and English-script checks. A transcript is
usable only if it passes those mechanical checks and a manual semantic review;
Latin characters alone do not establish translation fidelity.

If the speech clip yields usable English, run Qwen 3.5 4B with the same 12 frames
and that transcript. Compare its answer with the frozen frames-only and
large-v3-turbo conditions. Do not rerun the other VLM families.

## Spatial diagnostic

Use the question and the visible frames—not the official answer—to identify the
herpes-virus image in the color clip. Produce checksummed, high-resolution crops
from every sampled frame in which that image is visible. Keep the original 12
frames in their existing temporal order and append the crops with explicit
timestamps and a prompt note that they are detail views of earlier frames.

Run Qwen 3.5 4B once on this augmented evidence. Because the region is selected
manually for a known question, label the result `question-guided-crop-diagnostic`.
It may establish whether spatial resolution caused this failure, but it is not
eligible for the future 100-video benchmark until an automatic selection policy
is designed and tested.

## Decision rules

- Audio hypothesis supported: full large-v3 produces a faithful English rendering
  and the downstream Qwen answer moves materially toward the reference meaning.
- Spatial hypothesis supported: the crop-augmented answer matches or clearly
  improves toward the official color while the request remains label-free.
- If neither improves, retain the 12-frame Qwen baseline and do not add their
  cost to the 100-video plan.
- If a hypothesis is supported, propose a separate reusable policy before
  scaling; do not infer scale-readiness from this diagnostic.

## Verification and handoff

- Unit-test any new evidence/request serialization behavior before implementing
  it and observe the test fail first.
- Verify source/crop hashes, video IDs, timestamps, prompt contents, label
  exclusion, ROCm device reporting, pinned revisions, and terminal selection.
- Run the full test suite and `git diff --check` before reporting results.
- Report transcript text, predictions, metrics, latency, memory, and failures.
- Stop after these diagnostics. Do not start the 100-video run.
