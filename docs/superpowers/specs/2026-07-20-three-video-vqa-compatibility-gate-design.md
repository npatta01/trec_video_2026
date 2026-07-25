# Three-Video VQA Compatibility Gate Design

## Objective

Build and execute a reproducible, three-video portability gate before any
100-example benchmark. The gate must answer four questions with evidence:

1. Can the candidate model load and run on this AMD Radeon 8060S host?
2. Can it consume the intended visual and audio evidence without silently
   falling back to text-only inference?
3. Does it return a short, machine-readable answer for every example?
4. What latency, memory use, answer quality, and failure mode should determine
   whether the model advances to the 100-example pilot?

The 100-example run is explicitly out of scope until the user reviews the gate
results and authorizes it.

## Representative examples

Use three labeled 2025 training examples whose local media are present. This
avoids using the publicly bundled 2025 test answers while their reuse notice is
unclear and lets the gate calculate answer-quality diagnostics.

| Category | Video ID | Question | Why selected |
|---|---|---|---|
| Temporal visual action | `BDKkVYmWqdM` | What does a bicyclist do before landing his bike on a tennis court? | Requires ordering an action across a 21.3-second clip. |
| Spatial/color detail | `O60MtMuSpf8` | What color is the image of the herpes virus? | Tests whether sampled frames preserve a small visual attribute in a 14.6-second portrait clip. |
| Speech/audio | `Fk_j6re_Uiw` | What are the two voices arguing about? | A 19.0-second clip whose answer should benefit from speech evidence. |

All three files contain video and Opus audio streams. The manifest records the
question, reference answer, category, resolved media path, duration, and media
stream summary. Labels are passed to evaluation only, never to model prompts.

## Candidate conditions

The gate is staged to avoid unnecessary model downloads and compute.

### Required baseline conditions

1. `Qwen/Qwen3.5-4B` with the 12 shared frames and no transcript.
2. The same Qwen3.5-4B revision, frames, prompt, and decoding with the frozen
   Whisper transcript added.

These establish whether the simplest baseline works and whether the transcript
changes the audio-dependent answer.

### Candidate portability conditions

After the baseline passes, attempt the candidates previously selected for the
benchmark funnel:

- `allenai/Molmo2-4B` with the same frames and transcript.
- `Qwen/Qwen3.5-9B` with the same frames and transcript.
- `google/gemma-4-12B-it` with the same frames and transcript as the genuinely
  non-Qwen family comparison.
- `OpenGVLab/InternVL3_5-4B` using its capped native tiling path, recorded as a
  non-controlled spatial-evidence condition.
- `openbmb/MiniCPM-o-4_5` using offline native video plus audio, recorded as a
  non-controlled audio-evidence condition.

Each candidate first receives a load-and-one-example probe. Only a successful
probe advances to all three examples. A candidate that cannot run on ROCm is a
valid gate result; the harness records the exact error and continues. No
candidate silently substitutes CPU, a different checkpoint, or a different
precision.

## Evidence pipeline

The benchmark module exposes one small interface: given a manifest and run
configuration, produce an immutable run report. Its implementation hides media
probing, deterministic evidence extraction, runner invocation, resumption,
metrics, and provenance capture.

The controlled evidence is generated once and shared byte-for-byte:

- 12 chronological frames sampled at the midpoints of 12 equal-duration bins.
- Original aspect ratio retained; adapters may resize only according to their
  documented processor behavior.
- Timestamp labels paired with frames in the prompt.
- One timestamped English Whisper transcript per video, generated once and
  reused by every transcript condition.
- A single concise-answer prompt and deterministic decoding (`temperature=0`,
  no sampling, bounded output tokens) wherever the runtime supports it.

Cached evidence lives only beneath ignored `data/vqa/benchmarks/`. A cache
record includes source-media fingerprint, extractor version, parameters, and
output checksums so stale or mismatched evidence is rejected.

## Model-runner seam

Model implementations vary enough to justify a real adapter seam. The host
orchestrator writes one JSON request and invokes an isolated runner command.
Every runner returns one JSON result with:

- status (`passed`, `unsupported`, `failed`, or `timed_out`),
- answer or null,
- elapsed seconds,
- model ID and immutable revision,
- runtime, precision, device, and dependency versions,
- peak accelerator memory when available,
- structured error type and a bounded stderr excerpt.

The orchestrator validates the result schema and owns timeouts, retries,
resumption, and report assembly. Model-specific trusted remote code and
dependencies stay inside a pinned rootless Podman image or isolated environment
rather than entering the downloader's base dependency set.

## Runtime and hardware policy

The verified host has 122 GiB unified memory, 2.9 TiB free disk, ROCm-visible
`gfx1151` Radeon 8060S graphics, rootless Podman 5.7, and an already-cached
`rocm/pytorch:latest` image that successfully executes a GPU tensor operation.
The final report must replace mutable image tags with the resolved image digest.

Before weights are acquired, use `hf download --dry-run` to record file and
byte counts. Models are downloaded by immutable Hub revision into the existing
Hugging Face cache. Do not start a long-lived inference server or expose a
network port. Do not delete or prune any existing cache entries.

The gate has a 12-hour hard wall-clock limit. Each model probe has its own
timeout, and a timeout is reported rather than retried indefinitely. The gate
stops before the 100-example phase under every outcome.

## Evaluation and scoring

Three examples are too few to rank models statistically. Report per-example
results and descriptive aggregates only:

- normalized exact match,
- token F1,
- semantic metrics available through the pinned NIST 2025 scorer path,
- latency, peak memory, and failure count,
- a clearly labeled manual evidence check that the answer is supported by the
  frames/transcript rather than copied from a reference.

Add NIST's VQA scorer as a Git submodule at `external/nist-vqa`, pinned to the
audited commit. Invoke it only in an isolated temporary directory, never on an
untrusted answer string through a shell, and label its current AG evaluation as
`nist-2025-parity`, not official 2026 scoring. The 2026 validator is exposed
separately as `nist-2026-validate-ag`. If the external STS endpoint is
unavailable, record that metric as unavailable without invalidating the model
run. NDCG is not interpreted for single-answer gate outputs.

## Outputs and privacy

Canonical configuration and documentation are committed. Media, frames,
transcripts, model weights, raw logs, and generated predictions remain under
ignored local data/cache paths. The final sanitized Markdown report may include
questions, reference answers, model answers, metrics, timings, versions, and
bounded non-secret error excerpts, but never embeds video/audio or credentials.

Each run directory contains:

- resolved manifest and configuration,
- evidence checksums and transcript JSON,
- append-only per-attempt JSONL,
- normalized predictions JSONL,
- metric output,
- environment/provenance record,
- final Markdown and JSON summaries.

## Failure handling

- Missing/corrupt media fails preflight before any model is loaded.
- Frame or transcript failure is attached to that example and is never replaced
  with empty evidence.
- Out-of-memory, unsupported GPU kernels, dependency conflicts, invalid model
  output, and timeouts have distinct statuses.
- Partial outputs are append-only and resumable; a rerun does not overwrite a
  successful result with a failure.
- Secrets and full environment dumps are excluded from reports.

## Verification and review

Implementation proceeds test-first. Unit tests cover manifest validation,
deterministic timestamps, cache invalidation, prompt/reference separation,
runner-result validation, resume behavior, metric normalization, and report
generation. Integration tests use fake runners and tiny generated media before
real weights are loaded.

Before handoff:

1. Run the existing and new test suite.
2. Verify frame timestamps and transcript association manually for all three
   examples.
3. Execute the load probe and three-example gate under the runtime cap.
4. Re-run report generation from saved raw results.
5. Ask an independent advisor to review both the code diff and experimental
   findings for leakage, evidence mismatch, scoring errors, misleading claims,
   and missing provenance.
6. Address actionable findings and repeat affected verification.

## Deferred work

The deterministic 100-example manifest, full benchmark execution, training or
fine-tuning, ten-answer generation/ranking, official 2026 evaluation once NIST
publishes it, cloud-model calls, and public artifact publishing are deferred.
