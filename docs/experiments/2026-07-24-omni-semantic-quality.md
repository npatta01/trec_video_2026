# Omni semantic-quality correction

Date: 2026-07-24

## Decision

Stop at the three-example diagnostic panel. Do not promote Gemma 3n E4B-it
for semantic-quality evaluation and do not expand either non-Qwen model to
10–30 examples.

The existing Gemma run is genuine joint audio/vision compatibility evidence,
but it does not pass the predeclared semantic gate: one answer is correct, one
is partial, and one is wrong. The existing Phi-4-multimodal ROCm run is also
weak at zero correct, two partial, and one wrong.

The pinned CUDA Phi rerun is unresolved rather than semantically failed.
Three paid attempts stopped before inference; the final attempt reached an
NVIDIA A100 and the pinned model revision, then failed model loading because
the runtime omitted `scipy`. No CUDA Phi answer should be inferred from those
failure records.

## Frozen scope and gate

The Gemma and Phi diagnostic runs use the exact three-example contract at
`data/vqa/benchmarks/omni-compatibility-3/contract.json`.

- Canonical contract digest (normalized JSON SHA-256):
  `b6fa0c299cd3efafb2dec092ba5d7037d08b0a7eb3fc859b46358158b8b9725b`
- Existing Gemma predictions SHA-256:
  `9797c14694bf99a438a76eba27f64fd336cae06526e11d8d2faa99021423ef00`
- Existing Phi ROCm predictions SHA-256:
  `6a1ee0d5ea29d150f871f7942c6377643ad6a0d86b4df0953a339d7e6c890478`

The quality gate was written before any new CUDA outputs were revealed. A
model passes only if all three rows:

1. retain exact-request, joint first-forward audio/vision attestation;
2. produce non-empty, scoreable answers; and
3. receive zero `wrong` judgments and at least two `correct` judgments.

Expansion requires **both** Gemma and Phi to pass. Gemma's wrong lexical-speech
answer therefore stops expansion independently of the unresolved CUDA Phi
rerun. This is a diagnostic screening rule, not a statistically meaningful
leaderboard estimate.

## Label-isolated adjudication

Predictions were copied into an answer-only worksheet and identified only by
opaque row ID and operational role. The prediction files and their SHA-256
digests were frozen first. Plain text was then normalized without loading the
reference labels. Only after that freeze were the annotations opened for
semantic adjudication and lexical metric calculation.

The correction is deliberately narrow: non-empty plain text is a valid answer,
while JSON-looking or fenced-JSON output remains subject to strict parsing and
malformed structured output fails closed. Empty text is rejected. This removes
the earlier strict-JSON false negative without turning malformed structured
output into a silent pass.

## Per-example findings

Judgment labels are `C` (correct), `P` (partially correct), and `W` (wrong).

| Role | Reference meaning | Gemma 3n E4B-it | Judgment | Phi-4-multimodal, prior ROCm | Judgment |
|---|---|---|---:|---|---:|
| lexical speech | A bystander says she does not know | “The horse is 2012.” | W | “2012” | W |
| prosody | Greater emphasis/intensity plus gestures | Louder, emphatic or stronger language plus gestures | C | Repetitive hand-to-chest/face/mouth/microphone description | P |
| non-speech sound | A cat screaming/meowing | “Meowing” | P | “meow” | P |

| Model/run | Joint compatible | C/P/W | Exact match | Mean token F1 | Gate |
|---|---:|---:|---:|---:|---|
| Gemma 3n E4B-it, existing ROCm | 3/3 | 1/1/1 | 0.0000 | 0.0988 | Fail |
| Phi-4-multimodal, existing ROCm | 3/3 | 0/2/1 | 0.0000 | 0.0150 | Fail |
| Phi-4-multimodal, new CUDA | No inference output | — | — | — | Unresolved |

Compatibility is therefore not evidence of acceptable answer quality on this
panel.

## Matched existing baselines

The following rows were extracted for the same three video IDs from
`data/vqa/benchmarks/runs/native-audio-benchmark-v2/scored-predictions.jsonl`
(SHA-256
`c95669135b29551ef407e2889ac5e203053e944b00a0968cbe20da1fdfd0be7a`).
Manual C/P/W judgments use the same rubric as above.

| Baseline | Additional audio/text | Coverage | C/P/W | Exact match | Mean token F1 | METEOR |
|---|---|---:|---:|---:|---:|---:|
| MiniCPM-o | none (frames only) | 3/3 | 0/2/1 | 0.0000 | 0.4467 | 0.2411 |
| Qwen3.5 4B, 12 frames | none (frames only) | 3/3 | 0/1/2 | 0.0000 | 0.0290 | 0.0158 |
| Qwen3.5 4B, 24 frames | none (frames only) | 3/3 | 1/0/2 | 0.0000 | 0.1111 | 0.0535 |
| Qwen3.5 4B, 12 frames + Whisper | automatic transcript | 3/3 | 1/1/1 | 0.0000 | 0.2333 | 0.3243 |

An older MiniCPM-o audio-on file,
`data/vqa/benchmarks/runs/task4-cached-ten/bf16-v2/evaluation-predictions.jsonl`
(SHA-256
`27f3f397a26d1afa712416b8f820bae4fc90f394896a259469231ce807660738`),
contains only two of the three IDs. Its 0/1/1 judgment and 0.2500 mean token F1
are reported as 2/3 coverage and are not a comparable aggregate.

No Qwen3-Omni semantic output was available: the issue #5 CUDA attempt was
blocked before inference.

## CUDA/dstack execution evidence

The immutable configuration pinned:

- image:
  `docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755`
- model:
  `microsoft/Phi-4-multimodal-instruct`
- model revision:
  `93f923e1a7727d1c4f446756212d9d3e8fcc5d81`
- CUDA 12.4, PyTorch 2.5.1, Transformers 4.48.2, BF16
- one on-demand A100/H100/H200 with at least 80 GB VRAM
- RunPod or Vast.ai, `$2.00/hour` maximum, one-hour duration cap
- no application ports

The successful preview exposed four RunPod A100 80 GB offers; the displayed
offers were `$1.39`, `$1.49`, and `$1.49` per hour. Each submitted run selected
an on-demand RunPod A100-SXM4-80GB in US-MD-1 at `$1.49/hour`, with 250 GB host
RAM and a 100 GB disk.

| Run | UTC interval | Cost | Terminal finding | `SHA256SUMS` file SHA-256 |
|---|---|---:|---|---|
| `phi4mm-cuda-quality-3-20260724-a1` | 20:12:41–20:14:47 | $0.0489 | Input preflight incorrectly required an unused source video | `c598ffa6180f800763a35d525c55b579a39dce7cd80d539c85e2a7ad051e8207` |
| `phi4mm-cuda-quality-3-20260724-a2` | 20:16:13–20:18:44 | $0.0598 | The same source-video requirement remained at the probe boundary | `5d3b18d152a5a031ac6c4e34eb6df7151663b15cccdee383ec593916f2b6ff1c` |
| `phi4mm-cuda-quality-3-20260724-a3` | 20:20:27–20:22:30 | $0.0484 | Pinned model load failed because `scipy` was absent | `5e7fb63c427495db893e13830c770463e4bf9c84563c4a0f6d3ea861270d464a` |

Total billed cost was **$0.1571**. Every attempt stayed within the previewed
offer and fixed cap. Each private, run-specific HF Bucket prefix was synced
locally, and every downloaded file passed its recorded `SHA256SUMS` check.
Attempt `a3` attested Python 3.11.10, PyTorch 2.5.1+cu124, Transformers 4.48.2,
CUDA 12.4, and NVIDIA A100-SXM4-80GB. Its predictions file SHA-256 is
`bfb2615afed35c5e460bd52f88e5b9fd87e29f1ffa1b08cb82cfc71329de6022`;
all three rows are the same pre-inference `model_load` failure and contain no
answer.

No raw logs, predictions, media, model weights, credentials, or dataset files
are tracked by this change.

## Recommendation and next action

Do not promote Gemma from compatibility probing to a larger semantic-quality
run. Retain Phi's existing ROCm result only as weak three-row evidence; do not
claim a CUDA quality result.

After three consecutive pre-inference corrections, paid execution is paused.
If another run is explicitly authorized, first audit the pinned model's
dependency closure, add an exact binary-compatible SciPy pin with a local
import/model-load regression check, and preview a new immutable `a4` run under
the same `$2.00/hour` and one-hour caps. Even if that run passes, the frozen
two-model gate still forbids expansion because Gemma already failed.
