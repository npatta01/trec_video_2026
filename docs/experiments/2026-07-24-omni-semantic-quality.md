# Omni semantic-quality correction

Date: 2026-07-24

## Decision

Stop at the three-example diagnostic panel. Do not promote Gemma 3n E4B-it or
Phi-4-multimodal-instruct to a 10–30 example semantic-quality run.

Both models now have verified NVIDIA CUDA and AMD ROCm joint audio/vision
results on the exact frozen panel. Runtime compatibility is established, but
neither model passes the predeclared semantic gate:

- Gemma CUDA: `1 correct / 1 partial / 1 wrong`
- Phi CUDA: `0 correct / 2 partial / 1 wrong`

The gate requires zero wrong rows and at least two correct rows for each model.
Expansion also requires both models to pass.

## Frozen scope and gate

All compared runs use the three-example contract at
`data/vqa/benchmarks/omni-compatibility-3/contract.json`.

- Canonical normalized-JSON contract SHA-256:
  `b6fa0c299cd3efafb2dec092ba5d7037d08b0a7eb3fc859b46358158b8b9725b`
- Raw `contract.json` file SHA-256:
  `dfd2d765e0dc0947e81e0387e09b88a0b75221a91571f43ae9812fdd66151309`
- Existing Gemma ROCm predictions SHA-256:
  `9797c14694bf99a438a76eba27f64fd336cae06526e11d8d2faa99021423ef00`
- Existing Phi ROCm predictions SHA-256:
  `6a1ee0d5ea29d150f871f7942c6377643ad6a0d86b4df0953a339d7e6c890478`
- Gemma CUDA `a2` predictions SHA-256:
  `d82e4e70959a0df1e50cdf2146afb05d8d3d40033b6454b4313060b38f5c65b3`
- Phi CUDA `a4` predictions SHA-256:
  `f935265e4c15572d36e500ea2097ec83eebe88bc811a9d919cd805c27982b27d`

The quality gate was written before the new CUDA outputs were revealed. A
model passes only if all three rows:

1. retain exact-request, joint first-forward audio/vision attestation;
2. produce non-empty, scoreable answers; and
3. receive zero `wrong` judgments and at least two `correct` judgments.

This is a screening rule for a tiny diagnostic panel, not a statistically
meaningful leaderboard estimate.

## Output handling and adjudication

The original ROCm predictions were frozen and normalized before reference
labels were opened. CUDA predictions were likewise frozen and checksum-verified
before scoring, then judged with the same answer-only rubric. Because the
references had already been used for the original analysis, the CUDA
adjudication should not be described as a new blinded evaluation.

The parser correction remains narrow: non-empty plain text is a valid answer,
while JSON-looking or fenced-JSON output remains subject to strict parsing.
Malformed structured output fails closed and empty text is rejected.

Judgment labels are `C` (correct), `P` (partially correct), and `W` (wrong).

| Role | Reference meaning | Gemma CUDA | Judgment | Phi CUDA | Judgment |
|---|---|---|---:|---|---:|
| lexical speech | A bystander says she does not know | “The horse is 11 years old.” | W | “2012” | W |
| prosody | Faster, louder, more emphatic speech plus forceful gestures | Raised voice, emphatic gestures, stern expression | C | Hand-to-face/body-language description | P |
| non-speech sound | A cat screaming | “Meowing” | P | “meowing” | P |

Gemma's lexical evidence is internally inconsistent and its answer contradicts
the reference. Its prosody answer captures the central louder/emphatic speech
and gesture cues despite omitting faster speech. “Meowing” identifies the
animal vocalization but misses the reference's screaming intensity. Phi's
prosody answer notices visible distress cues but misses the requested
faster/louder/emphatic speech change and adds unsupported details.

## Aggregate quality

Exact match and token F1 use the repository's frozen
`vqa_evaluation.py` implementation.

| Model/run | Joint compatible | C/P/W | Exact match | Mean token F1 | Gate |
|---|---:|---:|---:|---:|---|
| Gemma, existing ROCm | 3/3 | 1/1/1 | 0.0000 | 0.0988 | Fail |
| Gemma, new CUDA `a2` | 3/3 | 1/1/1 | 0.0000 | 0.0702 | Fail |
| Phi, existing ROCm | 3/3 | 0/2/1 | 0.0000 | 0.0150 | Fail |
| Phi, new CUDA `a4` | 3/3 | 0/2/1 | 0.0000 | 0.0678 | Fail |

The CUDA runs confirm that the earlier conclusion is not an AMD-only runtime
artifact. Both models execute joint requests correctly on NVIDIA GPUs but
remain semantically weak on this panel.

## Matched existing baselines

The following rows were extracted for the same three video IDs from
`data/vqa/benchmarks/runs/native-audio-benchmark-v2/scored-predictions.jsonl`
(SHA-256
`c95669135b29551ef407e2889ac5e203053e944b00a0968cbe20da1fdfd0be7a`).
Manual C/P/W judgments use the same rubric.

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
are 2/3 coverage and are not a comparable aggregate.

No Qwen3-Omni semantic answer was available: the issue #5 CUDA attempt entered
generation on all three rows but failed on an FP32/BF16 dtype mismatch before
joint first-forward attestation.

## Runtime comparison

All successful CUDA rows attested CUDA 12.4, PyTorch 2.5.1+cu124, BF16, one
generation call, and 12 vision plus one audio element in the same first forward
pass.

| Model/platform | GPU | Model load | Lexical | Prosody | Sound | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|
| Gemma ROCm | Radeon 8060S | 9.02 s | 245.92 s | 28.19 s | 12.46 s | 16.88 GiB |
| Gemma CUDA `a2` | L40S 48 GB | 12.00 s | 19.45 s | 24.67 s | 5.87 s | 17.35 GiB |
| Phi ROCm | Radeon 8060S | 8.99 s | 53.45 s | 214.97 s | 114.04 s | 55.35 GiB |
| Phi CUDA `a4` | A100 80 GB | 15.82 s | 6.36 s | 20.01 s | 14.99 s | 55.33 GiB |

Phi CUDA generation was approximately 8.41×, 10.75×, and 7.61× faster than
the corresponding ROCm rows. Gemma CUDA was approximately 12.64×, 1.14×, and
2.12× faster. These are diagnostic observations, not pure hardware
benchmarks: the Gemma runs used adjacent Transformers versions, their generated
answers differ, and CUDA `a2` raised only Gemma's output ceiling.

## CUDA/dstack execution evidence

Every task used an immutable image digest, pinned model revision, on-demand
RunPod or Vast.ai capacity, no application ports, a one-hour duration cap, and
a private run-specific HF Bucket prefix.

Phi used Transformers 4.48.2 and an A100/H100/H200 80 GB pool capped at
`$2.00/hour`. Gemma used Transformers 5.14.1, timm 1.0.28, librosa 0.11.0,
and a closed NVIDIA pool with at least 40 GB VRAM capped at `$1.50/hour`.

| Run | UTC interval | Cost | Terminal finding | `SHA256SUMS` file SHA-256 |
|---|---|---:|---|---|
| Phi `a1` | 20:12:41–20:14:47 | $0.0489 | Preflight required an unused source video | `c598ffa6180f800763a35d525c55b579a39dce7cd80d539c85e2a7ad051e8207` |
| Phi `a2` | 20:16:13–20:18:44 | $0.0598 | Source-video requirement remained at probe boundary | `5d3b18d152a5a031ac6c4e34eb6df7151663b15cccdee383ec593916f2b6ff1c` |
| Phi `a3` | 20:20:27–20:22:30 | $0.0484 | Pinned model load lacked SciPy | `5e7fb63c427495db893e13830c770463e4bf9c84563c4a0f6d3ea861270d464a` |
| Phi `a4` | 21:09:48–21:13:38 | $0.0951 | Success; 3/3 joint rows | `a26be1889fccc22e1c694ad8d264f04b9a02374da802824d0af1ad994e08bac7` |
| Gemma `a1` | 21:29:20–21:33:07 | $0.0625 | 2 rows stopped at the 160-token ceiling | `af9ddece02b69f57f61750188e2e5e4d4f7a900b474627698780662f207b23b8` |
| Gemma `a2` | 21:39:41–21:43:52 | $0.0690 | Success; 3/3 joint rows | `4f75bc2fb8dc20fa895ae46bf88c851a8193d8d1fb9e915cdc59d8ed19067d55` |

Gemma `a1` isolated a deterministic output-length failure: lexical and prosody
each decoded exactly 160 tokens and ended mid-fenced JSON; sound ended
naturally at 39 tokens. `a2` changed only Gemma's ceiling from 160 to 320.
Every `a1` raw output is an exact prefix of its `a2` counterpart, and the same
greedy generation completed at 163, 237, and 39 tokens. Prompt, model revision,
media, precision, and decoding mode remained unchanged.

Total billed cost for all six issue runs was **$0.3837**. Phi `a4` plus both
Gemma attempts used **$0.2266** of the additional `$10.00` authorization.
Every downloaded artifact file passed its recorded checksum.

No raw logs, predictions, media, model weights, credentials, or dataset files
are tracked by this change.

## Recommendation

Keep both models as verified compatibility references, not promoted semantic
baselines. Do not run the 10–30 example expansion: both CUDA quality results
fail the frozen gate, and additional sampling was explicitly conditional on
both passing.
