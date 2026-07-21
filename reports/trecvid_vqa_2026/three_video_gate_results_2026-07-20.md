# Three-video VQA compatibility gate

Date: 2026-07-20

## Decision summary

The three-video gate is complete. Qwen 3.5 4B/9B, Molmo2 4B, InternVL3.5
4B, and Gemma 4 12B all completed every requested inference on the local AMD
Radeon 8060S through ROCm. MiniCPM-o 4.5 completed two of three and failed on
the speech-heavy example with a reproducible tensor-shape error.

This is a portability and pipeline test over three deliberately different
examples, not a statistical model ranking. It is too small to estimate
competition accuracy.

The strongest practical baseline remains Qwen 3.5 4B with 12 uniformly sampled
frames. Adding the quality-gated Whisper transcript did not improve these three
examples. The larger and specialist models did not change the qualitative
1-of-3 result, while requiring more memory or model-specific environments.

## Fixed examples and labels

These are 2025 training examples, so their labels are available locally. Labels
were kept out of every inference request and joined only during scoring.

| Video | Probe | Question | Official reference |
|---|---|---|---|
| `BDKkVYmWqdM` | temporal action | What does a bicyclist do before landing his bike on a tennis court? | The bicyclist drives his bike across a tennis net. |
| `O60MtMuSpf8` | spatial/color detail | What color is the image of the herpes virus? | Pale orange. |
| `Fk_j6re_Uiw` | speech/audio | What are the two voices arguing about? | The expected contents of the bag. |

## Result summary

“Qualitative” counts an answer as correct when it expresses the reference
meaning, even if exact-match scoring is zero. This judgment is included only to
make the three-example smoke test interpretable.

| Condition | Completed | Qualitative | Load | Per-video time after load | Peak GPU allocation |
|---|---:|---:|---:|---:|---:|
| Qwen 3.5 4B, 12 frames | 3/3 | 1/3 | 3.03 s | 10.1–338.4 s | 10.55 GiB |
| Qwen 3.5 4B, 12 frames + Whisper | 3/3 | 1/3 | shared | 9.9–10.8 s | 10.69 GiB |
| Qwen 3.5 9B, 12 frames + Whisper | 3/3 | 1/3 | 10.37 s | 11.4–379.8 s | 19.80 GiB |
| Molmo2 4B, 12 frames + Whisper | 3/3 | 1/3 | 6.98 s | 0.5–2.5 s | 9.90 GiB |
| InternVL3.5 4B, capped native tiling + Whisper | 3/3 | 1/3 | 4.33 s | 21.9–27.9 s | 23.14 GiB |
| MiniCPM-o 4.5, 12 frames + aligned native audio | 2/3 | 1/3 | 8.49 s | 16.8–22.1 s for passes | 19.57 GiB |
| Gemma 4 12B, 12 frames + Whisper | 3/3 | 1/3 | 7.08 s | 19.7–26.8 s | 25.74 GiB |

The 338–380 second Qwen first requests are ROCm kernel/JIT warm-up. Later
requests in the same loaded process take about 10–12 seconds. A production
batch must keep the model loaded; multiplying the first-request time by the
number of videos would be incorrect.

## Per-answer diagnostics

EM is normalized exact match. F1 is normalized token overlap. METEOR and
BERTScore reproduce the local portions of the pinned 2025 NIST evaluator. STS
was intentionally not run because the historical implementation sends answers
to a plaintext external HTTP service; single-answer NDCG is not meaningful.

| Condition | Video | Prediction | EM | F1 | METEOR | BERTScore F1 |
|---|---|---|---:|---:|---:|---:|
| Gemma 4 12B + Whisper | bike | He rides across the tennis net. | 0.000 | 0.500 | 0.527 | 0.924 |
| Gemma 4 12B + Whisper | color | The image of the herpes virus is reddish-pink and white. | 0.000 | 0.000 | 0.132 | 0.879 |
| Gemma 4 12B + Whisper | speech | The two voices are arguing about a “katta” (a type of firearm). | 0.000 | 0.154 | 0.187 | 0.872 |
| InternVL3.5 4B tiles + Whisper | bike | jump over tennis net | 0.000 | 0.364 | 0.199 | 0.913 |
| InternVL3.5 4B tiles + Whisper | color | Red | 0.000 | 0.000 | 0.000 | 0.887 |
| InternVL3.5 4B tiles + Whisper | speech | Pakistan | 0.000 | 0.000 | 0.000 | 0.831 |
| MiniCPM-o 4.5 native AV | bike | riding a bike across a tennis net | 0.000 | 0.667 | 0.607 | 0.943 |
| MiniCPM-o 4.5 native AV | color | The image of the herpes virus is pink. | 0.000 | 0.000 | 0.139 | 0.897 |
| MiniCPM-o 4.5 native AV | speech | **Failed:** tensor dimension 15 versus 14 | — | — | — | — |
| Molmo2 4B + Whisper | bike | Rides over the net | 0.000 | 0.200 | 0.160 | 0.878 |
| Molmo2 4B + Whisper | color | Red | 0.000 | 0.000 | 0.000 | 0.887 |
| Molmo2 4B + Whisper | speech | Pakistan's situation | 0.000 | 0.000 | 0.000 | 0.853 |
| Qwen 3.5 4B, frames only | bike | ride a bike across a tennis net | 0.000 | 0.667 | 0.607 | 0.946 |
| Qwen 3.5 4B, frames only | color | pink | 0.000 | 0.000 | 0.000 | 0.916 |
| Qwen 3.5 4B, frames only | speech | End of the trend | 0.000 | 0.286 | 0.280 | 0.902 |
| Qwen 3.5 4B + Whisper | bike | Riding a bike across a tennis net | 0.000 | 0.667 | 0.607 | 0.943 |
| Qwen 3.5 4B + Whisper | color | pink | 0.000 | 0.000 | 0.000 | 0.916 |
| Qwen 3.5 4B + Whisper | speech | Pakistan | 0.000 | 0.000 | 0.000 | 0.831 |
| Qwen 3.5 9B + Whisper | bike | riding a bike across a tennis net | 0.000 | 0.667 | 0.607 | 0.943 |
| Qwen 3.5 9B + Whisper | color | Brown. | 0.000 | 0.000 | 0.172 | 0.927 |
| Qwen 3.5 9B + Whisper | speech | Pakistan | 0.000 | 0.000 | 0.000 | 0.831 |

No answer is an exact string match, including the clearly correct bike/net
answers. That demonstrates why normalized EM cannot be the only decision
metric. Conversely, BERTScore remains above 0.83 for clearly wrong answers, so
it cannot be treated as correctness by itself either.

## Audio and evidence findings

- Every controlled condition used the same deterministic 12 bin-midpoint
  frames. The 36 cached images passed a manual contact-sheet inspection.
- Whisper large-v3-turbo produced a good English transcript for the bike clip:
  “He's riding a bike across a tennis net.”
- Whisper's color-clip transcript was rejected before inference as a repetitive
  hallucination (26.66 words/second).
- Whisper transcribed the speech clip primarily in Hindi script rather than
  translating it. It mentioned Pakistan and a bag-like term but did not recover
  the reference meaning. Supplying it caused several models to answer
  “Pakistan.”
- InternVL used up to two 448×448 tiles per selected frame (up to 24 tiles), not
  an additional set of sampled frames.
- MiniCPM received the same 12 frames interleaved with 12 midpoint-aligned 16 kHz
  audio chunks. This is selected-frame native audio, not raw end-to-end video
  decoding. Its speech-clip tensor-shape failure must be fixed before scaling.

## Reproducibility and compatibility

| Component | Pinned value |
|---|---|
| ROCm/PyTorch base | `rocm/pytorch@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e` |
| Main Qwen/Gemma Transformers | commit `cbf4d720ec734edb77d25452b2790c7e4be2f8d7` (`5.15.0.dev0`) |
| Molmo2 and InternVL Transformers | `4.57.1` |
| MiniCPM Transformers | `4.51.0` |
| NIST scorer submodule | `e06cb7854d7defe53e86412c2bba0ca5f636ea40` |
| BERTScore encoder | `FacebookAI/roberta-large@722cf37b1afa9454edce342e7895e588b6ff1d59`, layer 17 |
| Main runtime image | `sha256:510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5` |
| Molmo runtime image | `sha256:f5b655ab67cd62427819ac0992a626394f42cf786b4816132ae8610c47181fee` |
| InternVL runtime image | `sha256:64c040ec4e2cb76f055661c227c0fc456f8389091f4d3bad77bbcc643f28de12` |
| MiniCPM runtime image | `sha256:9fd8ffb14c39440b6a036781315558ae5ac6e93893582fff8d9eceaf8ec7a12a` |
| Gemma runtime image | `sha256:5bbd6f039622f70ad53d62395f189b07dc8fe28e2336c8aa462081abee9b7aaa` |

Model revisions are pinned in `configs/vqa/model_runs.json`. Raw append-only
attempts and transcripts remain under ignored `data/vqa/benchmarks/`; no media,
labels, or raw logs are committed.

## Recommendation before a 100-video run

Run Qwen 3.5 4B frames-only and Qwen 3.5 4B plus quality-gated Whisper first,
keeping the model loaded across the whole batch. They are the simplest and
lowest-risk comparison. Add Molmo2 as the inexpensive different-family check.
Do not include MiniCPM native audio until its speech-clip tensor mismatch is
understood. Treat the projected runtimes as approximate, but the measured
steady-state rates put each individual 100-video condition comfortably below
five hours on this host.

The 100-video experiment has not been started.
