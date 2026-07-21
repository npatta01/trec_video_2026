# Three-video VQA compatibility gate

Date: 2026-07-20

## Decision summary

The three-video gate is complete. Qwen 3.5 4B/9B, Molmo2 4B, InternVL3.5
4B, and Gemma 4 12B all completed every requested inference on the local AMD
Radeon 8060S through ROCm. MiniCPM-o 4.5 completed two of three and failed on
the speech-heavy example with an observed tensor-shape error.

This is a portability and pipeline test over three deliberately different
examples, not a statistical model ranking. It is too small to estimate
competition accuracy.

The strongest practical baseline remains Qwen 3.5 4B with 12 uniformly sampled
frames. The intended all-English Whisper condition was not achieved: quality
control accepted English for one clip and rejected two clips. In the Qwen 4B
frames-only counterfactual, that quality-gated ASR policy did not improve these
three examples. The larger and specialist models did not change the qualitative
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
| Qwen 3.5 4B, 12 frames + quality-gated ASR (1/3 present) | 3/3 | 1/3 | 3.03–4.79 s | 9.9–423.8 s | 10.69 GiB |
| Qwen 3.5 9B, 12 frames + quality-gated ASR (1/3 present) | 3/3 | 1/3 | 7.46–10.37 s | 11.4–379.8 s | 19.80 GiB |
| Molmo2 4B, 12 frames + quality-gated ASR (1/3 present) | 3/3 | 1/3 | 6.00–6.98 s | 0.5–2.5 s | 9.90 GiB |
| InternVL3.5 4B, capped native tiling + quality-gated ASR (1/3 present) | 3/3 | 1/3 | 3.75–4.33 s | 21.9–27.9 s | 23.14 GiB |
| MiniCPM-o 4.5, 12 frames + aligned native audio | 2/3 | 1/3 | 8.49 s | 16.8–22.1 s for passes | 19.57 GiB |
| Gemma 4 12B, 12 frames + quality-gated ASR (1/3 present) | 3/3 | 1/3 | 7.08–8.83 s | 18.5–25.7 s | 25.74 GiB |

The 338–424 second Qwen first requests are consistent with one-time ROCm
kernel/JIT warm-up, but the cause was not independently instrumented. Later
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
| Gemma 4 12B + gated ASR | bike | He rides across the tennis net. | 0.000 | 0.500 | 0.527 | 0.924 |
| Gemma 4 12B + gated ASR | color | The image of the herpes virus is reddish-pink and white. | 0.000 | 0.000 | 0.132 | 0.879 |
| Gemma 4 12B + gated ASR | speech | There are no voices arguing in the video. | 0.000 | 0.000 | 0.139 | 0.896 |
| InternVL3.5 4B tiles + gated ASR | bike | jump over tennis net | 0.000 | 0.364 | 0.199 | 0.913 |
| InternVL3.5 4B tiles + gated ASR | color | Red | 0.000 | 0.000 | 0.000 | 0.887 |
| InternVL3.5 4B tiles + gated ASR | speech | The two voices are arguing about the person's actions, specifically whether they are being lazy or not. | 0.000 | 0.000 | 0.181 | 0.876 |
| MiniCPM-o 4.5 native AV | bike | riding a bike across a tennis net | 0.000 | 0.667 | 0.607 | 0.943 |
| MiniCPM-o 4.5 native AV | color | The image of the herpes virus is pink. | 0.000 | 0.000 | 0.139 | 0.897 |
| MiniCPM-o 4.5 native AV | speech | **Failed:** tensor dimension 15 versus 14 | — | — | — | — |
| Molmo2 4B + gated ASR | bike | Rides over the net | 0.000 | 0.200 | 0.160 | 0.878 |
| Molmo2 4B + gated ASR | color | Red | 0.000 | 0.000 | 0.000 | 0.887 |
| Molmo2 4B + gated ASR | speech | The nature of reality | 0.000 | 0.286 | 0.149 | 0.911 |
| Qwen 3.5 4B, frames only | bike | ride a bike across a tennis net | 0.000 | 0.667 | 0.607 | 0.946 |
| Qwen 3.5 4B, frames only | color | pink | 0.000 | 0.000 | 0.000 | 0.916 |
| Qwen 3.5 4B, frames only | speech | End of the trend | 0.000 | 0.286 | 0.280 | 0.902 |
| Qwen 3.5 4B + gated ASR | bike | Riding a bike across a tennis net | 0.000 | 0.667 | 0.607 | 0.943 |
| Qwen 3.5 4B + gated ASR | color | pink | 0.000 | 0.000 | 0.000 | 0.916 |
| Qwen 3.5 4B + gated ASR | speech | End of the trend | 0.000 | 0.286 | 0.280 | 0.902 |
| Qwen 3.5 9B + gated ASR | bike | riding a bike across a tennis net | 0.000 | 0.667 | 0.607 | 0.943 |
| Qwen 3.5 9B + gated ASR | color | Brown. | 0.000 | 0.000 | 0.172 | 0.927 |
| Qwen 3.5 9B + gated ASR | speech | The trend. | 0.000 | 0.000 | 0.152 | 0.919 |

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
- Whisper transcribed the speech clip in Hindi script instead of producing the
  required English translation. The advisor caught that the initial rate-only
  check accepted it. A new language-script check rejected it, and every affected
  speech row above was rerun without that text.
- InternVL used up to two 448×448 tiles per selected frame (up to 24 tiles), not
  an additional set of sampled frames.
- MiniCPM received the same 12 frames interleaved with 12 midpoint-aligned 16 kHz
  audio chunks. This is selected-frame native audio, not raw end-to-end video
  decoding. Therefore the originally planned raw native-video MiniCPM condition
  remains untested. Its observed speech-clip tensor-shape failure must be fixed
  before scaling this selected-frame adapter.

## Compatibility history

The append-only log contains 65 attempts: 25 passes and 40 development-time
failures. Terminal results above select the latest pass for each
condition/video and never overwrite earlier failures. The failures were useful
portability evidence:

| Model | Intermediate failure | Resolution or terminal status |
|---|---|---|
| Qwen 3.5 | None | Main pinned runtime worked. |
| Molmo2 | Transformers 5 optional-processor, RoPE, cache-position, and causal-mask API breaks | Isolated documented Transformers 4.57.1 runtime; all three passed. |
| InternVL | Missing `timm`, stdout protocol contamination, unsupported SDPA, then Transformers 5 custom-code incompatibility | Added pinned `timm`, reserved stdout for JSON, selected eager attention, and isolated Transformers 4.57.1; all three passed. |
| MiniCPM-o | Removed `pkg_resources`, offline lazy processor reload, wrong generic processor class | Pinned setuptools/Transformers 4.51.0 and loaded MiniCPM's processor from the immutable local snapshot; two passed. The speech tensor mismatch remains. |
| Gemma 4 | Tokenizer construction under the non-canonical uppercase repository ID reported missing SentencePiece or tiktoken; Qwen-only pixel kwargs also broke processor construction | Added pinned `tiktoken==0.13.0`, used canonical `google/gemma-4-12b-it`, and omitted the Qwen pixel kwargs; all three passed. |

## Reproducibility boundary

Model revisions, container parents, and execution image references are now
digest-pinned. The runner now records the resolved runtime image reference;
the corrective Molmo/InternVL/Gemma attempts include it, while the corrective
Qwen attempts finished just before that instrumentation was added. Transcript
artifacts record `video_id` plus the source-audio SHA-256.

The current CLI still exposes only `preflight`; the completed experiment was
orchestrated through the tested Python seams and this analytical report was
assembled from the append-only log. A single documented `prepare/run/report`
command and a full resolved artifact bundle remain required before a 100-video
run. This report therefore documents an audited executed gate, not yet a
one-command reusable benchmark workflow.

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

After the missing reusable workflow is implemented, run Qwen 3.5 4B frames-only
and Qwen 3.5 4B plus quality-gated ASR first, keeping the model loaded across
the whole batch. They are the simplest and lowest-risk comparison. Add Molmo2
as the inexpensive different-family check. Do not include MiniCPM selected-frame
audio until its speech-clip tensor mismatch is understood, and treat raw native
video as a separate untested condition. Runtime projections exclude downloads
and setup failures; the successful measured steady-state rates put each
individual 100-video inference condition comfortably below five hours on this
host.

The 100-video experiment has not been started.
