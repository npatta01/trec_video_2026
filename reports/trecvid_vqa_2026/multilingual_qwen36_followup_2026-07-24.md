# Multilingual and Qwen3.6 follow-up

**Date:** 2026-07-24
**Scope:** What the completed frozen-30 v2 benchmark says about non-English
speech, and whether Qwen3.6 should be added to the next model comparison.

## Bottom line

1. **The v2 path did not assume English audio.** Its Whisper call explicitly
   set `task="translate"`, so automatically detected non-English speech was
   supposed to become timestamped English text before Qwen answered the
   English question.
2. **The run still does not validate multilingual accuracy.** The frozen
   manifest has no source-language field, the Whisper result does not retain
   detected language, and there is no reference transcript or translation.
   The 29/30 transcript acceptance result is a mechanical usability result,
   not a per-language speech-translation score.
3. **Qwen3.6 should have been considered.** The official 35B-A3B and 27B
   checkpoints were released on 2026-04-16 and 2026-04-22 respectively,
   three months before the 2026-07-21 benchmark plan. Their absence was not a
   release-timing issue; it was a shortlist/compute-scope oversight. The
   original experiment concentrated on 4B-class models and audio ablations,
   while Qwen3.6 currently has only substantially larger open checkpoints.
4. **Test Qwen3.6-27B first.** A one-video then three-video local ROCm probe is
   plausible on this 122 GiB unified-memory host. A full 30-row
   Qwen3.6-35B-A3B BF16 run is a larger-machine-preferred experiment because
   its resident weights are about 67.0 GiB before visual activations and
   runtime state, even though only 3B language parameters are active per
   token.

## What v2 actually does with non-English speech

The frozen implementation calls the Transformers Whisper pipeline with
timestamp output and `generate_kwargs={"task": "translate"}` in
[`vqa_whisper.py`](../../src/trec_video_2026/vqa_whisper.py). The exact source
file used for this audit has SHA-256
`3a8e94525a2111f76a22bf127a2cd6cc9e079c9269d5be2e3a5c6c00da384c41`.

This is important because Whisper distinguishes two tasks:

- transcription returns text in the source language;
- translation returns English text from non-English speech.

The official Whisper repository documents language identification and says
that a multilingual `large` model with `task="translate"` translates
non-English speech into English. The large-v3 card also says that source
language is detected automatically and that large-v3 is multilingual-only. It
warns that performance is uneven across languages and worse for lower-resource
languages and some accents
([official OpenAI Whisper repository](https://github.com/openai/whisper#command-line-usage),
[official Whisper large-v3 model card](https://huggingface.co/openai/whisper-large-v3#usage),
[limitations](https://huggingface.co/openai/whisper-large-v3#performance-and-limitations)).

After generation, the local quality gate accepts text unless it exceeds five
whitespace-delimited words per second or fewer than 80% of alphabetic
characters are Latin. Accepted text is attached to the otherwise identical
12-frame Qwen request; rejected text becomes `null`. The frozen run produced
29 accepted ASR outputs and rejected one repetition hallucination. This is
recorded in:

- `data/vqa/benchmarks/runs/native-audio-benchmark-v2/whisper-attempts.jsonl`
  (SHA-256
  `f893f3f9a15bc0b113cb1b1ee4db0845df13b798c6a99c18caf0bdd2a0cfbefa`);
- the approved
  [native-audio benchmark report](native_audio_model_benchmark_results_2026-07-21.md).

### What that result proves

- Whisper large-v3 translation completed reproducibly on ROCm/BF16 for all
  30 frozen audio requests.
- English text from 29 mechanically accepted results could be incorporated
  into the label-isolated Qwen evidence path.
- Across the mixed panel, this transcript arm improved aggregate exact match
  and token F1 relative to matched 12-frame visual-only Qwen.

### What it does not prove

- It does not establish which source languages were present. The resolved
  manifest records video ID, category, question, media metadata, and hashes,
  but no language label. Its SHA-256 is
  `f7d51e8e48bc4e3456277eff8deccd087095ef0f7d0d3501ad29d4445396b4ad`.
- It does not measure speech recognition or translation fidelity because the
  panel has no human source transcript or English translation reference.
- It does not show accuracy for any individual language, dialect, accent,
  code-switched clip, or low-resource language.
- The Latin-character rule is not a language detector or an accuracy metric.
  It can accept fluent but incorrect English, accept untranslated
  Latin-script languages, and reject legitimate names or mixed-script text.
- The downstream aggregate does not isolate multilingual benefit. It combines
  dialogue, non-speech sound/music, visual-control, spatial/OCR/color, and
  temporal/action questions, six of each.

The right interpretation is therefore: **the pipeline contains an intended
non-English-to-English route, but the completed panel validates compatibility,
not multilingual robustness.**

## Minimal multilingual evaluation

Use a **30-clip diagnostic panel**, explicitly marked as too small for a
leaderboard-quality multilingual claim:

1. Choose six human-verified source-language buckets with five clips each:
   English as the control; four languages observed in the actual TRECVID
   inventory; and one lower-resource or code-switched bucket. Do not choose
   the languages from Whisper output.
2. Require audible speech and at least one speech-dependent question per
   clip. Within every bucket include clean speech, noisy/music-overlaid
   speech, and at least one accent/dialect or code-switch case.
3. Create three human references without exposing them during inference:
   source language, verbatim source-language transcript, and faithful English
   translation. Retain the existing VQA answer annotation separately.
4. Run two mandatory matched conditions with the same frames and prompt:
   visual-only and visual plus Whisper large-v3 `task="translate"`.
   Optionally add same-language Whisper transcription followed by a separately
   pinned text translator; this distinguishes ASR failure from translation
   failure.
5. Score both layers:
   - source-language identification accuracy;
   - transcription WER/CER where a same-language arm exists;
   - English translation chrF plus a blinded human adequacy check;
   - downstream VQA EM, token F1, and METEOR, paired by clip.
6. Replace the Latin-script acceptance test for this panel with
   script-independent, recorded gates: no-speech probability, repetition or
   compression ratio, empty output, detected language, and a human-audited
   failure label. Never use the VQA reference answer to select a transcript.

Five examples per language can reveal broken language detection, empty
translation, hallucination, or catastrophic VQA regression. It cannot
estimate small per-language accuracy differences. Any promising condition
should therefore expand to at least 20 speech-critical clips per language
before becoming the default path.

## Why Qwen3.6 was not in the completed benchmark

The official Qwen release log dates `Qwen3.6-35B-A3B` to **2026-04-16** and
`Qwen3.6-27B` to **2026-04-22**
([Qwen3.6 official repository](https://github.com/QwenLM/Qwen3.6#news)).
The local benchmark plan is dated 2026-07-21 and scoped implementation to
Qwen3.5 4B/9B, Molmo2 4B, InternVL3.5 4B, Gemma 4 12B, MiniCPM-o 4.5, and
selected audio-native models
([local benchmark plan](../../docs/superpowers/plans/2026-07-21-phased-native-audio-model-benchmark.md)).

Consequently, the candid explanation is:

- **not timing:** Qwen3.6 was already public;
- **partly deliberate compute scope:** the requested first rung was a simple
  4B baseline and the experiment's stop rules favored cheap, non-redundant
  comparisons;
- **also an omission:** the shortlist should have recorded Qwen3.6 as the
  larger same-family rung, even if it deferred downloading or inference.

There is no official open 4B or 9B Qwen3.6 checkpoint in the current Qwen3.6
collection. The four official artifacts are BF16 and FP8 variants of 27B and
35B-A3B
([official Qwen3.6 collection](https://huggingface.co/collections/Qwen/qwen36)).

## What the official Qwen3.6 models support

| Official checkpoint | Architecture | Official repository weight bytes | Official video results |
|---|---|---:|---|
| `Qwen/Qwen3.6-27B` | Dense 27B language model plus vision encoder; 262,144-token native context | 55,563,006,400 BF16 bytes (51.75 GiB) | VideoMME with subtitles 87.7; VideoMMMU 84.4; MLVU 86.6; MVBench 75.5 |
| `Qwen/Qwen3.6-35B-A3B` | MoE, 35B total / 3B active language parameters plus vision encoder; 262,144-token native context | 71,903,776,776 BF16 bytes (66.97 GiB) | VideoMME 86.6 with and 82.5 without subtitles; VideoMMMU 83.7; MLVU 86.2; MVBench 74.6; LVBench 71.4 |
| `Qwen/Qwen3.6-27B-FP8` | Official FP8 quantization of 27B | 30,866,866,928 bytes (28.75 GiB) | Inherits the base checkpoint; the FP8 card does not publish a separate video table |
| `Qwen/Qwen3.6-35B-A3B-FP8` | Official FP8 quantization of 35B-A3B | 37,463,662,160 bytes (34.89 GiB) | Inherits the base checkpoint; the FP8 card does not publish a separate video table |

The byte totals above sum the `*.safetensors` sizes returned by the official
Hugging Face repository APIs
([27B](https://huggingface.co/api/models/Qwen/Qwen3.6-27B?blobs=true),
[35B-A3B](https://huggingface.co/api/models/Qwen/Qwen3.6-35B-A3B?blobs=true),
[27B-FP8](https://huggingface.co/api/models/Qwen/Qwen3.6-27B-FP8?blobs=true),
[35B-A3B-FP8](https://huggingface.co/api/models/Qwen/Qwen3.6-35B-A3B-FP8?blobs=true)).
Architecture and benchmark values come from the official
[27B model card](https://huggingface.co/Qwen/Qwen3.6-27B#model-overview) and
[35B-A3B model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B#model-overview).

Both are genuine video-capable vision-language models. The official API
example accepts `video_url`, defaults to 2 FPS, and allows the vLLM caller to
override frame sampling. The cards publish VideoMME, VideoMMMU, MLVU, and
MVBench results, so Qwen3.6 is relevant to TRECVID. The cards do **not**
document native audio input. Non-English speech still needs Whisper
translation, another ASR/translation route, or a separate audio-native model.

The official serving guidance recommends recent dedicated engines:
SGLang `>=0.5.10`, vLLM `>=0.19.0`, KTransformers, or latest Transformers.
Its full-context examples use tensor parallelism over eight GPUs, but that is
for the 262K context. A 12-frame TRECVID probe should cap context and output
length rather than treating the eight-GPU example as a minimum hardware
requirement
([Qwen3.6 serving and video guidance](https://huggingface.co/Qwen/Qwen3.6-35B-A3B#serving-qwen36)).

## Local-host recommendation

Read-only inspection on 2026-07-24 found 122 GiB total system/unified memory
and 117 GiB available. `rocminfo` identifies the Radeon 8060S as `gfx1151`.
The completed v2 run already demonstrated Qwen3.5-4B on this host with
Transformers `5.15.0.dev0`, PyTorch `2.10.0+rocm7.2.4`, ROCm/BF16, and no CPU
fallback. Its observed peak was 10.559 GiB at 12 frames and 15.302 GiB at
24 frames.

This makes **Qwen3.6-27B BF16 a plausible local compatibility test**, not a
promised production run. Start with one frozen video, then three, using the
same 12 frames, short context, batch size one, live device/dtype attestation,
and the existing stop rules. If it passes, a 10-row accuracy gate is more
informative than immediately running 30.

For **Qwen3.6-35B-A3B**, total resident weights matter even though only 3B
parameters are active per token. Its 66.97 GiB BF16 shard floor leaves
capacity on a 122 GiB unified-memory host, but visual activations, KV cache,
framework buffers, and ROCm kernel behavior are unmeasured. It may fit; it may
also become slower or less stable than the raw arithmetic suggests. Prefer a
larger discrete-GPU machine for the full benchmark, while allowing a
one-video local probe if Qwen3.6-27B succeeds first.

Do not assume the smaller FP8 downloads solve this on the Radeon. AMD's
ROCm 7.2.1 matrix lists `gfx1151` and this APU as supported but says only FP16
has been officially validated
([AMD ROCm 7.2.1 Ryzen support matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html)).
The prior benchmark empirically established BF16 for its pinned stack; it did
not establish Qwen3.6 FP8 kernels.

## Recommended next order

1. Add source-language metadata and reference translations to a 30-clip
   multilingual diagnostic; rerun visual-only versus Whisper-translation.
2. In parallel, perform a no-download planning check for Qwen3.6 integration,
   then a one-video Qwen3.6-27B BF16 ROCm probe only after explicitly
   authorizing the 55.6 GB model acquisition.
3. If the 27B probe is clean, run the same three videos and then a 10-row
   matched Qwen3.5-4B versus Qwen3.6-27B comparison.
4. Move Qwen3.6-35B-A3B and a full multilingual expansion to a larger machine
   unless local measurements establish safe memory and time headroom.
