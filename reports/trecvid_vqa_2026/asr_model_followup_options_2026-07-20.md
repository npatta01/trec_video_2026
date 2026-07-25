# ASR follow-up options after the three-video gate

Date: 2026-07-20

## Recommendation

Yes: run one more **three-video audio-only compatibility test**, with the
same cached 16 kHz mono WAV files and the same transcript quality gate. Do not
change the VLM yet. Rank the candidates as follows:

1. **Whisper large-v3 (`openai/whisper-large-v3`)** — the best next test.
2. **SeamlessM4T v2 Large (`facebook/seamless-m4t-v2-large`)** — the best
   different-family comparison, if its non-commercial license is acceptable.
3. **IndicConformer 600M + IndicTrans2 Indic-to-English 200M** — a diagnostic
   Hindi/Urdu-specialist cascade if the first two fail and the source language
   can be confirmed.

Only promote an ASR option into the 100-video benchmark if it (a) produces a
usable English rendering of the speech clip, (b) does not worsen the other two
clips, and (c) improves the downstream answer rather than merely passing the
script-based quality check.

## Why Whisper large-v3 should be first

The completed gate already requested `task="translate"`; this was not a prompt
configuration mistake. Nevertheless, `large-v3-turbo` is a pruned and
fine-tuned derivative of full `large-v3`: its decoder has **4 layers rather
than 32**, with the publisher describing the tradeoff as much higher speed for
minor quality degradation. It has 0.8B parameters. [Whisper large-v3-turbo
model card](https://huggingface.co/openai/whisper-large-v3-turbo)

Full `large-v3` is 1.55B parameters, explicitly supports speech translation to
English through `task="translate"`, and its weights and official repository are MIT
licensed. Its model card reports a 10–20% error reduction across many languages
relative to large-v2. [Whisper large-v3 model
card](https://huggingface.co/openai/whisper-large-v3)

This gives the cleanest controlled comparison: same Whisper preprocessing and
generation interface, but the unpruned decoder. It is not guaranteed to fix
this clip—Whisper's publisher warns that accuracy varies by language and that
hallucination/repetition can occur—but it is the cheapest experiment that can
answer whether turbo's decoder pruning caused the failed English translation.

Hardware impact should be modest on this host. The Hub's FP16 safetensors file
for full large-v3 is 3.09 GB, versus a 0.8B-parameter turbo model; locally,
turbo peaked at 2,159,425,536 bytes (about 2.01 GiB) on ROCm for each cached
clip. Full large-v3 should require materially more memory and decoding time,
but remains small relative to the 10–26 GiB VLM peaks already observed. The
exact ROCm peak and runtime are unknown until measured. [Whisper large-v3
files](https://huggingface.co/openai/whisper-large-v3/tree/main)

## Why SeamlessM4T is the second test

SeamlessM4T v2 Large is a 2.3B-parameter model designed explicitly for
speech-to-text translation as well as ASR. It accepts speech in 101 languages;
its support table lists both Hindi and Urdu as speech inputs and English as a
text target. That makes it the strongest direct architectural countercheck for
the problematic Hindi/Urdu-like clip. [SeamlessM4T v2 Large model
card](https://huggingface.co/facebook/seamless-m4t-v2-large)

The costs are higher than the Whisper swap: the Transformers safetensors total
about 9.24 GB, so its memory footprint is likely several times turbo's, and it
requires a separate adapter rather than the current Whisper runner. The host's
unified memory makes a three-clip test plausible, but ROCm compatibility is not
established. More importantly, the weights are **CC-BY-NC-4.0**, so it should
only be used if the project's intended use is compatible with a non-commercial
license. [SeamlessM4T v2 Large files and
license](https://huggingface.co/facebook/seamless-m4t-v2-large/tree/main)

## Why the AI4Bharat cascade is third

AI4Bharat's IndicConformer is a 600M-parameter, MIT-licensed hybrid CTC/RNNT ASR
model covering all 22 scheduled Indian languages, explicitly including Hindi
and Urdu. It transcribes rather than translates. [IndicConformer 600M model
card](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)

Its output can be translated with the MIT-licensed 200M-parameter
`indictrans2-indic-en-dist-200M`, which documents Hindi-to-English inference
and supports the Indic-to-English direction. [IndicTrans2 distilled 200M model
card](https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M)

This pair is potentially more language-specialized and is only about 800M
parameters in aggregate, but it is a less clean baseline:

- the ASR call requires an explicit source-language code, so Hindi versus Urdu
  must be identified or both must be tested;
- it introduces two separately fallible stages;
- both Hub repositories are gated, and the ASR uses trusted custom code;
- the documented GPU package path includes `onnxruntime-gpu`, so ROCm support
  must be proven rather than assumed.

Use this cascade to diagnose the hard clip, not as the first general-purpose
replacement.

## Current ASR models that are not direct replacements

Two newer open models are attractive for native-language transcription, but
their publishers do not document direct speech-to-English translation:

- **Qwen3-ASR 1.7B** is Apache-2.0 and documents language identification and
  speech recognition for 30 languages plus 22 Chinese dialects. Hindi is in
  its supported list, but Urdu is not, and its public interface is
  `transcribe`; the model card does not claim a translation task. It could
  replace IndicConformer on a confirmed Hindi clip, followed by IndicTrans2,
  but it does not answer the immediate direct-translation question as cleanly
  as full Whisper or SeamlessM4T. [Qwen3-ASR 1.7B model
  card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- **Omnilingual ASR** is Apache-2.0 and covers more than 1,600 languages, with
  model sizes from 300M CTC through 7B LLM-ASR. Its official examples produce
  source-language `transcriptions`, not English translations. The documented
  300M CTC model uses about 2 GiB VRAM on an A100, but the stack is based on
  fairseq2 and its ROCm behavior here is unknown. It is a valuable fallback for
  identifying/transcribing an unusual source language, followed by a text
  translator—not the second condition in this small direct-translation test.
  [Omnilingual ASR official
  repository](https://github.com/facebookresearch/omnilingual-asr)

## Proposed three-video decision test

Keep the existing turbo outputs frozen and generate only two new transcript
sets initially:

| Condition | Required output | Advance criterion |
|---|---|---|
| Whisper large-v3, auto language, `task="translate"` | Timestamped English text for all 3 clips | Speech clip is intelligible English and downstream Qwen 3.5 4B answer improves |
| SeamlessM4T v2 Large, speech-to-text, target `eng` | English text for all 3 clips | Same criterion; note non-commercial license in results |

If neither succeeds, first inspect/listen to the speech clip to confirm whether
the language is Hindi, Urdu, code-switched, or something else. Then test the
AI4Bharat cascade with the confirmed language (or a clearly labeled Hindi/Urdu
A/B). Reuse the existing quality checks, but add a downstream semantic check:
Latin script alone proves only that the result looks English, not that it is a
faithful translation.

## Evidence from the completed local gate

- The current implementation already passes
  `generate_kwargs={"task": "translate"}`.
- Turbo took 7.62–13.38 seconds per cached clip and peaked at about 2.01 GiB on
  ROCm.
- It produced usable English for one clip, a repetitive hallucination for one,
  and non-English-script output for the speech clip.

These local observations come from `src/trec_video_2026/vqa_whisper.py`, the
ignored transcript artifacts under `data/vqa/benchmarks/transcripts/`, and the
[three-video gate report](three_video_gate_results_2026-07-20.md). No models
were downloaded or executed for this follow-up research.
