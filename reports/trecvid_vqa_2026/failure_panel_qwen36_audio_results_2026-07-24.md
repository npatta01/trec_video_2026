# Failure-panel results: Qwen3.6, Whisper recovery, and native audio

**Date:** 2026-07-24
**Decision:** Keep a small, routed ensemble. Do not promote Qwen3.6-35B, Whisper
transcripts, or MiniCPM native audio as the global default.

## Executive result

The 12-video panel was deliberately selected from known failures: two
dialogue/prosody examples, four sound/music examples, four temporal-action
examples, and two visual controls. It is useful for failure diagnosis, but it
is not an unbiased estimate of full-test accuracy.

The strongest safe default tested is **Qwen3.5-4B with 24 sampled frames**.
**MiniCPM-o 4.5 with 12 frames and audio disabled** is an equally strong
alternative and wins more individual failures, but makes one worse temporal
error. **MiniCPM-o native audio** should be routed only to prosody and
non-speech-audio questions. A larger Qwen3.6 model did not improve aggregate
quality.

| Condition | Frames/audio | EM | token F1 | METEOR | Blind C/P/W |
|---|---|---:|---:|---:|---:|
| MiniCPM-o 4.5 | 12, audio off | **0.083** | **0.292** | **0.246** | 4 / 4 / 4 |
| Qwen3.5-4B | 24, visual | 0.000 | 0.175 | 0.134 | 4 / 4 / 4 |
| MiniCPM-o 4.5 diagnostic | 12 + native audio | 0.083 | 0.265 | 0.160 | 4 / 3 / 5 |
| Qwen3.5-4B | 12, visual | 0.000 | 0.106 | 0.122 | 1 / 4 / 7 |
| Qwen3.6-35B-A3B | 12, visual | 0.000 | 0.063 | 0.118 | 3 / 2 / 7 |
| Qwen3.5-4B + Whisper | 12 + transcript | 0.000 | 0.104 | 0.106 | 1 / 2 / 9 |

The first five-condition metric evaluation was run through the frozen,
independently reviewed scorer. The native-audio row is a later matched
diagnostic scored with the same metric implementations and pinned scorer
runtime; it was not inserted into the immutable five-condition evaluation.
Blind semantic grades use correct/partial/wrong and are reported separately
because lexical overlap sometimes rewards a semantically wrong answer.

## What the larger model showed

`Qwen/Qwen3.6-35B-A3B` at revision
`995ad96eacd98c81ed38be0c5b274b04031597b0` ran successfully on a dstack
H100-80GB in BF16. The runtime identifies the internal architecture as
`qwen3_5_moe`; that is the model implementation string, not a different public
model release.

The run was technically clean:

- all 12 label-free requests passed in the frozen order;
- all requests used the same 12 frames as the baseline;
- CUDA, H100 placement, BF16 parameter dtypes, model identity, output identity,
  workload return code, and artifact-sync return code were attested;
- downloaded checksums matched the private Hugging Face Bucket artifacts;
- approximate dstack cost was $0.31.

It recovered three failures relative to the 12-frame Qwen3.5 baseline: popping
sound, bottle opening, and the sour facial reaction. It did not improve over
the 24-frame model overall and lost the Christmas-ornament visual control by
answering only “a bird.” Its exact music and instrument answers were confident
but wrong.

**Conclusion:** model size alone is not the limiting factor on this panel.
Temporal coverage and modality-specific routing matter more.

## Whisper bug: diagnosis and repair

The suspect audio was not corrupt. `EMI_anpHII4` contains a valid 30.0118-second
PCM waveform. Whisper large-v3 deterministically produced a 284-word,
approximately 60-second repetition loop from the music/no-speech input.

The failure was caused by missing acceptance gates around a known class of ASR
hallucination:

- decoded timestamps could extend well beyond source duration;
- repeated phrases/windows were not rejected;
- boilerplate and implausible words-per-second were not rejected;
- empty/no-speech outputs were not treated as a valid rejected transcript;
- forced English translation discarded useful source-language diagnostics.

The repaired ASR seam now keeps native transcription and English translation
separate, records detected language and diagnostics, and rejects timestamp
overrun, excessive speech rate, no-speech/empty output, boilerplate,
compression anomalies, and repeated text. With these gates:

- large-v3's short “Thank you…” result was rejected for timestamp overrun;
- large-v3-turbo's empty result was rejected;
- the hallucinated transcript was not allowed into downstream VQA.

Whisper still should not be added globally. On this panel, the old Whisper
condition was a semantic net loss and changed the exact mask count from one to
two.

## Non-English speech

Qwen3-ASR-1.7B was tested as a different multilingual ASR family on ROCm. It
avoided Whisper's repetition loop and produced coherent native-language
transcripts for the French and Malay samples. The resulting two Qwen3.5 VQA
answers were still wrong or incomplete, so better transcription alone did not
solve answer selection.

This separates two problems:

1. **Speech recovery:** language-aware ASR can recover what was said.
2. **Question answering:** the VLM must ground the answer in the relevant
   utterance instead of inventing a nearby fact.

For speech-dependent questions, use timestamped multilingual ASR plus explicit
answer grounding. Do not force every transcript to English before quality
checks, and do not send rejected transcripts downstream.

## Native-audio diagnostic

MiniCPM-o 4.5 at revision
`1f761131fa83f5ed3cd6f2f22b225c4501d154fa` ran all 12 examples locally with
12 frames plus each full 16-kHz audio stream. The pinned container was
`localhost/trec-vqa-gate-minicpm@sha256:9fd8ffb14c39440b6a036781315558ae5ac6e93893582fff8d9eceaf8ec7a12a`.
The model attested ROCm, `cuda:0`, and BF16, with about 21.3 GB peak memory.

The first launch failed because the worktree symlink target was not mounted at
the canonical media path inside the container. No model inference ran in that
attempt. The corrected read-only canonical mount passed all 12. This was a
container-path error, not a ROCm compatibility problem.

| ID | Reference summary | Native-audio answer | Blind grade |
|---|---|---|---|
| NN7G28sBOlY | bystander says she does not know | “coucou Dior” | Wrong |
| NUmb2P3Bzu8 | louder/faster, emphatic gestures | raises voice and gestures emphatically | Correct |
| IRTTtnSRTyY | rubber bubbles popping | popping sound | Correct |
| IWSPAfCKtV8 | church bell | wedding march music | Wrong |
| MdMKHhFFqJQ | electronic instrument | organ | Partial |
| OzNdzgQi1CA | *Adagio for Strings* | background music is playing | Partial |
| CH8Dp7EgW4I | smaller boy dances | falls to the ground | Wrong |
| EtNOkUJd5w8 | opens bottle | opened bottle cap | Correct |
| Gp4LVTn3YbY | chews and says it is sour | drinks water | Wrong |
| KgOvCPOX-Ms | Kung Fu sword jump | standing apart | Wrong |
| EMI_anpHII4 | Christmas ornament | robin ornament | Partial |
| LusRurRrmf0 | one mask | one | Correct |

Relative to matched MiniCPM audio-off, native audio improved emotional prosody,
instrument-family plausibility, and music detection, but lost the church-bell
row and two temporal rows. On the first six audio-heavy rows, its blind ordinal
score rose from 4 to 6; on all 12, it fell from 12 to 11.

**Conclusion:** native audio is a useful category-gated specialist, not a
global switch. Audio conditioning can distract temporal visual reasoning.

## Failure audit and next experiments

An independent blind advisor reviewed every answer without model identities.
Its preferred operational ranking was:

1. Qwen3.5-4B, 24 frames — 4 correct / 4 partial / 4 wrong, no control loss.
2. MiniCPM-o 4.5, audio off — also 4/4/4 and the most pairwise wins, but one
   temporal regression.
3. Qwen3.6-35B-A3B — 3/2/7 and one control regression.
4. Qwen3.5-4B, 12 frames — 1/4/7.
5. Qwen3.5-4B + Whisper — 1/2/9 and one control regression.

The advisor's failure-specific next steps are:

| Failure type | Best next method | Why |
|---|---|---|
| Spoken factual answer | timestamped multilingual ASR + utterance grounding | frames cannot recover “doesn't know” |
| Emotion/prosody | native-audio specialist | words alone omit loudness, pace, and emphasis |
| Pops, bells, instruments | native audio or audio classifier with abstention | ASR is the wrong tool for non-speech sound |
| Exact composition title | audio fingerprinting/retrieval | general VLMs hallucinated plausible titles |
| After/before and brief actions | 24 frames, event-centered clips, then temporal video model | 24 frames recovered several short actions |
| Small crafted object | optional high-resolution crop/tiles | tiling is not justified globally |
| OCR | no experiment now | none of the audited failures depends on visible text |

## Recommended baseline architecture

Use a simple router rather than one condition for every question:

1. **Default:** Qwen3.5-4B with 24 uniform frames.
2. **Visual cross-check:** MiniCPM-o 4.5 with 12 frames, audio off.
3. **Prosody/non-speech route:** MiniCPM-o native audio; keep the visual answer
   when the audio specialist is outside its route or makes a precise unsupported
   claim.
4. **Speech route:** quality-gated multilingual ASR, followed by a prompt that
   requires the answer to be supported by a timestamped utterance.
5. **Exact music route:** fingerprint/retrieval or abstention; do not ask a
   general VLM to guess a title.
6. **Mandatory controls:** reject a configuration that loses the ornament or
   exact mask-count controls.

Before full-dataset inference, evaluate this router on a larger stratified
sample. A practical next gate is 30–100 examples with explicit speech,
non-speech, temporal, fine-detail, and control strata. Keep automatic EM/F1/
METEOR, but promote only when blinded semantic review agrees.

## Reproducibility record

- Panel manifest SHA-256:
  `ad76ec57b964bf391f763e7f82b1671475cddaa90814727d9c40ee7c6310ccbd`
- Inference manifest SHA-256:
  `d2c5515b8c7398f326931e2f993f3fd6eac8e70d9ef01d1a1bd4afcf3f6792e8`
- Frozen scorer SHA-256:
  `90a6da4ea4302f851a59316e98c82d5be979fd93f589fc1001967f17731fdf24`
- Frozen five-condition selection SHA-256:
  `fdf669d199d5d90fa37bacb533a2303a217180040e67583b5ef9a75d2d06912f`
- Qwen3.6 downloaded predictions SHA-256:
  `6728283c2c1e53427e5fc4531eea1144149377a213bd8c92c573d56d8593a825`
- Frozen evaluation SHA-256:
  `dd4c5c07b9b169446a87ba0c981c669b0ef207b718a799deb6f6d69fc911927b`
- Fixed Whisper probe artifact SHA-256:
  `92e93758603a83baa2a42630bf7ef39f2c6463c71a14328258374bbceb03004f`
  (`data/vqa/benchmarks/followups/failure-panel-2026-07-24/whisper-fixed-probes/emi-fixed-whisper-probes-20260724.json`)
- MiniCPM native-audio raw artifact SHA-256:
  `0f85d50e246c00a98278a280743c8ca9f4dfcabd01f0ab97c7a33feba7c9817f`
  (`data/vqa/benchmarks/followups/failure-panel-2026-07-24/native-audio-diagnostic/minicpmo45-audio-on-failure-panel-20260724-a2.json`)
- MiniCPM native-audio scored diagnostic SHA-256:
  `3a475b88f95b1fa5b6b73265bd92d40b3288e06a7a898d88dfa53a362c4ee4a3`
  (`data/vqa/benchmarks/followups/failure-panel-2026-07-24/native-audio-diagnostic/scored-minicpmo45-audio-on-failure-panel-20260724-a2.json`)
- Qwen3.6 private artifact prefix:
  `hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/qwen36-failure-panel-20260724-a10`

The scorer boundary, hashes, label isolation, order, runtime pinning, and metric
parity were independently reviewed before evaluation. A second independent
advisor performed blind semantic adjudication after scoring.
