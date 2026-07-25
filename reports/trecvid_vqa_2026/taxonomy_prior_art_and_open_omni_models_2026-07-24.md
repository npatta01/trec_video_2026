# TRECVID 2025 question routing and open-weight omni models

**Research date:** 2026-07-24
**Scope:** official TREC/NIST proceedings and first-party model cards/repositories only.

## Executive answer

- The question taxonomy is not an arbitrary invention. The official TRECVID 2025 overview used seven categories: temporal/order/attribute, causal/plot/goal, audio/multimodal, multi-hop, object–people interaction, counting, and common-sense. One of the strongest teams also classified questions before inference.
- Last year's systems used most of the ingredients under consideration: question-type prompting, Whisper transcripts, native audio-video input, denser frames, spatial tiling, multiple frame subsets, voting, logit pooling, and answer reranking.
- No official 2025 team paper describes a complete router that conditionally chooses the model family, native audio versus ASR, or frame density for each question. The closest precedent is NII_UIT's question-type-specific prompt routing.
- There are genuinely open-weight omni candidates. The most practical next non-Qwen-family probes are **Phi-4-multimodal-instruct** and **Gemma 3n E4B-it**. **Qwen3-Omni-30B-A3B-Thinking** is the strongest clearly documented native audio-video candidate for a larger dstack run. MiniCPM-o 4.5 is also genuinely end-to-end, but its language backbone is Qwen3-8B.

## 1. Did TRECVID 2025 systems use this taxonomy or related routing?

### Official taxonomy

The [official TRECVID 2025 VQA overview](https://trec.nist.gov/pubs/trec34/papers/Overview_vqa.pdf) defines seven question categories:

1. Temporal, order, and attribute
2. Causal, plot, and goal
3. Audio and multimodal
4. Multi-hop reasoning
5. Object and people interaction
6. Counting
7. Common-sense reasoning

This is useful as an **analysis taxonomy**. For inference routing, some categories should be made operational—for example, separate spoken-content questions from music/sound/prosody questions, because ASR only addresses the former.

### What the submitted systems actually did

| Official team | Taxonomy/routing | Audio treatment | Frame/evidence treatment | Aggregation |
|---|---|---|---|---|
| **NII_UIT** | Classified each question as temporal, counting, yes/no, or descriptive and changed its prompt profile | Whisper timestamped transcripts injected as text | 12 uniform frames plus spatial tiles | Output normalization |
| **CERTH-ITI** | No conditional question router documented | Native audio-video Qwen2.5-Omni-7B | Ten random frame subsets for every example | Majority vote for answer generation; mean logit pooling for multiple choice |
| **HLTCOE** | No taxonomy router documented | Compared Qwen2.5-VL and Qwen2.5-Omni; selected VL globally; the official overview labels the system “Qwen 2.5 VL + Whisper,” although the team paper does not document that Whisper stage | Uniform visual input | Learned candidate-answer reranker |
| **WHU-NERCMS** | No taxonomy router documented | Not described as a conditional audio branch | VideoLLaMA/Qwen2.5 model runs | Multiple runs/models, not question-conditioned |
| **KSLab** | No taxonomy router documented | Audio omitted and explicitly identified as a limitation | VideoLLaMA3-2B visual frames | Sample answers and confidence ranking |

Evidence:

- [NII_UIT, §§2.2–2.4 and §3.3, pp. 2–5](https://trec.nist.gov/pubs/trec34/papers/NII_UIT.vqa.pdf) documents 12-frame temporal densification, 2×2 or 3×3 tiling, Whisper transcripts, and temporal/counting/yes-no/descriptive prompt profiles. Its cumulative training-split ablation reports the largest final increment from question-type prompting: Top-1 `0.779 → 0.795`, MRR `0.866 → 0.876`.
- The same NII_UIT ablation shows only a small aggregate increment from Whisper: MC Top-1 `0.746 → 0.751` with MRR unchanged; its answer-generation STS changed `0.507 → 0.508`. This does **not** show that ASR is useless; it shows that applying it broadly produced a small average gain.
- [CERTH-ITI, §§3.1–3.3, pp. 5–6](https://trec.nist.gov/pubs/trec34/papers/CERTH-ITI.avs.vqa.pdf) used the Thinker component of Qwen2.5-Omni-7B, full-video audio embeddings, ten random frame-subset generations, and voting/pooling. Its paper reports the highest three answer-generation metrics among submissions.
- [HLTCOE, pp. 1–3](https://trec.nist.gov/pubs/trec34/papers/HLTCOE.vqa.pdf) compared Qwen2.5-VL and Qwen2.5-Omni and selected VL based on development metrics, then reranked generated answers. Its prompts explicitly account for multilingual ASR, but its implementation description does not clearly establish an ASR stage; this conflicts with the official overview's system summary and should not be overstated.
- [WHU-NERCMS, pp. 3–4](https://trec.nist.gov/pubs/trec34/papers/WHU-NERCMS.avs.vqa.pdf) and [KSLab, p. 3](https://trec.nist.gov/pubs/trec34/papers/kslab.vqa.pdf) do not document conditional routing. KSLab explicitly says excluding auditory information prevented answers based on dialogue, sound effects, or environmental audio.

### Conclusion for this repository

The taxonomy makes sense, but it should have two layers:

- **Stable reporting taxonomy:** the seven official categories, so results remain comparable and interpretable.
- **Operational routing tags:** `speech_content`, `non_speech_sound_or_music`, `prosody`, `temporal_order`, `counting`, `small_spatial_detail`, `global_visual`, and `multi_hop`. These tags describe which evidence path can answer the question.

This proposed evidence router goes beyond the published 2025 systems, but each component has direct prior art. The novelty is selecting components conditionally rather than applying every expensive component to every question.

## 2. What counts as an omni model?

Three materially different designs are often called “omni”:

1. **ASR + VLM pipeline:** speech is converted to text, then a visual model sees frames plus transcript. It loses non-speech sound, fine prosody, speaker overlap, and some timing information.
2. **One model with audio and vision encoders:** audio and frames enter the same language model, enabling joint reasoning, but video may still be represented as sampled images.
3. **Native audio-video model:** the supported interface accepts synchronized video and audio together and the model was trained for cross-modal/temporal reasoning.

For this task, designs 2 and 3 are more informative than ASR alone. ASR remains useful for exact multilingual dialogue.

## 3. Practical open-weight candidates as of 2026-07-24

| Candidate | Family / size | Joint audio-video status | License | Practical assessment |
|---|---|---|---|---|
| **Phi-4-multimodal-instruct** | Microsoft Phi, 5.6B | Same network accepts text, image, and audio; official image+audio “vision-speech” format. Video should be supplied as sampled frames rather than assumed to be a native video container. | MIT | **Best next non-Qwen probe.** Small enough for the local host. ROCm is plausible through PyTorch with eager/SDPA, but the official optimized path uses CUDA/FlashAttention, so require a 3-video compatibility test. |
| **Gemma 3n E4B-it** | Google Gemma, 8B raw / 4B effective | Official card declares text, image, video, and audio input in one model. Public examples are much clearer for individual image/audio modes than synchronized audio-video QA. | Gemma custom terms; gated acknowledgement | **Second non-Qwen probe.** Compact and multilingual. Test combined video/audio behavior rather than assuming that modality support guarantees strong temporal alignment. ROCm is plausible via Transformers BF16 but not officially attested. |
| **MiniCPM-o 4.5** | Qwen3-8B backbone, 9B total | Genuine end-to-end simultaneous continuous video+audio input; up to 10 fps video | Apache-2.0 | **Practical and already relevant, but not a non-Qwen family.** Official full-precision recommendation is NVIDIA; this repository has stronger direct runtime evidence than the model card alone. |
| **Qwen3-Omni-30B-A3B-Thinking** | Qwen MoE, 30B total / 3B active | Native audio, image, video, and text; official audio-visual QA examples and synchronized audio-in-video handling | Apache-2.0 | **Best large native-omni dstack candidate.** Not realistic for this local ROCm host in BF16. Official minimum is about 68.7 GB for 15 s and 77.8 GB for 30 s, so use an 80 GB GPU only for short clips or choose more headroom. |
| **Baichuan-Omni-1.5** | Named Baichuan, but Qwen2.5-7B language backbone | End-to-end image/video/audio/text, including continuous video/audio streaming | Apache-2.0 plus restrictive Baichuan community/commercial terms | Technically relevant but a lower-priority benchmark: not actually independent of Qwen, CUDA/FlashAttention-oriented setup, and less permissive operational terms. |

Primary sources:

- Microsoft states that [Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) has 5.6B parameters, MIT licensing, unified text/image/audio processing, a vision-speech input format, and multilingual ASR/translation support. Its card explicitly contrasts this joint representation with the older ASR-then-VLM pipeline.
- Google’s [Gemma 3n model card](https://ai.google.dev/gemma/docs/gemma-3n/model_card) states that Gemma 3n accepts text, image, video, and audio and generates text; the [E4B Hugging Face card](https://huggingface.co/google/gemma-3n-E4B-it) reports 8B raw parameters with 4B-effective operation.
- OpenBMB’s [MiniCPM-o repository](https://github.com/OpenBMB/MiniCPM-V#minicpm-o-45) documents a 9B end-to-end model built from SigLIP2, Whisper-medium, CosyVoice2, and Qwen3-8B, with simultaneous continuous video/audio processing and Apache-2.0 weights/code.
- Qwen’s [Qwen3-Omni repository](https://github.com/QwenLM/Qwen3-Omni) documents native audio-visual QA, Apache-2.0 checkpoints, and BF16 memory requirements by video duration.
- Baichuan’s [official repository](https://github.com/baichuan-inc/Baichuan-Omni-1.5) documents end-to-end video/audio support, the Qwen2.5-7B backbone, CUDA-oriented installation, and its additional community license conditions.

## Recommendation

Run a matched, label-blind compatibility panel in this order:

1. **Phi-4-multimodal-instruct:** sampled frames + native audio in one request.
2. **Gemma 3n E4B-it:** the same frames/audio/question, verifying that the runtime truly combines both modalities.
3. **Qwen3-Omni-30B-A3B-Thinking on dstack:** only after the small models pass input-contract checks; use a GPU with more than 80 GB for 30-second clips or reduce duration/frame load.

Score speech, sound/music, prosody, temporal, and visual controls separately. A better ASR model should be evaluated only on examples whose answer is actually present in speech; it cannot fix music identification, sound effects, prosody, missed visual events, or small-object failures.

## 4. Local audit of all 481 training questions

The repository contains 481 complete annotation records and 465 unique
case-normalized question strings. There are 16 repeated rows across six exact
duplicate-question groups. The largest repeated prompts are:

- “What is the video mainly about?” — 6 videos
- “What happens at the end of the video?” — 5 videos
- “What happens by the end of the video?” — 4 videos
- “What is the video about?” — 3 videos

A high-precision lexical pass finds 71 explicit counting prompts, 22 prompts
that explicitly request spoken language, 20 that explicitly mention sound or
music, and 15 that explicitly request visible text. These are overlapping,
lower-bound routing indicators rather than authoritative class counts. The
question alone can hide the required evidence: for example, a visually phrased
“what happens after” question can have an answer whose decisive evidence is a
spoken comment.

The four labels used for the 12-row failure panel—dialogue/speech,
sound/music, temporal/action, and visual controls—should therefore remain
**diagnostic panel strata**, not become the dataset's exclusive taxonomy. They
mix three different concepts:

1. **Evidence modality:** vision, spoken language, non-speech audio/music,
   visible text, or mixed evidence.
2. **Reasoning operation:** recognition/attribute, temporal order, counting,
   spatial reasoning, causal/goal reasoning, or multi-hop/common sense.
3. **Evaluation role and quality:** target failure, regression control,
   ambiguous/underspecified, subjective, or external-knowledge-dependent.

Use multi-label routing tags under the official seven reporting categories.
For example, “What text appears after Jiu Jitsu?” is both OCR and temporal;
“How many times does he shoot?” is visual, counting, and temporal. A router
should be allowed to abstain and use the general multimodal path when the
question text does not establish the required evidence.

## 5. Did ASR cause the observed failures?

The label-blind 12-row failure panel does not support ASR as the main global
bottleneck:

| Panel rows | Actual requirement | Can better ASR fix it? |
|---|---|---|
| 1 spoken-fact row | Faithful multilingual speech content | **Potentially yes** |
| 1 emotional-delivery row | Prosody plus visible gestures | Mostly no; preserve native audio |
| 4 sound/music rows | Effects, bells, instrument class, exact music identity | No |
| 4 temporal/action rows | The correct visual moment and event ordering | No |
| 2 visual controls | Object recognition and counting | No |

The direct ASR experiment reinforces that distinction. Whisper produced
repetition/boilerplate and timestamp-overrun failure modes on unsuitable audio,
which are now rejected by quality gates. Qwen3-ASR-1.7B avoided the repetition
loop and emitted coherent French and Malay text, but the downstream
Qwen3.5-4B answers were still wrong or incomplete: it answered that the horse
was 22 years old when the reference says the bystander did not know, and it
reduced emotional delivery to a serious facial expression.

Native audio was useful but not a global replacement either. On the same
12-row panel, MiniCPM-o 4.5 with audio enabled scored 4 correct, 3 partial, and
5 wrong, compared with 4/4/4 with audio disabled. It gained on the six
audio-focused rows but regressed on temporal/action rows. This supports
category-gated use of native audio, plus a faithful ASR branch specifically for
lexical speech questions.
