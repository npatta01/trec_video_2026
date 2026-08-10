# TRECVID VQA 2026 competitive gap: primary-source report

**Snapshot:** 2026-08-09. **Scope:** official NIST/TRECVID pages and proceedings, TREC 2025 team papers, and original model papers/code/docs. No 2026 leaderboard or participant-paper evidence was available in the sources reviewed.

## Bottom line

- **Verified:** The current 2026 task is no longer only the 2025-style ~30-second YouTube-Shorts problem. NIST plans to retain about 1,457 2025-annotated YouTube Shorts and add roughly 400 longer scenes/new Shorts; the longer licensed-movie subset is planned to be 20–25% of testing, averaging 1–2 minutes and reaching about 6 minutes. ([2026 VQA page](https://www-nlpir.nist.gov/projects/tv2026/vqa.html))
- **Verified:** The strongest 2025 AG result came from CERTH-ITI: native audio-visual Qwen2.5-Omni-7B fine-tuned on the 500-example TREC set, then ten predictions from different random frame subsets, majority-ranked for AG and mean-logit pooled for MC. CERTH had the top AG submission across the official metrics; NII_UIT led MC. ([CERTH paper](https://trec.nist.gov/pubs/trec34/papers/CERTH-ITI.avs.vqa.pdf), [AG results](https://trec.nist.gov/pubs/trec34/appendices/trec2025-vqa-gen.html), [MC results](https://trec.nist.gov/pubs/trec34/appendices/trec2025-vqa-mc.html))
- **Likely 2026 gap:** A fixed uniform frame budget plus one-pass answering is vulnerable on the longer subset. Strong long-video research converges on coarse-to-fine temporal search, event memory, and repeated evidence checking; TREC 2025 team papers mostly show dense sampling or frame-subset ensembling, not an explicit question-conditioned temporal locator.
- **Direct development-set answer:** **Yes for supervised development, with a qualification.** NIST says 2026 uses at least the same 500-video 2025 training set, with a human question, correct answer, and three plausible wrong answers per item. The 2025 test questions and ground truth are also now available, so they are usable as a local labeled development/benchmark pool. NIST does not explicitly rename the 2025 test set as the official 2026 dev split; do not treat it as a clean 2026 holdout. ([2026 data page](https://www-nlpir.nist.gov/projects/tv2026/data.html), [2026 VQA page](https://www-nlpir.nist.gov/projects/tv2026/vqa.html), [TRECVID past data](https://trecvid.nist.gov/trecvid.data.html), [dataset summary](https://trecvid.nist.gov/past.data.table.html))

## Official 2026 contract and schedule

The 2026 page describes seven question categories: temporal, causal, audio/multimodal, multi-hop, people/object interaction, counting, and common-sense/world knowledge. AG requires **10 ranked textual answers** plus per-answer generation time; MC requires a ranking of the four options. ([2026 VQA guidelines](https://www-nlpir.nist.gov/projects/tv2026/vqa.html))

The current **tentative** schedule is:

| Date | Official event |
|---|---|
| 17 Jul 2026 | AG testing data available |
| 10 Aug 2026 | AG submission page opens |
| **17 Aug 2026** | **AG submissions close** |
| 21 Aug 2026 | MC testing data available |
| 11 Sep 2026 | MC submission page opens |
| **18 Sep 2026** | **MC submissions close** |
| 28 Sep 2026 | AG and MC results released to teams |
| 16–20 Nov 2026 | TREC 2026 workshop in Maryland |

Source: [NIST TRECVID 2026 schedule](https://www-nlpir.nist.gov/projects/tv2026/schedule.html). At this snapshot date, the immediate operational deadline is AG submission on **17 August**; the page labels the schedule tentative.

One documentation wrinkle matters for planning: the current VQA page describes the expanded ~1,457 + ~400 composition, while the 2025 proceedings table labels only 1,457 items as “2026 Test.” Treat the current 2026 task page as the more relevant planning source and the final released topics as authoritative. ([2025 VQA overview, dataset table](https://trec.nist.gov/pubs/trec34/papers/Overview_vqa.pdf))

## What was actually competitive in 2025

The official overview reports seven participating groups, 20 AG runs, and nine MC runs. Most teams used VideoLLaMA3 or Qwen-based models; fine-tuning and off-the-shelf systems were both viable. ([2025 VQA overview](https://trec.nist.gov/pubs/trec34/papers/Overview_vqa.pdf))

| Team/system | Verified mechanism | Competitive signal and gap |
|---|---|---|
| **CERTH-ITI** | Fine-tuned the Thinker component of Qwen2.5-Omni-7B on the 500 TREC examples; trained with 10 frames per video. At inference, generated 10 greedy answers over random frame subsets; AG used majority voting, MC mean-pooled logits over option letters. | Top AG family in the official table and second MC family. This is the clearest evidence for test-time scaling plus native audio, but it samples the whole short video rather than doing explicit question-conditioned localization. ([paper](https://trec.nist.gov/pubs/trec34/papers/CERTH-ITI.avs.vqa.pdf)) |
| **NII_UIT** | Frozen Aria3x8B for AG and InternVL3.5-Instruct for MC; dense temporal sampling, spatial tiling, question-type prompts, answer normalization, and timestamped Whisper transcripts injected as text. The paper reports 12 frames/video. | **MC winner:** Top-1 0.7810, MRR 0.8671. This is the strongest direct evidence for dense evidence presentation, OCR/small-object zoom, and ASR augmentation. It is still mostly a dense fixed-input design, not a learned locator. ([paper](https://trec.nist.gov/pubs/trec34/papers/NII_UIT.vqa.pdf), [MC results](https://trec.nist.gov/pubs/trec34/appendices/trec2025-vqa-mc.html)) |
| **HLTCOE** | Generator first creates multiple candidates from prompt variants; a Qwen2.5-VL/Omni-based reranker learns a masked-pointer, rank-weighted listwise objective. Candidate ordering is derived from BERTScore against training references. | Direct evidence that AG is partly an **answer-ranking/grading** problem, not only generation. The released code is linked from the paper. ([paper](https://trec.nist.gov/pubs/trec34/papers/HLTCOE.vqa.pdf), [code](https://github.com/Charles201428/2025-VQA-AG/tree/research-framework-vqa)) |
| **WHU-NERCMS / kslab** | WHU generated a base answer with VideoLLaMA and ranked MC options by sentence-embedding cosine similarity. kslab fine-tuned VideoLLaMA3-2B with LoRA, sampled diverse AG answers and ranked by average token log-probability, then used generated-answer/option similarity for MC. | Useful low-complexity baselines, but both papers expose calibration risks: token confidence can favor short answers, and a generated “ground-truth” answer can contaminate option ranking. ([WHU paper](https://trec.nist.gov/pubs/trec34/papers/WHU-NERCMS.avs.vqa.pdf), [kslab paper](https://trec.nist.gov/pubs/trec34/papers/kslab.vqa.pdf)) |

The official AG protocol uses METEOR, BERTScore, STS, and NDCG; MC uses Top-1 accuracy and MRR. CERTH specifically reports that averaging over many AG responses can penalize extra responses even when rank 1 is correct, making candidate selection and ranking unusually important. ([2026 metrics/guidelines](https://www-nlpir.nist.gov/projects/tv2026/vqa.html), [CERTH discussion/results](https://trec.nist.gov/pubs/trec34/papers/CERTH-ITI.avs.vqa.pdf))

## Gap analysis for this repository

| Area | Verified evidence | Practical inference for the repo |
|---|---|---|
| **Full-video temporal localization** | NIST adds longer clips and explicitly tests temporal/order/attribute tracking. Qwen2.5-VL claims absolute-time encoding and second-level event localization. Independently, T* frames long-video QA as finding 1–5 relevant frames in a haystack and reports only 2.1% temporal F1 for existing selection on one subset. ([Qwen2.5-VL report](https://arxiv.org/abs/2502.13923), [T* report](https://arxiv.org/abs/2504.02259), [T* code](https://github.com/longvideohaystack/tstar)) | Keep the current 12-frame baseline as a control, but the high-value missing capability is **question-conditioned coarse-to-fine search**: timeline/ASR/cheap visual pass → candidate intervals → dense local frames → final VLM. Measure oracle intervals versus predicted intervals so a poor locator is not mistaken for a weak answerer. |
| **Dense/multistage evidence selection** | DeVi uses hierarchical segment captioning, temporal event memory, and self-consistency checking. LongVT uses a global skim → native `crop_video` tool → fine inspection loop. ([DeVi paper/code](https://arxiv.org/abs/2409.04388), [LongVT paper](https://arxiv.org/abs/2511.20785), [LongVT code](https://github.com/EvolvingLMMs-Lab/LongVT)) | Prefer a small evidence object per question: time interval, representative frames, local crops, ASR spans, and confidence. Preserve global context for “before/after” questions; do not replace it with only the top local crop. These are research-system patterns, not verified 2026 TREC team choices. |
| **Native audio + ASR** | CERTH feeds the native audio-visual Qwen2.5-Omni path; NII_UIT uses timestamped Whisper text. Qwen2.5-Omni explicitly time-aligns audio/video with TMRoPE; Whisper’s official code returns timestamped segments and supports multilingual transcription. ([Qwen2.5-Omni report](https://arxiv.org/abs/2503.20215), [Whisper code](https://github.com/openai/whisper)) | The robust design is **both**, not either/or: native audio for music/sound/speaker cues, timestamped ASR for lexical retrieval and evidence display. Route audio-heavy questions through this branch; do not make ASR the global default without a quality gate. |
| **Long-video handling** | The 2026 longer subset averages 1–2 minutes and can reach 6 minutes. Qwen3-VL reports a native 256K-token interleaved multimodal window and explicit text-based time alignment; STORM adds a temporal encoder and token compression for long videos. ([Qwen3-VL report](https://arxiv.org/abs/2511.21631), [STORM report](https://arxiv.org/abs/2503.04130)) | Use duration-aware budgets and hierarchical compression. A fixed frame count should be a baseline, not the only policy. The repo’s current short-video evidence should not be generalized to the movie subset without a duration-stratified check. |
| **Stronger VLMs** | 2025’s official overview names VideoLLaMA3 and Qwen2.5-Omni-7B as common choices; NII_UIT used Aria3x8B and InternVL3.5-Instruct. The original model sources now expose stronger long-context or native-audio options: VideoLLaMA3, Qwen2.5-VL/Omni, Qwen3-VL, and InternVL3. ([2025 overview](https://trec.nist.gov/pubs/trec34/papers/Overview_vqa.pdf), [VideoLLaMA3 code](https://github.com/DAMO-NLP-SG/VideoLLaMA3), [InternVL3 report/blog](https://internvl.github.io/blog/2025-04-11-InternVL-3.0/)) | Treat model swap as a controlled axis after evidence selection is stable. The likely competitive ensemble is heterogeneous—native-audio model + high-resolution/time-grounded VL model—not multiple prompts around one weak temporal view. No source verifies which of these models current 2026 teams will submit. |
| **Answer grading / ranking** | NIST scores ten AG answers with semantic and ranking metrics. CERTH uses vote counts; HLTCOE learns listwise reranking; WHU/kslab use embedding or log-probability heuristics. | Add a question-conditioned answer grader that scores entailment/semantic match against the selected evidence, then rank and deduplicate. Keep number/count/yes-no normalization explicit. The official NIST scorer remains the authority; a local grader is a selection tool, not a substitute for the official metrics. |
| **Ensembling and retrieval** | CERTH’s ten frame-subset ensemble is verified and effective. In the same team’s AVS system—not its VQA system—multiple VLM embeddings are fused, then top-N candidates are MLLM-reranked. T* similarly retrieves visual needles before final answering. ([CERTH paper](https://trec.nist.gov/pubs/trec34/papers/CERTH-ITI.avs.vqa.pdf)) | Build rank fusion across evidence views (uniform, ASR-hit, visual-hit, local crop) and across answerers, with calibrated abstention when evidence views disagree. For AG, rank quality matters more than emitting ten unfiltered variants. |

## Recommended priority, without assuming unverified 2026 practices

1. **First:** preserve the existing short-video baseline and add a label-blind temporal/evidence diagnostic: oracle-localized, predicted-localized, and uniform conditions.
2. **Second:** combine timestamped ASR retrieval with native-audio VLM inference for audio/multimodal questions; retain a visual-only control.
3. **Third:** add answer generation plus a learned/semantic reranker and official-score-compatible normalization; report Top-1, Top-5, and full-list behavior separately.
4. **Fourth:** evaluate stronger VL/omni backbones under the same evidence budget and duration strata. Only then consider expensive multi-pass ensembles.

## What remains unknown

- No official 2026 participant papers, runs, leaderboard, or final test composition were found as of 2026-08-09; claims about “likely” 2026 systems are extrapolations from the current task design, 2025 papers/results, and original model research.
- The current NIST page does not disclose the exact 2026 movie clips, automatic-QA construction, final category counts, or whether every longer item will have the same annotation protocol.
- TREC’s official VQA output has no timestamp/evidence field, so temporal localization is an internal strategy rather than a directly scored submission target.
- The 2025 labeled test set is available and useful for local development, but its reuse creates a familiar-data risk; only the newly released 2026 material can test generalization to the changed duration and automatic-QA mix.
