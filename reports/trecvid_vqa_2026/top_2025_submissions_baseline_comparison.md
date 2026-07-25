# TRECVID 2025 VQA: top systems versus a 12-frame + Whisper baseline

## Short answer

The proposed baseline is not far from the winning **multiple-choice (MC)** recipe. NII_UIT's top MC system also used **12 roughly uniform frames plus a Whisper transcript**, with a frozen model. Its important additions were a much larger backbone (**InternVL3.5-38B-Instruct**), spatial tiling/zoom-ins of every sampled frame, timestamped transcripts, question-type-specific prompting, and explicit ranking of all four choices.

The top **answer-generation (AG)** team, CERTH-ITI, used a more different approach: **Qwen2.5-Omni-7B**, native full-video audio rather than a Whisper transcript, LoRA supervised fine-tuning on the 500 provided examples, and inference-time ensembling over ten random frame subsets. Its strongest single-answer run was actually the simpler one-generation variant, while the ensemble helped when evaluated top-1 after discarding lower-ranked answers.

## What the official results say

The final NIST appendices report:

| Task | Leading submission | Official result |
|---|---|---|
| AG | CERTH-ITI `run_2` | METEOR 0.2604; BERTScore 0.8982; STS 0.3510 |
| AG | CERTH-ITI `run_4` | METEOR 0.2596; **BERTScore 0.9034**; STS 0.3485 |
| AG | CERTH-ITI `run_1` | METEOR 0.2596; BERTScore 0.9033; STS 0.3483 |
| MC | NII_UIT `HIGHEST_PIPELINE` | **Top-1 0.7810; MRR 0.8671** |
| MC | CERTH-ITI `run_2` | Top-1 0.7679; MRR 0.8629 |

Sources: [official NIST AG results](https://trec.nist.gov/pubs/trec34/appendices/trec2025-vqa-gen.html) and [official NIST MC results](https://trec.nist.gov/pubs/trec34/appendices/trec2025-vqa-mc.html).

There is no single unambiguous AG "winner" because the task reports multiple metrics: CERTH's single-generation run led METEOR and STS, while its ten-subset ensemble led BERTScore. CERTH was the leading team across the three headline similarity metrics, but not every metric was maximized by the same run.

## Pipeline comparison

| Component | Simple baseline | CERTH-ITI (top AG team; second MC) | NII_UIT (top MC) | HLTCOE (contrasting AG system) |
|---|---|---|---|---|
| Backbone | Suggested Qwen3-VL-4B | Qwen2.5-Omni-7B Thinker | InternVL3.5-38B-Instruct for MC; Aria-8x3.5B for AG | Qwen2.5-VL-7B generator and LoRA-tuned Qwen2.5-VL-7B reranker |
| Frames | 12 uniform | Ten random subsets at inference; each subset up to `2 fps × duration`; 10 frames during SFT | 12 roughly uniform frames | Paper does not specify frame count/sampling |
| Spatial detail | Global frames | Up to 16,384 visual tokens per generation | Each frame additionally tiled into local 2x2 or 3x3 crops | Not specified |
| Audio | Whisper text | Native full-video audio embeddings passed to Omni "as-is" | Timestamped Whisper text injected into prompt | Official overview says Whisper-v3-large; paper prompts include ASR transcript |
| Fine-tuning | None | LoRA SFT for 5 epochs on 500 correct QA pairs; wrong choices ignored | None; model weights frozen | Generator stayed pretrained in final setup; reranker LoRA-trained on TRECVID data |
| Inference/ranking | One concise answer | AG: greedy response per subset, exact-response vote count; MC: mean-pool option-letter logits over 10 subsets | MC: type-aware prompt asks model to rank choices; AG: short-answer normalization and paraphrastic variants | Multiple candidates from prompt variation; learned listwise pointer reranker |

Primary sources: [CERTH-ITI team paper](https://trec.nist.gov/pubs/trec34/papers/CERTH-ITI.avs.vqa.pdf), [NII_UIT team paper](https://trec.nist.gov/pubs/trec34/papers/NII_UIT.vqa.pdf), [HLTCOE team paper](https://trec.nist.gov/pubs/trec34/papers/HLTCOE.vqa.pdf), and the [official NIST track overview](https://trec.nist.gov/pubs/trec34/papers/Overview_vqa.pdf).

## What each team added beyond the baseline

### CERTH-ITI: training plus test-time frame-subset ensembling

- Used Qwen2.5-Omni-7B's Thinker, which consumes video, text, and audio jointly; the Talker was disabled.
- Fine-tuned with LoRA on the 500 released correct question-answer pairs for five epochs (single RTX 4090, rank 16, alpha 64, learning rate `3e-5`, batch size 32). Training used ten sampled frames per video.
- At inference, sampled ten random subsets from all video frames. Each subset could contain up to two frames per second times video duration. Full-video audio embeddings accompanied every subset.
- Generated greedily, so diversity came only from different visual subsets. AG answers were ranked by exact occurrence count; MC option-letter log probabilities were mean-pooled over subsets.
- Their own MC ablation reported 73.2% for the base model, 74.2% after multi-subset aggregation, 74.4% after fine-tuning, and 76.1% with both. Thus ensembling and SFT each helped modestly and combined additively on their analysis.
- Their AG ensemble did not beat the one-generation run on all official averages because the scorer averaged similarity over every submitted answer. In their post-hoc top-1 comparison, ensemble versus single generation improved METEOR 0.265 vs 0.260, BERTScore 0.905 vs 0.898, and STS 0.356 vs 0.351.

This is more compute-heavy than 12 frames + Whisper: up to ten model calls per question, native audio encoding on every call, and fine-tuning.

### NII_UIT: the winning MC system was essentially an enhanced 12-frame + Whisper baseline

- Frozen InternVL3.5-38B-Instruct, twelve roughly uniform frames, and timestamp-aligned Whisper text.
- Tiled every sampled frame into local regions (the paper gives 2x2 or 3x3 as examples) so small objects and peripheral details were enlarged, instead of presenting only global frames.
- Classified questions into temporal, counting, yes/no, or descriptive categories, then changed the reasoning/conciseness instruction accordingly.
- Asked the model to rank the answer options directly; no task-specific fine-tuning.
- On its reported training-split ablation, Top-1 moved from 0.746 (backbone baseline) to 0.751 with transcript, 0.761 with 12-frame temporal densification, 0.779 with spatial tiling, and 0.795 with question-type prompting. These are cumulative, internal results rather than isolated causal effects or official test scores.

The practical lesson is that **Whisper itself was a small average gain** in NII_UIT's ablation. More visual coverage, zoomed crops, and prompt routing accounted for larger cumulative gains. A 12-frame + Whisper run is therefore a very credible first baseline, not a toy system.

### HLTCOE: optimize candidate ranking, not just perception

- Used Qwen2.5-VL-7B to generate multiple answers; the final generator used temperature 0.7 and top-p 0.9.
- Produced diversity with multiple prompt templates rather than frame-subset sampling.
- Trained a separate Qwen2.5-VL-7B LoRA reranker on the 500-example AG set. Candidate targets were sorted by BERTScore to the reference, and a masked pointer/listwise objective emphasized the first three ranks.
- Their official AG run scored METEOR 0.2191, BERTScore 0.8893, and STS 0.2817, below CERTH's leading runs. The learned reranker did improve HLTCOE's test BERTScore-ranking measure (0.708 to 0.718), but mean semantic scores barely moved (BERTScore 0.896 to 0.896; METEOR 0.157 to 0.156 in their table).

This is useful mainly as a later-stage idea: first improve the top answer and evidence pipeline; only then spend effort learning how to rank multiple paraphrases.

## Recommended progression for this repository

1. **Baseline:** 12 uniform global frames + timestamped Whisper text + one concise-answer prompt.
2. **Highest-value cheap ablation:** keep the same model and compare global frames against question-guided high-resolution crops/tiles. Full 3x3 tiling of all 12 frames can explode the visual-token budget, so test 2x2 or selective tiling first.
3. **Prompt routing:** simple deterministic categories (temporal, counting, audio, descriptive) and short tailored instructions.
4. **Model scaling:** compare the 4B baseline to a larger current VLM under exactly the same inputs. NII_UIT's strongest result used 38B, so do not attribute its lead solely to preprocessing.
5. **Native-audio comparison:** if feasible, compare Whisper text with an audio-native model. Whisper captures speech but generally does not textualize non-speech cues such as music, impacts, or environmental sounds.
6. **Only after the above:** add multiple frame subsets and voting/pooling. CERTH's ablation suggests a real but modest benefit, at roughly ten times the inference calls.
7. **For AG submission formatting:** strongly validate whether returning multiple answers helps the 2026 scorer. In 2025, averaging over weak paraphrases penalized both CERTH and NII_UIT; the highest-scoring CERTH STS run returned one answer.

## Caveats and discrepancies

- The final NIST MC appendix reports NII_UIT at 0.7810 Top-1 / 0.8671 MRR and CERTH SFT at 0.7679 / 0.8629. Their papers report 0.774 / 0.859 and 0.761 / 0.855, respectively. Use the NIST appendix for final leaderboard values; the paper values appear to reflect an earlier or differently parsed scoring pass. CERTH documents a quote-escaping/parser issue affecting 0.7% of MC questions.
- The NII_UIT paper incorrectly describes the training dataset as 2,000 clips in one section. The NIST overview gives 500 training and 2,000 test examples, which also agrees with CERTH and HLTCOE. This does not undermine NII_UIT's claim that its final system was training-free, but its dataset-description paragraph should not be used as the source for split sizes.
- NII_UIT names the AG model inconsistently as `Aria3x8B` and `Aria-8x3.5B`; the official run tag is `Aria8x3.5B_VidLLaMa`. Its methods describe Aria as the AG reasoner but do not clearly explain any VideoLLaMA fusion implied by that run name.
- HLTCOE's paper does not specify frame sampling/count, so no claim should be made that its visual preprocessing matched a 12-frame baseline. The NIST overview, not the HLTCOE methods text, supplies the `Whisper-v3-large` designation.
- NIST describes 2025 as the first pilot year: 20 AG runs from seven teams and nine MC runs from five teams. These are competition results on one dataset, not a broad proof that one architecture is universally superior.
