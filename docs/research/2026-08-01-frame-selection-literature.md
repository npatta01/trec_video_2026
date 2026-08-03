# Uniform and query-guided frame selection for video QA

**Question.** Should an application run both uniform-12 and TimeLens-guided
hybrid-12 and let a final model choose, or combine/route their evidence before
answering?

## Bottom line

Running both is technically reasonable, and there is direct precedent for
multiple visual-context passes followed by answer aggregation. However, the
stronger default is to **select or fuse evidence before one answer call**. A
free-form model choosing between two answer strings is a new ensemble condition:
it needs its own blinded validation and costs two answer passes plus a judge.

For this project:

1. Keep **uniform-12 versus hybrid-12** as the equal-frame-budget scientific
   comparison.
2. For deployment, prefer one 12-frame global-plus-local input and one answer,
   or route global questions to uniform-12 and localized questions to hybrid-12.
3. Test a two-answer chooser only as a separately named higher-compute condition.

## What other systems did

| System | Evidence strategy | Answer strategy | Relevance here |
|---|---|---|---|
| TRECVID 2025 CERTH-ITI | Ten random frame subsets; native full-video audio; Qwen2.5-Omni-7B | One answer per view, then exact-answer voting for AG or option-logit pooling for MC | Direct competition precedent for running multiple visual views, but roughly ten answer calls rather than two. Its team paper reports modest MC gains from aggregation, while its simpler single-generation AG run led some official mean metrics. [CERTH paper](https://trec.nist.gov/pubs/trec34/papers/CERTH-ITI.avs.vqa.pdf), [official NIST results](https://trec.nist.gov/pubs/trec34/appendices/trec2025-vqa-gen.html) |
| TRECVID 2025 NII_UIT | About 12 time-spaced frames, local spatial tiles, timestamped Whisper transcript, and question-type prompts | Frozen model answers/ranks once over improved evidence | Closest competition precedent for retaining broad coverage and adding targeted detail before one answer. [NII_UIT paper](https://trec.nist.gov/pubs/trec34/papers/NII_UIT.vqa.pdf), [official MC results](https://trec.nist.gov/pubs/trec34/appendices/trec2025-vqa-mc.html) |
| VideoAgent | Starts with sparse visual information, iteratively retrieves query-relevant frames, and compiles the acquired evidence | The agent answers from one accumulated state | Supports adaptive acquisition and evidence fusion rather than independent uniform/retrieved answer judging; it reported using about 8.2--8.4 frames on average. [Paper](https://arxiv.org/abs/2403.10517) |
| VideoTree | Iteratively refines query-relevant keyframes and builds a coarse-to-fine tree | Aggregates hierarchical evidence into one LLM reasoning call | Supports global-to-local evidence fusion under an efficiency constraint. [Paper](https://arxiv.org/abs/2405.19209) |
| LongVU | Removes redundant frames, then uses the text query to preserve more detail in relevant frames while compressing other evidence | One generation over the adaptively compressed evidence | Supports retaining global context at lower cost while allocating detail to relevant evidence. [Paper](https://arxiv.org/abs/2410.17434), [official code](https://github.com/Vision-CAIR/LongVU) |
| RHS | Globally localizes relevant cues, then densely samples the selected local segments | Final answer is inferred from the refined frame set | Nearly the same global-to-local principle as the proposed hybrid sampler. [Paper](https://arxiv.org/abs/2503.12496) |

There is also direct literature precedent for answer-level selection. *From
Trial to Triumph* generated ten predictions from different bin-wise frame
samples and selected among them with an engineered self-reward combining answer
frequency, first-token confidence margin, and a complementary global/local
reasoning vote. It used 32 frames per prediction and ten predictions by default,
and reported gains across three video models. This validates the general idea,
but not a naive two-answer judge: with only two disagreeing answers, frequency
is tied, and its confidence mechanism is designed around answer-option logits.
[Paper](https://arxiv.org/abs/2503.20472)

## Recommended application design

```text
question + video
      |
      +-- global question ----> uniform-12 --------+
      |                                            |
      +-- localized question -> 6 global +         +--> one answer model
                                 6 TimeLens-local --+
```

This follows DIG's reported finding that uniform sampling is efficient and
effective for global questions, whereas localized questions benefit from
query-aware selection. [DIG paper](https://arxiv.org/abs/2512.04000)

If robustness justifies more compute, run both 12-frame passes only when the
router is uncertain or when the first answer is low-confidence. ReQuest uses a
related uncertainty-driven policy that triggers extra inference selectively
rather than on every question. [ReQuest paper](https://arxiv.org/abs/2607.01737)

## Scientific and compute boundary

| Condition | Selector calls | Answer calls | Judge calls | Valid claim |
|---|---:|---:|---:|---|
| Uniform-12 | 0 | 1 | 0 | Baseline answer quality |
| Hybrid-12 | 1 | 1 | 0 | Equal final-frame-budget selector effect |
| Fused-24 | 1 | 1 | 0 | Higher visual-token-budget system effect |
| Two answers + chooser | 1 | 2 | 1 | Ensemble/router effect, not selector-only effect |

Freeze and grade the two 12-frame answer passes independently before exposing
answers to a chooser. Then evaluate the chooser as a third condition. This keeps
the matched selector comparison interpretable and prevents the extra inference
budget from being mistaken for a frame-selection gain. Component/scene coverage
should also remain a separate outcome: KFS-Bench found that relevant-scene
coverage and balanced allocation both affect QA accuracy, while answer-model
errors make QA accuracy alone an imperfect sampler diagnostic.
[KFS-Bench](https://arxiv.org/abs/2512.14017)
