# AutoGaze for task-support and temporal-span verification

**Date.** 2026-08-15

**Question.** Would [NVlabs/AutoGaze](https://github.com/NVlabs/AutoGaze)
help issue #35, where a frozen question-blind scene map already contains the
needed evidence but the answer stage may select the wrong fact, action, or
temporal span?

## Bottom line

**Not as the next issue #35 treatment.** AutoGaze is a promising upstream
visual-token selector/compressor, but its released interface is question-blind
and its training objective is visual reconstruction. It returns patch positions
and masks for each frame; it does not return question-supported evidence
spans, event boundaries, cited facts, or a top-1 answer. It could be useful in
a separate evidence-recovery or local-MLLM efficiency experiment, especially
when high resolution, high frame rate, or long duration makes visual input too
large. It does not directly address the observed task-support failure when the
relevant evidence is already in the frozen map.

A zero-call paper walkthrough performed after this source review found that the
proposed task-support check cannot resolve the two decisive election failures:
one needs evidence absent from the frozen map, while the other needs an
unvalidated salience/text-type rule. Do not launch that paid follow-up from
these cases. AutoGaze should be reconsidered only as a separately named
retrieval/efficiency condition, with required-evidence recall measured before
answer quality.

## What AutoGaze actually provides

### Verified from official sources

- The project describes AutoGaze as a lightweight module that removes
  redundant video patches before a ViT or MLLM. Its stated mechanism is
  autoregressive selection of multi-scale patches that reconstruct the video
  under a reconstruction-error threshold or patch budget. It is not described
  as a question-conditioned selector. ([official project page](https://autogaze.github.io/),
  [paper abstract](https://arxiv.org/abs/2603.12254))
- The official quick start loads a video, runs the model with parameters such
  as `gazing_ratio` and `task_loss_requirement`, and returns `gazing_pos`,
  `num_gazing_each_frame`, and `if_padded_gazing`. The example uses 16 frames
  at 224x224; for longer or higher-resolution videos, the instructions say to
  process 16-frame chunks. ([quick start](https://github.com/NVlabs/AutoGaze/blob/main/QUICK_START.md))
- The integration guide passes those gaze outputs into a modified vision
  encoder. The encoder keeps only the gazed patches and applies an inter-frame
  attention mask; the guide then describes sending the resulting visual
  features to an MLLM. This is an encoder-integration path, not a text-only
  evidence annotator or a provider-side Gemini control. ([integration guide](https://github.com/NVlabs/AutoGaze/blob/main/INTEGRATION.md))
- At the inspected repository revision
  [`ba48d0f`](https://github.com/NVlabs/AutoGaze/tree/ba48d0f94ac2929d6fe3ee4380dc893aa6eed0ab),
  `AutoGaze.forward` accepts video plus optional target scales/patch size,
  `gazing_ratio`, and `task_loss_requirement`; it has no question or text
  argument. Its output contains `gazing_pos`, `gazing_mask`, per-frame counts,
  and padding masks. ([`autogaze.py`](https://github.com/NVlabs/AutoGaze/blob/ba48d0f94ac2929d6fe3ee4380dc893aa6eed0ab/autogaze/models/autogaze/autogaze.py#L275-L429))
- The same revision's default model configuration uses 224-pixel input and a
  16-frame maximum. It defines a 0.7 task-loss threshold, disabled by default
  and enabled explicitly in the quick start. ([`autogaze.yaml`](https://github.com/NVlabs/AutoGaze/blob/ba48d0f94ac2929d6fe3ee4380dc893aa6eed0ab/autogaze/configs/model/autogaze.yaml),
  [quick start](https://github.com/NVlabs/AutoGaze/blob/main/QUICK_START.md))
- The released VideoMAE task computes reconstruction loss and uses its negative
  as the gaze reward. The source does not define a QA-answer, temporal-span,
  citation-support, or question-conditional reward. ([`task_video_mae_reconstruction.py`](https://github.com/NVlabs/AutoGaze/blob/ba48d0f94ac2929d6fe3ee4380dc893aa6eed0ab/autogaze/tasks/video_mae_reconstruction/task_video_mae_reconstruction.py#L61-L160))
- The paper/project page report gains on long/high-resolution video benchmarks,
  including HLVid, and token/latency reductions. Those results establish a
  scalability claim for the tested NVILA/vision-encoder pipeline, not a claim
  about TRECVID answer grounding or temporal-span selection. ([paper](https://arxiv.org/abs/2603.12254),
  [project results](https://autogaze.github.io/#hlvid-high-resolution-long-form-video-benchmark))

### Inference for issue #35

- **Inference:** Because AutoGaze exposes frame-indexed patch selections, a
  downstream adapter could associate selected patches with decoded frame times.
  That might improve a local model's access to small spatial details or reduce
  visual-token cost. The adapter would still need to build semantic event/span
  records and answer-support checks outside AutoGaze.
- **Verified local correction:** The two decisive election failures do not
  share one downstream task-support mechanism. In one, the reference-level
  action is absent from the complete frozen map even though a different end
  action is cited. In the other, the reference text is present, but another
  displayed-text span also satisfies the underspecified question. See the
  private `task-support-paper-walkthrough-v1.json` and its tracked sanitized
  summary in the answer-election artifact bundle.
- **Inference:** AutoGaze is aimed at the upstream causal link and could be
  relevant to the first mechanism only through a new native-video evidence
  path. It cannot resolve the second mechanism or make a cited fact answer the
  requested span by itself.
- **Inference:** AutoGaze might help the separate rows where the full-panel
  audit found fine-detail or small-text omissions. That would be an evidence
  recovery hypothesis, not evidence that it repairs answer use. It must be
  scored with required-patch/span recall and answer quality separately.

## Unknowns and boundary conditions

- **Unknown:** AutoGaze's required-evidence recall on the TRECVID 2026 videos,
  especially tiny displayed text, hands/actions, and brief transitions, has not
  been measured here.
- **Unknown:** The released materials do not establish that reconstruction
  loss preserves the exact visual detail needed by a particular natural-language
  question. A patch can be visually redundant for reconstruction while still
  being decisive for OCR or a fine action distinction.
- **Unknown:** The official integration path requires a compatible local ViT/
  MLLM (the project gives NVILA-HD-Video as its example). It does not provide a
  way to insert selected patches into the already-frozen Gemini native-video
  calls used by the current benchmark.
- **Boundary:** The project reports no audio-selection mechanism. AutoGaze alone
  cannot recover speech, sound, or prosody evidence.
- **Boundary:** Mapping selected frames to time is not the same as verifying a
  temporal span. The current verifier must still check first/last observable
  evidence, temporal relation, and question support against direct evidence.

## Recommendation

1. **Do not add AutoGaze to issue #35, and do not launch the proposed text-only
   task-support follow-up from these two rows.** The paper contract resolved
   neither case. A future answer-use test first needs the treatment frozen,
   followed by an independent label-aware sufficiency audit proving that the
   requested fact and unique span are present while labels/direct evidence stay
   hidden from inference.
2. **If evidence recovery becomes the next bottleneck, run a separate no-answer
   canary first:** on a small held-out set with independently labeled required
   frames/crops, compare the baseline selector with AutoGaze at a declared patch
   budget. Record required-detail recall, per-frame/time coverage, token count,
   and failure cases before running any answer calls.
3. **Only after a positive retrieval canary**, compare a compatible local MLLM
   with and without AutoGaze under the same frozen questions, native audio
   treatment, answer prompt, and scoring. Treat this as a retrieval/efficiency
   experiment; it must not be conflated with the task-support/answer-election
   claim.

## Project context

The current routing record distinguishes retrieval failure from fusion/reasoning
failure. The latest evidence-aware election test found that candidate
availability was not the main bottleneck on its selected panel: the oracle had
a correct candidate on every row, while the fixed election emitted the correct
result on only 10/12 rows. The later paper walkthrough further split the two
decisive failures into one map omission and one span ambiguity, so neither
supports an immediate generic verifier run. See
[`research-routing.qmd`](../project-status/research-routing.qmd) and the dated
experiment report
[`2026-08-12-vqa-frontier-boundary-verifier.qmd`](../experiments/2026-08-12-vqa-frontier-boundary-verifier.qmd).

This note is a literature/code assessment only. No AutoGaze inference, model
download, credential use, or paid call was performed.
