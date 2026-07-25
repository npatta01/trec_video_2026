# Transcript-evidence diagnostic (issue #7)

## Outcome

The label-blind automatic-transcript ablation is complete. On the frozen
ten-row lexical-speech panel, adding Whisper large-v3 output to the same 12
frames raises manually adjudicated accuracy from **4/10 to 7/10**. There are
three rescues, no degradations, four rows that remain correct, and three rows
that remain incorrect.

This is evidence for keeping automatic transcripts in the VQA path. It is not
an oracle result: no automatic transcript, uploader subtitle, dataset prompt,
or multi-ASR consensus is relabeled as human-verified gold.

Of the three residual failures, two are attributable to VLM answer
selection/evidence use and one to ASR missing a non-lexical vocalization.
Accordingly, the next diagnostic priority should be VLM transcript grounding,
while the ASR quality gate should gain an explicit non-lexical-speech check.

## TREC panel result

| Metric | Frames only | Frames + automatic transcript | Paired delta |
| --- | ---: | ---: | ---: |
| Manual accuracy | 0.400 | 0.700 | +0.300 |
| Normalized exact match | 0.200 | 0.600 | +0.400 |
| Token F1 | 0.352 | 0.600 | +0.248 |
| METEOR | 0.288 | 0.547 | +0.259 |

The ten rows were scored only after all 20 predictions were frozen. Manual
adjudication treats equivalent durations and minor grammatical contractions as
correct, but rejects unrelated, contradicted, or wrong-span answers.

| Transition | Count | Rows |
| --- | ---: | --- |
| Rescued by automatic transcript | 3 | `LKiaCf1QZFw`, `J1CS47pBX_M`, `LSz_xBDBPNc` |
| Degraded with automatic transcript | 0 | — |
| Remained correct | 4 | `Ie-KCtcZ7uc`, `Mi-NpnCBegw`, `HD2_qedLUTE`, `QgzFuzktegE` |
| Remained incorrect | 3 | `NN7G28sBOlY`, `NW3Mhr76ClM`, `NrB4LPosWBk` |

### Residual-failure attribution

- `NN7G28sBOlY`: VLM answer-selection failure. The accepted English transcript
  contains the bystander's correct response, followed by later speculative
  age reasoning; the model selects the wrong span.
- `NW3Mhr76ClM`: ASR evidence failure. Whisper misses the target non-lexical
  chuckle and emits an unrelated outro phrase, which the current generic
  quality gate incorrectly accepts.
- `NrB4LPosWBk`: VLM evidence-use failure. Both the automatic transcript and
  the verified uploader subtitle end with the reference word, yet the model
  returns the same wrong word in frames-only, automatic-transcript, and
  human-caption-reference conditions.

These attributions are diagnostic, not population estimates. The panel has
only ten hand-selected lexical-speech rows.

## Label-isolated protocol

1. Freeze ten row IDs, source-media hashes, label-free questions, and 12
   chronological frame hashes per row.
2. Run `openai/whisper-large-v3` at immutable revision
   `06f233fe06e710322aca913c1bc4249a0d71fce1` in the pinned ROCm image.
3. Detect language using the Whisper model rather than metadata alone. The
   panel contains nine detected-English clips and one detected-French clip.
4. Require accepted language detection and selected-English transcript quality.
   All ten pass. One non-selected original-language branch records a timestamp
   overrun rather than hiding it.
5. Build 20 requests with identical model, revision, runtime, prompt, seed,
   decoding, frame paths, frame hashes, frame timestamps, and pixel bounds
   within each pair. Only the condition ID and transcript field differ.
6. Run a three-row/six-request probe, then the complete batch.
7. Require every raw runner row to echo the exact request SHA, recompute
   canonical request JSON from the live prompt/frames/transcript payload,
   verify immutable provenance, and freeze all 20 predictions.
8. Before scoring, recompute the complete request and prediction freeze and
   require exact equality with the stored freeze. Only then load TREC reference
   answers and adjudicate.

The VLM is `Qwen/Qwen3.5-4B` at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` in
`localhost/trec-vqa-gate-rocm@sha256:510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5`.
Inference uses 12 frames, 256²–512² pixels per frame, greedy decoding, a
32-token output cap, and seed `20260724`.

## One human-caption reference

Exactly one uploader-subtitle row was admitted: `NrB4LPosWBk`. Its uploader
track spans 0.000–24.948 seconds of a 25.028-second clip, for **99.68% verified
coverage**. The contract names it `frames_human_caption_reference`; it is not
an oracle condition.

The result is 0/1 in all three conditions. Because both automatic and uploader
text contain the final reference word, their identical wrong VLM answer is
useful negative evidence: transcript availability alone does not fix this row.

## External dataset-reference control

Primary-source comparison selected four tiny samples from the
[Audio-Visual Lombard GRID Speech corpus](https://spandh.dcs.shef.ac.uk/avlombard/).
The corpus is CC BY 4.0 and publishes synchronized audio and front-video
samples plus deterministic six-token recording prompts. The prompt is called
`dataset_recording_prompt_reference`, never a human oracle. `_WRONG` recordings
are excluded.

| Condition | Manual accuracy |
| --- | ---: |
| Frames only | 0/4 |
| Frames + automatic transcript | 4/4 |
| Frames + dataset reference | 4/4 |

Whisper passes language and quality gates on all four clips. Conservative
word-level WER is 0.333 on each clip because GRID writes letter and digit as
separate reference words while Whisper joins them (for example, `e 5` versus
`e5`); the lexical answer content is otherwise preserved. This control confirms
that the three transcript routes are wired correctly. It does not estimate
open-domain TRECVID performance.

LRS3 was not selected because the official site currently says downloads are
unavailable. MuAViC was not selected because its official recipe requires bulk
mTEDx/YouTube acquisition and some translations are pseudo-labels. Full
primary-source comparison and the exact tiny protocol are in
`external_gold_transcript_benchmark_research_2026-07-24.md`.

## Immutable evidence

All media, frames, audio, transcripts, requests, predictions, scores, and raw
logs remain beneath ignored `data/` and are not staged.

### TREC automatic ablation

| Artifact | SHA-256 |
| --- | --- |
| Panel manifest | `c13b2d7743449bd2f86f48b6a826c64a73a36df2c12b49e459cc82de798fc3b9` |
| Frame manifest | `94009d728b3f8e4810ad6d4a43f1abedbe60da4a464c663941e6b882ee1bfb63` |
| ASR requests | `d1c4bddcbc91d5a65ef40486b520436fb2f0966d0684fe84ce7bad54237c475f` |
| Automatic transcripts | `b017368822fd93528446356945db21cd7ca18b400a396a03b13fc355fed4ef01` |
| Matched VLM requests | `dfa03029293cefda74c104f81154fae266792dddb9ec6c48d9201d4648f4447b` |
| Raw VLM results | `bb18e0e3ab608ffb8ed962d8fae45a0999c7a5377f203e8a5b69d6063780b349` |
| Prediction freeze | `1b6111c936ca3f1a0a6171dfe69ae71db4a126d1a206415125b2add113371832` |
| Post-freeze scores | `7ee13eb406614a1943ce0c7c15d661a2395ad903b139125e72b99f4bee09f32f` |

The freeze internally records request-set SHA
`ea7f27229967320f334235c2b62507207d5718b2d6fe62a9c620ab3e54df8b5f`
and prediction SHA
`6d33b5c563161e3fb5ed1aefc46dab8c41957a3d9e94cd0ddc988cad62673ca3`.

### Human-caption and external controls

| Artifact | SHA-256 |
| --- | --- |
| Human-caption record | `fb6e39e8da2830c9149ff4803b8c13606c6693db3983d7d7b6d5cdc950beb445` |
| Human-caption prediction freeze | `f2a00c8f00cb843cac0ca93d80420607a58893a942b79a1a18dc83b28dbca6bf` |
| Human-caption score | `3094ef4112750a92aa67413348ac6cf204f7131856748be4013674fc559817b5` |
| External manifest | `deb5708cf7b98bddcf10c6ecdd934e1195bd07bc2b61cb43d3f54c7714fb2af8` |
| External dataset references | `d961917553d35818b87970dbc4a5a1aa7ba3a5714d201572999db8a374b2ffba` |
| External automatic transcripts | `af5931ae82d9903d4aadc0c4f0e9cf1c88df956d12069a69a22d632679ae4dee` |
| External prediction freeze | `88dc4b28626bb5c8905c671b30bf64e2fd60b72aaf697bb157bbe54529bb6ac0` |
| External score | `125c73fd2717f37b4e8f1296dbade813a16c23b9cf594f41dc66447d54ac7382` |

The executed VLM runner SHA is
`a1e0ccbf1bc15fda04fe811110d47eebc1f91b9c23a6a89c457ced0ab98340a6`;
its controlled-message helper SHA is
`ed2b85245551dedc484df8fd2e38a560429c795df89c0a270ed2150e3423c9cb`.
Post-freeze METEOR scoring ran in
`localhost/trec-vqa-gate-qwen-omni@sha256:effff2801d7677aa4894a641adc5793d6e5072a8cec71817bf925574862c88de`.

## Decision

- Keep the automatic-transcript condition: it rescues 3/10 rows with no
  observed degradation and improves every aggregate metric.
- Prioritize VLM transcript-grounding and answer-span selection next: two of
  three residual errors have the needed words in the transcript, and one is
  independently reproduced with a verified uploader caption.
- Add a non-lexical-vocalization check to ASR quality gating: the only residual
  ASR-attributed row passed a generic plausibility gate with the wrong content.
- Keep the strict human-oracle experiment blocked until independent
  transcriber/reviewer records exist. Automatic ASR, uploader captions,
  dataset prompts, and multi-ASR consensus remain separate provenance tiers.
