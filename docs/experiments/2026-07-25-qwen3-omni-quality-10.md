# Qwen3-Omni native-audio benchmark: where it helped and where it failed

## Technical summary

Qwen3-Omni is an open, natively multimodal mixture-of-experts model from the
Qwen team. We tested the
[`Qwen3-Omni-30B-A3B-Thinking`](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking)
checkpoint: a roughly 30-billion-parameter model that activates about
3 billion parameters per token, accepts text, images, audio, and video, and
returns text with an explicit reasoning stage. Unlike our
Qwen3.5-4B + Whisper baseline, it can consume the waveform directly instead
of receiving only an ASR transcript. The official
[Qwen3-Omni repository](https://github.com/QwenLM/Qwen3-Omni) describes the
family as a native audio-visual model rather than a VLM with an external
speech recognizer attached.

Our experiment did **not** feed Qwen3-Omni a continuous video stream. Each
question used 12 uniformly sampled frames; the treatment arm added the
video's native WAV audio. This is therefore a clean test of whether native
audio helps the same frame-based visual evidence, not a complete test of the
model's native-video capability.

The clearest result comes from the neutral-prompt A2 run:

| Neutral A2 condition | Correct / partial / wrong | Exact match | Token F1 |
|---|---:|---:|---:|
| 12 frames only | 3 / 3 / 4 | 0.1000 | **0.2295** |
| Same 12 frames + native audio | **5 / 2 / 3** | 0.1000 | 0.2204 |
| Audio minus frames | +2 / -1 / -1 | 0.0000 | -0.0091 |

The audio-on arm produced three semantically better answers: prosody `P→C`,
non-speech sound `P→C`, and a dance-after-shove action `W→P`. Both visual
controls remained correct. Audio did **not** fix the difficult French reply, the
end-of-video bell, another preceding-action question, or the incomplete sour
reaction. The official lexical-overlap F1 decreased slightly even though the
manual answer-quality categories improved, because the model often used
different wording from the short references.

**Decision:** keep Qwen3-Omni as a promising native-audio candidate, especially
for prosody and non-speech sound. Do not call it the new overall baseline yet.
The panel has only ten deliberately difficult videos, and A2 is formally
ineligible because one otherwise valid answer was wrapped in a Markdown JSON
fence: 19/20 outputs passed the unchanged strict parser.

## What model and input did we actually test?

| Component | This experiment |
|---|---|
| Checkpoint | `Qwen/Qwen3-Omni-30B-A3B-Thinking` |
| Model type | Multimodal mixture-of-experts “Thinking” model |
| Supported model inputs | Text, images, audio, and video |
| Inputs used here | Question + 12 sampled frames, with or without one WAV |
| Output used here | Short text answer and evidence in strict terminal JSON |
| Comparison being isolated | Same model and frames; native audio off vs on |
| Compute | One NVIDIA H200, BF16, CUDA |

The model is substantially larger than the 4B baseline, but the
mixture-of-experts design activates only part of it for each token. “Omni”
means it has native encoders and joint training for multiple modalities; it
does not mean that every kind of video question will automatically be solved.
Frame coverage, spatial detail, exact music recognition, output formatting,
and reasoning can still fail independently.

## Native audio helped sound character and vocal delivery, not every audio question

The following table is the most useful view of A2. `C`, `P`, and `W` mean
correct, partially correct, and wrong under the repository's answer-only
manual rubric.

| Question role | Frames | Frames + native audio | What changed |
|---|---:|---:|---|
| Spoken final synonym | C | C | Already solved visually/contextually; audio added nothing. |
| French reply about horse age | W | W | Both answered with a year instead of the bystander's reply. Native audio did not recover the requested proposition. |
| Emotional/prosodic change | P | **C** | Audio added the forceful tone that frames alone could not establish. |
| Cat vocalization | P | **C** | Audio distinguished a high-pitched scream-like sound from generic meowing. |
| Sound at the end | W | W | Audio hallucinated a song/chord instead of identifying the requested bell. |
| Action after a shove | W | **P** | The audio-on answer included dance-like movement, but still did not name the dance. |
| Action before placing balls | W | W | Both described an unrelated event; this remains a temporal-evidence failure. |
| Reaction after sour food | P | P | Neither answer recovered the complete verbal reaction. |
| Small ornament control | C | C | Native audio did not damage the spatial visual answer. |
| Face-mask count control | C | C | Native audio did not damage counting. |

This pattern supports a specialist interpretation:

- **Better:** vocal intensity/prosody and the character of a non-speech sound.
- **Sometimes helpful:** action questions where audio provides an indirect
  temporal cue, although the improvement was only partial.
- **Still weak:** exact spoken propositions in another language, exact
  endpoint sounds, and actions missing from the sampled frames.
- **No measured harm to visual controls in A2:** both control answers stayed
  correct.

## Comparison with our Qwen3.5-4B + Whisper benchmark

Our strongest completed 30-video baseline is Qwen3.5-4B with 12 frames plus a
quality-gated Whisper large-v3 transcript. Across its full frozen-30 panel it
scored `0.3667` exact match and `0.5068` token F1. Those numbers must **not** be
compared directly with Qwen3-Omni's ten-row aggregate: the Qwen3-Omni panel was
selected from difficult failures and is not the same cohort.

Only three Qwen3-Omni examples have an existing matched, manually adjudicated
Qwen3.5 + Whisper result. On that shared subset:

| System on the same three videos | Audio representation | C / P / W | Exact | Mean token F1 |
|---|---|---:|---:|---:|
| Qwen3.5-4B, 12 frames + Whisper large-v3 | ASR transcript text | 1 / 1 / 1 | 0.0000 | **0.2333** |
| Qwen3-Omni, 12 frames + native audio | Waveform consumed by the model | **2 / 0 / 1** | 0.0000 | 0.1531 |

The frozen-30 aggregate comes from
`data/vqa/benchmarks/runs/native-audio-benchmark-v2/evaluation.json`
(SHA-256
`f610dbd3d468f3b72074c7ace38dec95176c7f47fa213b5ce374d0a9429ad9c0`).
The shared three-row Qwen3.5 + Whisper judgments and metrics are documented in
the earlier
[`Qwen3-Omni compatibility report`](2026-07-24-qwen3-omni-dstack-compatibility.md);
the Qwen3-Omni A2 values reproduce the sanitized score artifact linked below.

The three shared rows are the French horse reply, emotional prosody, and the
cat sound. Both systems missed the horse reply. Qwen3-Omni was manually
correct on prosody and the scream-like cat sound, while Qwen3.5 + Whisper was
correct on prosody and only partial on the cat sound. This is exactly where
native audio should have an advantage: Whisper converts speech into words but
does not faithfully preserve vocal intensity or classify arbitrary
non-speech sounds.

The lower Qwen3-Omni token F1 on those same rows is not evidence that it was
semantically worse. Its longer paraphrases overlapped the short references
less literally. Conversely, the `2C/0P/1W` manual result is only three
examples and cannot establish general superiority. The defensible conclusion
is narrower: **Qwen3-Omni shows a useful native-audio signal on prosody and
sound character, while Qwen3.5 + Whisper remains the stronger validated
whole-panel baseline.**

## Why A1 and A2 tell different stories

The first immutable run, A1, completed successfully but used a prompt that
explicitly told both conditions to use audio—even when the frames-only arm
contained no audio tensors. It produced unsupported audio claims and a visual
counting regression. A1 therefore measures adding audio under an unnatural
audio-expecting prompt:

| A1 condition | C / P / W | Exact match | Mean token F1 |
|---|---:|---:|---:|
| Frames only | 5 / 1 / 4 | 0.1000 | 0.2163 |
| Frames + native audio | 3 / 2 / 5 | 0.1000 | 0.2600 |

A2 corrected only the instruction: both arms received the same
modality-neutral prompt, while the model, frames, audio, decoding, ordering,
and scoring stayed fixed. That makes A2 the better semantic comparison.
However, its one fenced-JSON response means it remains a diagnostic rather
than a strict promotion result.

## Verified facts

- Frozen run: `qwen3-omni-quality-10-20260725-a1`; all 20 records passed the
  condition, EOS, and strict-terminal-JSON contracts. The canary and terminal
  workload status passed.
- Prediction SHA-256:
  `9804197489450f1a31836ff1b80c773261c3917bea350243a5c374c9d52c89a8`.
  Contract SHA-256:
  `df016767b1a072770149fdc06fa47dfe7f17930b6c3145af4e92f55d58f4ddeb`.
- The sanitized score artifact binds every model answer to its condition,
  ignored annotation lookup path, and reference SHA-256; it contains no
  reference-answer text. It also records scorer SHA-256
  `65145375c8f49502ade4fd19a8b24acb3c76134b9f7961453b896220fcb731c1`.
- Frames-only attestation reported 12 visual elements and zero audio elements
  or audio tensors; audio-on reported the same 12 visual elements and one
  audio element. The unsupported audio-language issue below is therefore not
  evidence of modality leakage.
- The A1 H200 run was `1608.033806` seconds wall-clock, cost `$1.9609`, and ran
  at `$4.39/hour` on on-demand RunPod. Cumulative Qwen spend is
  now `$2.6269 + $1.9609 + $1.7253 = $6.3131` after A2.

## A1 sanitized automatic scores

| Condition | C / P / W | Exact match | Mean token F1 |
|---|---:|---:|---:|
| Frames only | 5 / 1 / 4 | 0.1000 | 0.2163 |
| Frames + native audio | 3 / 2 / 5 | 0.1000 | 0.2600 |
| Audio minus frames | — | 0.0000 | +0.0437 |

The official exact/F1 fields reproduce the frozen repository scorer and do not
equate digit and number-word forms. Clearly named per-example secondary fields
preserve that equivalence as a diagnostic only; they are not official metrics.
Full per-condition answers, metric values, annotation paths, reference hashes,
and answer-only judgments are in the sanitized artifact:
[`qwen3-omni-quality-10-score.json`](artifacts/qwen3-omni-quality-10-score.json).

## A1 manual answer-only adjudication

Judgments use `C` (correct), `P` (partial), and `W` (wrong). The short reasons
describe the adjudication without reproducing reference text.

| Diagnostic role | Frames only | Frames + audio | Concise adjudication |
|---|---:|---:|---|
| lexical speech | C | C | Both recover the requested final word. |
| repaired horse prompt | W | W | Both answer a different proposition rather than the requested reply. |
| prosody | C | C | Both identify the central vocal and gestural intensity cues. |
| non-speech sound | P | P | Both identify the animal vocalization, but not its specified form. |
| endpoint sound | W | W | Both name a different end sound. |
| temporal action | W | W | Both describe an intervening response rather than the requested action. |
| temporal action | W | W | Both give an unrelated preceding event or sequence. |
| cross-modal reaction | C | P | Audio-on retains chewing and sourness but omits the verbal response. |
| spatial visual control | C | C | Both identify an ornament type consistent with the target. |
| counting visual control | C | W | Audio-on gives a conflicting count. |

The paired manual transitions are `3 C→C`, `1 C→P`, `1 C→W`, `1 P→P`, and
`4 W→W`. No audio-focused row improved from `W` to `P/C` or from `P` to `C`.

## A1 gate result

The predeclared issue #17 promotion gate **fails**:

- All 20 condition contracts passed.
- Audio-on did not achieve a qualifying audio-focused manual improvement.
- One visual control regressed from `C` to `W` with audio on; the other stayed
  `C`.
- No inference label-leakage or frozen-provenance failure was found. The prompt
  mismatch below limits the intended visual-baseline interpretation, but does
  not invalidate the within-run contrast of adding audio under the fixed
  instruction.

## Limitations and follow-up

The horse row is an explicit prompt-repair diagnostic. Its question was
rewritten because the original annotation wording was declarative; this row is
not a frozen-prompt benchmark result.

The panel has ten videos and is a diagnostic, not a leaderboard estimate. Its
temporal rows use twelve frozen uniform frames and no native temporal-video
input, so failures cannot distinguish temporal reasoning from evidence-coverage
limits.

This was one non-blinded, answer-only reviewer after the frozen prediction
join. The references were already familiar from prior repository work; the
manual categories are useful diagnostic evidence, not an independent blinded
human evaluation.

There is also a treatment-context prompt mismatch. `omni_runtime.py` instructs
every request to use visual frames “and the audio together” and to cite both
kinds of cues, even when the quality runner disables audio. Six audited
frames-only rows relied on unavailable auditory evidence or claimed it:
`NN7G28sBOlY`, `NUmb2P3Bzu8`, `NkXPPJoZ-h8`, `IWSPAfCKtV8`,
`Gp4LVTn3YbY`, and `EMI_anpHII4`. Their zero-audio tensor attestations rule
out input modality leakage. Because the instruction is constant, the paired
run still isolates adding audio in this audio-expecting prompt context; the
mismatch instead makes frames-only an unnatural visual baseline and limits
generalization to a modality-neutral task. A neutral prompt rerun is required
for that intended benchmark interpretation.

The matched three-row baseline comparison above is intentionally separate
from the ten-row aggregate. It uses shared videos and the same manual rubric,
but the model prompts and audio representations are not identical, and the
three-row overlap is too small for a ranking claim.

## Neutral-prompt A2 correction

A2 used the same evidence, model, decoding, paired ordering, and conditions
with one modality-agnostic instruction in both arms. Its prediction SHA-256 is
`48f411538ba4e7118a08d9da5d4917f08bbf06f9694211e355c4d80ad678be08`;
its instruction SHA-256 is
`44b83ed11a2b9b8a62584302b126fb8eec13b03c55acb57d08b146f41c5e09a6`.

The strict result is failed and ineligible: the canary passed, 19 rows passed
strict parsing, and the `NkXPPJoZ-h8` audio-on row ended correctly but wrapped
its terminal JSON in a Markdown fence. A fail-closed post-hoc scorer recovers
only that exact identity and raw-output hash for semantic diagnosis; it does
not weaken A1's complete-run refusal or alter A2's terminal result.

| A2 diagnostic | C / P / W | Exact | Token F1 |
|---|---:|---:|---:|
| Frames only | 3 / 3 / 4 | 0.1000 | 0.2295 |
| Frames + native audio | 5 / 2 / 3 | 0.1000 | 0.2204 |
| Audio minus frames | — | 0.0000 | -0.0091 |

Manual transitions include prosody `P→C`, non-speech sound `P→C`, and temporal
dance `W→P`; both visual controls remain `C`. The neutral prompt reduces A1's
six audited frames-only unsupported auditory claims to one clear case:
`NkXPPJoZ-h8` still cites audio despite zero audio inputs. Thus the prompt fix
reduces, but does not eliminate, unsupported evidence language.

See the full
[`neutral A2 report`](2026-07-25-qwen3-omni-quality-10-neutral-a2.md) and
[`sanitized A2 score artifact`](artifacts/qwen3-omni-quality-10-neutral-a2-score.json).
The A2 run cost `$1.7253` and took `1414.836509` seconds. Another one-hour
`$4.50` run would put worst-case cumulative spend at `$10.8131`, above the
authorized `$10` cap, so no full rerun is authorized.

## Recommended next steps

1. Fix terminal JSON handling with constrained decoding or a narrowly tested
   parser correction, then rerun only the failed cat-sound sentinel as a cheap
   validation of that fix. This cannot change immutable A2's 19/20 status.
2. Formal strict promotion would require a fresh full 20-call paired run under
   one frozen configuration. That run is not authorized within the current
   worst-case budget.
3. With new budget approval, run Qwen3-Omni on a matched slice of the
   frozen-30 panel rather than another hand-picked failure panel. Preserve
   separate speech, non-speech sound, temporal, spatial, and visual-control
   strata.
4. Compare three arms on exactly the same questions and evidence:
   frames only, frames + Whisper transcript, and frames + native audio. This
   would show whether waveform access adds value beyond better ASR.
5. Keep specialist routes for failures the general omni model still misses:
   temporal retrieval for unseen actions and audio retrieval/classification
   for exact music or endpoint-sound questions.
