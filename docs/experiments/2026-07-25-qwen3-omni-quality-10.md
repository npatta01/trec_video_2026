# Qwen3-Omni paired quality-10 result

## Decision

Do not promote Qwen3-Omni to a broader quality benchmark. The immutable A1 run
completed successfully, but native audio produced no answer-only `W→P/C` or
`P→C` improvement and made one visual control worse. Its constant
audio-expecting prompt isolates adding audio within that instruction context,
but mismatches the frames-only treatment context.

The neutral-prompt A2 correction improves the post-hoc audio-on manual result
to `5C/2P/3W`, helps the temporal dance row, and removes the counting-control
regression. It nevertheless remains strictly failed and ineligible: exactly
one audio-on row returned its complete JSON answer inside a Markdown fence, so
only 19/20 records passed the unchanged parser. The scorer recovers that exact
pinned output solely for a clearly labeled semantic diagnostic.

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

## Sanitized automatic scores

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

## Manual answer-only adjudication

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

## Gate result

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

No matched external baseline is quantified here: the available earlier runs
do not share this frozen paired prompt/condition contract, so comparing their
aggregate scores would add a second confound.

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
