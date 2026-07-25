# Qwen3-Omni paired quality-10 result

## Decision

Do not promote Qwen3-Omni to a broader quality benchmark from this run. The
immutable run completed successfully, but native audio produced no answer-only
`W→P/C` or `P→C` improvement and made one visual control worse. More
importantly, the inherited prompt makes the frames-only condition an impure
visual baseline, so the observed paired difference must not be interpreted as
a clean audio ablation.

The working runner is preserved. A follow-up should rerun the paired panel
with a neutral, modality-agnostic evidence instruction in both conditions;
the only intended condition difference should then be the audio element.

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
  `50a23e39417c1885efc8320fb4fdaf0360e8b7e2e13b0e18f4ea3fc10b03a4a9`.
- Frames-only attestation reported 12 visual elements and zero audio elements
  or audio tensors; audio-on reported the same 12 visual elements and one
  audio element. The unsupported audio-language issue below is therefore not
  evidence of modality leakage.
- The H200 run was `1608.033806` seconds wall-clock, cost `$1.9609`, and ran
  at `$4.39/hour` on on-demand RunPod. Cumulative Qwen spend is
  `$2.6269 + $1.9609 = $4.5878`.

## Sanitized automatic scores

| Condition | C / P / W | Exact match | Mean token F1 |
|---|---:|---:|---:|
| Frames only | 5 / 1 / 4 | 0.2000 | 0.3163 |
| Frames + native audio | 3 / 2 / 5 | 0.1000 | 0.2600 |
| Audio minus frames | — | -0.1000 | -0.0563 |

The scorer treats the numeric forms `1` and `one` as equivalent. Full
per-condition answers, metric values, annotation paths, reference hashes, and
answer-only judgments are in the sanitized artifact:
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
- No inference label-leakage or frozen-provenance failure was found, but the
  prompt confound below prevents a clean causal interpretation.

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

There is also a direct prompt confound. `omni_runtime.py` instructs every
request to use visual frames “and the audio together” and to cite both kinds of
cues, even when the quality runner disables audio. Several frames-only outputs
then asserted audio support (including the prosody, non-speech, cross-modal,
and craft rows). Their zero-audio tensor attestations rule out input modality
leakage, but the language makes the `frames-only` answers an invalid clean
visual-control baseline. A neutral prompt rerun is required before attributing
any score difference to native audio.

No matched external baseline is quantified here: the available earlier runs
do not share this frozen paired prompt/condition contract, so comparing their
aggregate scores would add a second confound.
