# Qwen3-Omni quality-10 neutral A2 result

## Decision

The neutral A2 run is **strictly failed and ineligible for promotion**. It
produced 19 strict JSON passes and one parser failure. The failed row contains
a complete answer inside one terminal Markdown JSON fence, so a separately
labeled post-hoc semantic diagnostic recovers only that exact pinned output.
The diagnostic does not change the runner, parser, terminal status, or strict
eligibility.

The diagnostic is encouraging: native audio improves three manual categories,
including the temporal dance row, and neither visual control regresses.
However, these semantic conditions cannot override the strict output-contract
failure.

## Frozen provenance and recovery boundary

- Run: `qwen3-omni-quality-10-neutral-20260725-a2`.
- Predictions SHA-256:
  `48f411538ba4e7118a08d9da5d4917f08bbf06f9694211e355c4d80ad678be08`.
- Contract SHA-256:
  `df016767b1a072770149fdc06fa47dfe7f17930b6c3145af4e92f55d58f4ddeb`.
- Neutral instruction SHA-256:
  `44b83ed11a2b9b8a62584302b126fb8eec13b03c55acb57d08b146f41c5e09a6`.
- Strict result: canary passed; 19/20 rows passed; workload return code `1`;
  terminal dstack status `failed`.
- Exact failed identity: `NkXPPJoZ-h8`, `frames-native-audio`, pair index `3`,
  parser error after valid EOS and modality attestation.
- Exact failed raw-output SHA-256:
  `0dbb246367b2abdd7e1810d39597acf2cab9326c453e54db3299d07dd719872a`.
- Diagnostic scorer SHA-256:
  `73a3829f79cc8df6bfd89368288e6d6e085faad0b034150be89cefcd1105faf1`.

The recovery accepts only that run/video/condition/pair, instruction identity,
failure type, successful EOS/modality attestations, raw-output hash, and one
terminal `json` fence containing exactly non-empty `answer` and `evidence`
strings. Any changed identity, byte, hash, status, fence, key set, or trailing
text is rejected. The tracked artifact stores the recovered answer and hashes,
never annotation reference text.

## Post-hoc answer-only diagnostic

| Condition | C / P / W | Official exact | Official token F1 |
|---|---:|---:|---:|
| Frames only | 3 / 3 / 4 | 0.1000 | 0.2295 |
| Frames + native audio | 5 / 2 / 3 | 0.1000 | 0.2204 |
| Audio minus frames | — | 0.0000 | -0.0091 |
| Both conditions | 8 / 5 / 7 | 0.1000 | 0.2250 |

Official exact/F1 reproduce the frozen repository scorer. Their lexical
overlap does not determine the answer-only semantic category.

| Diagnostic role | Frames only | Frames + audio | Paired finding |
|---|---:|---:|---|
| lexical speech | C | C | No change. |
| repaired horse prompt | W | W | Both answer a different proposition. |
| prosody | P | C | Audio adds the central vocal escalation. |
| non-speech sound | P | C | Recovered audio answer identifies the scream-like character. |
| endpoint sound | W | W | Both miss the requested end sound. |
| temporal dance | W | P | Audio adds dance-like body movement, but does not identify the dance. |
| temporal action | W | W | Both give an unrelated preceding event. |
| cross-modal reaction | P | P | Both omit part of the verbal reaction. |
| spatial visual control | C | C | No regression. |
| counting visual control | C | C | The A1 audio-on count regression is removed. |

Paired transitions are `3 C→C`, `2 P→C`, `1 P→P`, `1 W→P`, and `3 W→W`.
Thus the post-hoc diagnostic satisfies the semantic portions of the original
gate, but remains ineligible because strict output conformance failed.

## Comparison with A1

Relative to the audio-expecting A1 run:

- A2 audio-on improves from `3C/2P/5W` to `5C/2P/3W`; manual changes include
  the non-speech row `P→C`, temporal dance `W→P`, and counting control `W→C`.
- A2 frames-only changes from `5C/1P/4W` to `3C/3P/4W`, primarily because the
  neutral answers no longer claim unavailable vocal evidence on the prosody
  row and no longer over-credit the cross-modal verbal response.
- A2 audio-on official exact is unchanged at `0.1000`; token F1 is `0.0396`
  lower than A1 despite the manual-category improvements.
- The neutral prompt reduces the six audited A1 frames-only unsupported
  auditory claims to one clear case: `NkXPPJoZ-h8` still says its evidence
  includes audio even though attestation proves zero audio inputs. The prompt
  repair therefore reduces but does not eliminate unsupported evidence
  language.

The sanitized artifact contains every model answer, condition, annotation
lookup path, reference hash, official metric, manual judgment, recovery hash,
paired delta, and A1 comparison:
[`qwen3-omni-quality-10-neutral-a2-score.json`](artifacts/qwen3-omni-quality-10-neutral-a2-score.json).

## Runtime, budget, and limitations

The on-demand RunPod H200 run took `1414.836509` seconds and cost `$1.7253`.
Actual cumulative Qwen spend is
`$2.6269 + $1.9609 + $1.7253 = $6.3131`.
A further full one-hour run has a `$4.50` worst case, which would raise the
cumulative worst case to `$10.8131` and exceed the authorized `$10` cap.
Therefore no full rerun is authorized from the remaining budget.

This remains a ten-video diagnostic with one non-blinded answer-only reviewer.
The horse question is an explicit repaired-prompt diagnostic. Temporal rows
use twelve uniform frames and no native temporal-video input, so evidence
coverage remains a confound. The single recovered row is suitable for the
reported post-hoc diagnostic only, not for strict promotion.
