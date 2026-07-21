# SeamlessM4T-v2 audio follow-up results

Date: 2026-07-21

## Decision

**Do not advance SeamlessM4T-v2.** On the one speech example, its English
output was materially incomplete and the downstream Qwen answer remained
wrong. Qwen changed from the 12-frame baseline's `End of the trend` to
`Where the girls are from`, but normalized exact match stayed 0, token F1 fell
from 0.286 to 0, and METEOR fell from 0.280 to 0.074.

This is a one-example causal diagnostic, not an accuracy estimate. The
100-video run was not started and no 100-video manifest was created or
executed.

## Controlled speech comparison

The official reference was absent from every inference request and joined from
the frozen labeled annotation only after the exact-capture Qwen process had
terminated and its attempt was indexed.

| Evidence | Prediction | EM | Token F1 | METEOR | BERTScore F1 |
|---|---|---:|---:|---:|---:|
| Frozen 12 frames only | End of the trend | 0.000 | 0.286 | 0.280 | 0.902 |
| 12 frames + Whisper large-v3 | Where the money is | 0.000 | 0.000 | 0.075 | 0.903 |
| 12 frames + SeamlessM4T-v2 large | Where the girls are from | 0.000 | 0.000 | 0.074 | 0.906 |

The frozen reference is `The expected contents of the bag.` The metrics were
computed with the repository's local evaluator: normalized exact match, token
F1, and NIST-style METEOR. BERTScore used the already-cached pinned
`FacebookAI/roberta-large@722cf37b1afa9454edce342e7895e588b6ff1d59`
encoder at layer 17 through an offline, read-only cache mount; no encoder was
downloaded. STS was not called, and singleton NDCG was not computed or
interpreted.

Relative to the stronger of the two frozen text metrics (the frames-only
baseline), Seamless changed EM by 0.000, token F1 by -0.286, and METEOR by
-0.206. Relative to the Whisper large-v3 condition, Seamless changed EM and
token F1 by 0.000 and METEOR by -0.001. The slightly higher BERTScore for the
wrong Seamless answer is another reminder that this similarity metric is not a
correctness judgment.

## Seamless transcript review

All three runner outputs passed the mechanical quality rule and exact
identity/provenance checks. Semantic status is a separate human evidence
review made without opening the official answer.

| Video | Exact English text | Semantic status | ASR time | Peak allocation |
|---|---|---|---:|---:|
| `BDKkVYmWqdM` | He's riding a bike across a tennis net. | accepted-semantically-consistent | 12.211 s | 3.174 GiB |
| `O60MtMuSpf8` | [Music playing] | accepted-semantically-consistent | 11.553 s | 3.023 GiB |
| `Fk_j6re_Uiw` | Where are these girls from? They look like Pakistani. Where are they from? | rejected-semantic-conflict | 11.864 s | 3.118 GiB |

The speech translation covers only the opening lines. The retained native
transcript spans the full 0.0–19.0 seconds and continues with repeated `katta`
utterances after that opening, so the Seamless output materially omits evidence
needed to answer the question. Its downstream condition was therefore required
but exploratory; a fluent English string was not treated as faithful audio
evidence.

## Qwen result and runtime cost

The label-free request retained the frozen question, exact 12 frame paths and
timestamps, null auxiliary fields, Qwen model/revision, and 32-token cap. Only
the condition ID and transcript changed from the frozen speech request. The
single offline ROCm run returned `Where the girls are from`.

| Component | Revision / runtime | Precision and device | Time | Peak allocation |
|---|---|---|---:|---:|
| SeamlessM4T-v2 large S2TT | `5f8cc790b19fc3f67a61c105133b20b34e3dcb76` | FP16, `cuda:0 (ROCm)` | 11.864 s | 3.118 GiB |
| Qwen 3.5 4B model load | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | BF16, `cuda:0 (ROCm)` | 3.155 s | included below |
| Qwen 3.5 4B answer | same | BF16, `cuda:0 (ROCm)` | 338.275 s | 10.569 GiB |

The observed summed runner/model time for this speech condition was 353.293
seconds. Both model stages used runtime image
`localhost/trec-vqa-gate-rocm@sha256:510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5`
with networking disabled. The Qwen stderr contains only the recorded fallback
and experimental ROCm attention warnings; the structured result passed.

SeamlessM4T-v2 large is licensed CC-BY-NC-4.0. This evaluation is therefore a
local non-commercial research diagnostic, not approval for a production
dependency or default baseline.

## Exact capture and provenance

The Qwen request was retained and leakage-audited before execution. Stdout and
stderr were redirected directly to their final ignored files and indexed as an
append-only terminal attempt.

| Artifact | SHA-256 |
|---|---|
| Qwen request | `638062a36c973f332941a332b9988cd9777b8b18ae5a39acf009d1683333a39d` |
| Qwen stdout/result | `6dc6cff9dffa9ba9808f9d2ff63700c0a061e7bb158d20f7b3696a190953354c` |
| Qwen stderr | `95d2fe748e6dd2eff2aaf4e741363aee4932587e246979c41dada9b6e20f5f4e` |
| Frozen labeled annotation, joined after inference | `d64675906e7aee39d76d1606fb656fbb9a5d6558cb54faaac2373ae7c35d73d3` |

The result embeds the exact request identity, pinned model ID and revision,
runtime image digest, BF16 precision, ROCm device, pixel bounds, 338.275-second
answer time, 3.155-second model load, and 11,347,926,016 peak allocated bytes.
The request retains `run_id=audio-spatial-2026-07-21` because the plan required
copying the frozen speech request and replacing only `condition_id` and
`transcript`; its unique condition/video identity and Seamless follow-up path
keep the new attempt unambiguous.

The Seamless snapshot is the pinned 11-file, 9,257,755,549-byte Transformers
artifact set recorded in the ignored execution ledger. The pinned image lacked
protobuf, so the reviewed ASR executions used the recorded read-only host
protobuf 6.33.6 pure-Python mount (108 files; aggregate SHA-256
`6121064e28b3d75706e25d9b2179366923d9165b6488bf137aeb1914964a8f5c`).
Two preliminary ASR failures remain preserved ahead of the successful terminal
records; no generated evidence was overwritten or committed.

## Recommendation

Keep the existing 12-frame baseline as the simple control. Seamless did not
satisfy either advancement requirement: its speech evidence was not
semantically credible and its downstream answer did not improve. If audio is
revisited, first establish a faithful translation on a small, independently
reviewable multilingual set before paying the roughly six-minute downstream
Qwen cost per diagnostic example.

Canonical design:
`docs/superpowers/specs/2026-07-21-seamless-audio-followup-design.md`

Canonical execution plan:
`docs/superpowers/plans/2026-07-21-seamless-audio-followup.md`
