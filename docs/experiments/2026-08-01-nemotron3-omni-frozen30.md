# Nemotron 3 Nano Omni frozen-30 baseline replacement

Date: 2026-08-01

Evidence validity: **Verified**

Decision disposition: **Not promoted under the tested condition**

## Completed evidence record

- **Primary theme:** model comparison and specialist candidates.
- **Canonical claim page and anchor:** this record, `#completed-evidence-record`.
- **Question:** on the exact frozen-30 panel, does native audio-video Nemotron
  meet the preregistered semantic and operational gates to replace the current
  Qwen3.5-4B 12-frame VQA baseline?
- **Shared evidence bundle:**
  `hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/nemotron3-omni-frozen30-20260801-a1/`
  (private; workload, sanitized audit, input, and grading prefixes).
- **Supersedes / superseded by:** supersedes the pending preregistration in
  this file; not superseded. The current Qwen baseline remains unchanged.

## Tested condition and panel

The candidate was
`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` at revision
`dc5f0b0bfddf8b6e0f5891475be9af05b80126fe`, served by official vLLM 0.20.0
CUDA image
`docker.io/vllm/vllm-openai:v0.20.0@sha256:77797441eae630c2e79eefa03957b3d61a278670f2a9928d64ce102e7a0790cc`.
Inference was non-thinking, greedy, short-answer only; no chain-of-thought was
requested or retained.

The panel was the pinned, label-free 30-row manifest
(`12fb622607be2bff0fc07789f5002fb14a51755bd18adabd9fb508d19230c499`), with
matched audio-off and audio-on conditions (60 predictions total). Qwen12 was
the independently verified first-30-row control from the pinned v2 freeze.
This is an end-to-end pipeline comparison: Qwen receives sampled frames while
Nemotron receives continuous normalized video with optional native audio.

## Gates and completion denominator

| Gate | Observed result |
|---|---|
| Exact-image native audio-video canary | 3/3 valid short answers; scheduler `nemotron3-f30-canary-20260801-a1`; cost `$0.1529` |
| Candidate freeze | **60/60** adjacent audio-off/audio-on predictions; manifest and `SHA256SUMS` verified |
| Qwen control freeze | Pinned full-file hash, order, and provenance verified before references loaded |
| Blind grading | Two independent 90-item packets; third grader adjudicated only 6 disagreements |
| Operational gate | **Pass**: status `done`, 34.832 minutes, `$0.5747`, `$0.99/hour`, verified private artifacts, no load-bearing privacy/audit gap |
| Semantic replacement gate | **Fail**: 8 improvements, 6 regressions, 16 ties; two visual-control regressions and a temporal category with more regressions than improvements |

## Observed result

The audio-on condition improved over Qwen12 on 8 rows and regressed on 6. The
six visual controls had 0 improvements, 2 regressions, and 4 ties. By category,
audio-on versus Qwen12 was:

| Category | Improvements | Regressions | Ties |
|---|---:|---:|---:|
| Dialogue and speech | 3 | 0 | 3 |
| Non-speech sound and music | 3 | 0 | 3 |
| Spatial, OCR, and color | 0 | 1 | 5 |
| Temporal and action | 2 | 3 | 1 |
| Visual controls | 0 | 2 | 4 |

The non-speech sound/music category therefore shows a useful specialist signal,
but it does not satisfy the baseline-replacement rule. Audio-on versus the
matched audio-off condition improved 2, regressed 0, and tied 4 in that
category.

Automatic aggregate metrics (same scorer for all conditions) were:

| Condition | Exact match | Token F1 | METEOR |
|---|---:|---:|---:|
| Qwen12 | 0.3000 | 0.4307 | 0.2645 |
| Nemotron audio-off | 0.2667 | 0.4009 | 0.2861 |
| Nemotron audio-on | 0.3333 | 0.4523 | 0.3423 |

The automatic guardrails pass: audio-on is +1 exact-match row, +0.0216 token
F1, and +0.0778 METEOR versus Qwen12. They cannot override the failed
semantic row/category/control boundaries.

## Immutable execution and trust evidence

The pre-run offer snapshot contained six physically distinct compatible
on-demand L40S offers: Vast.ai Türkiye `$0.55185`, Japan `$0.61852`, two
Taiwan offers `$0.81852`, and RunPod EU-NL-1/EUR-IS-2 at `$0.99` per hour.
The full run used RunPod EU-NL-1, one NVIDIA L40S (48 GiB), 32 CPUs,
128000 MiB host memory, and 102400 MiB disk. The full-run scheduler/evidence
ID was `nemotron3-omni-frozen30-20260801-a1`.

The sanitized scheduler, preview, cost, and workload metadata are durably
linked in the private `audit/` prefix. Its workload anchor SHA-256 is
`c7d6ab5dcb75069f64618ecd8028a44e40ef26f76137780a9fde1ec71aff9e53`; its
audit-bundle SHA-256 is
`f3cf15c213f2f81b0e7f712e0eeeb883869fa3296f00bc18e1b1b6aebf9b2438`.
The workload `SHA256SUMS` SHA-256 is
`960e035e3491f38f64f0f844ad5b384f1d9fe77a6cfe2fa1852851a6b2e36222`.

The independent trusted promotion root is the signed Git attestation commit
`10c3747e47a2fe1209401698e402d07b664a8818`, trust-root blob
`bd4a518b3e498a09ff6745329c696eb25e6012c3`, pinned by commit `8d98fdf` and
reviewed by `1314528+npatta01@users.noreply.github.com` with signer fingerprint
`SHA256:+w0DSDr+utElA1L6qX7urVSOt8r9kb221ZA8QDm3Gcc`.

Blind packets, submissions, adjudication, references, and the private score
report are under the private `grading/` prefix. The downloaded grading bundle
passed its own `SHA256SUMS`; that checksum file has SHA-256
`188eccd9acb19e7ef3d8f5fd8dff9682b66d458db420ed45950bf59e953477d4`.

## Limitations, rules-out boundary, and reconsideration

- **Limitations:** this is a condition-bound 30-row panel and an end-to-end
  evidence comparison, not a same-evidence model-only comparison. Hardware,
  normalization, prompt, decoding, and budget are fixed to the run above.
  METEOR uses pinned resource-free lexical tokenization so the score is
  reproducible without mutable external NLTK data; deltas remain within-run
  comparable, while absolute values are condition-bound.
- **What this rules out now:** promoting Nemotron as the current baseline for
  this tested pipeline; treating the sound/music signal or a canary alone as
  replacement evidence; and rerunning the valid run solely because quality was
  negative.
- **What this does not rule out:** a specialist sound/music route, a different
  panel or evidence-matched comparison, other Nemotron configurations, or a
  separately authorized future experiment with a new preregistration.
- **Audit gap:** none load-bearing. Raw media, rows, predictions, references,
  grades, and logs remain private by design; the retained sanitized audit
  preserves the preview and scheduler/region/duration/price/cost fields needed
  to reconstruct the operational decision.
- **Reconsider when:** a new, independently preregistered experiment changes
  the panel, evidence modality, runtime, or decision rule. This valid negative
  result ends the current track.

<details>
<summary>Collapsed method and privacy boundary</summary>

The collector verified the exact image, CUDA runner, native audio-video seam,
input bundle, normalized media, candidate manifest, prediction checksums,
completion denominator, Qwen comparator, blind packet bindings, third-grader
binding, and trusted operational attestation before scoring. No credentials,
chain-of-thought, raw competition rows, or private labels are present in this
repository record or generated report.

</details>
