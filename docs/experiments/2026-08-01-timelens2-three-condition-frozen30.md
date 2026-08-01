# TimeLens2 three-condition frozen-30 — stopped at canary

- **Primary theme / related themes:** Visual Evidence is primary; Integrity and
  Models are related.
- **Question:** Does TimeLens2-guided temporal sampling improve matched VQA
  answers over uniform-12 when tested as hybrid-12 (six global plus six local)
  or combined-18 (all 12 uniform plus the same six local frames)?
- **Frozen comparison:** The label-isolated development panel was prepared from
  `configs/vqa/native_audio_benchmark_30_inference.json` at SHA-256
  `12fb622607be2bff0fc07789f5002fb14a51755bd18adabd9fb508d19230c499`.
  The intended denominator was 30 selector requests and 90 answers across
  uniform-12, hybrid-12, and combined-18. No reference answers or annotations
  were opened.
- **Disposition:** **Incomplete; no quality result.** The exact-image canary
  failed twice for distinct technical reasons, consuming the one predeclared
  corrective attempt. The real frozen-30 workload was not preview-submitted,
  no private-media inference output was produced, and no grading occurred.
  This is neither a positive nor a valid negative result for TimeLens2-guided
  sampling and does not change the uniform-12 default.

## Exact runtime and offer evidence

Both canaries used
`MCG-NJU/TimeLens2-2B@90dd30b3eeff3520a30c830eff2209f4ad788ca9`,
`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, and
`docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755`.
The dstack resource names were shortened to satisfy its 41-character limit;
the durable artifact identities remained descriptive and immutable.

- **A1:** `tl2-3cond-canary-20260801-a1` previewed nine eligible offers.
  Three physically distinct Vast.ai RTX 6000 Ada 48 GB offers were shown in
  California and Texas at `$0.6319`, `$0.7078`, and `$0.7344` per hour. The
  selected California node had 120 GB host RAM and cost `$0.0395` actual.
- **A2:** `tl2-3cond-canary-20260801-a2` previewed eight eligible offers.
  Three physically distinct Vast.ai RTX 6000 Ada 48 GB offers were shown in
  California and Texas at `$0.6319`, `$0.7252`, and `$0.7344` per hour. The
  selected California node had 120 GB host RAM and cost `$0.0265` actual.
- **Budget:** These attempts cost `$0.0660`. Total TimeLens2 spend is now
  `$0.9246 / $4.00`, leaving `$3.0754`; the central reserve was not used.

## Failure evidence and completion

- **A1:** TimeLens completed its synthetic selector inference with raw output
  `[[0.0, 2.0]]`, 1.299 seconds latency, 4,328,924,672 bytes peak allocation,
  bfloat16 precision, and an NVIDIA RTX 6000 Ada device. Qwen loading then
  failed because Transformers `4.57.6` did not recognize the checkpoint's
  `qwen3_5` architecture. Completion was **1 / 1 selector** and **0 / 2
  canary answers**.
- **Corrective A2:** A test-first runtime gate pinned Transformers
  `5.15.0.dev0` at official source revision
  `cbf4d720ec734edb77d25452b2790c7e4be2f8d7`. That revision contains dense
  `qwen3_5` support and maps it to `AutoModelForImageTextToText`; the live
  capability attestation passed before model download. TimeLens again
  completed with raw output `[[0.0, 2.0]]`, 1.280 seconds latency,
  4,333,073,408 bytes peak allocation, bfloat16, and the same GPU. Qwen weights
  loaded, but the 18-frame synthetic branch referenced `frame-18.png` after
  the decoder had produced only 17 files at the requested near-end timestamps.
  Completion was **1 / 1 selector** and **0 / 2 canary answers**.
- **Real workload:** `tl2-3cond-frozen30-20260801-a1` was never submitted.
  Completion is **0 / 30 selectors** and **0 / 90 answers**.
- **Persistent error:** The exact-image canary has a synthetic end-of-video
  frame fixture mismatch. Because A2 was the sole corrective technical
  attempt, the track stopped rather than silently running an unvalidated
  private-media workload.

The runtime correction is supported by the official pinned
[Transformers `qwen3_5` source](https://github.com/huggingface/transformers/tree/cbf4d720ec734edb77d25452b2790c7e4be2f8d7/src/transformers/models/qwen3_5)
and its
[`AutoModelForImageTextToText` mapping](https://github.com/huggingface/transformers/blob/cbf4d720ec734edb77d25452b2790c7e4be2f8d7/src/transformers/models/auto/modeling_auto.py#L1035-L1117).

## Private artifact verification

- **A1 output:**
  `hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/timelens2-three-condition-canary-20260801-a1/output`.
  All 20 payloads verified after fresh download. `SHA256SUMS` SHA-256:
  `5ac46af6b724e3bbd962b56a48ce9bbb7af4448f6820dd9f6027151ba6059f8b`.
- **A1 audit:** same run prefix under `/audit`. All seven payloads verified.
  `AUDIT_SHA256SUMS` SHA-256:
  `2a5a67a60b90c36c1d60eebdd3b2cfe0f994c14b0c19ae77f68ef2076ed9c3b0`.
- **A2 output:**
  `hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/timelens2-three-condition-canary-20260801-a2/output`.
  All 21 payloads verified after fresh download. `SHA256SUMS` SHA-256:
  `1c29de6b2c6cd5a57eb602b07327494bd6272f82b686f2f67c6758e9341e8f7e`.
- **A2 audit:** same run prefix under `/audit`. All five payloads verified.
  `AUDIT_SHA256SUMS` SHA-256:
  `ea715cb297b881a398305b3b58af8bfa20a84cdd1db4450d4efd3e1a8b4f3080`.

The private bundles contain synthetic canary assets, failure records, ordinary
logs, declined previews, submission transcripts, sanitized status, and
checksums. They contain no competition media, reference labels, or credentials.
During local control-plane diagnosis an ordinary dstack startup message exposed
the local admin token; it was immediately rotated before either canary was
submitted. No Hugging Face or provider credential was exposed.
