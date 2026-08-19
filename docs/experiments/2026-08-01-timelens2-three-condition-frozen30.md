# TimeLens2 three-condition frozen-30 — selector gate not promoted

- **Primary theme / related themes:** Visual Evidence is primary; Integrity and
  Models are related.
- **Question:** Does TimeLens2-guided sampling improve matched VQA answers over
  uniform-12 as hybrid-12 (six global plus six local) or combined-18 (all 12
  uniform plus the same six local frames)?
- **Frozen comparison:** The label-isolated panel came from
  `configs/vqa/native_audio_benchmark_30_inference.json` at SHA-256
  `12fb622607be2bff0fc07789f5002fb14a51755bd18adabd9fb508d19230c499`.
  The intended denominator was 30 selector requests and 90 answers. The
  immutable input contained 30 media payloads plus the prepared request file;
  all checksums passed before both real attempts.
- **Disposition:** **Selector compatibility not promoted; matched answer
  quality incomplete.** The exact-image canary passed, but the corrected real
  attempt accepted 29 of 30 strict interval results. One TimeLens response was
  rejected as out of duration, so frame extraction and all 90 answers were
  correctly withheld. Reference answers and annotations were never opened.
  The uniform-12 default therefore remains unchanged.

## Exact runtime and offers

Every attempt used
`MCG-NJU/TimeLens2-2B@90dd30b3eeff3520a30c830eff2209f4ad788ca9`,
`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, and
`docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755`.
The corrected runtime was Transformers `5.15.0.dev0` at official revision
`cbf4d720ec734edb77d25452b2790c7e4be2f8d7`, qwen-vl-utils `0.0.14`, PyAV
`16.0.1`, and CUDA `12.4` with bfloat16 inference.

- **Canary A3:** `tl2-3cond-canary-20260801-a3` previewed eight offers. Three
  shown physically distinct Vast.ai RTX 6000 Ada 48 GB nodes were priced at
  `$0.6319`, `$0.7252`, and `$0.7344` per hour. The selected California node
  had 120 GB host RAM at `$0.63185/h`; actual cost was `$0.0318`.
- **Frozen A1:** `tl2-3cond-frozen30-20260801-a1` previewed eight offers. Three
  shown RTX 6000 Ada 48 GB nodes in California and Texas were `$0.6319`,
  `$0.7078`, and `$0.7344/h`. The selected 120 GB California node was
  `$0.63185/h`; actual cost was `$0.0416`.
- **Frozen A2:** `tl2-3cond-frozen30-20260801-a2` previewed seven offers. The
  shown distinct choices included RTX 6000 Ada 48 GB nodes in California and
  Texas at `$0.7078` and `$0.7344/h`, plus a RunPod L40S 48 GB node in the
  Netherlands at `$0.99/h`. The selected 128 GB California node was
  `$0.70778/h`; actual cost was `$0.0922`.
- **Budget:** All three-condition attempts A1/A2/A3 and frozen A1/A2 cost
  `$0.2316`. Total TimeLens2 spend is `$1.0902 / $4.00`, leaving `$2.9098`.
  The central reserve was not used.

## Gate and completion evidence

- **Canary A1:** The selector passed, but Transformers `4.57.6` did not support
  dense `qwen3_5`; completion was 1/1 selectors and 0/2 answers.
- **Canary A2:** The corrected Transformers runtime loaded both models, but the
  synthetic fixture requested a final timestamp after its last decodable frame;
  completion was 1/1 selectors and 0/2 answers.
- **Canary A3:** The fixture was corrected test-first to request all 18 actual
  source PTS values and to reject missing decoder output immediately. The
  selector returned raw `[[0.0, 2.0]]`; both 12-frame and 18-frame Qwen calls
  completed. Qwen peak allocation was 9,631,145,984 bytes, and selector peak
  allocation was 4,333,073,408 bytes. Completion was **1/1 selectors and 2/2
  answers**.
- **Frozen A1:** All 30 rows were attempted; 26 passed. Four exact-media rows
  were rejected because FFprobe-frozen durations differed by 54--76 ms from
  PyAV packet-end measurements under a 50 ms tolerance. This was a technical
  cross-probe validation inconsistency, not an OOM. Completion was **26/30
  selectors and 0/90 answers**.
- **Frozen A2:** A test-first 100 ms cross-probe tolerance retained validation
  while accepting the observed sub-frame deltas. All 30 rows were attempted;
  29 produced accepted strict results with one interval each. Median selector
  latency was 2.580 seconds (range 0.893--10.234), peak allocation was
  5,785,068,032 bytes, and execution was bfloat16 on RTX 6000 Ada. One response
  was rejected because interval 0 was outside the measured video duration.
  Completion was **29/30 selectors and 0/90 answers**.

The predeclared inference freeze required 30 selectors and 90 answers. It was
not reached, so no grading packet was created and no hybrid-versus-uniform
quality rule was evaluated. A third paid run was not submitted: repeating a
technically valid strict-schema failure would violate the no-quality-retry
rule. A future experiment would need a separately predeclared selector
abstention/fallback policy; it cannot be retrofitted onto this result.

## Validity and audit gaps

- Raw, normalized, sorted, and merged representations remained separate. The
  out-of-duration interval was not clamped, buffered, or normalized into
  acceptance.
- A2 exposed an **audit gap**: the rejected row's raw model text was discarded
  by the per-row exception recorder, so the exact invalid interval cannot be
  reconstructed from immutable artifacts. The durable record proves the
  strict parser's error class and message but not the rejected numeric text.
  A post-run regression fix now retains raw output, duration, latency, memory,
  revisions, and exact phase for future rejected attempts. It does not repair
  or upgrade A2 retrospectively.
- The A1 summary's final phase label was also coarser than the actual duration
  check. Post-run phase checkpoints now distinguish duration validation,
  inference, raw parsing, normalization, and the completion gate.
- No competition labels, reference answers, raw media, credentials, or private
  logs are included in Git.

## Private artifact verification

- **Frozen input:**
  `hf://buckets/Npatta01/trec_video_2026/experiments/timelens2-three-condition-frozen30-20260801-a1/input`.
  All 31 payloads verified after fresh download. `SHA256SUMS` SHA-256:
  `1e03593461416c2141045523ae82cf2c40de1a97f0eba1ca0290323fb9df656a`.
- **Canary A3 output:**
  `hf://buckets/Npatta01/trec_video_2026/experiments/timelens2-three-condition-canary-20260801-a3/output`.
  All 22 payloads verified; manifest SHA-256
  `df5e7f85976e5a64a52121b5167c559690f80613cc368e8ea0f8ec1667359eae`.
- **Canary A3 audit:** same A3 run identity under `/audit`. All five payloads
  verified; audit manifest SHA-256
  `f435a800331f061e2a0b7ab38d27695711ffe4bd24244844e6671679410ea457`.
- **Frozen A1 output:**
  `hf://buckets/Npatta01/trec_video_2026/experiments/timelens2-three-condition-frozen30-20260801-a1/output`.
  All eight payloads verified; manifest SHA-256
  `f525b5de47cd7d944b38143178546d4fac0b762b4a1de6d24236d25dc27d02e2`.
- **Frozen A1 audit:** same A1 run identity under `/audit`. All five payloads
  verified; audit manifest SHA-256
  `1c012f77ab6e3def33cce2f313fb6e5d72b484e846fd1c8833b8b51f62a73ee7`.
- **Frozen A2 output:**
  `hf://buckets/Npatta01/trec_video_2026/experiments/timelens2-three-condition-frozen30-20260801-a2/output`.
  All eight payloads verified; manifest SHA-256
  `c846857e7267fbe571c85189a5ac84953e9676ef6372b1cc919ce1917066ffb4`.
- **Frozen A2 audit:** same A2 run identity under `/audit`. All five payloads
  verified; audit manifest SHA-256
  `af5ce67b330aa9f2db1a847803878f995ca91356ad9b203489cbd25003f7e6c7`.

Earlier canary A1/A2 bundles remain under their immutable run identities with
the previously verified manifests. The official corrected runtime is supported
by the pinned [Transformers `qwen3_5` source](https://github.com/huggingface/transformers/tree/cbf4d720ec734edb77d25452b2790c7e4be2f8d7/src/transformers/models/qwen3_5)
and its [`AutoModelForImageTextToText` mapping](https://github.com/huggingface/transformers/blob/cbf4d720ec734edb77d25452b2790c7e4be2f8d7/src/transformers/models/auto/modeling_auto.py#L1035-L1117).
