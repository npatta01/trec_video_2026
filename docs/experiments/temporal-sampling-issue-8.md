# Issue 8: Question-directed temporal sampling

## Decision

**Stop; do not promote question-directed 12-frame sampling.** On the frozen
10-question temporal diagnostic, it tied uniform 12 frames in blinded
correct/partial/wrong (C/P/W) adjudication and trailed uniform 24 frames.
Although it used half the visual-token proxy of the 24-frame condition, it did
not meet either quality requirement in the promotion gate.

## Matched conditions

All 30 predictions used Qwen/Qwen3.5-4B at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, the digest-pinned ROCm runtime
`localhost/trec-vqa-gate-rocm@sha256:510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5`,
native decoded frame resolution bounded to 256²–512² pixels per frame, the
same prompt, seed 0, and greedy decoding. The label-blind panel covered
before, after, beginning, and end questions. The directed sampler retained
global context frames and concentrated the balance in the requested temporal
neighborhood.

| Condition | C / P / W | Ordinal score | Exact | Token F1 | METEOR | Runtime | Visual-token proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform 12 | 2 / 5 / 3 | 9 | 0.100 | 0.347 | 0.320 | 1,008.7 s | 31,457,280 |
| Uniform 24 | 4 / 3 / 3 | 11 | 0.200 | 0.413 | 0.380 | 780.1 s | 62,914,560 |
| Directed 12 | 2 / 5 / 3 | 9 | 0.100 | 0.374 | 0.333 | 94.7 s | 31,457,280 |

Runtime is observed wall-clock model time, not a stable throughput estimate;
the small panel had highly variable per-row latency. Visual-token proxy is
frame count × the fixed maximum pixels-per-frame bound.

Paired C/P/W transitions show no net gain over uniform 12: two C→C, four
P→P, two W→W, one W→P, and one P→W. Against uniform 24, directed 12 retained
two correct rows but regressed one C→P and one C→W; it made one W→P gain.

## Reproducibility and label isolation

The compatibility gate completed three label-blind rows before the full run.
The full 30-row prediction artifact was SHA-256 frozen before any annotation
was opened; semantic metrics were then calculated with the pinned scorer
runtime and C/P/W grades were assigned from a condition-blind packet before
unblinding condition IDs.

Ignored local artifacts under `data/vqa/benchmarks/temporal-diagnostic-10/`:

| Artifact | SHA-256 |
| --- | --- |
| inference manifest | `d2cfec431652bcf7c6c83be59a4932fc0ed75016d6a3dc973e49cebdd9ed0241` |
| requests | `6ae7ee945f6e1d8bda1ee329aff2e5de314de44da3a6916e475230183520ce95` |
| compatibility predictions | `12de73a4e80ccccce9057329468dc115177c78663428705819c86afccd948a31` |
| frozen full predictions | `e09792dc191f4ef498138b56ad966470e813db3da33b91353acd514aa3c4212e` |
| prediction-freeze record | `1389e76647992bdd9e947029c98098b56b58631f8c24f4b4aebeec32db1038c7` |
| pinned semantic metrics | `1c93455a1ccf154ca66e663f87e8611a7fc6f75955325d88dcdd2fc81c2ff53f` |
| blinded C/P/W packet | `17669cc4d59f3c2c0b249be18e0b22b3ece91a74d4699ec6577ce982bd2972ed` |
| final evaluation | `a6d40c4240c53afc1f6eba25b2abf83d5a52b0ae3771d97d19f15ada7b028261` |

No frames, media, raw logs, answers, or labels are tracked by this change.
