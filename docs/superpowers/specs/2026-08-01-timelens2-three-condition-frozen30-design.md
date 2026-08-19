# TimeLens2 Three-Condition Frozen-30 Design

## Objective

Determine whether question-guided TimeLens2 local evidence should replace part
of, or supplement, the current uniform 12-frame input for the pinned Qwen
answer model. This is a label-isolated development comparison on an existing
30-row project panel, not a pristine external confirmation.

## Frozen inputs and revisions

- Label-free source:
  `configs/vqa/native_audio_benchmark_30_inference.json` at SHA-256
  `12fb622607be2bff0fc07789f5002fb14a51755bd18adabd9fb508d19230c499`.
- The source must contain exactly 30 distinct video identities and only
  `category`, `media`, `question`, and `video_id` per example.
- A private prepared source binds each identity and question to the exact media
  SHA-256 and measured duration before upload. It contains no annotation path,
  reference answer, grade, transcript, or prior prediction.
- Selector: `MCG-NJU/TimeLens2-2B` at revision
  `90dd30b3eeff3520a30c830eff2209f4ad788ca9`.
- Answer model: `Qwen/Qwen3.5-4B` at revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Runtime image:
  `docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755`.
- Runtime dependency pins remain Transformers `4.57.6`, qwen-vl-utils
  `0.0.14`, PyAV `16.0.1`, Accelerate `1.12.0`, imageio-ffmpeg `0.6.0`,
  uv `0.11.3`, and Hugging Face CLI `1.26.0`.

## Experimental conditions

The selector runs once per row. Its strict raw result is preserved and checked
before interval normalization. The answer model then runs once per condition:

1. **uniform-12:** twelve equal-bin midpoint frames over the full clip.
2. **hybrid-12:** six equal-bin global frames plus six local frames distributed
   across normalized TimeLens intervals by deterministic Hamilton allocation.
3. **combined-18:** all twelve uniform frames plus the same six TimeLens-local
   frames used by hybrid-12.

Every condition is chronological and contains the declared number of distinct
decoded images. Shared uniform frames may be reused across conditions; this is
intentional and is not a collision. Within-condition collisions move only a
local slot to the nearest unused actual source-frame PTS strictly inside its
own predicted interval. Global timestamps never move. A row fails closed when
no valid local replacement exists.

No temporal padding is added outside predicted intervals. Raw selector output,
normalized intervals, requested and resolved timestamps, frame hashes,
latency, peak memory, device, precision, model revisions, runtime revisions,
and request hashes are retained.

## Label isolation and evaluation

The dstack workload receives only the prepared label-free source and exact
media. It freezes all 30 selector results, 90 answers, recipes, resolved frame
manifests, attempts, and completion denominators before terminating.

Only after the downloaded inference freeze and checksums verify may a separate
local evaluation process open
`configs/vqa/native_audio_benchmark_30.json`. Evaluation constructs a
condition-blind packet, assigns correct/partial/wrong grades with the existing
project grading contract, and then reveals condition identities for paired
comparisons. No inference rerun is permitted because of answer quality.

The primary comparisons are hybrid-12 versus uniform-12 and combined-18 versus
uniform-12. Hybrid-12 versus combined-18 is secondary. Component-scene coverage
is not claimed because this 30-row panel has no independently accepted
relevant-scene oracle.

## Decision rule

The result is valid only if all 30 selector rows and all 90 answers complete
and verify. For each TimeLens condition, an improvement is a higher ordinal
grade (`wrong < partial < correct`) than uniform-12 on the same row; a
regression is lower.

A TimeLens condition becomes the preferred development default only if it has:

- at least four improved rows versus uniform-12;
- at most two regressed rows;
- a net improvement of at least three rows; and
- no decrease in total correct answers.

If both pass, prefer the condition with more correct answers, then larger net
improvement, then fewer frames. A complete result missing the thresholds is a
valid negative under this condition. Complementary wins between uniform-12 and
hybrid-12 may justify a separately designed two-answer ensemble; this run does
not silently add an answer judge.

## Runtime, privacy, and cost

- CUDA/dstack only; no ROCm and no local model download.
- Before private media, run an exact-image, exact-revision canary that loads
  both models and exercises 12- and 18-frame Qwen inference on generated media.
- Immutable names are
  `timelens2-three-condition-canary-20260801-a1` and
  `timelens2-three-condition-frozen30-20260801-a1`; `-a2` is reserved for at
  most one narrowly corrective technical attempt per gate.
- Use one A40-or-better 48 GB GPU, at least 96 GB host RAM, on-demand capacity,
  hard `max_price: 1.00`, bounded native no-capacity retry, and hard durations
  of 30 minutes for the canary and two hours for the experiment.
- Preview every configuration with `echo "n" | dstack apply -f ...`, retain
  the complete output, and require physically distinct compatible offers when
  practical. Submit once, detached; keep the workload foregrounded.
- Total remaining TimeLens base allocation is `$3.1414`; worst-case configured
  exposure for one canary and one experiment is at most `$2.50`. The central
  reserve is not authorized.
- Keep `HF_TOKEN` only in dstack secrets. Do not use diagnose output on a
  secret-bearing run.
- Durable private artifacts live under
  `hf://buckets/Npatta01/trec_video_2026/experiments/<run-name>/`.
  List, download to a fresh ignored directory, and checksum-verify them.
- `.timelens2-private/` stays locally excluded. Raw competition rows, media,
  answers, labels, grading packets, and logs stay out of Git. Only one uniquely
  named sanitized evidence record may be committed.

## Verification boundary

Write and observe failing tests before adapter, schema, runner, or config
changes. Tests cover exact source binding, forbidden labels, all three frame
budgets, shared-frame reuse, within-condition collision repair, immutable
freezes and denominators, model/runtime pins, CUDA/resource/cost limits, and
private path exclusion. Run model-specific tests, relevant VQA runner tests,
privacy scans, and `git diff --check` before each implementation commit and
before final handoff.
