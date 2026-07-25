# Qwen3-Omni Stratified Quality-10 Design

Date: 2026-07-25

Issue: #17

## Objective

Determine whether the working
`Qwen/Qwen3-Omni-30B-A3B-Thinking` audio/vision runner materially benefits
from native audio on a ten-video diagnostic panel without regressing
visual-only questions.

This is a diagnostic experiment, not a leaderboard estimate. It preserves the
successful issue #5 runtime and changes only the panel and the presence or
absence of audio.

## Considered approaches

### A. Paired ten-video audio ablation — selected

Run every video twice: once with 12 frozen frames plus native audio and once
with the identical frames but no audio element. Keep model, revision, runtime,
question, frame evidence, seed, and generation policy identical.

This costs 20 inference calls, but it directly isolates audio benefit and
provides matched visual controls. Based on issue #5 timing, it should fit
comfortably inside a one-hour task and the remaining Qwen budget.

### B. Ten audio-on calls only

This is cheaper, but any success could come from the frames rather than audio.
It also cannot test issue #5's visual-control regression requirement. Reject.

### C. Paired thirty-video benchmark

This would give broader coverage but substantially increases adjudication,
runtime variance, and failure-recovery scope before the ten-row design proves
useful. Defer until the paired ten-row diagnostic passes its gate.

## Frozen panel

The selected rows already have downloaded video, extracted audio, uniform
12-frame evidence, existing baselines, and prior inspection evidence.

| Video ID | Stratum | Purpose |
|---|---|---|
| `Mi-NpnCBegw` | lexical speech | English spoken-word recovery |
| `NN7G28sBOlY` | lexical speech | multilingual dialogue with an explicit repaired question |
| `NUmb2P3Bzu8` | prosody | vocal intensity plus visible gesture |
| `NkXPPJoZ-h8` | non-speech sound | issue #5 cat-vocalization anchor |
| `IWSPAfCKtV8` | endpoint sound | church bell near the end |
| `CH8Dp7EgW4I` | temporal/action | action after a shove |
| `EtNOkUJd5w8` | temporal/action | action immediately before another event |
| `Gp4LVTn3YbY` | cross-modal reaction | visible reaction plus spoken “too sour” evidence |
| `EMI_anpHII4` | spatial visual control | completed-object detail |
| `LusRurRrmf0` | counting visual control | cross-frame entity count |

The horse row uses: “What does the bystander answer when asked how old the
horse is?” The original annotation text is declarative and caused Qwen to
answer `True`; the repaired row is explicitly marked as a prompt-repair
diagnostic and must not be presented as a frozen-prompt result.

Exact music-title identification is excluded because prior inspection showed
that it needs audio fingerprinting or music retrieval. Rubber-bubble and
instrument rows are deferred so the ten-row panel retains speech, temporal,
cross-modal, spatial, and counting coverage.

## Conditions and ordering

Each video has two condition records:

1. `frames-only`: 12 frozen frames, no audio element.
2. `frames-native-audio`: the same 12 frames plus the frozen WAV.

Conditions are adjacent per video and use the same fixed seed. The runner
records the condition before generation and attests:

- exactly 12 processed visual elements in both conditions;
- zero processed audio elements and no audio tensor keys for `frames-only`;
- one processed audio element and the expected audio tensor keys for
  `frames-native-audio`;
- one generation call and the same first-forward boundary;
- actual EOS termination and one strict terminal JSON answer.

The first row is a canary. If either condition fails model load, modality
attestation, EOS termination, or strict parsing, the run stops before spending
on rows 2–10 and still syncs its failure artifacts.

## Input and label isolation

The tracked experiment code contains IDs, roles, questions, expected
condition structure, and immutable runtime identifiers. A local builder joins
the ignored shared dataset to construct a label-free contract containing
relative evidence paths and file hashes.

Reference answers, distractors, prior predictions, transcripts, and scores
must not enter the contract, dstack upload, prompt, or inference process.
Predictions are written incrementally, frozen, and hashed before post-run
scoring joins ignored training annotations.

No dataset contents, media, raw logs, raw predictions, model weights,
credentials, or reference text may be committed.

## Runtime and artifact contract

- Base branch: issue #5 / draft PR #14.
- Model revision:
  `2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b`.
- Runtime image:
  `docker.io/qwenllm/qwen3-omni@sha256:71896cf6533abc610ac298f26289ff3a4ea4bd1fc835f370ece9e5c31e3181d8`.
- EOS: `<|im_end|>` / `151645`.
- Padding: `<|endoftext|>` / `151643`.
- Visual evidence: 12 uniform frozen frames; no native temporal-video input.
- GPU envelope: one supported NVIDIA GPU with more than 90 GB VRAM,
  at least 16 CPUs, 192 GB host RAM, and 200 GB disk.
- Providers: on-demand RunPod or Vast.ai; no ports.
- Price cap: `$4.50/hour`.
- Duration cap: one hour.
- Worst new cost: `$4.50`.
- Prior Qwen spend: `$2.6269`.
- Worst cumulative spend: `$7.1269`, below the authorized `$10` hard cap.
- Remote prefix:
  `hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/qwen3-omni-quality-10-20260725-a1`.

The run prefix is immutable and must be empty before submission. The task
syncs a manifest, preflight, incremental predictions, summary, terminal
status, and checksums to the existing private bucket. The collector downloads
only that prefix into the ignored canonical benchmark directory and verifies
both remote and local hashes.

## Scoring and decision gate

After prediction freeze:

1. Compute normalized exact match and token F1 with the frozen repository
   scorer.
2. Apply answer-only `C`/`P`/`W` judgments using the existing issue #12
   rubric.
3. Compare each audio-on answer to its paired audio-off answer.
4. Compare both conditions to matched existing Qwen3.5/Whisper or visual
   baselines where available.

Promote Qwen3-Omni to a broader quality benchmark only if:

- all 20 calls pass their declared condition contract, EOS, and strict JSON;
- audio-on changes at least one audio-focused row from `W` to `P/C` or from
  `P` to `C`;
- neither visual-control row becomes a worse `C/P/W` category with audio on;
- there is no label leakage or provenance failure.

Otherwise preserve the working runner and park broad expansion with a
failure-specific recommendation.

## Error handling and verification

- Fail closed on contract, file-hash, model/image, condition, tensor,
  termination, parsing, output-path, or remote-prefix mismatch.
- Preserve structured failure records and successfully sync partial artifacts.
- Do not retry configuration, authentication, model, or workload failures.
- Use dstack's bounded native retry only for `no-capacity`.
- Preview without submission and require at least two distinct eligible offers
  when practical.
- Submit one immutable run exactly once.
- Verify repository tests, focused tests, Ruff, shell syntax, contract
  construction, secret scan, remote listing, artifact checksums, scoring
  reproduction, and independent review before publishing the draft PR.
