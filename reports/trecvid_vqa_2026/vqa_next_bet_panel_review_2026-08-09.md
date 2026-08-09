# VQA next-bet independent panel review — 2026-08-09

## Decision

Run one full-panel, dual-axis diagnostic on the 475 annotated VQA rows with
local media. The experiment must test both reference-assisted evidence timing
and reference-blind answer election. It must not use the 1,883-row test set.

The panel consisted of Codex Sol xhigh, Claude Fable/Max, and Gemini 3.1 Pro
High. All three independently rejected the proposed 120-row design as too weak
for a `+0.015` composite-C gate. Codex and Gemini recommended all 475 rows for
the selector and answer arms. Claude recommended all 475 rows for election and
160 rows only if selector calls are materially expensive. Because the selected
local model can process the complete panel in parallel and all videos are at
most 30 seconds long, the project adopts the full-475 design.

## Consensus

- Keep `Qwen/Qwen3.5-4B` with the current answer policy frozen as the matched
  answerer. The completed factorial did not support a blanket 8B switch or a
  blanket question-aware prompt.
- Add a real no-Whisper condition. The earlier six-cell proposal could not
  isolate transcript presence from transcript timing.
- Separate three quantities:
  1. reference-assisted evidence-selection headroom;
  2. reference-blind election over the four already-deployable global-evidence
     candidates; and
  3. reference-blind election after oracle-assisted evidence construction.
- Treat the per-row metric maximum as a ceiling only. It is inflated by
  max-of-candidates selection and is not deployable.
- Preserve selected frame timestamps, decoded PTS values, hashes, contact
  sheets, transcript provenance, and with/without-Whisper outputs.
- Use METEOR, BERTScore F1, STS, and their mean composite `C`; keep exact match
  diagnostic only. Use paired row bootstrap intervals.
- If election succeeds, multipass generation and ranking are the backbone even
  if a locator also succeeds. The competition output needs ranked alternatives,
  and the existing four candidates already show substantial complementarity.

## Disagreement resolved

Claude proposed a larger materiality threshold (`+0.030 C`) on 160 selector
rows. Codex and Gemini proposed `+0.015 C` on all 475. The project adopts all
475 and retains the `+0.015 C` threshold with a positive paired 95% lower bound.
This keeps the user-requested larger, confidence-producing experiment and
avoids a second run caused only by inadequate power.

## Selected local models

- Silver-oracle selector and evidence-aware judge:
  `Qwen/Qwen3-Omni-30B-A3B-Thinking@2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b`.
  It runs on local dstack GPU infrastructure, so competition media and labels
  are not sent to a paid external model provider. The repository already has a
  verified 30/30 runtime path for this exact revision. The new canary must still
  prove native-video structured-output compatibility.
- Matched answerer:
  `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
  with the frozen current prompt, decoding, and per-frame pixel limits.

The selector is a **silver oracle**, not a human oracle: it sees the reference
answer and may still localize incorrectly. Any evidence arm derived from it is
diagnostic and nondeployable.

## First bounded action

Freeze the protocol and run a ten-row operational canary spanning the declared
question strata. The canary may inspect schema, range validity, extraction,
reference isolation, and artifact checksums, but not treatment metrics. Continue
to all 475 only if all ten rows produce valid selector records after at most one
predeclared schema retry and every answerer-visible request passes the leakage
audit.
