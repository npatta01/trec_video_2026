# Question-Driven Atomic Counting Evidence Design

**Date:** 2026-08-14  
**Status:** Completed diagnostic; tested treatment not promoted

**Outcome:** The six-row canary rescued 1/5 failures and preserved 1/1 control
at `$0.253924`. It failed the 3/5 rescue gate; projected 39-row cost was
`$4.3874`, above the `$3` ceiling, so no scale-up was submitted. A three-lane
advisor review recommends retaining atomic records but replacing model-selected
units and universal cardinality with a frozen CountSpec and typed deterministic
reducers. That is a separate, initially zero-cost next mechanism.

## Objective

Test the smallest evidence-recovery treatment for Gemini native-AV counting
failures: one label-blind, question-driven pass over the complete native video
that emits atomic, timestamped count records. A deterministic local reducer
derives the answer from those records. The treatment targets missing or
collapsed evidence and count-unit binding; it is not a generic second answer or
answer-diversity method.

## Corrected source census and panel

The frozen 475-row Gemini outputs and labels produce 66 lexical count
candidates. Re-auditing the parser and the source evidence gives 38 correct
count answers, 26 confirmed wrong answers, one genuine count-unit ambiguity,
and one question that mentions a changing number but does not request a count.
The earlier 37/27/2 split had two offsetting parser errors: a correct answer was
made wrong by choosing a later numeral from its reference, and the word
`forty` was outside the parser vocabulary and hid a wrong answer.

The challenge freeze therefore contains the 26 confirmed failures plus the
one ambiguous row as an explicitly unscored diagnostic. Twelve correct
controls are selected deterministically, including the corrected false
failure. Rescue is measured on 26 rows; control preservation is measured on
12. The diagnostic row cannot affect either gate.

## Treatment contract

Each request contains only the question, the already-frozen question-blind
event map as navigation context, complete native video/audio, media hashes,
and duration. It contains no reference, baseline prediction, correctness
status, answer hypothesis, panel role, or expected count.

The model must classify the count unit and emit one record per counted unit:

- `unique_entities_objects`
- `simultaneously_visible_instances`
- `appearances_across_scenes`
- `edited_shots`
- `repeated_actions_transitions`
- `background_people`
- `parts_components`

Every record has a stable record ID, representative timestamp or bounded
span, scene description, and concise inclusion rationale. Exclusions are
separate records. The provider does not emit a final count field. A strict
local validator rejects duplicate IDs, invalid or out-of-range timestamps,
unknown fields, non-atomic records, and malformed JSON. The treatment answer
is the number of validated included records.

## Experimental sequence

1. Freeze the corrected census, 27-row challenge panel, 12 controls, source
   hashes, and a recursively label-blind request manifest.
2. Run a six-row representative canary: five confirmed failures spanning
   distinct count mechanisms and one corrected control.
3. Continue only if all responses validate, the control is preserved, at
   least three of five failures are rescued, and the projected total remains
   below the USD 3 ceiling.
4. If the canary passes, run the remaining frozen rows with the identical
   model, prompt, schema, media settings, and direct API path.
5. Freeze predictions before joining labels. Then score and inspect failures.

The hard cost stop uses observed token usage and current official Gemini list
prices. No request may start when its conservative projected cost could take
the cumulative run above USD 3.

## Metrics and decision rule

Report METEOR, BERTScore-F1, STS, and their equal-weight semantic composite for
the frozen baseline and treatment. Also report exact-count rescue rate on the
26 confirmed failures, control preservation on the 12 controls, the ambiguous
diagnostic separately, count-unit strata, schema validity, and map-sufficiency
failure categories.

Promotion requires at least 16 of 26 confirmed failures rescued (60%) and at
least 11 of 12 controls preserved (90%). The semantic metrics are diagnostic,
not substitutes for the exact-count gate. A valid negative result is frozen
and reported rather than repaired with a new prompt.

## Provenance and privacy

Complete questions, references, requests, raw responses, provider metadata,
media paths, hashes, selection records, and post-freeze scoring stay under the
canonical ignored benchmark root
`data/vqa/benchmarks/direct-gemini-counting-evidence-v1/`. Only aggregate
counts, metrics, decision, hashes, method, and non-sensitive failure taxonomy
are projected into tracked artifacts and the owning QMD report. The frozen
475-row panel is never rerun or modified.

## Non-goals

- No full-panel Gemini rerun.
- No generic self-consistency, candidate generation, or answer reranking.
- No question-driven temporal/speech verifier (issue #35).
- No appearance-aware ranked-output synthesis (issue #36).
- No detector/tracker production system, global prompt rewrite, or label-aware
  inference.
