# Question-blind frontier scene-map and boundary-verifier diagnostic


Date: 2026-08-12

Evidence validity: **Diagnostic; valid negative result for the tested
same-model verifier**

Decision disposition: **Not promoted under the tested condition**

## Completed evidence record

- **Primary theme:** routing, fusion, and answer selection.
- **Related theme:** visual and temporal evidence.
- **Question:** can Gemini 3.1 Pro first create a reusable,
  question-blind native-AV event/state map and then, in one separate
  question-blind pass, verify and repair that map’s temporal coverage
  and boundaries?
- **Hypothesis:** a full-AV verifier that receives an untrusted frozen
  draft should restore a known omitted state and correct materially
  displaced event boundaries without needing the VQA question.
- **Falsifier:** the verifier leaves either known temporal error intact,
  introduces a material false correction, or fails its strict contract.
- **Panel:** one spent 14.528-second labeled-training video. Neither
  annotation pass received the question, label, reference answer, prior
  answer, or metric.
- **Tracked artifact index:**
  `docs/experiments/artifacts/vqa-frontier-boundary-verifier-20260812/`.

## Why the screen stopped

The one-pass v5 draft was structurally valid, but a dense 4-fps
native-frame audit found a material temporal error. It ended a sustained
state several seconds early, displaced the following transition into
that interval, omitted a later interaction, and assigned confidence
`1.0` despite coarse boundaries. The pilot audit froze
`revise_contract`, so the planned seven-video screen remained
unauthorized. The event semantics, media, question, and answer remain
only in the private evidence bundle.

An independent panel—Sol, Claude Fable, and Gemini 3.1 Pro
High—unanimously recommended the same smallest decision test: one
question-blind verification pass over the frozen bad draft and complete
native AV. Before the paid call, the runner was changed so request seeds
use content-addressed media identities, and a Sol review required exact
prediction/audit cross-binding, provider/model and cost evidence,
one-attempt semantics, and complete artifact-chain checks. After the
terminal evidence was hardened, the current focused Phase 0 and verifier
suite passed 135 tests.

## Observed verifier result

Gemini returned a syntactically parseable structured response for one
call, with 4,246 input tokens, 4,031 output tokens, and observed cost
`$0.056864`. There was no retry.

The response failed the strict local contract because every event review
used an empty uncertainty string. Its nested corrected neutral map was
nevertheless locally valid, so it could be inspected without repairing
or relabeling the terminal provider result.

That inspection failed the scientific gate:

| Predeclared check                                    | Result |
|------------------------------------------------------|--------|
| Restore the extended sustained state                 | Failed |
| Move the following transition until after that state | Failed |
| Preserve/capture the late interaction                | Failed |
| Reduce confidence when timing is uncertain           | Failed |
| Avoid material false corrections                     | Failed |

The verifier reproduced the same early endpoint and overlapping
transition contradicted by dense native frames. It retained confidence
`1.0`, reported no uncertainty, and did not add the omitted late
interaction. This is a negative mechanism result, not merely a
JSON-format failure.

## Cost and completion boundary

The final v5 pilot plus verifier cost `$0.102772`. Including all
preserved prompt/schema development attempts and the stopped partial v4
screen, observed Phase 0 provider cost was approximately `$0.972344`,
well below the `$25` cap.

The experiment did not run or score the seven-video answer screen.
Therefore it reports no METEOR, BERTScore, STS, or NDCG comparison and
makes no answer- quality claim. The single-video result tests only
whether this same-model verification mechanism can catch a known
failure.

## What this rules out now

- Do not scale the tested same-model draft-plus-verifier architecture to
  all labeled videos.
- Do not treat a second Gemini pass as an independent temporal-quality
  gate.
- Do not use self-reported confidence or round timestamps as acceptance
  evidence.
- Do not run the v4 verifier, fresh category videos, or the seven-video
  answer screen under this unchanged mechanism.

## What it does not rule out

The result does not show that reusable question-blind scene maps are a
bad architecture. It does not rule out a different frontier annotator, a
genuinely independent verifier, explicit frame/timestamp boundary
evidence, overlapping window coverage for longer clips, or later
question-driven evidence retrieval and native-AV answering. It also does
not estimate failure prevalence from one video.

## Next decision test

Keep the reusable annotation architecture, but change the
quality-control mechanism. The smallest next test is the same known-bad
draft plus one independent verifier or explicit temporal-grounding
model. Advance only if it:

1.  cites direct onset/offset evidence;
2.  restores the known missing interval and late interaction;
3.  preserves a known-good map without destructive edits; and
4.  passes the same dense native-media audit.

Only after that mechanism passes should it run on two or three category-
diverse videos and then feed a separate top-1 answerer. Candidate
ranking and top-10 serialization remain downstream concerns.

## Artifact retention and reproducibility

The complete private evidence remains under:

`/home/npatta01/data/competitions/trec_video_2026/data/vqa/benchmarks/vqa-frontier-phase0-20260812/`

The additive `pilot-audit-v5-v2/` bundle revalidates the frozen draft
while binding the preserved raw response and both per-row reviewer
manifests. Its audit and freeze SHA-256 values are
`5757c2ad3a80dfc00f1e9fff14dea65403dc2718042edd66ed3d7d41e2c5f5c6` and
`f0740a98c1110c098af58ce18842586c1903ff1576f18fc7d30c28e62a344c26`.

The final `verifier-v5/` directory preserves the manifest, frozen draft,
request, two-record attempt ledger, raw response, failed terminal
result, additive terminal freeze, diagnostic audit, and diagnostic
freeze. The terminal freeze SHA-256 is
`0df48028449fbcdb9ceffbd72e1ca23b2b6d06c6791d15cd15b726dadea8836b`; the
diagnostic audit SHA-256 is
`72d3c6be270ea24c811a5b8db076f8e48bed157d9514b008cc91fbd805a7c49e`.
These hardening steps were offline revalidation only. No provider rerun
or GPU provisioning is needed to inspect the result.
