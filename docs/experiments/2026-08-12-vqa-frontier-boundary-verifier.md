# Question-blind frontier scene-map and boundary-verifier diagnostic


Date: 2026-08-12

Evidence validity: **Diagnostic; negative only for the tested OpenRouter
same-model verifier, with delivery and schema-projection confounds**

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
interaction. This is a negative result for this exact
OpenRouter-delivered, draft-anchored verifier—not evidence that Gemini
native video is incapable. The provider schema also allowed empty
strings that the stricter local contract rejected.

## Direct Google API follow-up

Two identical question-blind Gemini 3.1 Pro Preview calls were then run
through Google’s direct Developer API on the same preserved WebM. Both
used explicit 4-fps video sampling, high media resolution, high
thinking, temperature `0`, plain text, no draft, and no output schema.
Their responses were byte-identical and each request used 15,801
prompt/media tokens, compared with 4,246 total input tokens for the
OpenRouter verifier.

The direct timeline recovered the complete event sequence, including the
late interaction and final embrace. It placed the major transition near
10 seconds, although one endpoint remained about 0.75–1.0 seconds early.
This overturns any Gemini-capability interpretation of the earlier
failure and points instead to delivery/sampling, draft anchoring, and
prompt/schema complexity.

A subsequent question-plus-native-AV answer call still returned the
immediate micro-event rather than the labeled broader outcome. A cheap
text-only repair over the correct frozen timeline did the same. The
remaining canary failure is therefore question interpretation/answer
granularity, not missing video evidence.

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

- Do not scale the tested OpenRouter same-model draft-plus-verifier
  condition.
- Do not treat a second Gemini pass as an independent temporal-quality
  gate.
- Do not use self-reported confidence or round timestamps as acceptance
  evidence.
- Do not run the v4 verifier, fresh category videos, or the seven-video
  answer screen under this unchanged mechanism.

## What it does not rule out

The result does not show that reusable question-blind scene maps or
Gemini native video are bad choices. Direct Gemini annotation passed the
coverage test on this clip, but one video cannot estimate prevalence and
its boundaries were still coarse.

## Next decision test

Keep the reusable annotation architecture and use Google’s direct
native-AV route. The next test is a small category-balanced set, with
question-blind timeline quality and top-1 answer quality measured
separately. Reuse each frozen timeline to test question decomposition
and a separate text answerer without paying to reprocess video.
Candidate ranking and top-10 serialization remain downstream concerns.

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

The `direct-gemini-timeline-v1/` directory preserves the direct-API
timeline and answer calls, metadata, cleanup records, human audit, and
checksums. Its audit SHA-256 is
`fd0b88b7acd5cb61cdba867da5b2974033f22940c6a706410b3049fc7fe7750c`.
