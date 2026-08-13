# Independent advisor review

Sol reviewed a sanitized packet containing the frozen protocol, aggregate
metrics, stratum metrics, and mechanism-audit counts. It received no raw media,
credentials, questions, labels, answers, or private logs.

## Decision

- **No-go:** another 475-video treatment or immediate implementation of a
  six-stage specialist architecture.
- **Go:** test one narrow seam—frozen map to claim/span evidence bindings to a
  concise answer, with `needs_verification` when evidence is absent.
- Treat the map as an optional retrieval index, not mandatory answer context.
- Do not add a generic generative consistency pass. The tested same-model
  verifier already repeated a known error; use deterministic validation only.

## Ranked tests and gates

1. Draw a deterministic question-stratified 60-row sample independent of metric
   outcome. Blindly adjudicate randomized Gemini/Qwen answers, then inspect map
   sufficiency. Continue to grounding if at least 8/60 clean rows are wrong or
   partial despite sufficient map evidence; continue to retrieval work if at
   least 6/60 have genuine map omissions. If more than 6/60 are materially
   reference-ambiguous, repair evaluation first.
2. On the existing 12 audited map-sufficient failures plus six supported
   controls, compare plain question-plus-map answering with structured
   claim/span binding. Require at least 7/12 repairs, a four-case net advantage,
   at most one control regression, and no unsupported claims.
3. On eight audited map omissions plus four map-sufficient controls, allow at
   most one targeted AV verification. Require at least 5/8 repairs, no control
   regression, no increased unsupported-claim rate, and a predeclared usage cap.

Because Gemini 3.1 Pro Preview can drift, any later provider A/B must pin the
exact revision or rerun a matched direct control.
