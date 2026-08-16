# Multi-interpretation answer-election diagnostic

This is the sanitized evidence index for the 2026-08-14 fresh-panel follow-up
to issue #35. It tested whether three evidence-citing interpretations plus a
fixed local election rule improve emitted top-1 answers over one unique-event
interpretation. It used only frozen question-blind scene maps: no native video,
new evidence, retry, counting specialist, or shared router.

Evidence validity is **Diagnostic** and the decision disposition is **Not
promoted under the tested condition**. All 24 one-attempt calls passed their
contracts, and all 12 candidate sets (36 candidates) plus every emitted change
were manually inspected after both arms were frozen.

The 12 rows are genuinely disjoint from issue #35 v2/v3 and the seven-row
canary. The original manifest bound v2 and the canary but did not name v3;
additive `selection-exclusion-audit.json` binds the exact v3 protocol and proves
all 11 v3 rows were already covered by the frozen 24-row v2 exclusion. A
separate panel-isolation limitation remains: label-aware row selection preceded
the committed treatment runner/election implementation. The negative result is
therefore diagnostic, not a clean fresh-panel performance estimate.

The multi arm produced correct answers on 4/4 premise-variance rows, but it
improved only 1 row over the single arm (gate: at least 2). Both arms routed
only 3/4 evidence-insufficient safeguards to verification (gate: 4/4), and the
multi arm preserved only 3/4 controls (gate: 4/4). No emitted answer lacked
support from its cited frozen evidence.

A later zero-call paper walkthrough tested the proposed entity–fact–time/span
support rule on the two decisive election failures before authorizing another
24 calls. It resolved 0/2. One row's reference-level action is absent from the
frozen map, so a label-blind text-only verifier cannot detect the omission. The
other question is satisfied by two displayed-text spans and needs an untested
salience/text-type preference to choose the reference span. The proposed paid
follow-up was therefore stopped before inference. See
`task-support-paper-walkthrough.json` for the sanitized result.

Every row contained at least one manually correct candidate (oracle coverage
12/12), compared with 10/12 correct elected multi outputs. This oracle result
is diagnostic only and was not used in the gate. The gap localizes the failure
to election: diversity exposed safe alternatives, but the frozen rule selected
one cited wrong-for-reference action on an evidence-insufficient row and turned
two conflicting text candidates into a control abstention.

`summary.json` contains sanitized completion, gate, answer-only role-aware metrics,
candidate diagnostics, mechanism findings, and integrity hashes. It contains
no questions, references, map text, predictions, raw responses, credentials,
or media. Complete private evidence remains at:

`/home/npatta01/data/competitions/trec_video_2026/data/vqa/benchmarks/multi-interpretation-answer-election-issue35-v1/`

The original private score and additive metric-provenance v2 are preserved.
The additive v3 score introduced the explicit v3-exclusion binding and stopped
treating `NEEDS VERIFICATION` as an ordinary semantic answer; every aggregate
reports its answer-only denominator. V4 and final v5 rebind that unchanged
calculation to the finalized scorer bytes after formatting and a tested helper
refactor. STS and the semantic composite remain `null`: the historical
UMBC-compatible service is unavailable, and a different local model would not
be a comparable replacement.
