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

The multi arm produced correct answers on 4/4 premise-variance rows, but it
improved only 1 row over the single arm (gate: at least 2). Both arms routed
only 3/4 evidence-insufficient safeguards to verification (gate: 4/4), and the
multi arm preserved only 3/4 controls (gate: 4/4). No emitted answer lacked
support from its cited frozen evidence.

Every row contained at least one manually correct candidate (oracle coverage
12/12), compared with 10/12 correct elected multi outputs. This oracle result
is diagnostic only and was not used in the gate. The gap localizes the failure
to election: diversity exposed safe alternatives, but the frozen rule selected
one cited wrong-for-reference action on an evidence-insufficient row and turned
two conflicting text candidates into a control abstention.

`summary.json` contains sanitized completion, gate, role-aware metrics,
candidate diagnostics, mechanism findings, and integrity hashes. It contains
no questions, references, map text, predictions, raw responses, credentials,
or media. Complete private evidence remains at:

`/home/npatta01/data/competitions/trec_video_2026/data/vqa/benchmarks/multi-interpretation-answer-election-issue35-v1/`

The original private score is preserved. Additive `postfreeze-score-v2.json`
adds exact metric-runtime and BERT-model hashes without changing any numerical
result. STS and the semantic composite remain `null`: the historical
UMBC-compatible service is unavailable, and a different local model would not
be a comparable replacement.
