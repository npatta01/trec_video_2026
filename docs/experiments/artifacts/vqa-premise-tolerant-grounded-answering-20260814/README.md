# Premise-tolerant grounded answering follow-up

This is the sanitized evidence index for the 2026-08-14 issue #35 follow-up.
It tested whether separating harmless locator variance from answer-blocking
evidence gaps repairs false abstentions over the exact frozen question-blind
map lines. It used no native video, new evidence, multiple answers, counting
specialist, or shared router.

Evidence validity is **Diagnostic** and the decision disposition is **Not
promoted under the tested condition**. Eleven one-attempt text-only calls completed
and all 11 rows were manually inspected. The treatment rescued 1/5 declared
false abstentions (gate: at least 4), correctly completed 1/2 detectable
insufficiency routes (gate: 2), and preserved 1/3 controls (gate: 3). Three raw
responses failed the structured contract. No accepted final answer was
unsupported by its citations.

Post-freeze sensitivity found that one declared false-abstention row resolved
the locator but reasonably abstained because the causal “why” was not explicit.
The frozen gate remains 1/5; descriptively, the repair rescued 1/4 confirmed
false-abstention cases.

The failure is narrower than generic “reasoning.” The policy continued to
treat paraphrased or inaccurate question locators as answer-blocking even when
the map contained one uniquely relevant event. More deliberation under that
policy did not fix evidence use. The known map-omission case also confirms that
a map-only answerer cannot detect a missing event when the map supplies a
plausible incomplete answer.

`summary.json` contains the sanitized gate, metrics, completion record, and
mechanism findings. Complete private requests, raw responses, metadata,
terminal records, freezes, manual audit, and score remain at:

`/home/npatta01/data/competitions/trec_video_2026/data/vqa/benchmarks/grounded-map-answering-issue35-premise-v3/`

Independent review found that the original prediction/evidence freezes did not
bind the separate role/threshold gate. The observed gate exactly matched the
committed selection. An additive `gate-integrity.json` now binds that unchanged
gate and selection to both original freezes without rewriting them; future
runs bind the gate in the prediction freeze itself.

STS and the semantic composite remain `null`. The discontinued UMBC service
was the frozen historical implementation; substituting a different model after
prediction freeze would not be a comparable metric.
