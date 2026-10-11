# Advisor panel review — 2026-10-05

Scope: technical evidence and reader/visual clarity, read-only independent reviews of the draft and rendered pages. No new inference or publication.

## Technical evidence advisor
- Confirmed aggregate and paired MC counts against records.
- Distinguish MC1 inheritance from MC3 independent disagreement: applied.
- Do not claim the selected frames reconstruct model inputs or establish sampling failure: applied.
- Describe Qwen calls as the same video payload with/without transcript; native-audio consumption remains unverified: applied.
- Mention AG1's unavailable-video question-only fallback: applied.
- Replace machine-specific graphics paths: applied, relative to docs/paper. Assets remain local/ignored; this is a local draft, not a publication bundle.

## Clarity and visual advisor
- Preserve shared real AG/MC examples: applied, one two-panel figure with actual answers and all MC choices.
- Make MC2's checker and reuse of MC3 predictions explicit: applied while retaining the user's preferred separate-row layout.
- Remove insect vignette and repeated winner statements: applied.
- Shorten per-case interpretation and duplicate analysis: applied; retained a brief shared-versus-inherited distinction to address the user's question.
- Dark frame print legibility remains a limitation; preserved unaltered evidence rather than implying unseen foreground detail. The selected frame is paired with the exact model claim, not a causal diagnosis.

## Verification
Local XeLaTeX build passes twice; six-page saved PDF checked. Desktop compiler does not support the local image assets, so the saved PDF is the review artifact. Exact examples were checked against frozen AG/MC submission JSON. Sampling failure is not established by these cases.

Final technical corrections applied: AG3/AG4 METEOR tie; AG4 generates variants from source rank-one answers rather than the submitted lower-rank list; NDCG wording/content confound; exploratory-development boundary; lexical how-many subgroup label; consistent video payload terminology; historical evidence-boundary text marked superseded. Model references and publication-ready asset permissions remain author-stage work; nothing was published.

## Focused Claude Code adversarial pass — 2026-10-05

Scope: current TeX and QMD/JSON evidence, read-only via `cc:adversarial-review`, working-tree review. User requested practical improvements without new experiments or broad redesign. Reviewer verdict: **needs-attention**; stated that the reported numbers match the supporting evidence and all proposed fixes require wording/presentation changes only. Dispositions below supersede earlier presentation descriptions in this file.

1. **High: weather example could imply text generally beats video.** Accepted the clarity issue, not a general method conclusion. MC1–MC4 now occupy separate rows with method labels; the paragraph explicitly states that MC4 is correct, case selection is illustrative, this outcome pattern occurs in 20/2,428 rows, and MC3 improves the overall total by140. No assertion about why MC2 selected or retained a route without its gate trace.
2. **Medium: omitted AG1 rainy score weakens the shared example.** Added AG1 rainy, STS .552021 / METEOR0 / BERTScore .999436 (table rounded). This directly connects the scoring and grouping panels. Removed the redundant Q106 BERTScore comparison from the paper; historical evidence remains.
3. **Medium: AG abstract presents unchanged first-answer scores as a learned result.** Clarified rank-one preservation is by design. Verified Q106 AG1 answer2 appears at AG3 rank3 beneath Gemini's4; body now makes the fixed-priority limitation concrete. Replaced lower-rank-quality wording with lower ranked-list scores, preserving the wording/ordering distinction.
4. **Medium: development result could be mistaken for an AG2 ablation.** Retained useful 2025 evidence, consistently called it selected development data, and stated explicitly in the abstract that it is not an ablation of submitted AG2. Did not introduce another lexical-group result.
5. **Medium: small BERTScore lead and unclear case referent.** Added the .000896 margin and stated significance was not tested; retained the factual numerical ranking. Did not adopt the reviewer's stronger claim that the lead is meaningless, which was not established. Redundant case-score sentence removed.
6. **Low: inspected-count certainty and incomplete subgroup comparison.** Specified1-fps inspection, bounded the observation to inspected frames, and added MC2/MC4 how-many totals205/204 versus MC3's200. No causal or exhaustive visual-audit claim.
7. **Low: typos and missing implementation detail.** Closed the abstract quotation, removed duplicated video-payload wording, completed the architecture caption sentence, named Gemini3.7 Flash for AG4 from the generator script, and clarified answer-option availability.

No new inference, experiment, or publication. Added local rebuild instructions because the private source frame remains ignored. Verification: frozen Q106 ranks and weather scores checked; official subgroup/pattern totals checked; two-pass XeLaTeX build and rendered-page review. Final draft remains six pages. Native editor image limitation persists. These are fixes assessed and integrated by Codex, not a claim of a second independent approval.

## Shared-error analysis review — 2026-10-05

Focused read-only Claude Code adversarial review of the new paper section and audit QMD/JSON returned **approve**. It confirmed the271/194/77 counts and the two in-context One/One bed and One/One person is wearing a hat equivalences against the audit record, finding the claims appropriately bounded. Two low-priority suggestions were addressed:

- Explicitly identify Q1610 as a possible wording overlap, rather than describing every other inspection as visual/timing ambiguity.
- Show additional equivalent pairs within the two original option sets. Used precise examples (Two/Two beds; None/No one is wearing a hat), not the reviewer's overly broad suggestion that every count has two forms, since the actual lists do not support that universal statement.

Also clarified that the hat frame shows two people at an easel, one painting. No prevalence, false-reference-fact, all-team-error, or corrected-leaderboard claim was added. Review did not independently replay every video. Local rendering checks remain the primary agent's responsibility.

## Astra first pass and extended panel — 2026-10-10

Requested/effective profile: **extended**. First, a separate GPT-6 Astra xhigh advisor reviewed the source, rendered pages and evidence. After its accepted changes, the extended panel received one frozen source/evidence packet with the same focus: **Improve clarity, visual variety, and evidence-grounded argument without adding density or unsupported analysis.** Panel reviews were source-only; the parent performed rendered PDF inspection.

| Lane | Model | Auth | Outcome | Main contribution |
|---|---|---|---|---|
| Sol | gpt-6-sol, xhigh | ChatGPT subscription | Answered | Calibrate count heading; avoid attributing AG4 decline to rewording alone; trim appendix repetition. |
| Astra | gpt-6-astra, xhigh | ChatGPT subscription | Answered | Clarify saved MC3 reuse, move AG4 constraints into methods, close the main argument. |
| Claude | opus, max | Claude.ai subscription | Answered | Replace misleading text-only shorthand, emphasize larger METEOR difference, say groups rather than merges, remove source color inconsistency. |
| Gemini | gemini-3.8-flash-high | Antigravity OAuth; billing tier unverified | Answered | Check chart spacing and color semantics; preserve reference-text uncertainty. |
| Qwen | opencode-go/qwen3.8-max | Runtime preflight unavailable | Unavailable | No inference or substitute model. |
| Kimi | opencode-go/kimi-k3 | Runtime preflight unavailable | Unavailable | No inference or substitute model. |

Packet classification: private, task-scoped draft and compact evidence only. Training use was not permitted; every configured lane declared provider_data_use=none. No public-use lanes or training endpoints were enabled. Reviewers were read-only. The runner disposed of temporary OpenCode runtimes; no Qwen/Kimi inference session ran and no session cleanup receipt was produced. The private packet/result files were removed after synthesis; this sanitized record preserves the dispositions.

Accepted: compact source-priority layout and MC routes, paired corrections/losses chart (150/37 then69/42), ten-rank AG4 ladder, explicit audit screening scope, softer counting headline, saved-answer MC1 wording, exact-source/duplicate constraints in AG4 methods, larger METEOR contrast, concise overall takeaway, and shorter Appendix B.

Not adopted: more analysis/MC4 bars, speculative score causes, a same-settings causal comparison of AG3/MC3, or additional repeated numerical bridges. The chart and existing abstract already provide incremental and total differences. Gemini's suggested cross-reference collision is not real: fig:shared and sec:shared-errors are distinct labels. Its suggested label crowding was checked against the render; the positive-count/net spacing had already been repaired. Keep the explicit AG-reference boundary without speculating about missing references. The evidence supports local option equivalence, not prevalence or wrong underlying counts.

Verification: directly rejoined all four archived MC CSVs on the same2,428 IDs; totals1912/2025/2052/2047 and both new transitions match the saved JSON. Rebuilt the seven-page PDF twice with XeLaTeX, checked warnings and whitespace, and inspected all pages. No new model inference, source media alteration, publication, or push. Final model citations and author sign-off remain pre-existing draft tasks.
