# Multi-Interpretation Answer Election Design

## Objective

Test whether three evidence-citing interpretations improve answer use over one
unique-event interpretation when the question wording and frozen scene map do
not align exactly. This is a fresh answer-generation/election diagnostic, not
evidence recovery, counting specialist #34, or shared router #36.

## Decision being tested

Multiple candidates help only if a label-blind election rule can convert their
additional coverage into a better emitted top-1 answer without increasing
unsupported answers or control regressions. Oracle-best candidate performance
is diagnostic and can never satisfy the promotion gate.

## Fresh panel

Freeze 12 previously uninspected rows from the completed 475-row Gemini panel,
excluding every row used by grounded-map issue #35 v2/v3 and earlier seven-row
canaries. Use a deterministic, code-recorded selection over existing frozen
questions, maps, direct predictions, references, and automatic metrics:

- four likely premise-variance failures: low direct-answer semantic score,
  reference-level support present in the map, and a temporal or entity locator
  whose wording differs between question and map;
- four insufficiency/ambiguity safeguards: low direct-answer score and the
  reference-level requested fact absent or non-unique in the map;
- four stability controls: supported direct answer and reference-level fact
  explicit in the map.

Panel adjudication is label-aware but provider-isolated. Freeze row IDs, roles,
exact question/map evidence, source hashes, and selection notes before calls.
No selected row may influence prompt or selector development after selection.

## Arms

Hold model revision, evidence, temperature `0.0`, seed `42`, high thinking,
one provider response, and no-retry behavior fixed.

### C — Single unique-event resolution

Generate one cited answer or `NEEDS VERIFICATION`. The prompt permits harmless
locator variance only when the evidence contains exactly one event compatible
with the requested relation and answer type. It forbids inventing causal
motives, purposes, unique counts, speech, or omitted events.

### A — Three candidates plus deterministic election

Generate exactly three structurally independent candidates in one response:

1. literal-premise interpretation;
2. unique-event/paraphrase interpretation;
3. conservative verification interpretation.

Each candidate contains interpretation, premise status, cited claim, conflicts,
verification flag, and concise answer. Generation never selects a winner.

The local deterministic election receives candidates and frozen evidence only:

1. reject schema-invalid, unknown-citation, claim/answer-mismatch, or forbidden
   `answer_blocking` answers;
2. normalize remaining non-verification answers;
3. if two or more supported candidates agree, emit their shortest answer;
4. if exactly one supported answer remains and every other valid candidate is
   verification, emit that answer;
5. if supported candidates conflict, or none remains, emit
   `NEEDS VERIFICATION`.

The election does not use references, direct predictions, roles, metrics, or a
model call. Persist all candidates, rejection reasons, normalized groups, and
the emitted result.

## Integrity and artifacts

- Keep provider protocols label-free and roles/gates in separate pre-inference
  freezes bound to the tracked selection.
- Use one call per row per arm, no retry, with raw response and metadata written
  before parsing.
- Freeze every terminal and row artifact for both arms before loading labels or
  scores.
- Bind v2/v3 exclusion lists, 475-row source artifacts, both arm freezes, gate,
  and scorer inputs by SHA-256.
- Complete private evidence stays under the canonical shared dataset; only a
  sanitized summary, hashes, code, tests, and reports are tracked.

## Evaluation

Score the two emitted top-1 arms and the frozen direct baseline with METEOR,
BERTScore-F1, STS when the historical implementation is genuinely available,
and the semantic composite only when all three components exist. Report metrics
for each role and all rows, but do not treat the all-row mean as the mechanism
gate because correct verification sentinels are intentionally non-reference
answers.

Separately report candidate coverage: whether any of the three candidates would
be judged correct. This oracle-best coverage is a generation ceiling only.
Manually inspect every treatment/control difference and every candidate set.

## Predeclared gate

The multi-candidate arm advances over the single unique-event arm only if all
conditions hold on the frozen 12 rows:

- at least three of four premise-variance rows are correct emitted top-1 answers;
- at least two premise-variance rows are improvements over the single arm;
- all four insufficiency/ambiguity safeguards emit verification in both arms;
- all four stability controls are preserved by the multi-candidate arm;
- zero unsupported emitted answers;
- all 12 candidate sets and every changed emitted row are manually inspected;
- oracle coverage is never substituted for emitted top-1 performance.

Operational or contract failures remain in denominators. A positive result is
diagnostic evidence for this frozen election rule, not general answer diversity.

## Outcomes and next actions

- Gate passes: preserve the rule and propose a larger fresh matched validation;
  do not merge it into a shared router.
- Oracle coverage rises but emitted top-1 fails: generation diversity has
  headroom, but election is the bottleneck; preserve the negative top-1 result.
- Candidate coverage does not rise: stop multi-answer work and retain the
  single unique-event policy or move to explicit temporal/speech verification.
- Safeguards or controls fail: stop; diversity is unsafe under the tested rule.

## Verification

Use test-first development for selection, candidate validation, election,
freezes, scorer bindings, role-aware gate, and privacy-safe summary. Run focused
and neighboring tests, Ruff, formatting, `py_compile`, JSON/checksums, deliberate
tamper tests, Quarto regeneration, privacy scans, full tests with baseline
classification, and independent review before publishing any result.
