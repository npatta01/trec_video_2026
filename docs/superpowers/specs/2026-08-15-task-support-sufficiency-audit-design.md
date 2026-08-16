# Task-support sufficiency audit and verifier design

## Objective

Test whether a narrowly constrained, evidence-citing answerer can repair
answer-use failures when a frozen question-blind Gemini map uniquely contains
the requested fact. The experiment remains about answer reasoning and evidence
use. It does not test new evidence acquisition, native-video retrieval, a
counting specialist, or a shared router.

## Design decision

Use all 475 completed full-panel rows as the eligible source pool. Previously
inspected rows are not excluded. Instead, every candidate carries explicit
exposure flags for the 30-row hard-case audit, grounded-answering v2/v3,
multi-interpretation election, seven-row native-AV canary, scope control, and
whole-video synthesis follow-up. Results must be reported separately for
previously inspected and not-previously-inspected rows. A mixed or reused panel
is a selected mechanism diagnostic, never a fresh-panel generalization claim.

The treatment contract must be frozen and committed before references or
label-aware sufficiency verdicts are used to form the panel.

## Frozen treatment contract

The future provider sees only the question and cited lines from the already
frozen question-blind map. It must return exactly one structured result with:

- `question_interpretation`: requested entity, relation/action/attribute,
  temporal or speech qualifier, and expected answer unit;
- `cited_evidence`: one or more evidence IDs and the claim each supports;
- `conflicts_or_missing_qualifiers`: explicit ambiguity, contradiction, or
  absent requested fact;
- `needs_verification`: boolean;
- `answer`: one concise top-1 answer, or the exact sentinel
  `NEEDS VERIFICATION`.

The treatment may answer only when one map-supported interpretation uniquely
entails the requested entity–fact–qualifier tuple. Harmless entity aliases are
allowed only when identity and requested fact are unique. Multiple compatible
spans, missing requested facts, unresolved entity aliases, or unsupported
qualifiers must route to verification. No native video, new evidence, labels,
references, direct predictions, multiple candidate answers, or post-hoc
salience rule may enter inference.

Freeze `treatment-freeze.json` before selection. It binds the exact prompt,
schema, deterministic validator, runner hash, decision rules, promotion gate,
and the source hashes for the 475-row protocol and frozen maps. The provider
condition is `gemini-3.1-pro-preview`, temperature `0.0`, thinking level `HIGH`,
maximum output `2048` tokens, and one attempt per row. The historical issue #35
runners remain byte-for-byte unchanged.

## Offline candidate and sufficiency audit

Build a write-once candidate sheet for all 475 rows from the frozen full-panel
protocol, maps, direct predictions, post-freeze scores, annotation hashes, and
exposure manifests. The provider-safe projection contains no reference,
correctness label, sufficiency verdict, or direct prediction.

After the treatment freeze exists, construct deterministic audit queues:

1. Low-score queue: `temporal` and `audio_or_text` rows with direct semantic
   composite at most `0.45`, sorted by composite ascending and then video ID.
2. Control queue: the same strata with direct semantic composite at least
   `0.70`, sorted by composite descending and then video ID.
3. Audit the first 24 low-score rows and first 8 control rows. Do not stop early
   when role quotas are met.

For each of these 32 rows, an independent label-aware audit compares the
question, frozen reference, complete frozen map, and frozen direct answer. It
records the requested tuple, exact supporting map lines, conflicts, exposure
flags, and one verdict:

- `answer_sufficient_failure`: the direct answer is wrong or materially
  incomplete and exactly one map-supported answer entails the reference-level
  requested tuple;
- `insufficient_or_ambiguous`: the map lacks the requested fact or supports
  multiple unresolved answers;
- `supported_control`: the direct answer is correct and the map uniquely
  supports it;
- `reference_or_question_issue`: the label/question prevents a unique validity
  judgment.

No native-video rerun or new evidence acquisition is allowed. A row that
cannot be adjudicated from the frozen records is not promoted to an
answer-sufficient target.

## Panel feasibility gate

The zero-call audit authorizes inference only if it yields at least four rows
for each of these roles:

- answer-sufficient failure target;
- insufficiency/ambiguity safeguard;
- supported stability control.

Select the first four eligible rows for each role in deterministic queue order.
Freeze the 12-row role manifest, exact evidence lines, exposure flags, audit
verdicts, treatment hash, source hashes, and an audit checksum. If any quota is
not met, preserve the audit as a valid negative feasibility result and issue no
provider calls.

## Provider run and evaluation

If the offline gate passes, run exactly one deterministic Gemini request for
each of the 12 frozen rows, with no retry and no alternative-answer generation.
Freeze request, raw response, metadata, parsed result, terminal state, and
predictions before loading references for scoring.

The predeclared promotion gate requires:

- at least three of four answer-sufficient failures rescued;
- all four insufficiency safeguards routed to verification;
- all four controls preserved;
- zero unsupported emitted answers;
- 12/12 terminal rows and manual inspection of every changed row.

Report METEOR, BERTScore-F1, STS when the frozen comparable service/cache is
available, and the existing semantic composite only on emitted answer rows.
Report verification-route accuracy and role-specific results separately; do
not score the sentinel as an ordinary semantic answer. Also report results by
prior-exposure status. Preserve a failed gate as a negative result.

## Durable artifacts and reporting

Private write-once artifacts live under a new benchmark root in the canonical
shared dataset. The tracked bundle contains only sanitized counts, hashes,
schema descriptions, gate outcomes, and aggregate metrics—never row IDs,
questions, references, predictions, or evidence text. Update the existing
dated frontier report and the routing theme; do not create a new research
theme.

Evidence validity and decision disposition use the repository's standard
axes. A positive zero-call audit establishes only that a clean verifier panel
can be formed. A positive provider result establishes only the tested
answer-use mechanism on the selected panel; it does not promote a generic
router or establish population prevalence.

## Verification

Tests must cover write-once behavior, treatment-before-audit ordering,
source/exposure hash drift, all-475 eligibility, label-free provider
projection, deterministic queue order, exact role quotas, evidence-line
validation, tamper detection, sentinel-aware metrics, and exposure-stratified
reporting. Run focused and neighboring tests, Ruff, JSON validation, checksums,
privacy scans, generated-report consistency, and independent review before any
push or issue update.
