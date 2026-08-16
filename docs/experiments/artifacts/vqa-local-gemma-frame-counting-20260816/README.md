# Local Gemma blind frame-counting smoke test — sanitized index

This bundle indexes the stopped 2026-08-16 zero-provider-cost blind structured
frame-counting smoke test for issue #34.

The frozen panel contained one confirmed failure and one correct control, the
only two untouched rows eligible under the narrow label-independent question
router. The locally cached Gemma 3n model completed both contact-sheet
generations, but both outputs were Markdown-fenced. The predeclared strict
parser rejected both without repair. Post-freeze review also found that the
two calls requested states for 55 and 113 frames, violating the frozen maximum
of 11 frames per model call. The intended canary was therefore not executed;
no individual-frame challenges, second repeat, quality metrics, or promotion
decision were run.

- Local model generations: 2.
- Paid provider calls: 0.
- Provider cost: `$0.00`.
- Evidence validity: `Invalid`.
- Decision disposition: `Not promoted under the tested condition`.
- Submission fallback changed: no.

The fenced outputs are factual observations of the deviating condition, not
usable evidence for the planned canary or for model visual quality. The eligible
panel was also too small to authorize a primary submission change. The deadline
decision is unchanged: stop this branch and retain the validated fallback.

`summary.json` contains sanitized aggregates and cryptographic bindings.
Complete private evidence remains in the canonical benchmark root
`local-gemma-frame-counting-canary-v2`; it is intentionally not tracked here.
