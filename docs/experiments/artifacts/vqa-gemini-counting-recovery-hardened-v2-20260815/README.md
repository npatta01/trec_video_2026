# Hardened Gemini counting recovery repeat — sanitized index

This bundle indexes the authorized one-row repeat of the repaired Gemini
native-video counting recovery workflow for issue #34.

The run was restricted to the first planned failure row and stopped before the
other two manifest rows. All three required stages completed, the result was
frozen before reference unlock, and the authenticated exact prediction was
`2` against reference count `4`.

- Completion: 1/1 authorized row; two other manifest rows untouched.
- Rescue: 0/1 (`0%`).
- Control preservation: unavailable; no control row ran.
- Paid provider calls: 3.
- Cost: `$0.111700` under the `$0.34` one-row cap.
- Evidence validity: `Diagnostic`.
- Decision disposition: `Not promoted under the tested condition`.

Post-freeze frame inspection found that Gemini identified the two middle ball
appearances but missed appearances at both temporal edges. The challenger
nevertheless certified the incomplete ledger. The result therefore rejects
the unchanged model-asserted full-span coverage mechanism; no retry or scale-up
is warranted.

`summary.json` contains only sanitized aggregate evidence and cryptographic
bindings. Complete private evidence remains in the canonical benchmark root
`direct-gemini-counting-recovery-hardened-v2`; it is intentionally not tracked
here.
