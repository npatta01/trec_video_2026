# Gemini counting recovery canary — sanitized index

This bundle indexes the incomplete 2026-08-15 question-driven, multi-pass
Gemini native-video counting canary for issue #34.

The first of three developmental rows was attempted. Its locator and verifier
calls completed, but the mandatory full-video challenger returned truncated
JSON. The row therefore remained incomplete, the remaining two rows were not
started, and no semantic metrics or rescue/control rates were computed.

- Completion: 0/3 rows; 1/3 attempted.
- Paid provider calls: 3.
- Estimated direct-Google cost: `$0.156048`.
- Evidence validity: `Incomplete`.
- Decision disposition: `Not promoted under the tested condition`.

The malformed-response terminals preserve raw text and token-derived costs but
not provider response IDs or returned model-version fields. The runner was
hardened after this immutable run; that change was not rerun.

An additive private provenance audit also records that the three OS-exclusive
paid-call claims are not self-authenticating because of a legacy hash namespace
collision, and stage-to-media binding is indirect through the manifest and
upload record. The post-run runner hardening is not experimental evidence.

`summary.json` contains only sanitized aggregate evidence and cryptographic
bindings. Complete private evidence remains in the canonical benchmark root
`direct-gemini-counting-recovery-v1`; it is intentionally not tracked here.
