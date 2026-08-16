# Structured frame-presence counting feasibility — sanitized index

This bundle records a zero-provider-cost, post-label feasibility check for
issue #34. A fixed endpoint-inclusive grid of 55 decoded frames (7.5 fps) was
reviewed with strict `present`/`absent` decisions, then a deterministic
reducer counted contiguous presence episodes.

The reducer returned **4**, matching the known reference on the one reviewed
failure row. This is a diagnostic human-oracle result, not promotion evidence:
the row was already unblinded and no model produced the frame decisions.

The audit also corrects the interpretation of the preceding hardened Gemini
negative. Gemini's first broad range contained two separate ball appearances
that its verifier collapsed into one; it then omitted the fourth appearance
near 6.5 seconds. The earlier claim that the model missed both temporal edges
came from a seeked inspection sheet whose displayed timestamps restarted at
zero. The frozen prediction, score, 0/1 rescue result, and stop decision do not
change.

- Selected frames: 55, including the first and last decoded frames.
- Effective sampling: 7.5 fps; maximum selected-frame PTS gap `0.134s`.
- Deterministic result: exact 4 episodes.
- Provider calls and cost: 0 / `$0.00`.
- Evidence validity: `Diagnostic`.
- Decision disposition: `Diagnostic signal`.

`summary.json` contains sanitized observations and cryptographic bindings.
Complete frames, contact sheets, ledger, result, and provenance remain private
in the canonical benchmark root
`direct-gemini-counting-frame-presence-feasibility-v1`.
