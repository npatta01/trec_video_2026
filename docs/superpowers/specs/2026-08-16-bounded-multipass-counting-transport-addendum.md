# Bounded Multi-Pass Counting Transport Addendum

**Date:** 2026-08-16  
**Applies to:** `direct-gemini-bounded-multipass-counting-v4`

The v3 fixed arm stopped on its second row because the CountSpec call returned
HTTP 503 `UNAVAILABLE` before any response artifact existed. The immutable v3
root remains an incomplete result with one of twelve rows complete. Its
conservative cumulative exposure, including all earlier v1/v2 work, is
`$1.310788`.

One fresh, full, label-blind replicate is permitted. It must:

- use the unchanged v3 panel, questions, media, model, prompts, schemas,
  deterministic temporal scheduler, reducers, sampling rules, and ceilings;
- preserve v1 through v3 without alteration;
- carry `$1.310788` forward as settled prior exposure before its first call;
- retain the existing no-retry rule, including for another provider error;
- rerun the representative smoke and all rows rather than splice successful
  outputs from different roots;
- keep labels unavailable until both treatment arms have authenticated freezes;
  and
- report v3 separately as an incomplete transport failure.

If v4 encounters any provider, integrity, schema, authentication, or budget
failure, it stops. No further fresh replicate is authorized by this addendum.

This amendment changes only transport recovery after a response-free provider
failure. It does not permit content-based retries, answer selection, prompt
changes, row replacement, or additional refinement.
