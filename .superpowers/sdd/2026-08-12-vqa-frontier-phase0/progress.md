# SDD ledger — plan: docs/superpowers/plans/2026-08-12-vqa-frontier-phase0.md

Setup: existing linked worktree verified on `codex/gemini-native-av-canary`.
Setup: shared `data` symlink resolves to the canonical dataset root.
Setup: pre-flight plan conflict scan clean.
Setup: relevant baseline suite clean — 149 passed in 1.97s.
Task 1: minor (deferred): provider-schema projection tests do not recursively prove all unsupported keywords are removed.
Task 1: minor (deferred): frozen-record test checks tuple conversion/type but not mutation failure or hash stability.
Task 1: reviewer cannot verify downstream provider wiring and label isolation from this contract-only diff; Tasks 4–9 own those checks.
Task 1: fix round 1/5 (2 addressed, 0 open — runtime/local-schema parity; validated guided-map input; commits 0f418d2..a5714b5).
Task 1: minor (deferred): question prompt input now inherits a 4096-character cap even though the question is outside the output schema; immaterial for this frozen dataset.
Task 1: complete (commits c79139c..a5714b5, review clean with 3 deferred minors).
Task 2: review ruling — condition IDs `direct_answer` and `map_guided_answer` are mandated verbatim by the Task 2 brief; reviewer saw an erroneous controller-supplied constraint naming them `*_native_av`. No code change.
Task 2: review ruling — private questions are allowed and required by the answering architecture; “without labels” and recursive rejection apply to references/gold/labels, not questions. Questions remain private under canonical data.
Task 2: minor (deferred): forbidden-key matching does not reject every compound token such as `incorrect_answer`; exact reference/label/gold/oracle keys are covered.
Task 2: fix round 1/5 (4 addressed, 0 open — Qwen provenance, full canary freeze validation, private-root output, independent selection test; commits c3125bd..ed08185).
Task 2: minor (deferred): unused `PILOT_BINDING_KEYS` import in CLI test module; no runtime impact.
Task 2: complete (commits a5714b5..ed08185, review clean with 2 deferred minors and 2 recorded rulings).
Task 3: fix round 1/5 (4 addressed, 0 open — descriptor-relative legacy writes, no-follow verification traversal, exact record shape, strict UTF-8; commits 88a3f18..c3bdc93).
Task 3: complete (commits ed08185..c3bdc93, review clean).
Task 4: minor (deferred): 1,747-line runner and combined dispatch/parser increase audit surface; final branch review must assess maintainability after correctness fixes.
Task 4: fix round 1/5 (7 addressed, 4 open — billing claims/cost/cleanup/retry/duplicates/chains/panel/request/transport; commits da83d81..dc6e488).
Task 4: fix round 2/5 (4 addressed, 2 open — completed-claim race, OpenRouter truncation evidence, freeze totals, Task 2 ordinals; commits dc6e488..36a30fd).
Task 4: fix round 3/5 (2 addressed, 0 open — malformed-content retry classification and zero-cost skipped loser; commits 36a30fd..0ad8fce).
Task 4: complete (commits c3bdc93..0ad8fce, review clean with runner-size minor deferred).
Task 5: fix round 1/5 complete — evidence bytes and complete frame records are deterministically bound and reverified; source/evidence symlinks are rejected; media paths are root-relative under `rows/<video_id>`; source-timeline PTS and fallback provenance are validated; pilot usability follows the exact complete/no-contradiction rule while diagnostic failure freezes remain allowed.
Task 5: fix round 1/5 verification — focused audit 21 passed; prescribed combined suite 114 passed in 27.45s; Ruff and `git diff --check` passed; no residual concern in owned scope.
