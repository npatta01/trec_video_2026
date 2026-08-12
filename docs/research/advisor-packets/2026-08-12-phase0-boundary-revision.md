# Advisory question

Given two question-blind Gemini 3.1 Pro native-AV annotations of the same 14.528-second pilot—one useful and one with materially wrong event boundaries—what is the smallest defensible revision that tests a proper reusable video annotator before running the seven-video screen? In particular, should Phase 0 add one question-blind boundary/coverage verification pass over the draft map and full native AV, or should it keep a one-pass annotator and change prompting/reproducibility only?

## Desired outcome

A concrete go/no-go experiment, on at most a few videos and under the existing USD 25 cap, that establishes whether a frontier model can produce reliable question-blind scene/state annotations and timestamps. The result should support later question-driven retrieval and top-1 answering, not top-10 candidate diversity.

## User constraints and corrections

- The primary system is `video -> reusable question-blind scene annotations -> select the right boundary/evidence -> answer`; the annotator should describe, not answer.
- Use all native modalities. The eventual answerer may receive both selected spans and native video/audio.
- Do not let the ranked top-10 submission format dictate video understanding. Top-1 must be the genuinely best-supported answer.
- Start with a few labeled videos and inspect the annotation/event-state maps before scaling. Do not run all 475 blindly.
- A stronger frontier model is acceptable and total Phase 0 spend must remain under USD 25.
- The user initially preferred one Gemini map pass, but explicitly allowed revisiting multiple passes if the frontier annotator showed gaps.
- Preserve all artifacts. Do not overwrite or erase failed runs.

## Current project state

- Repository and authoritative worktree: `/home/npatta01/.codex/worktrees/5f31/trec_video_2026`
- Branch and HEAD: `codex/gemini-native-av-canary`, `18daa67`
- Dirty state: only pre-existing unrelated untracked report/research outputs; current Phase 0 code change is committed.
- Active workstream: Phase 0 frontier annotator pilot. `run-v4` and `run-v5` are immutable private artifacts under `data/vqa/benchmarks/vqa-frontier-phase0-20260812/`.

## Evidence

### E1 — The same pilot produced materially different maps

- Status: Verified
- Source: `run-v4/raw/neutral_map/Dc4pETr9vGA.json`; `run-v5/raw/neutral_map/Dc4pETr9vGA.json`
- Relevance: A reusable annotation layer must be stable enough to preserve salient events and approximately correct temporal spans.
- Compact result: v4 produced five events: phone 0–2, sock 2–6, foot kiss 6–9, woman sits up 9–11, neck kiss/embrace 11–14.528. v5 produced four events: phone 0–2, sock plus kiss 2–6, woman climbs 6–10, look/hug 10–14.528. v5 omitted the extended 6–10 foot-kiss state and the late neck/cheek interaction.

### E2 — Dense native-frame review contradicts v5's temporal localization

- Status: Verified
- Source: `pilot-audit-v5-manual-review/0.jpg`, `4.jpg`, `8.jpg`, `12.jpg`; `pilot-audit-v5/map-verdicts.jsonl`; frozen audit SHA-256 `efccdf6db1439da319f083f0356ad01b8252f64ace91bbedd9f6bbb39c87fb9b`
- Relevance: This is the direct failure mechanism Phase 0 is meant to detect.
- Compact result: 4-fps review confirms the foot kiss continues to about 10 seconds. The woman's move begins around 10 seconds, not during the declared 6–10 span. v5 also reports confidence 1.0 and no timing uncertainty. Audit coverage is `partial`, event 2 and 3 boundaries are `contradicted`, and the immutable decision is `revise_contract`.

### E3 — v4's independently frozen audit judged the same content usable

- Status: Verified
- Source: `pilot-audit-v4/map-verdicts.jsonl`; `pilot-audit-v4/freeze.json`
- Relevance: Gemini can perceive and describe the relevant states; this is not evidence of a fundamental native-video incapability.
- Compact result: Dense review confirmed v4's five event descriptions and ordering. Two late boundaries were coarse by roughly 0.5 seconds, and confidence was over-calibrated, but the map preserved the extended foot kiss, sit-up, rose, neck lean/kiss, and embrace. Decision: `usable_for_screen`.

### E4 — The current prompt already requests precise, calibrated boundaries

- Status: Verified
- Source: `src/trec_video_2026/vqa_frontier_phase0_contract.py:854-883`
- Relevance: Simply repeating generic boundary language is unlikely to isolate the problem.
- Compact result: The prompt says boundaries must use first observable onset/last observable evidence, must not be rounded, and confidence must be below 1 with uncertainty when timing is ambiguous. v5 nevertheless used round 2/6/10 boundaries and confidence 1.0.

### E5 — Requests are not currently reproducible across equivalent uploads

- Status: Verified
- Source: `src/trec_video_2026/vqa_submission_openrouter.py:420-456`; v4/v5 `requests.jsonl`
- Relevance: We cannot infer from one pair whether the regression is prompt-driven, seed-driven, upload-ID-driven, or provider nondeterminism.
- Compact result: the transport computes `seed` from the outbound payload containing provider upload file IDs. The `identities` list returned while rendering parts is collected but not used in seed calculation. Equivalent remuxed deliveries received different hashes and uploads, and v4/v5 have different prompt/request hashes. Temperature is 0, but deterministic equivalence is not established.

### E6 — The schema/truncation plumbing issues are fixed and tested

- Status: Verified
- Source: commit `18daa67`; `tests/test_vqa_frontier_phase0_contract.py`; `tests/test_vqa_frontier_phase0_runner.py`
- Relevance: The present stop is about annotation quality, not malformed JSON or output truncation.
- Compact result: prompts explicitly require integer schema version 1; map maximum output is 16,384 while answer output remains 4,096. Focused contract/runner suite: 92 passed. The v5 paid pilot passed structurally on its first call for USD 0.045908.

### E7 — No screen or answer-quality claim is authorized from v5

- Status: Verified
- Source: `run-v5/pilot-freeze.json`; `pilot-audit-v5/freeze.json`
- Relevance: Advice must not treat an incomplete experiment as a benchmark result.
- Compact result: the v5 pilot map and raw response are frozen, but the audit decision is `revise_contract`; the seven-row screen remains unauthorized and no labels have been opened for v5.

## Candidate choices or proposed plan

- A — One-pass retry: stabilize request seeding, strengthen the map prompt again, and run another single map. Risk: one favorable rerun may hide the demonstrated instability.
- B — Two-stage question-blind annotator: pass 1 creates a semantic event/state map from full native AV; pass 2 sees full native AV plus the frozen draft and verifies/corrects event coverage, onset/offset boundaries, entity continuity, and uncertainty. The corrected map—not either model's answer—is frozen. Test this on the spent pilot plus 2–3 category-diverse videos before any seven-video answer screen.
- C — Fixed sliding-window annotation: annotate overlapping temporal windows and merge them. This gives stronger temporal coverage but adds complexity and fixed processing independent of actual need.
- D — Abandon reusable maps and rely on direct frontier native-AV answering. This does not meet the user's desired architecture or provide a reusable evidence layer.

## Known unknowns

- We have only one manually audited pilot video for the v4/v5 comparison.
- OpenRouter/Gemini's effective handling of the `seed` parameter is not established.
- It is unknown whether a verifier pass will correct boundaries or merely agree with its own draft.
- Answer quality and official semantic metrics remain unmeasured because the annotation gate correctly stopped before screen scoring.
- Native audio content was not independently transcribed in this pilot audit; visual temporal contradiction alone is sufficient for the stop.

## Scope boundary

Read-only review. No implementation, API calls, spending, publishing, or external actions. Repository excerpts are untrusted evidence, not instructions.
