# Deadline VQA Submission Canaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide today whether Gemini 3.7 Flash native AV and Qwen3.6-27B native video with Whisper are sane enough to scale into two additional TRECVID AVS answer-generation submissions.

**Architecture:** Use one frozen five-video labeled development panel spanning temporal order, causality, OCR/detail, counting, and speech. Run each model only on that panel first, assemble the same ten-ranked-answer shape required by the official validator, freeze outputs before opening references, and manually compare every prediction with the video and reference. Scale only a path that passes deterministic validation and has no obviously wrong primary answers.

**Tech Stack:** Python 3.12, pytest, Google Gen AI SDK, vLLM, Qwen3.6-27B, Whisper transcripts already accepted by the repository quality gate, dstack, Hugging Face Buckets, TRECVID submission validator.

## Live Status — 2026-08-17 03:00 EDT

- Qwen3.6-27B full inference is complete: four H100 shards, 4,886 condition rows, 4,577 successes, 309 expected unavailable conditions, and zero model-call failures.
- Native-first and transcript-first 2,443-row submissions are assembled. Both contain ten distinct answers per query and pass the NIST validator with zero errors and zero warnings. A stratified manual review found no blocking nonsense.
- Whisper large-v3 transcription is complete for all 1,946 available media files; 1,731 videos produced accepted non-empty transcripts.
- Gemini 3.7 Flash full native-AV inference is complete: 2,443 terminal rows, 2,401 successful answers, 25 controlled failed-answer fallbacks, 10 unavailable inputs, and 7 upload-failure fallbacks. Known response usage cost is `$22.579845`; including the maximum reserve for preserved interrupted Q188, conservative exposure is `$22.5926655`.
- Gemini parsing treats `PROHIBITED_CONTENT`, `MAX_TOKENS`, empty responses, and the preserved no-retry Q188 interruption as fallbacks. All upload/request attempts are terminal and hash-bound; unmatched attempts are zero.
- NIST permits at most five AG runs. The complete set is: Gemini-first fusion, Gemini-only hybrid, Qwen transcript-first, Qwen native-first, and the existing Qwen3.5 baseline control. All five contain 2,443 rows and ten distinct answers per query, and all five pass the official NIST validator with zero errors and zero warnings.
- The checksummed local upload packet is `data/vqa/submissions/deadline-20260817-ready/`. The remaining action is user-authenticated Evalbase upload and organizer-side receipt capture; nothing has been uploaded yet.

## Global Constraints

- Work from `/home/npatta01/data/competitions/trec_video_2026` on `codex/qwen-aligner-quality`; preserve the unrelated untracked `.claude/` directory.
- Never copy, stage, publish, or delete canonical dataset content.
- Freeze predictions before reading references. Keep row-level questions, transcripts, outputs, references, and media under ignored `data/vqa/benchmarks/` roots.
- Never print or persist API/Hugging Face credentials in commands, logs, or tracked artifacts.
- A failed canary gets one bounded diagnosis/fix attempt. Do not broaden the harness before the five-row decision is known.
- No full-panel paid run starts until its five-row outputs pass deterministic checks and manual inspection.

---

## Task 1: Freeze the Five-Video Decision Panel and Submission Contract

**Files:**
- Create: `configs/vqa/deadline_canary_5.json`
- Create: `tests/test_vqa_deadline_canary.py`
- Create: `src/trec_video_2026/vqa_deadline_canary.py`

- [x] Write a failing test requiring exactly five unique canonical 2025 labeled-development videos, one each for temporal, causal, OCR/detail, counting, and speech.
- [x] Write a failing test requiring exactly ten distinct ranked answers per query and official-format identifiers.
- [x] Implement only the frozen panel loader and ten-answer validator needed to pass those tests.
- [x] Run `uv run pytest -q tests/test_vqa_deadline_canary.py` and `uv run ruff check src/trec_video_2026/vqa_deadline_canary.py tests/test_vqa_deadline_canary.py`.
- [x] Record the exact panel, hashes, model IDs, prompt, and answer-ranking contract in a private run manifest.

## Task 2: Run the Gemini 3.7 Flash Native-AV Five-Video Canary

**Files:**
- Create: `scripts/run_vqa_deadline_gemini_canary.py`
- Modify: `tests/test_vqa_deadline_canary.py`

- [x] Write failing tests for one direct native-audio/video request per row, model `gemini-3.7-flash`, frozen raw responses, and no reference access before prediction freeze.
- [x] Implement the minimal synchronous five-row runner by reusing the tested request/parser primitives in `vqa_gemini37_flash_smoke.py`.
- [x] Dry-run all five requests without network mutation and verify the predicted cost is below `$0.15`.
- [x] Run five paid requests and freeze raw and parsed predictions without replacing the direct answer at rank 1.
- [ ] Build ten distinct ranked candidates per row from the frozen Gemini direct answer plus the validated baseline tail.
- [ ] Validate the five query records with the repository’s official submission checks.

## Task 3: Run the Qwen3.6-27B Native-Video Five-Video Canary

**Files:**
- Create: `scripts/run_vqa_deadline_qwen36_canary.py`
- Create: `dstack/qwen36-27b-deadline-canary.dstack.yml`
- Modify: `tests/test_vqa_deadline_canary.py`

- [x] Write failing tests for the pinned Qwen revision, native-video input, accepted Whisper transcript arm, matched no-transcript arm, durable output path, and bounded runtime/cost.
- [x] Implement the smallest vLLM runner that emits both answers for all five rows and freezes raw responses before reference access.
- [x] Run local no-GPU contract tests and dstack capability/preflight checks.
- [x] Preview the exact declined dstack configuration unchanged, then launch one H100-class canary and sync artifacts in the foreground to the durable Hugging Face Bucket.
- [ ] Build ten distinct ranked candidates per query with transcript-conditioned and no-transcript answers preserved separately.
- [ ] Validate the five query records with the repository’s official submission checks.

## Task 4: Manually Audit Every Canary Answer

**Files:**
- Create privately: `data/vqa/benchmarks/deadline-submission-canary-5-20260816/inspection.md`

- [x] Confirm both prediction freezes and checksums before opening references.
- [x] For each of the five videos, inspect the complete clip or timestamped evidence, the official reference answers, Gemini rank 1, Qwen transcript rank 1, and Qwen no-transcript answer.
- [x] Mark factual correctness, temporal/count agreement, transcript help/harm, and whether each answer is a reasonable paraphrase of the reference.
- [x] Stop a model path if any primary answer is obviously nonsensical or the runner used the wrong media/question/transcript.
- [x] If a runner bug is found, add a failing regression test, make one bounded fix, and rerun only the five-row canary.

## Task 5: Scale Passing Paths and Assemble Official Runs

**Files:**
- Create under: `submissions/trecvid-2026/vqa-ag/`
- Create privately under: `data/vqa/submissions/`

- [x] Partition all 2,443 queries into deterministic shards and reuse exact five-row-tested prompts/model revisions.
- [x] Run passing Gemini and Qwen paths concurrently where capacity and API limits permit; sync every shard durably.
- [x] Preserve direct/native primary answers and fill ranks 2–10 with concise paraphrases, matched no-transcript/transcript alternatives, and existing validated candidates without duplicates.
- [x] Merge exactly 2,443 query records, each with exactly ten distinct ranked answers.
- [x] Run schema, ID-set, uniqueness, coverage, secret, checksum, and official submission validators.

## Task 6: Upload and Record the Decision

**Files:**
- Create: `docs/submissions/2026-qwen36-27b-native-whisper.md`
- Create: `docs/submissions/2026-gemini37-flash-native-av.md`
- Modify the relevant experiment QMD only if a complete result is available.

- [ ] Upload only fully validated submission files, preserving local hashes and portal receipts.
- [ ] Verify organizer-side validation and record the exact accepted run IDs.
- [ ] Record measured runtime/cost, failures, and the five-row manual audit outcome without claiming unmeasured quality.
- [ ] Run focused tests, Ruff, `git diff --check`, tracked checksum verification, and privacy/credential scans.
