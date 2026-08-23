# VQA MC Adaptive Inspection Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a label-isolated, exactly-two-call Gemini agent that selects one native-AV inspection per TRECVID MC video and produces an evidence-bearing final ranking.

**Architecture:** A provider-independent core module defines and validates the planner and final-answer contracts. A new Gemini runner owns uploads, two sequential inference stages, immutable freezes, and cost accounting; a separate scorer is the only component that reads labels. The existing direct-Gemini and Qwen evidence remains unchanged.

**Tech Stack:** Python 3.13, `google-genai` 2.x, pytest, Ruff, existing TRECVID VQA MC request/label artifacts.

**Spec:** `docs/superpowers/specs/2026-08-21-vqa-mc-adaptive-inspection-agent-design.md`

## Global Constraints

- Use exactly 20 canonical rows and reject label-bearing inference inputs.
- Run exactly one 4-FPS/low full-video planner call and one validated inspection call per row.
- Inspection FPS is one of `1.0`, `4.0`, or `8.0`; resolution is `low` or `high`.
- Never silently clamp an invalid interval; record and use the deterministic 4-FPS/high full-video fallback.
- Create private evidence only under ignored `data/`; tracked reports contain aggregates only.
- Reject the run before credentials or inference if worst-case cost exceeds `$0.75`.
- Do not modify or overwrite frozen direct-Gemini or Qwen evidence.
- This externally managed worktree is detached; do not push or create a branch during execution.

---

### Task 1: Provider-independent agent contract

**Files:**
- Create: `src/trec_video_2026/vqa_experiments/vqa_mc_adaptive_agent.py`
- Create: `tests/test_vqa_mc_adaptive_agent.py`

**Interfaces:**
- Consumes: `OPTION_IDS`, `parse_ranking`, and `score_rankings` from `trec_video_2026.vqa_experiments.vqa_mc_poc20`.
- Produces: `PLANNER_SCHEMA`, `FINAL_SCHEMA`, `InspectionPlan`, `parse_stage1`, `parse_stage2`, `fallback_plan`, `validate_plan`, `stage1_prompt`, `stage2_prompt`, and `choose_final_ranking`.

- [ ] **Step 1: Add failing tests for valid contracts and prompts**

```python
def test_stage1_parses_ranking_and_generic_inspection():
    parsed = parse_stage1({
        "ranking": ["B", "A", "C", "D"],
        "inspection": {
            "start_seconds": 1.25,
            "end_seconds": 4.75,
            "fps": 8.0,
            "resolution": "high",
            "evidence_gap": "rapid_or_repeated_action",
            "evidence_question": "Count each completed repetition.",
        },
    }, duration_seconds=6.0)
    assert parsed["ranking"] == ["B", "A", "C", "D"]
    assert parsed["inspection"].fps == 8.0

def test_stage2_prompt_hides_stage1_top_two_order():
    prompt = stage2_prompt(row(), plan(), ["B", "A", "C", "D"])
    assert "Top-two set: A, B" in prompt
    assert "B, A" not in prompt
```

- [ ] **Step 2: Run the contract tests and verify missing symbols fail**

Run: `PYTHONPATH=src pytest -q tests/test_vqa_mc_adaptive_agent.py`

Expected: collection or import failure because the new module does not exist.

- [ ] **Step 3: Implement the strict schemas and parser**

```python
ALLOWED_FPS = frozenset((1.0, 4.0, 8.0))
ALLOWED_RESOLUTIONS = frozenset(("low", "high"))
EVIDENCE_GAPS = frozenset((
    "text_detail", "rapid_or_repeated_action", "speech_or_sound",
    "global_composition", "temporal_order", "other",
))

@dataclass(frozen=True)
class InspectionPlan:
    start_seconds: float
    end_seconds: float
    fps: float
    resolution: str
    evidence_gap: str
    evidence_question: str
    source: str
```

Implement exact-key checks, finite-number checks, a 1-ms interval minimum,
duration bounds, evidence timestamp validation, and deterministic fallback
`InspectionPlan(0.0, duration, 4.0, "high", "other", ..., "fallback")`.

- [ ] **Step 4: Add failure and final-selection tests**

Cover non-finite values, reversed/out-of-bounds intervals, unsupported FPS,
unsupported resolution/gap, blank evidence questions, invalid evidence
timestamps, stage-two fallback to stage one, and double-invalid rows.

- [ ] **Step 5: Run the focused tests**

Run: `PYTHONPATH=src pytest -q tests/test_vqa_mc_adaptive_agent.py`

Expected: all tests pass.

---

### Task 2: Two-stage Gemini runner and immutable evidence

**Files:**
- Create: `scripts/run_vqa_mc_gemini_adaptive_agent_poc20.py`
- Create: `tests/test_vqa_mc_adaptive_agent_runner.py`

**Interfaces:**
- Consumes: Task 1 contracts; `load_requests`, `_delivery_path`, `_upload`, `_response_text`, `_api_key`, and `_write_json` behavior from the existing direct runner, copied only where import-safe reuse is unavailable.
- Produces: `estimate_worst_case_cost`, `preflight`, `build_stage1_request`, `build_stage2_request`, `run`, and CLI modes `--preflight` / `--run`.

- [ ] **Step 1: Add failing request-metadata and cost tests**

```python
def test_stage1_is_full_video_4fps_low():
    request = runner.build_stage1_request(row(), upload())
    video = request.contents[0].parts[0]
    assert video.video_metadata.fps == 4.0
    assert video.video_metadata.start_offset is None
    assert video.media_resolution.level.name.endswith("LOW")

def test_stage2_uses_validated_interval_fps_and_resolution():
    request = runner.build_stage2_request(row(), upload(), plan(), list("BACD"))
    video = request.contents[0].parts[0]
    assert video.video_metadata.start_offset == "1.250s"
    assert video.video_metadata.end_offset == "4.750s"
    assert video.video_metadata.fps == 8.0
```

Also assert the response schemas, canonical-only request count, worst-case
40-call estimate, and `$0.75` rejection.

- [ ] **Step 2: Run runner tests and verify failure**

Run: `PYTHONPATH=src pytest -q tests/test_vqa_mc_adaptive_agent_runner.py`

Expected: import failure because the runner does not exist.

- [ ] **Step 3: Implement preflight and request builders**

Use `types.VideoMetadata(fps=..., start_offset=..., end_offset=...)` and
per-part `PartMediaResolutionLevel`. Calculate a conservative input ceiling
from 246.9 seconds or loaded request durations using 312 tokens/second for
stage one and 2,272 tokens/second for worst-case stage two, plus 500 text
tokens per call and 2,048 output tokens per call.

- [ ] **Step 4: Implement the two-stage freeze order**

Create stage-specific raw directories and write, in order:

```text
run-manifest.json
stage1/raw/*.json
stage1/raw-freeze.json
stage1/predictions-freeze.json
inspection-requests-freeze.json
stage2/raw/*.json
stage2/raw-freeze.json
final-predictions-freeze.json
summary.json
```

Parse stage one only after its raw freeze. Freeze every validated or fallback
inspection request before the first stage-two call. Parse stage two only after
its raw freeze. Preserve per-stage usage and observed price.

- [ ] **Step 5: Add a fake-client end-to-end runner test**

Use one fake row and two canned strict responses. Assert two provider calls,
one upload, immutable freeze hashes, stage-two clipping metadata, final source,
and that an invalid planner produces a recorded fallback rather than skipping
stage two.

- [ ] **Step 6: Run focused runner tests**

Run: `PYTHONPATH=src pytest -q tests/test_vqa_mc_adaptive_agent_runner.py`

Expected: all tests pass.

---

### Task 3: Label-isolated scoring

**Files:**
- Create: `scripts/score_vqa_mc_adaptive_agent_poc20.py`
- Create: `tests/test_vqa_mc_adaptive_agent_scoring.py`

**Interfaces:**
- Consumes: frozen request rows, frozen labels, frozen external Gemini control,
  `stage1/predictions-freeze.json`, `final-predictions-freeze.json`, and
  `inspection-requests-freeze.json`.
- Produces: one private score JSON with metrics, transitions, plan
  distributions, source hashes, and row-level audit records.

- [ ] **Step 1: Add failing scorer tests**

```python
def test_scorer_counts_repairs_and_regressions():
    result = scorer.score(...)
    assert result["transitions"] == {
        "repaired": 1,
        "regressed": 1,
        "unchanged_correct": 0,
        "unchanged_wrong": 0,
    }
    assert result["inspection_distribution"]["fps"] == {"8.0": 1, "4.0": 1}
```

Also test request/label identity drift, duplicate/missing predictions, invalid
ranking penalties, fallback/source counts, and source hashes.

- [ ] **Step 2: Run scorer tests and verify failure**

Run: `PYTHONPATH=src pytest -q tests/test_vqa_mc_adaptive_agent_scoring.py`

Expected: import failure because the scorer does not exist.

- [ ] **Step 3: Implement scoring and the exclusive-output CLI**

Reuse `score_rankings` and compute per-stratum metrics. A repair is stage-one
wrong/final correct; a regression is stage-one correct/final wrong. Do not
drop invalid rows from the denominator.

- [ ] **Step 4: Run scorer tests**

Run: `PYTHONPATH=src pytest -q tests/test_vqa_mc_adaptive_agent_scoring.py`

Expected: all tests pass.

---

### Task 4: Local verification and paid-run gate

**Files:**
- Modify only if tests reveal a scoped defect in Task 1-3 files.

**Interfaces:**
- Consumes: all new code and the existing MC implementation.
- Produces: fresh test, lint, symlink, and no-cost preflight evidence.

- [ ] **Step 1: Run the focused adaptive suite**

Run: `PYTHONPATH=src pytest -q tests/test_vqa_mc_adaptive_agent.py tests/test_vqa_mc_adaptive_agent_runner.py tests/test_vqa_mc_adaptive_agent_scoring.py`

Expected: all pass.

- [ ] **Step 2: Run existing MC regression tests**

Run: `PYTHONPATH=src pytest -q tests/test_vqa_mc_poc20.py tests/test_vqa_mc_poc20_runners.py tests/test_vqa_mc_poc20_scoring.py`

Expected: all pass.

- [ ] **Step 3: Run lint and diff checks**

Run: `ruff check src/trec_video_2026/vqa_experiments/vqa_mc_adaptive_agent.py scripts/run_vqa_mc_gemini_adaptive_agent_poc20.py scripts/score_vqa_mc_adaptive_agent_poc20.py tests/test_vqa_mc_adaptive_agent*.py && git diff --check`

Expected: exit zero.

- [ ] **Step 4: Verify the dataset link and run no-cost preflight**

Run:

```bash
readlink -f data
PYTHONPATH=src python scripts/run_vqa_mc_gemini_adaptive_agent_poc20.py \
  --preflight --canonical-only
```

Expected: canonical shared-data path, 20 requests, 40 calls, 20 deliveries,
and worst-case cost at or below `$0.75`. No credential is loaded.

---

### Task 5: Execute, score, and record the experiment

**Files:**
- Create privately: `data/vqa/benchmarks/vqa-mc-adaptive-agent-poc20-20260821-a1/`
- Create: `docs/experiments/2026-08-21-vqa-mc-adaptive-agent-poc20.qmd`
- Create: `docs/experiments/artifacts/vqa-mc-adaptive-agent-poc20-20260821/README.md`
- Create: `docs/experiments/artifacts/vqa-mc-adaptive-agent-poc20-20260821/summary.json`
- Modify QMD sources for the owning routing theme and experiment ledger only if the result changes their current decision; regenerate siblings rather than hand-editing them.

**Interfaces:**
- Consumes: verified runner, frozen requests/labels/control, and authorized `$0.75` cap.
- Produces: private immutable inference evidence, private row-level score, and sanitized tracked aggregates.

- [ ] **Step 1: Run the adaptive arm once**

Run:

```bash
PYTHONPATH=src python scripts/run_vqa_mc_gemini_adaptive_agent_poc20.py \
  --run --canonical-only \
  --output data/vqa/benchmarks/vqa-mc-adaptive-agent-poc20-20260821-a1
```

Expected: 20/20 stage-one calls and 20/20 stage-two calls complete or every
terminal failure is preserved; no semantic retries.

- [ ] **Step 2: Verify freeze hashes and completion before labels are read**

Use a read-only JSON/hash audit to confirm 20 unique video IDs at each stage,
40 raw records, frozen request hash identity, inspection interval validity,
and summary totals.

- [ ] **Step 3: Score after the inference freeze**

Run the new scorer with the separate labels, existing direct-Gemini canonical
prediction freeze, adaptive stage-one/final freezes, and inspection freeze.
Write the result with exclusive creation under the private experiment root.

- [ ] **Step 4: Interpret against the predeclared decision rule**

Report observed stage-one/final/control accuracy and MRR, repairs, regressions,
strata, plan distribution, fallback/source counts, cost, and limitations.
Classify the prototype as promising or not promoted without changing the final
submission solely from this selected panel.

- [ ] **Step 5: Write and render the aggregate result**

Keep questions, options, media, row predictions, evidence text, provider raw
responses, and private locators out of tracked files. Render QMD-derived
Markdown/HTML, verify links and privacy, and run `git diff --check`.

- [ ] **Step 6: Run final scoped verification**

Re-run Task 4 tests and lint, validate aggregate JSON against the private score,
and state the known unrelated baseline failures separately.
