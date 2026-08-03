# Relevant-Scene Oracle Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to implement this plan task-by-task.
> Production behavior follows `superpowers:test-driven-development`; steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit all ten frozen temporal-panel videos to establish whether six-
local hybrid and twelve-local scene-only inputs actually contain and preserve
the adjudicated relevant evidence, then freeze and share the Stage 1 selection
or stop decision before any model inference.

**Architecture:** Add exact-preferred-set hybrid/scene-only planning and
single-round repair while preserving the legacy planner, then add a separate
pure-Python audit-contract module for frozen scene semantics, frame-first
reviews, full-video reviews, third-party adjudication, deterministic condition
selection, and a sanitized aggregate scoreboard. A new ignored private
orchestrator binds the immutable failed-gate evidence without modifying it,
extracts exact decoded frames, creates opaque packets/contact sheets, and
freezes every Stage 1 artifact. This plan intentionally stops before request
construction, GPU inference, answer grading, or transcript work.

**Tech Stack:** Python 3.12, pytest, stdlib JSON/hash/secrets/subprocess,
`ffprobe`, `ffmpeg`, ImageMagick, Ruff, Quarto, Git.

## Global Constraints

- The controlling specification is
  `docs/superpowers/specs/2026-07-30-relevant-scene-oracle-design.md` at or after
  commit `ad4f045`.
- The primary theme is **Visual and Temporal Evidence**. The result is an
  exploratory selected-development-panel diagnostic, not a prevalence,
  baseline, deployment, or 2026-performance claim.
- Audit all ten frozen rows. Keep all ten visible in completion denominators and
  record an explicit reason for every ineligible, incompatible, unconstructible,
  failed, or uncertain row.
- Access the only physical dataset through the ignored `data` symlink, which
  must resolve to
  `/home/npatta01/data/competitions/trec_video_2026/data`. Never copy, move,
  delete, stage, or commit dataset contents.
- Create a new ignored run directory
  `data/vqa/benchmarks/temporal-relevant-scene-oracle-10/`. Treat
  `data/vqa/benchmarks/temporal-required-moment-oracle-10/` as immutable source
  evidence and never invoke its mutating commands.
- Keep media, questions, IDs, annotations, intervals, extracted frames, contact
  sheets, reviewer answers, candidate mappings, and raw logs private and
  ignored. Tracked fixtures use synthetic IDs and synthetic hashes only.
- Stage 1 uses `uniform-12`, `oracle-hybrid-12` (six global plus six local), and
  `oracle-local-12` (twelve local) frame plans. Every plan contains twelve
  unique decoded frame identities sorted by actual PTS. Both candidates use the
  exact adjudicated `preferred_set_index`; the legacy shortest-set planner is
  preserved only for historical reproducibility.
- Freeze preferred-set component IDs/roles and all required motion, ordering,
  duration, transition, and co-occurrence relationships before candidate frames
  exist. Candidate success is judged only against that exact preferred set.
- Auditors see exact timestamps in chronological order, matching
  `build_controlled_messages`. Candidate identities and local/global origin are
  withheld during frame-first review, but timestamp distribution makes this
  partial rather than guaranteed condition blinding.
- Auditors see only the bounded-pixel PNGs reserved for later model input, never
  higher-resolution extracted frames.
- Both independent frame-first files freeze before either auditor receives a
  full-video packet. Both independent full-video files freeze before a third
  auditor adjudicates disagreements.
- Permit one simultaneous, predeclared repair round with at most one replacement
  per failed preferred-set interval. Only `partial` or `not_visible` local-frame
  gaps are repairable. Repair stays inside the adjudicated interval and binds
  proposed/resolved time, exact PTS, frame index, source hash, extraction
  identity, and frame hash.
- Stage 2 may select at most one treatment. Select hybrid only if it passes
  every prospective paired row; otherwise select scene-only only if it passes
  every prospective paired row; otherwise stop. Fewer than eight labelable,
  still-representable rows also stops.
- Do not build model requests, start Podman, access references, generate model
  predictions, grade model answers, or prepare transcript work in this plan.
- QMD is editable report source. Generated Markdown and HTML are regenerated,
  never hand-edited. Only sanitized derived HTML may be refreshed on the
  existing tailnet-only portal.

---

### Task 1: Add exact-preferred-set candidate planning and repair

**Files:**
- Modify: `src/trec_video_2026/vqa_experiments/temporal_bottleneck.py:345`
- Modify: `tests/test_temporal_bottleneck.py:370`

**Interfaces:**
- Consumes: `_normalized_sufficient_sets`, `_hamilton_local_allocations`,
  `_desired_times`, `_bind_slots`, and normalized decoded-frame inventories.
- Produces:
  `relevant_scene_candidate_plans(duration: float, sufficient_sets: list[list[list[float]]], preferred_set_index: int, inventory: list[dict[str, Any]]) -> dict[str, dict[str, Any]]`.
- Produces:
  `repair_relevant_scene_candidate_plan(candidate_plan: dict[str, Any], inventory: list[dict[str, Any]], *, repairs: list[dict[str, Any]]) -> dict[str, Any]`.
- Preserves the existing
  `oracle_hybrid_decoded_frame_plan(...) -> list[dict[str, Any]]` result exactly.

- [ ] **Step 1: Write failing exact-preferred-set planner tests**

  Add synthetic tests with no private values:

  ```python
  def test_candidate_plans_use_exact_preferred_set_and_bind_local_targets():
      inventory = _decoded_inventory([index / 10 for index in range(121)])
      plans = _candidate_plans(
          12.0,
          [[[0.2, 0.4]], [[1.0, 2.0], [4.0, 6.0]]],
          1,
          inventory,
      )
      plan = plans["oracle-local-12"]["frames"]

      assert len(plan) == 12
      assert {frame["slot_kind"] for frame in plan} == {"local"}
      assert plans["oracle-local-12"]["preferred_set_index"] == 1
      assert len({(frame["stream_index"], frame["frame_index"]) for frame in plan}) == 12
      assert [frame["actual_pts_seconds"] for frame in plan] == sorted(
          frame["actual_pts_seconds"] for frame in plan
      )
      assert all(
          1.0 <= frame["actual_pts_seconds"] <= 2.0
          or 4.0 <= frame["actual_pts_seconds"] <= 6.0
          for frame in plan
      )
      assert {frame["audit_target"]["interval_index"] for frame in plan} == {0, 1}
  ```

  Also assert Hamilton allocation for unequal interval durations, exact
  preferred-set selection when a shorter alternative exists, earlier-PTS tie
  breaking, audit-target metadata in `frame_plan_sha256`, exact provenance,
  invalid preferred-index rejection, and failure when twelve unique in-interval
  decoded frames cannot be formed.

- [ ] **Step 2: Run the new tests and verify RED**

  Run:

  ```bash
  uv run pytest tests/test_temporal_bottleneck.py -k 'candidate_plan or repair_relevant' -q
  ```

  Expected: failure because `relevant_scene_candidate_plans` is absent.

- [ ] **Step 3: Implement the exact-preferred candidate planner**

  Add a new private helper for Stage 1 without changing the legacy hybrid
  planner:

  ```python
  def _relevant_scene_candidate_plan(
      duration: float,
      sufficient_sets: list[list[list[float]]],
      preferred_set_index: int,
      inventory: list[dict[str, Any]],
      *,
      local_count: int,
      global_count: int,
  ) -> dict[str, Any]:
      """Bind one preferred set to a candidate and its audit targets."""
  ```

  `relevant_scene_candidate_plans` calls it with `(local_count=6,
  global_count=6)` and `(local_count=12, global_count=0)`. Each local frame binds
  `sufficient_set_index`, `interval_index`, interval bounds, and an opaque
  component slot in `audit_target`; those fields enter `frame_plan_sha256`.
  Preserve existing uniqueness, provenance, and ordering rules. Leave
  `oracle_hybrid_decoded_frame_plan` unchanged for historical reproducibility.

- [ ] **Step 4: Write failing constrained-repair tests**

  Cover one 0.1-second event-center replacement inside the original bound
  interval, collision avoidance, earlier-PTS tie breaking, original and
  replacement plan hashes, and rejection of global-frame repair, outside-
  interval time, imprecise time, unknown target hash, and a second repair.

- [ ] **Step 5: Implement repair and verify GREEN**

  Each repair item contains `target_frame_plan_sha256`,
  `resolved_time_seconds`, and `replacement_reason`. The function derives every
  interval from the target frame's bound `audit_target`; callers cannot supply
  broader intervals. It applies the complete list as one simultaneous round and
  returns a new candidate plan plus one repair record, leaving the original plan
  immutable.

  Run:

  ```bash
  uv run pytest tests/test_temporal_bottleneck.py -q
  uv run ruff check src/trec_video_2026/vqa_experiments/temporal_bottleneck.py tests/test_temporal_bottleneck.py
  git diff --check
  ```

- [ ] **Step 6: Commit the planner**

  ```bash
  git add src/trec_video_2026/vqa_experiments/temporal_bottleneck.py tests/test_temporal_bottleneck.py
  git commit -m "feat: plan exact relevant-scene candidates"
  ```

### Task 2: Add Stage 1 audit and selection contracts

**Files:**
- Create: `src/trec_video_2026/vqa_experiments/relevant_scene_oracle.py`
- Create: `tests/test_relevant_scene_oracle.py`

**Interfaces:**
- Produces
  `validate_scene_semantics(payload: dict[str, Any], adjudication: dict[str, Any]) -> dict[str, Any]`.
- Produces
  `validate_frame_first_audit(payload: dict[str, Any], packet: dict[str, Any], *, expected_reviewer: str) -> dict[str, Any]`.
- Produces
  `validate_full_video_audit(payload: dict[str, Any], packet: dict[str, Any], adjudication: dict[str, Any], scene_semantics: dict[str, Any], frame_first_freeze: dict[str, Any], *, expected_reviewer: str) -> dict[str, Any]`.
- Produces
  `validate_audit_adjudication(payload: dict[str, Any], left: dict[str, Any], right: dict[str, Any], packet: dict[str, Any], adjudication: dict[str, Any], scene_semantics: dict[str, Any], *, round_name: str) -> dict[str, Any]`.
- Produces
  `build_stage2_selection_record(adjudication: dict[str, Any], scene_semantics: dict[str, Any], audit_adjudication: dict[str, Any], candidate_condition_map: dict[str, Any], *, bindings: dict[str, Any]) -> dict[str, Any]`.
- Produces
  `build_stage1_scoreboard(adjudication: dict[str, Any], audit_adjudication: dict[str, Any], selection: dict[str, Any], *, bindings: dict[str, Any]) -> dict[str, Any]`.
- Reuses canonical hashing and file freeze/verification behavior from
  `temporal_bottleneck.py`; it does not execute subprocesses or read media.

- [ ] **Step 1: Write failing scene-semantics tests**

  Require every labelable row's exact preferred set to have one stable component
  per interval and all required relationships frozen before candidate plans:

  ```python
  row = {
      "video_id": "synthetic-1",
      "preferred_set_index": 0,
      "components": [{
          "component_id": "component-1",
          "interval_index": 0,
          "roles": ["target"],
      }],
      "required_relationships": [{
          "relationship_id": "relation-1",
          "type": "transition",
          "component_ids": ["component-1"],
      }],
  }
  ```

  Reject missing/extra preferred intervals, alternative-set components,
  duplicate IDs, unbound relationship components, invalid role/type values,
  ordering with fewer than two components, and timestamps that do not follow
  the bound base-adjudication freeze.

- [ ] **Step 2: Implement and verify scene-semantics validation**

  Use closed component roles `anchor`, `target`, `context`, `precondition`,
  `outcome`, and `cooccurrence`; use relationship types `motion`, `ordering`,
  `duration`, `transition`, and `cooccurrence`. Preserve private descriptive
  material only in ignored artifacts. The private orchestrator additionally
  refuses semantics freeze when any frame-plan artifact already exists.

- [ ] **Step 3: Write failing frame-first validation tests**

  Use a two-row synthetic opaque packet and require this shape:

  ```python
  payload = {
      "schema_version": 1,
      "role": "scene_auditor",
      "reviewer": "auditor-left",
      "packet_sha256": "a" * 64,
      "started_at": "2026-07-30T12:00:00Z",
      "ended_at": "2026-07-30T12:05:00Z",
      "rows": [{
          "opaque_row_id": "row-alpha",
          "candidate_order": ["candidate-k", "candidate-m"],
          "candidates": [{
              "opaque_candidate_id": "candidate-k",
              "response_kind": "not_answerable",
              "response": None,
              "confidence": "low",
              "available_relations": ["ordering"],
          }],
      }],
  }
  ```

  Require exact row/candidate order from the packet. Reject missing rows,
  duplicate opaque IDs, unknown candidates, non-increasing timestamps, a text
  response paired with `not_answerable`, absent confidence, noncanonical
  relation values, answer/reference/prediction/grade/condition keys, and any
  path or condition identity copied into reviewer output.

- [ ] **Step 4: Verify frame-first tests RED, then implement validation**

  Run:

  ```bash
  uv run pytest tests/test_relevant_scene_oracle.py -k 'frame_first' -q
  ```

  Normalize without changing row order, compute `audit_sha256`, and make
  revalidation idempotent.

- [ ] **Step 5: Write failing full-video and adjudication tests**

  Require every row to record:

  ```python
  {
      "opaque_row_id": "row-alpha",
      "still_representation": "still_frame_representable",
      "representation_classes": ["ordering", "transition"],
      "candidates": [{
          "opaque_candidate_id": "candidate-k",
          "intervals": [{"opaque_interval_id": "interval-1", "visibility": "visible"}],
          "relations": [{"relation": "ordering", "preservation": "preserved"}],
          "answer_sufficiency": "answer_sufficient",
          "repair_proposals": [],
      }],
  }
  ```

  Reject full-video work whose `started_at` does not follow the bound
  frame-first freeze, mismatched adjudication/packet hashes, missing preferred-
  set intervals, any alternative-set rescue, invalid visibility,
  preservation, representation, or sufficiency values, repairs outside their
  interval, repairs not recorded to 0.1 seconds, and adjudication not bound to
  exactly two distinct frozen auditor hashes. Require a third reviewer and
  `round_name` equal to `initial` or `repair`.

- [ ] **Step 6: Implement full-video and third-auditor validators**

  Define exact frozen vocabularies:

  ```python
  CANDIDATES = ("oracle-hybrid-12", "oracle-local-12")
  VISIBILITY_VALUES = frozenset({"visible", "partial", "not_visible", "uncertain"})
  RELATION_VALUES = frozenset({"static_state", "motion", "ordering", "duration", "transition", "cooccurrence", "uncertain"})
  PRESERVATION_VALUES = frozenset({"preserved", "not_preserved", "not_applicable", "uncertain"})
  SUFFICIENCY_VALUES = frozenset({"answer_sufficient", "answer_insufficient", "uncertain"})
  STILL_REPRESENTATION_VALUES = frozenset({"still_frame_representable", "still_frame_incompatible", "uncertain"})
  ```

  Derive `candidate_passes` only when the exact preferred set has all-local
  `visible` components, every relationship frozen in scene semantics is
  `preserved`, and the complete candidate is `answer_sufficient`. `uncertain`
  never passes. Also compute raw independent-auditor agreement/disagreement
  counts before adjudication.

- [ ] **Step 7: Write failing selection and scoreboard tests**

  Cover these literal decisions:

  ```python
  @pytest.mark.parametrize(
      ("cohort", "hybrid", "local", "decision"),
      [
          (7, 7, 7, "stop_before_inference"),
          (8, 8, 8, "select_oracle_hybrid_12"),
          (8, 7, 8, "select_oracle_local_12"),
          (8, 7, 7, "stop_before_inference"),
      ],
  )
  def test_stage2_selection_is_predeclared(cohort, hybrid, local, decision):
      assert _selection(cohort, hybrid, local)["decision"] == decision
  ```

  Assert the scoreboard contains only aggregate counts, hashes, validity,
  decision, and all-ten denominators. It must reject or omit video IDs,
  questions, intervals, paths, candidate secrets, frame hashes, reviewer answer
  text, and repair timestamps.

- [ ] **Step 8: Implement selection and sanitized aggregation**

  Use the entire labelable, `still_frame_representable` cohort. Never create a
  favorable subset by dropping candidate failures. Include counts for all ten,
  eligibility, still incompatibility/uncertainty, per-candidate visible,
  relation-preserved, answer-sufficient, pass-after-repair, final pass, and the
  exact Stage 2 decision.

- [ ] **Step 9: Verify and commit the contract module**

  ```bash
  uv run pytest tests/test_relevant_scene_oracle.py tests/test_temporal_bottleneck.py -q
  uv run ruff check src/trec_video_2026/vqa_experiments/relevant_scene_oracle.py tests/test_relevant_scene_oracle.py
  git diff --check
  git add src/trec_video_2026/vqa_experiments/relevant_scene_oracle.py tests/test_relevant_scene_oracle.py
  git commit -m "feat: add relevant-scene audit contract"
  ```

### Task 3: Build the private Stage 1 orchestrator and frozen packet boundary

**Files:**
- Create, ignored/private:
  `data/vqa/benchmarks/temporal-relevant-scene-oracle-10/run_relevant_scene_stage1.py`
- Create, ignored/private:
  `data/vqa/benchmarks/temporal-relevant-scene-oracle-10/protocol.json`
- Create, ignored/private: SHA-256 sidecars, secret opaque maps, inventories,
  frame plans/manifests, packets, contact sheets, and validation records under
  the same run directory.

**Interfaces:**
- Consumes the immutable prior packet, frozen A/B/C annotations, their sidecars,
  the current tracked source hashes, and the canonical shared media.
- Provides CLI commands:
  `init`, `freeze-adjudication INPUT`, `probe-plan-extract`,
  `build-frame-first-packets`, `freeze-frame-first REVIEWER INPUT`,
  `freeze-scene-semantics INPUT`, `build-full-video-packets`,
  `freeze-full-audit REVIEWER INPUT`,
  `freeze-audit-adjudication ROUND INPUT`, `repair`, `select`, `scoreboard`, and
  `verify`.
- Every mutating command refuses overwrite. `verify` is read-only.

- [ ] **Step 1: Create the new run boundary and initialization command**

  Use `write_json_new`, `write_sidecar_new`, and `verify_sidecar` equivalents
  that open with exclusive creation. `init` must verify the data symlink,
  source sidecars, prior A/B/C annotation hashes, prior stop-decision hash, Git
  revision, tracked source hash, `ffprobe`/`ffmpeg`/ImageMagick identities, and
  write a protocol whose decision rules match the approved spec. It references
  prior artifacts by path and hash; it does not copy or mutate them.

- [ ] **Step 2: Implement and dry-check adjudication and semantics freeze**

  `freeze-adjudication` validates an all-ten `ROLE_ADJUDICATOR` file with
  `validate_annotation`, requires the three frozen annotation hashes in A/B/C
  order, writes it under `annotations/adjudicated.json`, and stops if fewer than
  eight rows are labelable. `freeze-scene-semantics` then freezes preferred-set
  components, roles, and required relationships before any frame plan exists.
  The dry-check uses a temporary synthetic packet and leaves the real run
  unchanged.

- [ ] **Step 3: Implement exact probe, plan, and extraction stages**

  For each labelable row, run `ffprobe` with
  `best_effort_timestamp_time`, stream index, and time base; bind the raw probe
  bytes to the source media hash; then construct `uniform-12`,
  `oracle-hybrid-12`, and `oracle-local-12` from the exact preferred-set
  planner. Extract by decoded frame index with `ffmpeg`, require one nonempty
  PNG per slot, and freeze actual PTS, stream and frame index, requested time,
  time base, source hash, plan hash, extraction command identity, and PNG hash.
  Do not build inference requests.

- [ ] **Step 4: Implement opaque frame-first packets and contact sheets**

  Generate CSPRNG opaque row/candidate IDs and reviewer-specific frozen
  presentation orders. Store the secret mapping separately with restrictive
  permissions. Resize each extracted frame once to the declared maximum-pixel
  representation, freeze that PNG as the later model-input candidate, and do
  not expose full-resolution extraction to auditors. For each candidate, render
  timestamp-labeled copies and a complete twelve-frame 4-by-3 contact sheet with
  ImageMagick. Packet paths and filenames are opaque and do not reveal hybrid/
  local identity. Frame-first
  packets include only opaque IDs, the private question, timestamps, and
  candidate contact sheets—no media path, adjudicated intervals, local/global
  marks, references, prior outcomes, or condition map.

- [ ] **Step 5: Enforce the phase boundary and full-video packets**

  Refuse `build-full-video-packets` until two distinct frame-first reviews
  validate and freeze. Full-video packets add the source-video path, opaque
  interval IDs, actual PTS, and candidate-local target sheets. They continue to
  omit references, model material, grades, condition mappings, and prior
  condition outcomes.

- [ ] **Step 6: Implement review freeze, repair, selection, and scoreboard commands**

  Validate each reviewer artifact through Task 2. Permit `repair` exactly once,
  only for `partial` or `not_visible` preferred-set component failures with an
  in-interval 0.1-second resolved event center. Apply one simultaneous
  replacement per failed interval; reject repairs for incompatibility,
  uncertainty, relationship failure, or answer insufficiency. Re-extract the
  nearest unused decoded frame inside the same interval with earlier-PTS tie
  break, regenerate opaque sheets, and bind all replaced hashes. `select` calls
  `build_stage2_selection_record`; `scoreboard` calls
  `build_stage1_scoreboard` and freezes a private detailed record plus the
  sanitized aggregate.

- [ ] **Step 7: Run private dry-check and freeze the script identity**

  ```bash
  python data/vqa/benchmarks/temporal-relevant-scene-oracle-10/run_relevant_scene_stage1.py verify --synthetic
  sha256sum data/vqa/benchmarks/temporal-relevant-scene-oracle-10/run_relevant_scene_stage1.py
  ```

  Record the script digest in `protocol.json`, verify the whole private run is
  ignored, and confirm `git status --short` shows no `data/` path.

### Task 4: Adjudicate all ten rows and extract both candidate inputs

**Files:**
- Create, ignored/private:
  `data/vqa/benchmarks/temporal-relevant-scene-oracle-10/annotations/adjudicated.json`
- Create, ignored/private: all frozen inventories, plans, PNGs, manifests,
  opaque maps, and initial frame-first packets.

**Interfaces:**
- Consumes only the blind packet and frozen A/B/C annotations during
  adjudication; references, predictions, grades, selector outcomes, and prior
  answer outcomes remain unavailable.
- Produces complete all-ten dispositions and two candidate inputs for every
  labelable row before any scene auditor starts.

- [ ] **Step 1: Initialize and verify the immutable inputs**

  Run `init`, then `verify`. Confirm the new protocol binds the old stop result
  as historical evidence while explicitly authorizing a separate exploratory
  adjudication; confirm no previous bundle changed.

- [ ] **Step 2: Freeze exploratory adjudication and scene semantics**

  A reviewer acting as adjudicator C inspects the frozen A/B/C annotations and
  full videos, selects eligibility and minimal sufficient sets without
  averaging boundaries, records all ten rows, and binds all three input hashes.
  The same pre-frame phase freezes preferred-set component roles and every
  required relationship record. Validate, sidecar, and freeze both outputs
  before frame probing. If fewer than eight rows are labelable, create the Stage
  1 stop record and move to Task 6.

- [ ] **Step 3: Probe decoded inventories and construct all plans**

  Run `probe-plan-extract` through the plan-freeze boundary first. Verify every
  eligible row has twelve unique identities in all three conditions and that
  every candidate-local frame remains inside its adjudicated interval.

- [ ] **Step 4: Extract frames and verify provenance**

  Complete extraction, recalculate every PNG hash, compare the extracted frame
  identity with the plan, and verify no duplicate PTS/hash appears within a
  condition. Freeze the manifest before building audit packets.

- [ ] **Step 5: Build two reviewer-specific frame-first packets**

  Verify each packet contains all labelable rows, two opaque candidates per
  row, chronological timestamp labels, and no full-video path, intervals,
  condition names, local/global marks, expected answers, or outcome material.

### Task 5: Run independent audits, one repair round, and freeze Stage 1

**Files:**
- Create, ignored/private: two frame-first review files, two full-video review
  files, third-party adjudications, optional repair artifacts, condition
  selection, detailed audit record, and sanitized `stage1-scoreboard.json`.

**Interfaces:**
- Consumes the opaque packets from Task 4 and the validation/decision functions
  from Task 2.
- Produces exactly one `select_oracle_hybrid_12`,
  `select_oracle_local_12`, or `stop_before_inference` decision bound to the
  all-ten audit evidence.

- [ ] **Step 1: Collect and freeze two frame-first audits**

  Dispatch two fresh auditors independently. Each sees only its frame-first
  packet, inspects candidates in frozen presentation order, and records a short
  answer or `not_answerable`, confidence, and available temporal relations.
  Freeze both files before either auditor receives a full-video packet.

- [ ] **Step 2: Collect and freeze two full-video audits**

  Give the same auditors their full-video packets only after Step 1 verifies.
  Each independently records question-level still representability,
  representation classes, interval visibility from local target sheets,
  relation preservation in the complete candidate, answer sufficiency, and
  repair proposals. Freeze both files before sharing them.

- [ ] **Step 3: Third-party adjudicate initial disagreements**

  A fresh third auditor sees both frozen full-video reviews and the permitted
  source material, but no candidate identity, references, model outputs, grades,
  or prior condition outcomes. It resolves every disagreement and freezes the
  initial aggregate.

- [ ] **Step 4: Execute at most one repair round**

  If the initial aggregate contains repairable failures, resolve one event-
  center time per failed interval, run `repair` once, rebuild only affected
  complete candidate packets, and have both original auditors repeat the
  candidate-blind affected-row audit. Freeze a third-party repair adjudication.
  If there are no repairable failures, freeze an explicit
  `repair_not_required` record.

- [ ] **Step 5: Freeze the selection and sanitized scoreboard**

  Run `select` and `scoreboard`. Verify the decision uses the entire
  labelable/still-representable cohort and exactly implements the hybrid-first,
  scene-only-fallback, otherwise-stop rule. Verify the public aggregate has no
  private identifiers or paths and reports all-ten denominators separately for
  extraction, representation, and auditor answer sufficiency, plus raw pre-
  adjudication agreement/disagreement counts for each audit dimension.

- [ ] **Step 6: Stop before Stage 2**

  Verify no inference request, compatibility prediction, model prediction,
  reference, model grade, Podman run log, or transcript artifact exists in the
  new run. Share the Stage 1 result before any Stage 2 planning or execution.

### Task 6: Record, render, independently verify, and hand off Stage 1

**Files:**
- Modify: `docs/experiments/2026-07-30-temporal-required-moment-oracle.md`
- Modify: `docs/project-status/research-visual-evidence.qmd`
- Modify: `docs/project-status/experiment-ledger.qmd`
- Regenerate only: sibling project-status Markdown and
  `reports/project-status/*.html`
- Refresh sanitized derived HTML only under
  `/home/npatta01/codex-rendered/plans/trecvid-vqa-current-state/`

**Interfaces:**
- Consumes only the sanitized scoreboard and immutable artifact hashes.
- Produces one condition-bound Stage 1 claim, one current-theme update, and a
  user-facing checkpoint. It does not report a model-answer result.

- [ ] **Step 1: Write the dated Stage 1 evidence record**

  Record the user correction, question, candidate definitions, panel and data
  boundary, adjudication method, frame-first/full-video audit order, extraction
  counts, representation counts, answer-sufficiency counts, repair count,
  still-incompatible count, selection/stop decision, completion denominator,
  evidence validity, limitations, immutable private artifact hashes, and next
  decision. Explicitly state that no GPU inference or transcript work ran.

- [ ] **Step 2: Update only the owning QMD sources**

  Add the result under Visual and Temporal Evidence, link it from the ledger,
  and keep uniform-12 as the operational baseline. Do not duplicate the full
  claim in Routing and do not hand-edit generated Markdown or HTML.

- [ ] **Step 3: Run focused and full verification**

  ```bash
  uv run pytest tests/test_temporal_bottleneck.py tests/test_relevant_scene_oracle.py -q
  uv run pytest -q
  uv run ruff check src/trec_video_2026/vqa_experiments/temporal_bottleneck.py src/trec_video_2026/vqa_experiments/relevant_scene_oracle.py tests/test_temporal_bottleneck.py tests/test_relevant_scene_oracle.py
  git diff --check
  ```

  Validate every private sidecar and binding, compare an isolated QMD
  regeneration, check links/fragments, scan tracked diffs and rendered output
  for private paths/questions/IDs/frames/answers/logs, and confirm no `data/`
  file is tracked.

- [ ] **Step 4: Independently review the scientific and privacy boundary**

  Require one reviewer to check code/contract conformance and another to check
  aggregate arithmetic, claim wording, privacy, and the no-inference/no-
  transcript boundary. Resolve every actionable finding and rerun affected
  checks.

- [ ] **Step 5: Render and verify the private portal copy**

  Regenerate sanitized HTML, update the existing portal index, verify the live
  HTTPS artifact returns the intended content, confirm Tailscale remains
  tailnet-only, and verify that no dataset, raw artifact, or unrelated file is
  reachable.

- [ ] **Step 6: Commit and hand off the Stage 1 result**

  Commit only tracked code, tests, authored report sources, and regenerated
  report outputs. Report the extraction and representation results separately,
  give the exact selected-treatment or stop decision, state that Stage 2 has
  not started, and stop before transcript work.
