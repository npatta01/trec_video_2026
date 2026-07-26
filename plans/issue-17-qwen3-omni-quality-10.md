# Qwen3-Omni Stratified Quality-10 Implementation Plan

> **Historical status:** The referenced PR is merged and this quality work is
> complete. The final neutral A2 run produced 19/20 contract-valid outputs, so
> that run was ineligible for its predeclared promotion decision. A repaired
> comparison can reconsider Qwen3-Omni. See
> [Project Status](../docs/project-status/index.md).

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run and evaluate a paired 10-video Qwen3-Omni native-audio ablation
without exceeding the cumulative `$10` Qwen budget.

**Architecture:** A label-free contract builder resolves ten fixed video IDs
to the ignored shared dataset and emits immutable path/file hashes. The
quality runner reuses the proven a4 model/generation code but executes paired
`frames-only` and `frames-native-audio` conditions with condition-specific
modality attestation. A separate scorer joins annotations only after prediction
freeze and emits a sanitized, reproducible report.

**Tech Stack:** Python 3.12 locally, pinned Qwen3-Omni CUDA container,
PyTorch/Transformers in the image, dstack task orchestration, private Hugging
Face Bucket artifacts, pytest, Ruff, Bash, GitHub draft PR.

## Global Constraints

- Base changes on commit `d6773437125822441b82076da4ea3da4664c6e2b`.
- Preserve model revision
  `2f443cfc4c54b14a815c0e2bb9a9d6cbcd9a748b`.
- Preserve runtime image digest
  `sha256:71896cf6533abc610ac298f26289ff3a4ea4bd1fc835f370ece9e5c31e3181d8`.
- Preserve EOS ID `151645`, padding ID `151643`, fixed seed `20260724`,
  checkpoint sampling, and the strict terminal-JSON parser.
- Use exactly 12 frozen frames for each video in both conditions.
- Supply exactly one audio element only in `frames-native-audio`.
- Keep labels, distractors, transcripts, and prior answers outside inference.
- Do not commit dataset contents, media, raw logs, raw predictions, model
  weights, credentials, or reference text.
- Use immutable run `qwen3-omni-quality-10-20260725-a1` and private remote
  prefix
  `hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/qwen3-omni-quality-10-20260725-a1`.
- Keep dstack at `$4.50/hour` for at most one hour; cumulative worst case is
  `$7.1269`, below the approved `$10` hard cap.
- Do not submit while the dstack server is unreachable.
- Preview first and submit the immutable run exactly once.

---

### Task 1: Label-free panel and paired-condition contract

**Files:**

- Create:
  `src/trec_video_2026/vqa_experiments/qwen3_omni_quality_contract.py`
- Create: `tests/test_qwen3_omni_quality_contract.py`
- Generate ignored:
  `data/vqa/benchmarks/qwen3-omni-quality-10/contract.json`

**Interfaces:**

- Consumes: ignored frozen30 manifest, frozen 12-frame request file, WAV files,
  and the ten fixed IDs from the design.
- Produces:
  `build_contract(data_root: Path) -> dict[str, Any]`,
  `validate_quality_contract(contract: Mapping[str, Any], data_root: Path,
  *, verify_files: bool) -> None`, and
  `quality_contract_sha256(contract: Mapping[str, Any]) -> str`.

- [ ] **Step 1: Write failing identity, schema, label-exclusion, and hash tests**

```python
def test_quality_contract_has_ten_examples_and_two_conditions():
    contract = build_contract(Path("data"))
    assert [row["video_id"] for row in contract["examples"]] == list(VIDEO_IDS)
    assert contract["conditions"] == [
        {"condition_id": "frames-only", "audio_enabled": False},
        {"condition_id": "frames-native-audio", "audio_enabled": True},
    ]
    assert all(len(row["frame_paths"]) == 12 for row in contract["examples"])


def test_quality_contract_is_label_free():
    serialized = json.dumps(build_contract(Path("data"))).lower()
    assert "correct_answer" not in serialized
    assert "reference" not in serialized
    assert "distractor" not in serialized
    assert "transcript" not in serialized
```

- [ ] **Step 2: Run focused tests and verify they fail**

Run:
`uv run pytest -q tests/test_qwen3_omni_quality_contract.py`

Expected: collection fails because the contract module does not exist.

- [ ] **Step 3: Implement the fixed ten-row builder and fail-closed validator**

The fixed order is:

```python
VIDEO_IDS = (
    "Mi-NpnCBegw",
    "NN7G28sBOlY",
    "NUmb2P3Bzu8",
    "NkXPPJoZ-h8",
    "IWSPAfCKtV8",
    "CH8Dp7EgW4I",
    "EtNOkUJd5w8",
    "Gp4LVTn3YbY",
    "EMI_anpHII4",
    "LusRurRrmf0",
)
```

Resolve questions and evidence recipes from ignored files, override only the
horse question with the approved explicit wording, convert every path to a
safe `vqa/...` relative path, hash the contract inputs, and reject absolute or
parent-traversing paths. Do not read annotations.

- [ ] **Step 4: Generate and independently inspect the ignored contract**

Run:

```bash
uv run python -m trec_video_2026.vqa_experiments.qwen3_omni_quality_contract \
  --data-root data \
  --output data/vqa/benchmarks/qwen3-omni-quality-10/contract.json
```

Expected: one contract with 10 examples, 120 frame hashes, 10 audio hashes,
two conditions, no label-bearing keys, and a printed canonical digest.

- [ ] **Step 5: Run focused tests and commit**

Run:
`uv run pytest -q tests/test_qwen3_omni_quality_contract.py`

Commit:
`git commit -m "feat: freeze Qwen3 Omni quality panel"`

### Task 2: Condition-aware Qwen generation and quality runner

**Files:**

- Modify:
  `src/trec_video_2026/vqa_experiments/qwen3_omni_dstack.py`
- Create:
  `src/trec_video_2026/vqa_experiments/qwen3_omni_quality_dstack.py`
- Modify: `tests/test_qwen3_omni_dstack_config.py`
- Create: `tests/test_qwen3_omni_quality_dstack.py`

**Interfaces:**

- Consumes: Task 1 contract and the existing a4
  `generation_special_token_ids`, `_capture_first_forward`,
  `normalize_generation_output`, and runtime loader.
- Produces:
  `build_messages(example, data_root, *, audio_enabled: bool = True)`,
  `attest_qwen_inputs(..., audio_enabled: bool = True)`,
  `_generate_one(..., audio_enabled: bool = True)`, and
  `run_quality_probe(...) -> dict[str, Any]`.

- [x] **Step 1: Add failing audio-off message and tensor-attestation tests**

```python
def test_frames_only_message_and_attestation_omit_audio():
    content = build_messages(
        example(), Path("/dstack/input"), audio_enabled=False
    )[0]["content"]
    assert [item["type"] for item in content].count("image") == 12
    assert [item["type"] for item in content].count("audio") == 0

    attestation = attest_qwen_inputs(
        frames_only_inputs,
        first_forward_input_keys=set(frames_only_inputs),
        audio_enabled=False,
    )
    assert attestation["audio_element_count"] == 0
    assert attestation["audio_input_keys"] == []
```

- [x] **Step 2: Run focused tests and verify failure**

Run:

```bash
uv run pytest -q \
  tests/test_qwen3_omni_dstack_config.py \
  tests/test_qwen3_omni_quality_dstack.py
```

Expected: failure because shared generation functions do not accept
`audio_enabled`.

- [x] **Step 3: Generalize only the modality seam**

Keep `audio_enabled=True` as the compatibility default. With audio disabled,
omit the message audio element and require audio tensors to be absent. With
audio enabled, preserve the exact a4 requirements. Do not alter EOS, padding,
sampling, seed, output normalization, model load, dtype, or strict parser.

- [x] **Step 4: Implement the paired quality runner**

For each example, run `frames-only` then `frames-native-audio`. Add
`condition_id`, `audio_enabled`, `pair_index`, and condition-specific request
hash to every record. Stop after the first pair unless both canary calls pass
the modality, EOS, and parsing contract. Continue writing predictions after
every later call so failures remain durable.

- [x] **Step 5: Test compatibility defaults and paired failure behavior**

Run:

```bash
uv run pytest -q \
  tests/test_qwen3_omni_dstack_config.py \
  tests/test_qwen3_omni_quality_dstack.py
```

Expected: both the unchanged a4 compatibility suite and new quality suite pass.

- [x] **Step 6: Commit**

Commit:
`git commit -m "feat: add paired Qwen3 Omni quality runner"`

### Task 3: Immutable dstack wrapper, config, and collector

**Files:**

- Create: `dstack/qwen3-omni-quality-10.dstack.yml`
- Create: `scripts/run-qwen3-omni-quality-dstack.sh`
- Create: `scripts/collect-qwen3-omni-quality-dstack.sh`
- Create: `tests/test_qwen3_omni_quality_dstack_config.py`

**Interfaces:**

- Consumes: Task 1 ignored contract and 130 exact evidence files; Task 2 module.
- Produces: one immutable private remote run and one ignored local artifact
  directory.

- [x] **Step 1: Write failing static configuration and wrapper tests**

Assert the exact run ID/prefix, image/model revisions, 120 PNG mappings, 10 WAV
mappings, no ports, two providers, on-demand policy, 90+ GB VRAM, 192+ GB RAM,
one-hour duration, `$4.50/hour` cap, bounded `no-capacity` retry, absolute
interpreter, pinned `uv`/HF CLI, empty-prefix check, partial-failure
finalization, foreground workload, and no `--delete`.

- [x] **Step 2: Run focused tests and verify failure**

Run:
`uv run pytest -q tests/test_qwen3_omni_quality_dstack_config.py`

Expected: failure because the quality dstack files do not exist.

- [x] **Step 3: Implement the wrapper and config**

Use:

```yaml
type: task
name: qwen3-omni-quality-10-20260725-a1
max_price: 4.50
max_duration: 1h
retry:
  on_events: [no-capacity]
  duration: 30m
```

Map only the runner source, contract, 119 direct PNGs, three pinned transport
chunks for the sole over-limit PNG, and 10 exact WAVs. Reconstruct the original
PNG byte-for-byte before validation. Bind the verified `/usr/bin/python3`, keep
Python foregrounded, sync partial results in finalization, and preserve
workload failure over sync success.

- [x] **Step 4: Implement the collector**

Accept only `qwen3-omni-quality-10-20260725-a1`. Download the exact private
prefix into
`data/vqa/benchmarks/qwen3-omni-quality-10/<run-id>`, verify remote
`SHA256SUMS`, sanitize only allowlisted dstack metadata, and write/verify a
local artifact-hash manifest.

- [x] **Step 5: Run static verification and commit**

Run:

```bash
uv run pytest -q tests/test_qwen3_omni_quality_dstack_config.py
bash -n scripts/run-qwen3-omni-quality-dstack.sh
bash -n scripts/collect-qwen3-omni-quality-dstack.sh
```

Commit:
`git commit -m "feat: add bounded Qwen3 Omni quality task"`

### Task 4: Pre-run freeze, review, and submission gate

**Files:**

- Modify: `plans/issue-17-qwen3-omni-quality-10.md`

**Interfaces:**

- Consumes: Tasks 1–3.
- Produces: reviewed commit, empty namespaces, acceptable declined preview, or
  an explicit blocker.

- [x] **Step 1: Run all cheap verification**

Run:

```bash
uv run pytest -q
uv run ruff check .
bash -n scripts/run-qwen3-omni-quality-dstack.sh
bash -n scripts/collect-qwen3-omni-quality-dstack.sh
git diff --check
```

- [x] **Step 2: Verify data and secret boundaries**

Confirm the worktree `data` symlink resolves to the canonical dataset, no
dataset file is tracked, the contract is label-free, only intended input files
are mapped, tracked files contain no token value, and both local/remote run
prefixes are empty.

- [x] **Step 3: Obtain independent standards and specification review**

Reviewers must check the paired condition semantics, canary stop, a4
non-regression, strict parsing, budget arithmetic, immutable paths, artifact
durability, secret exposure, and test coverage. Fix all Critical or Important
findings and re-review.

- [x] **Step 4: Preview without submission**

Run:

```bash
echo "n" | dstack apply --no-repo \
  -f dstack/qwen3-omni-quality-10.dstack.yml
```

Do not submit if the server is unreachable, validation fails, no offers exist,
or the output lacks at least two distinct eligible offers when practical.

- [x] **Step 5: Commit the reviewed pre-run state**

Commit:
`git commit -m "chore: freeze Qwen3 Omni quality run"`

### Task 5: Execute once and collect immutable artifacts

**Files:**

- Generate ignored:
  `data/vqa/benchmarks/qwen3-omni-quality-10/qwen3-omni-quality-10-20260725-a1/`

**Interfaces:**

- Consumes: approved Task 4 preview.
- Produces: terminal dstack record plus checksum-verified private artifacts.

- [x] **Step 1: Submit once**

Run:

```bash
dstack apply --no-repo \
  -f dstack/qwen3-omni-quality-10.dstack.yml -y -d
```

- [x] **Step 2: Verify the unique run and monitor**

Use `dstack ps -v`, `dstack run get <run-id> --json` through an allowlisted
projection, and normal `dstack logs`. Never print diagnostic secret-bearing
environment output.

- [x] **Step 3: Collect after terminal state**

Run:
`bash scripts/collect-qwen3-omni-quality-dstack.sh`

- [x] **Step 4: Verify artifact counts and hashes**

Require terminal status, workload return code, exact cost, 20 prediction
records unless the canary stopped the run, remote file count, remote
`SHA256SUMS`, local artifact hashes, executed runner hash, model/image IDs, and
contract digest.

### Task 6: Score, review, and publish findings

**Files:**

- Create:
  `src/trec_video_2026/vqa_experiments/qwen3_omni_quality_scoring.py`
- Create: `tests/test_qwen3_omni_quality_scoring.py`
- Create:
  `docs/experiments/2026-07-25-qwen3-omni-quality-10.md`
- Create:
  `docs/experiments/artifacts/qwen3-omni-quality-10-score.json`
- Modify: `plans/issue-17-qwen3-omni-quality-10.md`

**Interfaces:**

- Consumes: frozen ignored predictions and ignored annotations.
- Produces: sanitized score artifact, report, reviewed branch, and draft PR.

- [x] **Step 1: Write failing scoring tests**

Test prediction-hash binding, annotation-path/reference-hash binding, no
reference text in tracked output, numeric equivalence for `1`/`one`, paired
condition joins, aggregate exact/F1, C/P/W placeholders that must be filled
from answer-only adjudication, and refusal to score incomplete/unfrozen input.

- [x] **Step 2: Implement and run the scorer after freeze**

The scorer may read ignored annotations only in this phase. Store model
answers, condition IDs, annotation lookup paths, reference hashes, numeric
metrics, manual judgments, scorer source hash, and aggregate paired deltas.
Never store reference text.

- [x] **Step 3: Write the evidence-backed report**

Separate verified facts, manual judgments, inferences, and unknowns. Disclose
the horse prompt repair, temporal evidence-coverage limitations, small-panel
size, any non-blinded adjudication, exact runtime/cost, and whether the
predeclared issue #17 gate passed.

- [x] **Step 4: Reproduce scores and run final verification**

Run:

```bash
uv run python -m \
  trec_video_2026.vqa_experiments.qwen3_omni_quality_scoring --verify
uv run pytest -q
uv run ruff check .
git diff --check
```

- [x] **Step 5: Obtain independent post-run findings and code review**

Fix all Critical or Important findings, rerun verification, and record a clean
review result.

- [x] **Step 6: Commit, push, and open a draft PR**

Commit:
`git commit -m "docs: report Qwen3 Omni quality benchmark"`

Push branch `codex/issue-17-qwen3-omni-quality-10` and open a draft PR based on
`codex/issue-5-qwen3-omni-dstack`. Update issue #17 with the exact decision,
cost, hashes, verification, report, and PR links.

### Task 7: Neutral-prompt paired correction

Repeat only the same frozen ten paired examples with one identical,
modality-agnostic instruction in both conditions. This removes the a1
audio-expecting treatment-context mismatch without changing evidence, model,
runtime, decoding, or scoring.

- Run ID: `qwen3-omni-quality-10-neutral-20260725-a2`
- Private prefix:
  `hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/qwen3-omni-quality-10-neutral-20260725-a2`
- Incremental worst-case spend: `$4.50`
- Cumulative worst case: `$4.5878 + $4.50 = $9.0878`, below the approved
  `$10` cap.

- [x] Add a neutral instruction profile and tests proving both conditions use
  identical text while only the audio element/tensors differ.
- [x] Add immutable a2 dstack config/wrapper/collector coverage while
  preserving the a1 configuration history and artifacts.
- [x] Run the full suite, Ruff, Bash, diff, data/secret/namespace checks, and
  independent specification/standards review.
- [x] Decline a preview and require at least two eligible on-demand offers
  when practical.
- [x] Submit a2 exactly once with `--no-repo`; the canary passed and artifacts
  were checksum-verified, but strict completion failed at 19/20 because one
  answer used a terminal Markdown JSON fence.
- [x] Score with the same frozen repository metric and answer-only C/P/W
  rubric, compare a2 against a1, update the report, and re-review.

## Current status

- [x] User approved the 10-video Qwen3-Omni quality test.
- [x] Issue #17 created.
- [x] Isolated worktree created from issue #5 and canonical data symlink
  verified.
- [x] Clean baseline: 187 tests passed.
- [x] Design documented and committed.
- [x] Task 1 complete: label-free contract digest
  `df016767b1a072770149fdc06fa47dfe7f17930b6c3145af4e92f55d58f4ddeb`;
  21 focused and 208 full tests passed; task review clean after two fix
  rounds.
- [x] Task 2 complete: paired runner; 24 focused and 211 full tests passed;
  task review clean.
- [x] Task 3 complete: bounded dstack task and collector; task review clean
  after one fix round.
- [x] Task 4 complete: 225 full tests, Ruff, Bash, and diff checks pass;
  standards and specification reviews clean after four fix rounds. Declined
  preview showed 7 eligible on-demand offers, including three distinct H200
  offers at `$4.39/hour`; no run was submitted.
- [x] Task 5 complete: immutable run finished `done` with 20/20 valid calls
  on one on-demand H200; cost `$1.9609`, elapsed `1608.033806` seconds;
  checksum-verified private artifacts collected.
- [x] Task 6 scoring/report complete and independently reviewed; a1 promotion
  gate failed.
- [x] Task 7 complete: a2 strict result failed at 19/20; pinned post-hoc
  semantic diagnostic is `3C/3P/4W` frames-only versus `5C/2P/3W` audio-on;
  independent semantic and code reviews passed.
- [x] Draft PR #18 opened against `codex/issue-5-qwen3-omni-dstack`.

## Next action

Review draft PR #18. Before any broader benchmark, harden terminal JSON
extraction or use constrained decoding and validate the correction with the
single failed A2 sentinel row.
