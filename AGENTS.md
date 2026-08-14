# Repository agent instructions

## Adding or updating research

Before proposing, running, interpreting, or recording an experiment, read the
[Adding Experiments and Results guide](docs/project-status/adding-experiments.qmd).
Choose one primary theme for each current result; related themes should link to
that record rather than copy it. QMD is the editable report source: generated
Markdown and HTML must not be hand-edited.

## Shared local dataset

- The only physical dataset copy is
  `/home/npatta01/data/competitions/trec_video_2026/data`.
- Every linked Git worktree must access it through an ignored `data` symlink.
- Before work that needs the dataset, verify that `data` resolves to the
  canonical directory above. If it is missing, run:

  ```bash
  /home/npatta01/data/competitions/trec_video_2026/scripts/link-shared-data.sh --all
  ```

- The helper is idempotent. If it reports a conflicting path, stop and report
  the conflict; never remove or overwrite that path automatically.
- Never copy, move, delete, stage, or commit dataset contents.

## Full-panel VQA artifact reference

The completed 2026-08-10 full-panel AV event-map experiment is documented in
[`docs/experiments/2026-08-10-vqa-plaintext-full-panel.qmd`](docs/experiments/2026-08-10-vqa-plaintext-full-panel.qmd).
Its compact, reviewable evidence-of-record files are under
[`docs/experiments/artifacts/vqa-plaintext-full-20260810/`](docs/experiments/artifacts/vqa-plaintext-full-20260810/):

After this branch is merged to `main`, the same repository-relative links are
available from the shared checkout at
`/home/npatta01/data/competitions/trec_video_2026/`:

- report: `/home/npatta01/data/competitions/trec_video_2026/docs/experiments/2026-08-10-vqa-plaintext-full-panel.qmd`;
- artifact bundle: `/home/npatta01/data/competitions/trec_video_2026/docs/experiments/artifacts/vqa-plaintext-full-20260810/`.

Do not copy the tracked bundle manually between worktrees; merging the branch
to `main` makes it available in the shared checkout. The raw inspection media
is intentionally worktree-local and ignored by Git; use the local bundle in
the producing worktree or regenerate/link it separately if shared raw-media
inspection is required.

- `summary.json`, `correction.json`, `maps.jsonl`, and `results.jsonl` preserve
  the complete 475-row timeline/selector run and hashes.
- `selector-quality.json` contains the post-freeze METEOR, BERTScore-F1, STS,
  and explicitly degenerate NDCG summary.
- `selected-evidence/evidence.json`, `manifest.json`,
  `requests.frames.json`, and `requests.whisper.json` preserve selected
  timestamps, decoded PTS provenance, hashes, and the two answer-input arms.
- The local `selected-evidence/frames/`, `audio/`, and `contact-sheets/`
  directories are the complete private inspection bundle (5,700 JPEG frames,
  475 WAV files, and 475 contact sheets). They are intentionally ignored by
  Git because they are private derived media and total roughly 782 MB; do not
  delete them when inspecting this run. The manifests and hashes remain the
  audit record.

Do not reprovision dstack to recover this run. Inference is complete and the
remote development pod was stopped. If the local inspection bundle is absent,
classify that as an audit gap and use the tracked manifests rather than
silently rerunning or changing the experiment.

## Frontier Gemini canary artifact reference

The completed 2026-08-11 two-video native-AV Gemini event-map canary is
documented in
[`docs/experiments/2026-08-11-frontier-video-map-diagnostic.qmd`](docs/experiments/2026-08-11-frontier-video-map-diagnostic.qmd).
Its sanitized evidence-of-record is under
[`docs/experiments/artifacts/frontier-video-map-20260811/`](docs/experiments/artifacts/frontier-video-map-20260811/).

The complete private artifacts are already in the canonical shared dataset and
are accessible from every correctly linked checkout. Do not copy them between
worktrees:

- first frozen Gemini response and provenance:
  `/home/npatta01/data/competitions/trec_video_2026/data/vqa/benchmarks/frontier-video-map-20260811/run-webm-v5-observable/`;
- second frozen Gemini response and provenance:
  `/home/npatta01/data/competitions/trec_video_2026/data/vqa/benchmarks/frontier-video-map-20260811/run-webm-v6-second/`;
- combined two-row audit with overview sheets, 15 timestamped event frames,
  maps, and private post-freeze metrics:
  `/home/npatta01/data/competitions/trec_video_2026/data/vqa/benchmarks/frontier-video-map-20260811/canary-audit-v1/`.

The first provider result is terminally recorded as `invalid_json`, but its raw
paid response is preserved and passes the strict event-map validator after
removing one exact outer Markdown JSON fence. Do not silently relabel or rerun
it. The second result passed end to end with the deterministic envelope parser.
Both original MKV sources were losslessly stream-copied to WebM; delivery and
source packet hashes prove that the audio/video elementary streams were
unchanged.

The private audit packet validates but is not reviewer-frozen: its empty verdict
files are placeholders, and the report's qualitative inspection must not be
misrepresented as a completed blinded audit. The two-row metrics are diagnostic,
single-answer NDCG is degenerate, and no independent audio check was performed.
Reka did not run because the supplied authenticated credential returned HTTP
403. Do not retry it without a new credential or confirmed account permission.

## Frontier scene-map boundary-verifier artifact reference

The completed 2026-08-12 question-blind Gemini scene-map and boundary-verifier
diagnostic is documented in
[`docs/experiments/2026-08-12-vqa-frontier-boundary-verifier.qmd`](docs/experiments/2026-08-12-vqa-frontier-boundary-verifier.qmd).
Its sanitized tracked index is under
[`docs/experiments/artifacts/vqa-frontier-boundary-verifier-20260812/`](docs/experiments/artifacts/vqa-frontier-boundary-verifier-20260812/).

After this branch is merged to `main`, those tracked records are available from
the shared checkout at:

- report: `/home/npatta01/data/competitions/trec_video_2026/docs/experiments/2026-08-12-vqa-frontier-boundary-verifier.qmd`;
- sanitized index: `/home/npatta01/data/competitions/trec_video_2026/docs/experiments/artifacts/vqa-frontier-boundary-verifier-20260812/`.

The complete private evidence is already in the canonical shared dataset at
`/home/npatta01/data/competitions/trec_video_2026/data/vqa/benchmarks/vqa-frontier-phase0-20260812/`.
Do not copy, stage, or publish that directory. In particular:

- `run-v5/` preserves the final one-pass question-blind annotation draft;
- `pilot-audit-v5/`, additive hardened `pilot-audit-v5-v2/`, and
  `pilot-audit-v5-manual-review/` preserve the frozen `revise_contract`
  decision, raw-response and reviewer-row bindings, and dense 4-fps review
  sheets;
- `verifier-v5/` preserves the sole paid verifier call, raw response, result,
  additive failed-terminal freeze, diagnostic audit, and diagnostic freeze;
- `verifier-v5-pre-review/` and `verifier-v5-pre-freeze-fix/` are preserved
  offline preparations and did not issue paid calls.
- `direct-gemini-timeline-v1/` preserves the direct Google API follow-up. It
  recovered complete event coverage but its answer stage missed the labeled
  temporal granularity.

The tested OpenRouter same-model draft-plus-verifier mechanism has terminal
decision `stop_two_stage`; do not rerun or scale it. Do not generalize that
result to Gemini native-video capability: the direct Google follow-up recovered
the complete sequence. The next test is a small category-balanced direct-API
annotation sample with answering evaluated separately over frozen timelines.

That seven-category test is now complete. Its private evidence-of-record is:

`/home/npatta01/data/competitions/trec_video_2026/data/vqa/benchmarks/direct-gemini-evidence-canary-v1/`

Start with `inspection.md`; it links the complete timelines, selected spans,
timestamped frames/contact sheets, native AV deliveries, three frozen answer
paths, and post-freeze references/metrics. Do not rerun Gemini to recover this
result. Direct native AV was the strongest tested answer path; the derived
timeline-only and timeline-plus-frame paths were not promoted.

Two matched follow-ups on those seven rows are complete at:

- `/home/npatta01/data/competitions/trec_video_2026/data/vqa/benchmarks/direct-gemini-scope-control-v1/`;
- `/home/npatta01/data/competitions/trec_video_2026/data/vqa/benchmarks/direct-gemini-whole-video-synthesis-v1/`.

The answer-only scope control and the synthesis-plus-map answer treatment both
failed their promotion gates. The C synthesis itself correctly represented the
global repeated-diving structure and all seven syntheses passed deterministic
timestamp grounding, but final answering still missed temporal semantics and
exact speech. Preserve both roots; do not rerun them. Start with C's
`inspection.md` or the tailnet-only Gemini evidence inspector. The next test is
question-driven temporal/speech verification over frozen global synthesis,
not another global prompt rewrite and not a 475-row scale-up.

## Gemini native-AV 475-row artifact reference

The completed 2026-08-13 Gemini 3.1 Pro native-AV Batch benchmark is documented
in the full-panel section of
[`docs/experiments/2026-08-12-vqa-frontier-boundary-verifier.qmd`](docs/experiments/2026-08-12-vqa-frontier-boundary-verifier.qmd).
Its sanitized index is under
[`docs/experiments/artifacts/vqa-gemini-native-av-full-panel-20260813/`](docs/experiments/artifacts/vqa-gemini-native-av-full-panel-20260813/).

The complete private evidence is already in the canonical shared dataset at
`/home/npatta01/data/competitions/trec_video_2026/data/vqa/benchmarks/direct-gemini-batch-475-v1/`.
Do not copy, stage, publish, or rerun it. Start with `predictions-freeze.json`,
`postfreeze-score.json`, `maps.question-blind.json`, and
`failure-audit-v1/manual-review.json`. One initial direct-answer provider error
was retried exactly once and passed; both terminal snapshots are preserved.
The final panel has 475/475 direct answers and 475/475 question-blind maps.

Gemini's semantic composite is `0.5046` versus `0.4962` for the matched
Qwen+Whisper baseline; the paired `+0.0083` interval crosses zero, and counting
regresses. The hard-row audit shows both answer-stage failures despite useful
maps and map omissions on fine counting/detail. The next experiment is a small
frozen question-driven retrieval and targeted-verification test—not another
global prompt rewrite or full-panel Gemini run.

## Question-driven atomic counting evidence artifact reference

The completed 2026-08-14 issue #34 atomic counting canary is documented in the
counting section of
[`docs/experiments/2026-08-12-vqa-frontier-boundary-verifier.qmd`](docs/experiments/2026-08-12-vqa-frontier-boundary-verifier.qmd).
Its sanitized index is under
[`docs/experiments/artifacts/vqa-gemini-counting-evidence-20260814/`](docs/experiments/artifacts/vqa-gemini-counting-evidence-20260814/).

The complete private evidence is already in the canonical shared dataset at
`/home/npatta01/data/competitions/trec_video_2026/data/vqa/benchmarks/direct-gemini-counting-evidence-v1/`.
Start with `inspection.md`. The corrected census is 38 correct, 26 confirmed
wrong, one ambiguous, and one not-count row; the frozen panel has those 26
failures, the ambiguity, and 12 controls. The valid OpenRouter-delivered canary
rescued 1/5 failures and preserved 1/1 control at `$0.253924`; it failed the
3/5 gate, and projected scale also exceeded the `$3` ceiling. Do not rerun or
scale the unchanged generic atomic enumerator. The zero-cost next seam is a
frozen CountSpec plus typed deterministic reducers over already-purchased
records; any new paid micro-canary requires separate authorization.
