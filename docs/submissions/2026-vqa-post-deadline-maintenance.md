# TRECVID 2026 VQA Post-Deadline Maintenance

## Closure status

The 2026 Answer Generation submission cycle is closed. Evalbase reported
server-side validation success for all four tracked run tags:

| Run tag | Treatment | Tracked upload SHA-256 |
|---|---|---|
| `q35_4b_u12` | Qwen3.5-4B uniform-12 control | `c01873efb9a5f5cf1090368f1bed4fd91b6ecbf51d806f8670ecabc3d82322a7` |
| `g37_q36_fusion` | Gemini 3.7 Flash and Qwen3.6 multiview fusion | `f3ee540c4e865a45973f8f8886363a8501f6d6a58861eb12dfa58986efd147bb` |
| `q36_dualview` | Qwen3.6 transcript/native dual view | `d8ccd445dd1cb3dd7cbd54454c012184f9fe1a21d3187fb7b068e09c90fc6a1a` |
| `g37_grouped_unique` | Grouped grounded rewrites | `d6e3731f19e4dca575815afcd4ed1b163126936b8bb1bbacfd8c675e4473741d` |

The tracked files under `submissions/trecvid-2026/vqa-ag/` are immutable
upload records. Changing an answer requires a new run tag and a new local and
organizer validation cycle. Historical executed-source hashes and frozen
artifact hashes must not be rewritten to match a later integration commit.

PR [#43](https://github.com/npatta01/trec_video_2026/pull/43) integrated the
deadline work through merge commit
`215060ae06eeb0f00f611807b65aace35de92f82`. The individual run records remain
the authoritative source for methods, hashes, and upload notes.

## What is maintained

The supported entry points remain the CLIs listed in the repository
[README](../../README.md#supported-cli). Deadline-specific runners,
assemblers, rescue utilities, grouped-rewrite tools, and dstack configurations
remain tracked as historical reproducibility tooling. Their presence does not
make them supported general-purpose interfaces and does not authorize another
paid or GPU run.

These historical files are deliberately retained because they are connected
by imports, focused tests, frozen request and provenance contracts, dated
implementation plans, and experiment records. Deleting or moving a single
runner can break reconstruction of the submitted artifacts even when that
runner is not part of the supported CLI table.

## Future cleanup boundary

A later retirement PR may archive deadline-only tooling only after it:

1. inventories imports, tests, plans, reports, and configuration references;
2. preserves the exact tracked submission files and their run records;
3. preserves private frozen evidence in place and never stages dataset files;
4. replaces live references with explicit archival links where needed;
5. passes the focused submission, assembly, rescue, and grouped-rewrite tests;
6. verifies that no historical executed-source or artifact hash changed; and
7. keeps reusable baseline code separate from one-off deadline policy.

No runner, configuration, frozen artifact, or private evidence is retired by
this documentation-only cleanup.

## Test-health boundary

The submission closure is covered by focused validator, assembler, rescue,
and grouped-rewrite tests. Repository-wide test health is a separate
maintenance concern. A fresh post-merge run of
`uv run --extra gemini pytest -q` completed with 2,474 passed, one skipped, and
21 failed tests. The failures were outside the submission workflow:

- one dstack source-archive size assertion;
- one optional NLTK import in the Nemotron scoring script; and
- 19 downloader tests whose temporary runtime fixtures do not satisfy the
  newer secure-directory precondition.

The 105 focused submission tests passed separately. A future test-hygiene PR
should fix or update these three baseline categories without coupling them to
the four validated submission records.

## Maintainer checklist

- Start with the [submission ledger](README.md) and individual run record.
- Verify the tracked upload SHA-256 before copying or inspecting a submission.
- Treat `data/` as private, ignored evidence and never stage its contents.
- Do not rerun paid or GPU workflows solely to reconstruct an already frozen
  result.
- Use a new branch, run tag, and evidence root for any future experiment.
