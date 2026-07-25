# NIST VQA scorer reuse audit — 2026-07-20

## Bottom line

**Reuse it, but do not treat the current evaluator as the final 2026 scorer.** Pin the current repository revision, use its new Answer Generation (AG) JSON validator for submission-format checks, and use the 2025 CSV evaluators only as a historical-parity oracle on labeled development data. Put a small adapter and safety boundary around them rather than importing or copying the scripts directly.

Audited revision: [`e06cb7854d7defe53e86412c2bba0ca5f636ea40`](https://github.com/gawad-nist/VQA/tree/e06cb7854d7defe53e86412c2bba0ca5f636ea40), committed 2026-07-17. This is the current `main` revision at the audit date.

## What is current and what is not

| Component | Status at the pinned revision | Appropriate use now |
| --- | --- | --- |
| `validate.ag.run.py` | Added 2026-07-17; validates the 2026 AG JSON shape and exactly ten answers | Yes: pre-submission validation |
| `eval.vqa.ag.py` | Explicitly documented as **2025**; consumes old CSV runs and does not report 2026 top-1/top-5/top-10 cutoffs | Historical/local development parity only |
| `eval.vqa.mc.py` | Explicitly documented as **2025**; consumes old CSV runs | Historical/local development parity only |
| `calc.sts.pl` | 2025 STS helper calling the UMBC service over HTTP | Avoid in a normal in-process path; isolate if exact parity is required |

The repository README still says it is based on the 2025 guidelines and that scoring must be updated for 2026 ([README lines 4–10](https://github.com/gawad-nist/VQA/blob/e06cb7854d7defe53e86412c2bba0ca5f636ea40/README.md#L4-L10)). NIST's 2026 page likewise says the scoring code “will be updated,” while specifying AG reporting at top 1, 5, and 10 ([2026 evaluation](https://www-nlpir.nist.gov/projects/tv2026/vqa.html#Evaluation_and_Metrics)). Therefore the evaluator is not yet the authoritative final 2026 implementation. The new validator is nonetheless aligned with the published 2026 AG JSON contract, which requires ten ranked answers ([2026 run format](https://www-nlpir.nist.gov/projects/tv2026/vqa.html#Run_Submission_Format:)).

One useful side effect: the repository includes a 2,000-row `vqa.gt.csv` with 2025 `QueryID, VideoID, Question, CorrectAnswer` records ([file](https://github.com/gawad-nist/VQA/blob/e06cb7854d7defe53e86412c2bba0ca5f636ea40/vqa.gt.csv)). That provides public 2025 AG reference answers, although the scripts exclude 18 unavailable-video query IDs. Treat the file's reuse terms separately from the source-code notices noted below.

## Exact interfaces and outputs

### 2026 AG validator

```bash
python validate.ag.run.py RELEASED_QUERIES.json SUBMISSION.json \
  --report-json validation-report.json
```

The validator is standard-library-only. By default it requires:

- released topics: a nonempty JSON array; integer `Q_ID >= 1`; nonempty string `Video_ID` and `Question`; unique query IDs;
- submission: one entry for every released query, no unknown or duplicate `Q_ID`, and an exact `Video_ID` match;
- exactly ten answer objects per query, with ranks 1–10 exactly once;
- each answer has only `Rank`, `Answer`, and `Time` unless `--allow-extra-fields` is used;
- nonempty, distinct answer text after case-folding and whitespace normalization;
- finite, nonnegative numeric time (booleans are rejected).

Out-of-order ranks and leading/trailing answer whitespace are warnings. Invalid JSON, an invalid released-query file, or report-write failure exits `2`; a validly parsed but invalid submission exits `1`; success exits `0`. `--warnings-as-errors`, `--answers-per-query`, and `--max-messages` are also supported. The optional JSON report contains validity, expected/submitted counts, the answer count, and error/warning arrays ([validator CLI and exit behavior](https://github.com/gawad-nist/VQA/blob/e06cb7854d7defe53e86412c2bba0ca5f636ea40/validate.ag.run.py#L653-L803); [answer checks](https://github.com/gawad-nist/VQA/blob/e06cb7854d7defe53e86412c2bba0ca5f636ea40/validate.ag.run.py#L279-L435)). Paths in messages use zero-based JSON array indexes even though `Q_ID` begins at 1, as the README warns.

### 2025 AG evaluator

```bash
python eval.vqa.ag.py GROUND_TRUTH.csv RUNS_DIRECTORY
```

Inputs are headerless CSV:

- ground truth: `QueryID,VideoID,Question,CorrectAnswer`;
- each run: `QueryID,VideoID,Rank,Answer,time`.

For each non-skipped query, it normalizes case/whitespace, computes STS, METEOR, and BERTScore F1 for each submitted response, averages each answer-quality metric across **all submitted responses**, and computes NDCG using METEOR and BERTScore as gains. It then averages query scores. It does not compute separate top-1/top-5/top-10 reports and ignores submitted time and video-ID correctness ([AG evaluator](https://github.com/gawad-nist/VQA/blob/e06cb7854d7defe53e86412c2bba0ca5f636ea40/eval.vqa.ag.py#L53-L175)).

It writes beside each input run:

- `<run>_diagnostic.csv`: one row per query;
- `<run>_per_response.csv`: one row per response;
- `evaluation_results.csv` in the runs directory: one row per discovered `.csv` file.

### 2025 MC evaluator

```bash
python eval.vqa.mc.py GROUND_TRUTH.csv RUNS_DIRECTORY
```

The ground truth has the same four columns; each headerless run is `QueryID,VideoID,Rank,Response`. Answer comparison is exact and case-sensitive. It reports top-1 accuracy and MRR, writes `<run>_diagnostic.csv`, and creates `evaluation_results.csv` ([MC evaluator](https://github.com/gawad-nist/VQA/blob/e06cb7854d7defe53e86412c2bba0ca5f636ea40/eval.vqa.mc.py#L23-L111)).

## Dependencies and network behavior

The repository has no package metadata, requirements file, lock file, tests, or release. Static imports and subprocesses require:

- Python: `pandas`, `numpy`, `nltk`, `bert-score`, `scikit-learn`;
- executables: Perl and `wget`;
- runtime network: `nltk.download('wordnet')`; BERTScore model retrieval/cache when absent; one plaintext-HTTP UMBC STS request **per submitted answer**.

The STS helper is an external API wrapper, not a local model ([`calc.sts.pl`](https://github.com/gawad-nist/VQA/blob/e06cb7854d7defe53e86412c2bba0ca5f636ea40/calc.sts.pl#L3-L19)). On 2026-07-20, a direct identical-sentence smoke request returned `1.0`, but availability, latency, and output can change. The AG script downloads only `wordnet`; depending on the installed NLTK release/cache, `word_tokenize` may additionally require tokenizer resources. Dependency versions and the BERTScore model are not pinned, so exact reproducibility is not currently guaranteed.

## Verified hazards and assumptions

These are reasons to wrap the code, not reasons to discard it.

1. **Command injection in AG STS calls.** Normalized ground truth and model responses are interpolated into a shell command passed to `os.popen`; quotes and shell metacharacters are not escaped ([lines 104–106](https://github.com/gawad-nist/VQA/blob/e06cb7854d7defe53e86412c2bba0ca5f636ea40/eval.vqa.ag.py#L104-L106)). Do not run the upstream AG evaluator directly on arbitrary model text in a privileged or secret-bearing environment.
2. **MC skipped-query denominator bug.** The loop skips 18 hard-coded 2025 IDs, but `total_queries` is set before skipping; those 18 remain in both metric denominators ([lines 45–79](https://github.com/gawad-nist/VQA/blob/e06cb7854d7defe53e86412c2bba0ca5f636ea40/eval.vqa.mc.py#L45-L79)). A synthetic two-query test with one perfect scored query and one skipped ID returned `0.500`, not `1.000`.
3. **Directory rescanning is unsafe for repeat runs.** Both evaluators ingest every `.csv` in the input directory, while also writing `.csv` outputs into it. A second invocation can score generated diagnostics and the prior summary as if they were runs. Always use a fresh, single-purpose input directory.
4. **2025-only exclusions are embedded.** Both evaluators hard-code the same 18 unavailable 2025 IDs. They are inappropriate for a custom split or 2026 data unless deliberately preserved for historical parity.
5. **Limited validation.** The evaluators do not reject duplicate/missing ranks, duplicate queries, wrong `VideoID`, malformed answer counts, or extra response rows. Duplicate ground-truth query IDs silently overwrite in dictionaries. The validator is much stricter, but currently covers AG JSON only.
6. **Missing/failed STS can be excluded rather than penalized.** Empty STS service output becomes `NaN`; `nanmean` can omit it from averages, potentially changing the evaluated population ([lines 105–127 and 149–154](https://github.com/gawad-nist/VQA/blob/e06cb7854d7defe53e86412c2bba0ca5f636ea40/eval.vqa.ag.py#L105-L154)).
7. **Working-directory assumption.** AG invokes `perl calc.sts.pl` by a relative path, so it must run with the upstream checkout as the current directory or the STS result is empty.
8. **No final 2026 cutoff semantics yet.** The present AG implementation averages every answer and computes full-list NDCG; it cannot be assumed to match NIST's eventual top-1/top-5/top-10 aggregation.
9. **Single-response NDCG is forced to 1.0.** `ndcg_from_gains` returns `1.0` whenever a query has exactly one response, regardless of that response's relevance ([lines 43–51](https://github.com/gawad-nist/VQA/blob/e06cb7854d7defe53e86412c2bba0ca5f636ea40/eval.vqa.ag.py#L43-L51)). This makes the historical evaluator unsuitable for using NDCG to assess a one-answer baseline.
10. **CSV missing values become text.** Each run dataframe is converted to strings before `fillna`, so a missing value can become the literal answer `"nan"` rather than an empty response ([AG lines 72–79](https://github.com/gawad-nist/VQA/blob/e06cb7854d7defe53e86412c2bba0ca5f636ea40/eval.vqa.ag.py#L72-L79)).

## License signals (not legal advice)

- `eval.vqa.ag.py` and `eval.vqa.mc.py` each contain a NIST-developed-software notice permitting use, copying, distribution, modification, and derivative works if the notice is retained; modified works must identify the date/nature of changes and acknowledge NIST ([AG notice](https://github.com/gawad-nist/VQA/blob/e06cb7854d7defe53e86412c2bba0ca5f636ea40/eval.vqa.ag.py#L1-L17); [MC notice](https://github.com/gawad-nist/VQA/blob/e06cb7854d7defe53e86412c2bba0ca5f636ea40/eval.vqa.mc.py#L1-L17)).
- The repository has no root `LICENSE`, and GitHub reports no detected repository license. `validate.ag.run.py`, `calc.sts.pl`, and `vqa.gt.csv` do not carry that per-file notice at the pinned revision.
- Consequently, vendoring the two noticed evaluator files is clearer than vendoring the repository wholesale, but the reuse status of the validator, STS helper, and bundled data is less explicit. Seek NIST clarification before redistributing those files. Keeping an external pinned checkout avoids copying them into this repository, though it does not itself resolve all use questions.

## Recommended integration in this repository

### Preferred design

1. **Own the stable interface.** Add a repository CLI such as `vqa-score` that reads our prediction JSONL/internal records and emits either 2025 scorer CSV or official 2026 submission JSON. Keep model output, metric results, and submission serialization separate.
2. **Pin upstream metadata, not floating `main`.** Record commit `e06cb7854d7defe53e86412c2bba0ca5f636ea40` and a source-archive checksum in a small lock/config file. Make the adapter accept an explicit checkout path or fetch into a Git-ignored cache. Re-audit before changing the pin.
3. **Expose two clearly named modes.** `nist-2025-parity` runs the unmodified historical evaluator in a locked, isolated environment; `nist-2026-validate-ag` runs the official validator. Never label the former “2026 official score.”
4. **Isolate parity execution.** Use a clean temporary run directory, a minimal environment without credentials, the upstream checkout as `cwd`, explicit timeouts, and captured stdout/stderr. Prefer a rootless container or similarly constrained subprocess because of the shell-interpolation issue and external HTTP request.
5. **Use an optional locked dependency group.** Keep scoring packages out of the downloader's base dependencies; add a `scoring` group with exact versions and verify/cache NLTK and BERTScore assets. Record model/resource identifiers with each score.
6. **Add local contract tests.** Cover quote/metacharacter answers, commas/newlines, missing/duplicate ranks, duplicate answers, missing queries, wrong video IDs, the 18-query denominator behavior, offline STS failure, clean-directory enforcement, and top-k slices.
7. **Retain raw per-answer metrics.** Our adapter should store one row per query/rank/metric. When NIST publishes the revised evaluator, we can reproduce its cutoff aggregation without rerunning expensive model inference.
8. **Gate submissions with the official validator.** Run it in CI against a small fixture and as the final local preflight against the real generated JSON. Use the default ten-answer rule; do not override `--answers-per-query` for competition submissions.

### Integration choice comparison

| Pattern | Assessment |
| --- | --- |
| PyPI/VCS dependency | **Not suitable:** this is not an installable Python package and exposes no stable library API. |
| Git submodule | Traceable pin and no source rewriting, but adds worktree/submodule friction and does not solve missing dependency locks or uneven license notices. Reasonable only if the team accepts that operational cost. |
| Vendor selected scripts | Convenient and offline, but requires retained notices/modification records; validator/STS/data licensing needs clarification. Also risks drifting from NIST updates. |
| Adapter + external pinned checkout/cache | **Recommended now:** clean separation, easy pin/re-audit, and lets us replace the scorer when NIST publishes the 2026 revision. |
| Reimplement all metrics immediately | Not recommended for official parity. It risks subtle divergence before NIST finalizes cutoff semantics. A safe local scorer may be useful for iteration, but compare it against the pinned oracle on fixtures and label it “development,” not official. |

## Decision checkpoint

Proceed now with the adapter, the pinned 2026 AG validator, and a quarantined 2025-parity mode. Wait to freeze any “official 2026 scorer” API or leaderboard comparison until NIST updates the evaluator. At that point, diff the new revision, rerun golden fixtures, verify metric/model versions and offline behavior, and update the pin deliberately.
