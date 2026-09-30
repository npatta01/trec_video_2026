# TREC VQA 2026 official NP_Labs results

Organizer attachments and participant-only leaderboard files from George Awad’s September 28, 2026
“TREC VQA Track Results” email, captured September 29. Added to the repository
at the user's explicit request. The two ZIPs total 10,097,810 bytes (9.63 MiB),
so they are stored directly rather than as release assets.

## Files

- [NP_Labs.zip](NP_Labs.zip): original AG attachment; four runs with per-query
  scores under `human_answer_only` and `all_references` reference modes.
- [NP_Labs-MC.zip](NP_Labs-MC.zip): original MC attachment; four official run
  summaries and their per-query CSV files.
- [attachment-manifest.json](attachment-manifest.json): original byte sizes,
  SHA-256 checksums, and ZIP entry lists.
- [mc-official-summary.json](mc-official-summary.json): extracted organizer
  summary objects, preserving the reported values.
- [ag-derived-means.json](ag-derived-means.json): locally calculated arithmetic
  means of organizer per-query fields across 2,428 queries for each run and
  reference mode. These are derived aggregates, not the official leaderboard
  workbook. Includes cutoffs 1, 5, and 10 for STS, METEOR, and BERTScore F1,
  with best-answer, mean-top-k, and NDCG statistics.

## MC scores

| Run | Correct / scored | Top-1 accuracy | MRR |
|---|---:|---:|---:|
| g37_mc_direct | 2052 / 2428 | 84.514% | 0.905653 |
| g38_mc_direct | 2047 / 2428 | 84.308% | 0.905368 |
| g37_mc_target | 2025 / 2428 | 83.402% | 0.896845 |
| g37q36_mc_text | 1912 / 2428 | 78.748% | 0.864605 |

All four MC summaries report success, 15 excluded queries out of 2,443,
zero missing scored queries, zero warnings, and zero errors.

## Provenance and limits

[Source email](https://mail.google.com/mail/u/#all/1a0e88a14176c072)
requires access to the recipient's mailbox. Original ZIP bytes are preserved;
no inference or scoring model was rerun. The raw email and its headers are not
included. The archives contain organizer-provided per-query evaluation data;
repository inclusion does not establish permission for broader redistribution.

The participant-only leaderboard files are now included unchanged:

- [AG workbook](VQA_AG_effectiveness_scores.xlsx), 43,034 bytes.
- [MC leaderboard](vqa.mcq.scores.txt), 3,359 bytes.
- [Results and organizer instructions](results.qmd), the editable report source.
- [Complete extracted leaderboards and ranks](leaderboard-summary.json).

Our best MC run is second of 42 by accuracy and third by MRR. Our two g37 AG
runs tie for second of 44 on METEOR@1 in both reference modes. See the report
for all NP_Labs results, metric-specific ranks, ties, and verification details.
All four source files fit directly in the repository; no release is needed.
