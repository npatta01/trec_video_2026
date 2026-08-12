# Frontier boundary-verifier artifact index

This tracked directory is the sanitized index for the 2026-08-12
question-blind Gemini scene-map and boundary-verifier diagnostic. It contains
no questions, expected answers, raw provider responses, credentials, or media.

The complete private evidence is retained at:

`/home/npatta01/data/competitions/trec_video_2026/data/vqa/benchmarks/vqa-frontier-phase0-20260812/`

Important private subdirectories:

- `run-v4/` and `pilot-audit-v4/`: preserved earlier usable draft and audit;
- `run-v5/`: final one-pass draft run;
- `pilot-audit-v5/`: frozen `revise_contract` audit;
- `pilot-audit-v5-manual-review/`: dense 4-fps inspection sheets;
- `verifier-v5/`: the one terminal verifier request, raw response, result,
  diagnostic audit, and diagnostic freeze;
- `verifier-v5-pre-review/` and `verifier-v5-pre-freeze-fix/`: offline-only
  prepared artifacts superseded before any paid request.

The decisive private records are:

- `pilot-audit-v5/freeze.json`: freezes the one-pass draft decision as
  `revise_contract`;
- `verifier-v5/result.json`: terminal verifier result;
- `verifier-v5/raw-response.json`: complete preserved provider output;
- `verifier-v5/diagnostic-audit.json`: separates the strict contract failure
  from the scientific boundary failure;
- `verifier-v5/diagnostic-freeze.json`: freezes the diagnostic decision as
  `stop_two_stage`.

No Gemini rerun, dstack pod, or label access is needed to inspect this result.
The raw files remain private and untracked; merging this branch exposes this
sanitized index and the report from the shared checkout.
