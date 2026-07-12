# TRECVID 2026 Video Question Answering

Research and submission-planning materials for the TRECVID 2026 Video Question Answering (VQA) task. The report is written for readers who are new to video QA and uses progressive disclosure: a short executive overview first, followed by task rules, 2025 results, team architectures, fine-tuning guidance, public models, literature, and an implementation plan.

## Report

- [Open the self-contained interactive report](reports/trecvid_vqa_2026/report.html)
- [Canonical report artifact](reports/trecvid_vqa_2026/artifact.json)
- [Sources, reconciliation decisions, and QA notes](reports/trecvid_vqa_2026/source_notes.md)

The HTML report is self-contained and can be opened directly in a modern browser. It includes a contents card, nine numbered chapters, accessible architecture diagrams, sortable evidence tables, and source details.

## What the report covers

- The 2026 Answer Generation and Multiple Choice tasks, output contracts, scoring, data, and tentative schedule.
- The complete published 2025 submission inventory and the strongest current appendix results.
- Plain-language explanations of video-language model components and four team architecture patterns.
- Fine-tuning choices: frozen baselines, ranker-only training, LoRA, QLoRA, and full-model tuning.
- Training-data design, leakage controls, GPU planning bands, and a staged experiment plan.
- Official public-weight video models and originator-released task fine-tunes, including access and license caveats.
- A 25-work competition-adjacent literature review tied to concrete TRECVID experiments.
- A recommended shared evidence engine with separate AG and MC output heads.

## Repository layout

```text
reports/
└── trecvid_vqa_2026/
    ├── report.html          # portable interactive report
    ├── artifact.json        # canonical manifest and bounded data snapshot
    ├── source_notes.md      # evidence ledger, calculations, and QA contract
    ├── assets/              # accessible SVG architecture explainers
    ├── queries/             # reproducible SQL-shaped evidence mirrors
    └── *.mjs                # deterministic report update/delivery scripts
```

## Rebuild the report

Run the update scripts in this order so content expansions occur before the chapter labels are applied:

```bash
cd reports/trecvid_vqa_2026
node expand_literature.mjs
node embed_diagrams.mjs
node expand_finetuning.mjs
node add_navigation.mjs
node deliver_report.mjs
```

`deliver_report.mjs` uses the Data Analytics report runtime installed in the local Codex environment and an installed Chrome/Chromium browser for desktop and phone-width verification. The committed `report.html` does not require that runtime to be read.

## Evidence boundary

Competition facts were checked against the official NIST pages and TREC 2025 proceedings through July 11, 2026. Model cards, fine-tuning documentation, and the literature supplement were checked July 12, 2026. See the report's source controls and `source_notes.md` for exact links, denominator reconciliation, limitations, and reproducibility notes.

Primary task page: [NIST TRECVID 2026 VQA](https://www-nlpir.nist.gov/projects/tv2026/vqa.html).
