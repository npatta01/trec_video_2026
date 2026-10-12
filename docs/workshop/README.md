# TRECVID 2026 workshop speaker proposal

Prepared proposal for the VQA session, which the organizer's reminder says is
planned for November 18, 2026. This record does not establish that the proposal
has been emailed or that a speaking slot has been confirmed.

- **Talk:** NP Labs at TRECVID 2026: Experiments with Off-the-Shelf Video QA Models
- **Registered team:** NP Labs
- **Speaker:** Tanwir Zaman
- **Speaker email:** zaman.tanwir@gmail.com
- **Affiliation:** Independent researcher
- **Presentation mode:** In person

The abstract describes the submitted answer-generation and multiple-choice
variants and proposes discussing successes, failures, difficult question types,
and selected open-weight/paid-model output comparisons.

[Prepared PDF](trecvid2026-speaker-proposal.pdf) ·
[Editable LaTeX source](trecvid2026-speaker-proposal.tex)

## Rebuild

Requires XeLaTeX and the TeX Gyre Pagella font. Run from the repository root:

```bash
proposal_build_dir=$(mktemp -d /tmp/trecvid-speaker-build.XXXXXX)
xelatex -interaction=nonstopmode -halt-on-error -output-directory="$proposal_build_dir" docs/workshop/trecvid2026-speaker-proposal.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory="$proposal_build_dir" docs/workshop/trecvid2026-speaker-proposal.tex
cp "$proposal_build_dir/trecvid2026-speaker-proposal.pdf" docs/workshop/trecvid2026-speaker-proposal.pdf
```

[Official speaker-proposal instructions](https://www-nlpir.nist.gov/projects/tv2026/speakerproposals.html)
