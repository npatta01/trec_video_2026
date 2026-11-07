# Local paper draft

Edit `trecvid2026-draft.tex`. The saved PDF is the review artifact. This is a private draft, not a publication package; final model references and author review remain outstanding.

The shared example is Q1835 / 8083.webm. Exact frozen answers, grouping trace, and official scores are recorded in `question-level-analysis.json` (`unified_worked_example`); the editable evidence narrative is `question-level-analysis.qmd`. The shared-error cohort and inspected-case dispositions are in `shared-error-audit.json` and the corresponding QMD section. Earlier cases are retained as historical evidence. `advisor-review.md` records review dispositions.

## Rebuild

Requires XeLaTeX (including TikZ), FFmpeg, and the authorized local dataset. Run from the repository root. Verify `data` resolves to `/home/npatta01/data/competitions/trec_video_2026/data`; do not replace a conflicting path.

The extracted frame is ignored by Git. Regenerate it only if absent:

```bash
mkdir -p docs/paper/private-frames
ffmpeg -v error -n -ss 2 -i data/vqa/2026/testing/videos/8083.webm -an -frames:v 1 docs/paper/private-frames/weather-02.jpg
ffmpeg -v error -n -ss 50 -i data/vqa/2026/testing/videos/Archipelago-42.webm -an -frames:v 1 docs/paper/private-frames/hat-50.jpg
```

Build without writing temporary TeX files into the source directory:

```bash
paper_build_dir=$(mktemp -d /tmp/trec-paper-build.XXXXXX)
cd docs/paper
xelatex -interaction=nonstopmode -halt-on-error -output-directory="$paper_build_dir" trecvid2026-draft.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory="$paper_build_dir" trecvid2026-draft.tex
cp "$paper_build_dir/trecvid2026-draft.pdf" trecvid2026-draft.pdf
```

The desktop standalone compiler cannot access the external frame. Use the saved PDF for visual review. The PDF embeds the authorized example frame; no publication or push is implied by this local commit.
