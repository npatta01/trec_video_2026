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


## Workshop slides

The editable Reveal.js talk is [trecvid2026-talk.qmd](trecvid2026-talk.qmd).
Slides 1–21 are the live talk. Slide 22 opens the appendix, which includes
the full official answer-generation and multiple-choice leaderboards.
The tables load from the repository's organizer score extract during rendering.

The private HTML build needs the shared dataset, FFmpeg, ImageMagick,
Poppler, and Quarto. It also needs the archived
`448_video-plus-transcript.json` request from the frozen Qwen3.6 shard 3.
Set `speech_request` to the local path of that archived file before running
the following commands from the repository root. The preparation script
checks the archived request against the saved checksums and writes the
transcript excerpts beside the ignored presentation media.

```bash
frame_dir=docs/paper/private-frames
frame_build_dir=$(mktemp -d)
mkdir -p "$frame_dir"
if [ ! -f "$frame_dir/weather-02.jpg" ]; then
  pdfimages -f 4 -l 4 -j docs/paper/trecvid2026-draft.pdf "$frame_build_dir/weather"
  install -m 600 "$frame_build_dir/weather-000.jpg" "$frame_dir/weather-02.jpg"
fi
if [ ! -f "$frame_dir/hat-50.jpg" ]; then
  pdfimages -f 5 -l 5 -j docs/paper/trecvid2026-draft.pdf "$frame_build_dir/hat"
  install -m 600 "$frame_build_dir/hat-000.jpg" "$frame_dir/hat-50.jpg"
fi
rm -f "$frame_build_dir/weather-000.jpg" "$frame_build_dir/hat-000.jpg"
rmdir "$frame_build_dir"
python docs/workshop/prepare-talk-media.py --speech-request "$speech_request"
if [ ! -f "$frame_dir/vehicle-02.jpg" ]; then
  ffmpeg -nostdin -v error -n -ss 2 -i data/vqa/2026/testing/videos/7040.webm \
    -map 0:v:0 -an -frames:v 1 -vf scale=960:-2 -q:v 3 "$frame_dir/vehicle-02.jpg"
fi
if [ ! -f "$frame_dir/vehicle-clip.mp4" ]; then
  ffmpeg -nostdin -v error -n -i data/vqa/2026/testing/videos/7040.webm \
    -map 0:v:0 -map 0:a:0 -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p \
    -c:a aac -b:a 96k -movflags +faststart "$frame_dir/vehicle-clip.mp4"
fi
if [ ! -f "$frame_dir/camera-q25.jpg" ]; then
  ffmpeg -nostdin -v error -n -ss 2.4 -i data/vqa/2026/testing/videos/8329.webm \
    -map 0:v:0 -an -frames:v 1 -vf scale=720:-2 -q:v 3 "$frame_dir/camera-q25.jpg"
fi
if [ ! -f "$frame_dir/flags-q576.jpg" ]; then
  ffmpeg -nostdin -v error -n -ss 5.2 -i data/vqa/2026/testing/videos/7465.webm \
    -map 0:v:0 -an -frames:v 1 -vf scale=720:-2 -q:v 3 "$frame_dir/flags-q576.jpg"
fi
if [ ! -f "$frame_dir/chairs-q1744.jpg" ]; then
  ffmpeg -nostdin -v error -n -ss 4 -i data/vqa/2026/testing/videos/7766.webm \
    -map 0:v:0 -an -frames:v 1 -vf scale=720:-2 -q:v 3 "$frame_dir/chairs-q1744.jpg"
fi
if [ ! -f "$frame_dir/bench-q120.jpg" ]; then
  ffmpeg -nostdin -v error -n -ss 2 -i data/vqa/2026/testing/videos/7433.webm \
    -map 0:v:0 -an -frames:v 1 -vf scale=720:-2 -q:v 3 "$frame_dir/bench-q120.jpg"
fi
for item in "8384 2.5 fade-q1" "7156 4 dance-q2181" "6950 4 voice-q448" "6861 2.2 number-q1573" "7194 3 cattle-q109" "6850 7 square-q55" "Memphis-18 11 bed-q1369"; do
  read -r video time name <<< "$item"
  if [ ! -f "$frame_dir/$name.jpg" ]; then
    ffmpeg -nostdin -v error -n -ss "$time" -i "data/vqa/2026/testing/videos/$video.webm" \
      -map 0:v:0 -an -frames:v 1 -vf scale=720:-2 -q:v 3 "$frame_dir/$name.jpg"
  fi
done
for item in "7766 chairs-q1744" "7433 bench-q120" "8329 camera-q25" "7465 flags-q576" "8384 fade-q1" "7156 dance-q2181" "6950 voice-q448" "6861 number-q1573"; do
  read -r video name <<< "$item"
  if [ ! -f "$frame_dir/$name.mp4" ]; then
    ffmpeg -nostdin -v error -n -i "data/vqa/2026/testing/videos/$video.webm" \
      -map 0:v:0 -map 0:a:0 -vf "scale='min(960,iw)':-2" -c:v libx264 -preset fast -crf 22 -pix_fmt yuv420p \
      -c:a aac -b:a 96k -movflags +faststart "$frame_dir/$name.mp4"
  fi
done
quarto render docs/workshop/trecvid2026-talk.qmd --to revealjs
```

The HTML embeds the verified Q1835 frames and a presentation encode with the
source audio track. Keep the rendered export private. Slide 1 uses a single
ImageGen emblem of a video player, a question bubble and a sun-shaped answer. Slide 4 starts
muted and offers playback and sound controls. Slide 6 shows the 12 saved input
frames, a replay of the source clip with its audio, and the Whisper text from
Q1835's frozen request. The answer-generation section plays the full Q1744,
Q120, Q1, Q2181, Q448, and Q1573 clips. Each example shows the frames-only
Qwen answer, both Qwen3.6 video answers, and the Gemini answer. The score
beside each answer comes from the organizer’s all-references METEOR for
the exact submitted wording. The multiple-choice section plays Q447, Q25, and Q576 and uses
sampled frames for the cyclist, town square, hat, and bed examples. The wordmarks identify
the model families. [Qwen](https://commons.wikimedia.org/wiki/File:Qwen_Logo.svg)
and [Gemini](https://commons.wikimedia.org/wiki/File:Google_Gemini_logo_2025.svg).

### Visual reference

The cream canvas, centered cover, roadmap, and illustrated chapter breaks
follow the [final TREC RAG talk](https://npatta01.github.io/trec_rag_2026/reports/2026-trec-rag-talk.html).
The original TRECVID section art in `slide-assets` was generated with built-in
ImageGen. The prompts describe warm hand-drawn scenes on cream paper: a clip
becoming two answer slips, a clip leading to ranked choices, and two evaluation
scorecards. The illustration files are independent of the test footage.

### Annotate the HTML deck

Choose **Annotate**, then click a spot on a slide. To leave a note on the
whole slide, choose **Comments** and **Add slide note**. Reopen a comment from
a pin or the list to edit or delete it. Comments are saved in that browser
when local storage is available. Use **Comments → Export comments** and send
the JSON file with your feedback.
