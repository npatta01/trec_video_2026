# Gemini Evidence Inspector Design

Date: 2026-08-12

## Objective

Make the seven-video Gemini canary understandable without reading raw Markdown
or opening artifacts separately. The inspector must show why an answer was
judged good, okay, wrong, or ambiguous and make it easy to compare native AV,
timeline-only, and timeline-plus-frame answering.

## Scope

Build one private, read-only static HTML page. It has no backend, framework,
editing, persistence, or new model calls. It reads only the already-completed
seven-video artifact bundle and does not copy media into Git.

## Page structure

The page has a compact video picker followed by one selected-video workspace:

1. Playable native video.
2. Question, post-freeze reference, and overall assessment.
3. Three answer cards for direct native AV, timeline-only, and timeline plus
   selected frames. Each card shows the answer, human verdict, explanation,
   and semantic composite.
4. Question-blind timeline events with timestamps. Selecting an event seeks
   the video to its start time.
5. Question-selected spans and extracted frames. Each frame shows its
   timestamp; selecting it seeks the video to that time.

Verdicts use both text and color. `Good` means supported and semantically
correct; `okay` means useful but incomplete or unnecessarily verbose; `wrong`
means it misses the requested answer; `ambiguous` means the reference or task
interpretation needs adjudication.

## Data and privacy

A small generator reads the canonical private bundle at:

`/home/npatta01/data/competitions/trec_video_2026/data/vqa/benchmarks/direct-gemini-evidence-canary-v1/`

It emits a private self-contained HTML shell under
`$HOME/codex-rendered/plans/`. The HTML references the existing private media
through the authorized tailnet-only portal; it does not embed or publish raw
video, frames, questions, references, or provider responses in the Git
repository. The canonical source remains the private artifact bundle.

## Interaction and responsive behavior

- Desktop uses a narrow video picker and a wide evidence workspace.
- Mobile stacks the picker, video, answers, timeline, and frames.
- The current video and selected evidence timestamp remain visually obvious.
- Native browser video controls are retained.
- Keyboard activation works for video choices, timeline events, and frames.
- A missing media file produces a visible unavailable message without hiding
  the remaining evidence.

## Implementation boundary

Add one repository generator script that validates the seven expected rows and
their required artifacts before rendering. The generated private HTML is not
tracked. Add focused tests for data projection, verdict presence, and safe
relative media URLs. Update the private portal index with one link after the
rendered page passes local and tailnet checks.

## Verification

- Seven videos appear exactly once.
- Every video shows all three answers and all four verdict dimensions.
- Questions, references, timelines, selected spans, frame timestamps, and
  assessment explanations match the frozen artifact files.
- Timeline and frame controls seek the selected video correctly.
- The page works at phone and desktop widths.
- The live HTTPS page returns successfully and remains tailnet-only.
- No credential, raw provider response, or unrelated dataset content appears
  in the generated page or tracked changes.
