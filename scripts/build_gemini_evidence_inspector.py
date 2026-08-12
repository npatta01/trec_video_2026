#!/usr/bin/env python3
"""Build a private static inspector for the seven-row Gemini evidence canary."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Sequence


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from score_direct_gemini_evidence_canary import (  # noqa: E402
    CONDITIONS,
    MANUAL_ASSESSMENTS,
)


ANSWER_LABELS = {
    "direct_answer": "Direct native AV",
    "timeline_answer": "Timeline only",
    "frame_answer": "Timeline + selected frames",
}


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"Missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _media_url(prefix: str, *parts: str) -> str:
    if not prefix.startswith("/") or "://" in prefix or ".." in PurePosixPath(prefix).parts:
        raise ValueError("media prefix must be a same-origin absolute URL path")
    return str(PurePosixPath(prefix, *parts))


def load_cases(run_root: Path, media_prefix: str) -> list[dict[str, Any]]:
    """Validate and project the frozen private artifacts for the inspector."""

    metric_artifact = _read_json(run_root / "postfreeze-metrics.json")
    metric_rows = {
        (row["video_id"], row["condition"]): row for row in metric_artifact["rows"]
    }
    row_ids = {path.parent.name for path in (run_root / "rows").glob("*/complete.json")}
    expected_ids = set(MANUAL_ASSESSMENTS)
    if row_ids != expected_ids:
        raise ValueError(
            f"Expected exactly seven frozen rows; missing={sorted(expected_ids-row_ids)}, "
            f"extra={sorted(row_ids-expected_ids)}"
        )

    cases = []
    for video_id in sorted(expected_ids):
        row_root = run_root / "rows" / video_id
        complete = _read_json(row_root / "complete.json")
        timeline = _read_json(row_root / "timeline" / "events.json")
        frame_manifest = _read_json(row_root / "selected-frames" / "manifest.json")
        videos = sorted((row_root / "delivery" / "delivery-media").glob("*.webm"))
        if len(videos) != 1:
            raise ValueError(f"Missing required artifact: one WebM for {video_id}")
        assessment = MANUAL_ASSESSMENTS[video_id]
        answers = []
        for condition in CONDITIONS:
            scored = metric_rows.get((video_id, condition))
            if scored is None:
                raise ValueError(f"Missing required artifact: metric {video_id}/{condition}")
            verdict, reason = assessment[condition]
            answers.append(
                {
                    "condition": condition,
                    "label": ANSWER_LABELS[condition],
                    "answer": scored["prediction"],
                    "verdict": verdict,
                    "reason": reason,
                    "semantic_composite": scored["metrics"]["semantic_composite"],
                }
            )
        frames = []
        for frame in frame_manifest:
            frame_path = row_root / "selected-frames" / frame["path"]
            if not frame_path.is_file():
                raise ValueError(f"Missing required artifact: {frame_path}")
            frames.append(
                {
                    "timestamp_seconds": frame["timestamp_seconds"],
                    "url": _media_url(
                        media_prefix,
                        "rows",
                        video_id,
                        "selected-frames",
                        frame["path"],
                    ),
                }
            )
        cases.append(
            {
                "video_id": video_id,
                "category": complete["category"],
                "question": complete["question"],
                "reference": metric_rows[(video_id, CONDITIONS[0])]["reference"],
                "overall": assessment["overall"],
                "timeline_verdict": assessment["timeline"][0],
                "timeline_reason": assessment["timeline"][1],
                "answers": answers,
                "timeline": timeline,
                "selected_spans": complete["selected_spans"],
                "frames": frames,
                "video_url": _media_url(
                    media_prefix,
                    "rows",
                    video_id,
                    "delivery",
                    "delivery-media",
                    videos[0].name,
                ),
            }
        )
    return cases


def _page_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def render_html(cases: list[dict[str, Any]]) -> str:
    """Render the dependency-free, read-only evidence inspector."""

    choices = "\n".join(
        f'<button class="case-choice" data-index="{index}" type="button">'
        f'<span class="case-category">{html.escape(case["category"])}</span>'
        f'<span>{html.escape(case["video_id"])}</span></button>'
        for index, case in enumerate(cases)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>Gemini evidence inspector</title>
  <style>
    :root {{ font-family: ui-sans-serif, system-ui, sans-serif; color: #172033; background: #eef3f8; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; overflow-x: hidden; }}
    button {{ font: inherit; }}
    .shell {{ display: grid; grid-template-columns: 13rem minmax(0, 1fr); min-width: 0; min-height: 100vh; }}
    nav {{ padding: 1rem; background: #132033; color: #fff; }}
    nav h1 {{ margin: 0 0 .35rem; font-size: 1.05rem; }}
    nav p {{ margin: 0 0 1rem; color: #b9c7dc; font-size: .82rem; line-height: 1.4; }}
    .case-choice {{ display: flex; width: 100%; margin: .4rem 0; padding: .7rem; flex-direction: column; gap: .18rem; text-align: left; color: #dce7f7; background: #1c2c43; border: 1px solid #334764; border-radius: .65rem; cursor: pointer; }}
    .case-choice.active {{ color: #10213a; background: #dbeafe; border-color: #93c5fd; }}
    .case-category {{ font-size: .68rem; font-weight: 800; letter-spacing: .05em; text-transform: uppercase; opacity: .78; }}
    main {{ width: min(100%, 88rem); min-width: 0; padding: clamp(1rem, 3vw, 2rem); }}
    .top {{ display: grid; grid-template-columns: minmax(18rem, 1fr) minmax(18rem, 1fr); gap: 1rem; }}
    video {{ display: block; width: 100%; min-height: 16rem; max-height: 31rem; background: #05080d; border-radius: .85rem; }}
    .panel {{ min-width: 0; padding: 1rem; overflow-wrap: anywhere; background: #fff; border: 1px solid #d8e1ec; border-radius: .85rem; box-shadow: 0 .4rem 1.2rem rgb(23 32 51 / 6%); }}
    .eyebrow {{ color: #2563eb; font-size: .75rem; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; }}
    h2 {{ margin: .35rem 0 .7rem; font-size: clamp(1.25rem, 3vw, 1.8rem); }}
    .reference {{ padding: .75rem; background: #f1f5f9; border-radius: .65rem; }}
    .overall {{ margin-top: .75rem; line-height: 1.5; }}
    .section-title {{ margin: 1.3rem 0 .65rem; }}
    .answer-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .75rem; }}
    .answer-card {{ border-top: .35rem solid var(--verdict); }}
    .answer-head {{ display: flex; justify-content: space-between; gap: .5rem; align-items: start; }}
    .badge {{ padding: .22rem .5rem; color: #fff; background: var(--verdict); border-radius: 999px; font-size: .72rem; font-weight: 800; text-transform: uppercase; }}
    .answer-text {{ font-size: 1.05rem; font-weight: 700; line-height: 1.4; }}
    .why {{ color: #526176; line-height: 1.5; }}
    .score {{ color: #64748b; font-size: .78rem; }}
    .good {{ --verdict: #16834a; }} .okay {{ --verdict: #b77908; }} .wrong {{ --verdict: #c93636; }} .ambiguous {{ --verdict: #7656c8; }}
    .evidence {{ display: grid; grid-template-columns: minmax(18rem, .9fr) minmax(20rem, 1.1fr); gap: 1rem; }}
    .timeline-list {{ display: grid; gap: .45rem; }}
    .seek {{ width: 100%; padding: .65rem; text-align: left; color: inherit; background: #f8fafc; border: 1px solid #d8e1ec; border-radius: .55rem; cursor: pointer; }}
    .seek:hover, .seek:focus {{ border-color: #2563eb; outline: 2px solid #bfdbfe; }}
    .time {{ display: block; margin-bottom: .2rem; color: #2563eb; font-size: .75rem; font-weight: 800; }}
    .frame-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .6rem; }}
    .frame {{ padding: 0; overflow: hidden; background: #0f172a; border: 2px solid transparent; border-radius: .6rem; cursor: pointer; }}
    .frame:hover, .frame:focus {{ border-color: #2563eb; }}
    .frame img {{ display: block; width: 100%; aspect-ratio: 16/10; object-fit: contain; }}
    .frame span {{ display: block; padding: .4rem; color: #fff; font-weight: 750; }}
    .media-error {{ display: none; margin-top: .5rem; color: #b42318; font-weight: 700; }}
    @media (max-width: 900px) {{ .answer-grid {{ grid-template-columns: 1fr; }} .evidence {{ grid-template-columns: 1fr; }} }}
    @media (max-width: 760px) {{ .shell {{ display: block; }} nav {{ position: static; width: 100%; overflow: hidden; }} .case-list {{ display: flex; max-width: 100%; overflow-x: auto; gap: .4rem; }} .case-choice {{ min-width: 9rem; }} .top {{ grid-template-columns: minmax(0, 1fr); }} main {{ width: 100%; padding: .8rem; }} video {{ min-height: 12rem; }} }}
    @media (prefers-color-scheme: dark) {{ :root {{ color: #e9f0fa; background: #0c1421; }} .panel {{ background: #142033; border-color: #2a3b54; box-shadow: none; }} .reference, .seek {{ background: #1c2a40; border-color: #344863; }} .why, .score {{ color: #b7c5d8; }} }}
  </style>
</head>
<body>
  <div class="shell">
    <nav aria-label="Video cases">
      <h1>Evidence inspector</h1>
      <p>Compare what Gemini saw, selected, and answered.</p>
      <div class="case-list">{choices}</div>
    </nav>
    <main>
      <div class="top">
        <section>
          <video id="native-video" controls preload="metadata"></video>
          <div id="media-error" class="media-error">Video unavailable. The remaining evidence is still shown.</div>
        </section>
        <section class="panel">
          <div id="case-category" class="eyebrow"></div>
          <h2 id="question"></h2>
          <div class="reference"><strong>Reference:</strong> <span id="reference"></span></div>
          <div id="overall" class="overall"></div>
        </section>
      </div>
      <h2 class="section-title">What each path answered</h2>
      <section id="answer-grid" class="answer-grid"></section>
      <div class="evidence">
        <section>
          <h2 class="section-title">Question-blind timeline</h2>
          <div id="timeline-assessment" class="panel"></div>
          <div id="timeline-list" class="timeline-list"></div>
          <h2 class="section-title">Question-selected spans</h2>
          <div id="span-list" class="timeline-list"></div>
        </section>
        <section>
          <h2 class="section-title">Frames sent to the answerer</h2>
          <div id="frame-grid" class="frame-grid"></div>
        </section>
      </div>
    </main>
  </div>
  <script id="case-data" type="application/json">{_page_json(cases)}</script>
  <script>
    const cases = JSON.parse(document.getElementById('case-data').textContent);
    const video = document.getElementById('native-video');
    const el = id => document.getElementById(id);
    const text = (tag, value, className = '') => {{ const node = document.createElement(tag); node.textContent = value; node.className = className; return node; }};
    function seekVideo(seconds) {{ video.currentTime = Number(seconds); video.play().catch(() => {{}}); video.scrollIntoView({{behavior: 'smooth', block: 'center'}}); }}
    function seekButton(start, end, description) {{ const button = text('button', '', 'seek'); button.type = 'button'; button.dataset.seek = start; button.append(text('span', `${{Number(start).toFixed(3)}}–${{Number(end).toFixed(3)}}s`, 'time'), text('span', description)); button.addEventListener('click', () => seekVideo(start)); return button; }}
    function renderCase(index) {{
      const item = cases[index];
      document.querySelectorAll('.case-choice').forEach((button, i) => button.classList.toggle('active', i === index));
      video.src = item.video_url; video.load(); el('media-error').style.display = 'none';
      el('case-category').textContent = `${{item.category}} · ${{item.video_id}}`;
      el('question').textContent = item.question; el('reference').textContent = item.reference; el('overall').textContent = item.overall;
      el('answer-grid').replaceChildren(...item.answers.map(answer => {{
        const card = text('article', '', `panel answer-card ${{answer.verdict}}`);
        const head = text('div', '', 'answer-head'); head.append(text('strong', answer.label), text('span', answer.verdict, 'badge'));
        card.append(head, text('p', answer.answer, 'answer-text'), text('p', answer.reason, 'why'), text('div', `Semantic composite: ${{answer.semantic_composite.toFixed(3)}}`, 'score')); return card;
      }}));
      el('timeline-assessment').replaceChildren(text('strong', `${{item.timeline_verdict.toUpperCase()}}: `), document.createTextNode(item.timeline_reason));
      el('timeline-list').replaceChildren(...item.timeline.map(event => seekButton(event.start_seconds, event.end_seconds, event.description)));
      el('span-list').replaceChildren(...item.selected_spans.map(span => seekButton(span.start_seconds, span.end_seconds, span.reason)));
      el('frame-grid').replaceChildren(...item.frames.map(frame => {{ const button = text('button', '', 'frame'); button.type = 'button'; button.dataset.seek = frame.timestamp_seconds; const image = document.createElement('img'); image.src = frame.url; image.alt = `Selected frame at ${{frame.timestamp_seconds.toFixed(3)}} seconds`; image.loading = 'lazy'; button.append(image, text('span', `${{frame.timestamp_seconds.toFixed(3)}}s`)); button.addEventListener('click', () => seekVideo(frame.timestamp_seconds)); return button; }}));
      history.replaceState(null, '', `#${{item.video_id}}`);
    }}
    video.addEventListener('error', () => el('media-error').style.display = 'block');
    document.querySelectorAll('.case-choice').forEach(button => button.addEventListener('click', () => renderCase(Number(button.dataset.index))));
    const initial = Math.max(0, cases.findIndex(item => `#${{item.video_id}}` === location.hash)); renderCase(initial);
  </script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--media-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = load_cases(args.run_root, args.media_prefix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(cases), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cases": len(cases)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
