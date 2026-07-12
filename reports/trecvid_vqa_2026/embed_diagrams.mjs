#!/usr/bin/env node

// Embed deterministic SVG architecture diagrams into the canonical report
// artifact as self-contained image blocks. Re-run this script after changing
// any SVG in assets/ and then rebuild report.html with deliver_report.mjs.
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = dirname(fileURLToPath(import.meta.url));
const artifactPath = join(root, "artifact.json");
const artifact = JSON.parse(readFileSync(artifactPath, "utf8"));
const blocks = artifact?.manifest?.blocks;

if (!Array.isArray(blocks)) {
  throw new Error("artifact.json is missing manifest.blocks");
}

function blockById(id) {
  const block = blocks.find((candidate) => candidate.id === id);
  if (!block) throw new Error(`Expected report block ${id}`);
  return block;
}

function embeddedSvg(asset) {
  const path = join(root, "assets", asset);
  const svg = readFileSync(path, "utf8");
  if (!/<svg\b/.test(svg)) throw new Error(`${asset} is not an SVG document`);
  return `data:image/svg+xml;base64,${Buffer.from(svg, "utf8").toString("base64")}`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function imageBlock({ id, asset, title, alt, caption }) {
  const body = [
    "<style>",
    "*{box-sizing:border-box}",
    "body{margin:0;background:transparent;color:#172033;font:16px/1.5 system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif}",
    "figure{width:min(760px,100%);margin:0 auto;padding:0}",
    "h3{margin:0 0 10px;font-size:20px;line-height:1.25;color:#172033}",
    "img{display:block;width:100%;height:auto;border:1px solid #c9d4e3;border-radius:16px;background:#f8fbff}",
    "figcaption{margin:10px 2px 0;color:#4b5870;font-size:14px;line-height:1.45}",
    "@media(max-width:480px){body{font-size:15px}h3{font-size:18px}figcaption{font-size:13px}}",
    "</style>",
    "<figure>",
    `<h3>${escapeHtml(title)}</h3>`,
    `<img alt="${escapeHtml(alt)}" decoding="async" loading="lazy" src="${embeddedSvg(asset)}">`,
    `<figcaption>${escapeHtml(caption)}</figcaption>`,
    "</figure>",
  ].join("");
  return { id, type: "html", layout: "full", body };
}

function replaceGroupAfter(anchorId, ids, replacements) {
  const removal = new Set(ids);
  const retained = blocks.filter((block) => !removal.has(block.id));
  const anchorIndex = retained.findIndex((block) => block.id === anchorId);
  if (anchorIndex < 0) throw new Error(`Expected anchor block ${anchorId}`);
  retained.splice(anchorIndex + 1, 0, ...replacements);
  blocks.splice(0, blocks.length, ...retained);
}

blockById("model_primer_takeaway").body = `## A submission is a pipeline, not just a model name

A **model architecture** is a blueprint, a **checkpoint** is one trained set of weights, and a **submission pipeline** adds media decoding, frame selection, audio handling, prompting, ensembling, normalization, ranking, and JSON validation. Two teams can use the same model family and obtain very different results because the surrounding pipeline controls what evidence the model ever sees.

**How to read the image below:** the sampler chooses pages from a flipbook, the vision encoder turns patches into numeric “visual words,” the connector translates those words for the language model, and the final head converts raw output into the exact competition contract. A failure in any box can dominate the choice of backbone.`;

blockById("team_flow_patterns").body = `## Four architecture patterns explain all seven teams

The images below group the published systems by **how they gather evidence and turn it into a ranked answer**. They are architecture maps, not performance rankings. Where tcna or TJUMI did not publish an intermediate stage, the diagram labels that gap instead of guessing.`;

blockById("architecture_takeaway").body = `## Build one evidence engine and two thin task heads

The highest-leverage shared work is media reliability: download or locate each video, record availability and hashes, normalize codecs, extract timestamped automatic speech recognition (ASR) and optical character recognition (OCR) once, and cache global frames, shot-aware keyframes, and local tiles. A question router then chooses the evidence budget. The image below turns that plan into one phone-readable flow.

The competition-adjacent literature makes the routing rule more specific: guarantee broad coverage, add question-relevant moments, preserve high resolution only around likely evidence, retain both ASR and native audio, and spend extra views only when answer agreement is weak. For causal questions, keep the cause and outcome timestamps with each candidate rather than relying on an unsupported rationale.

For AG, generate 20–30 candidates efficiently, normalize without changing meaning, cluster near-duplicates, and rank exactly ten. For MC, preserve the four option strings byte-for-byte and aggregate direct option scores across evidence views. Both outputs pass a strict schema gate that fails loudly on missing IDs, invalid ranks, duplicate options, missing times, or fallback error text.`;

replaceGroupAfter("model_primer_takeaway", ["model_primer_image"], [
  imageBlock({
    id: "model_primer_image",
    asset: "model-primer.svg",
    title: "From a video file to two competition outputs",
    alt: "A video branches into visual-frame sampling and audio processing. Those streams join the question and multiple-choice options at a multimodal language model, then split into an Answer Generation head that ranks ten answers and a Multiple Choice head that ranks four options.",
    caption: "A backbone is only the center box. Sampling, audio, and the output head determine what it sees and whether its answer satisfies the task contract.",
  }),
]);

replaceGroupAfter("team_flow_patterns", [
  "team_direct_image",
  "team_multiview_image",
  "team_evidence_image",
  "team_generate_rank_image",
], [
  imageBlock({
    id: "team_direct_image",
    asset: "team-direct.svg",
    title: "1. One backbone answers directly — WHU, tcna, and TJUMI",
    alt: "Video and question pass through the team's reported preprocessing and then a VideoLLaMA or Gemini backbone. The generated answer goes to submission. WHU's multiple-choice branch adds semantic matching against the options.",
    caption: "This is the simplest published pattern. tcna and TJUMI did not disclose enough detail to reconstruct every intermediate stage, so the image leaves those stages explicitly unpublished.",
  }),
  imageBlock({
    id: "team_multiview_image",
    asset: "team-multiview.svg",
    title: "2. Ask the model through several visual windows — CERTH-ITI",
    alt: "Video plus full audio are sampled into ten random frame views. Qwen2.5-Omni processes all ten views. Answer Generation combines repeated answers by voting, while Multiple Choice pools option scores across views.",
    caption: "The model gets ten different looks at the clip. Repeated answers or consistently high option scores count as stronger evidence.",
  }),
  imageBlock({
    id: "team_evidence_image",
    asset: "team-evidence.svg",
    title: "3. Improve the evidence while freezing the reasoner — NII_UIT",
    alt: "The video yields about twelve time-spaced frames and zoomed tiles. Audio becomes a timestamped Whisper transcript, and the question becomes a question-type prompt. The merged evidence enters frozen Aria for Answer Generation or frozen InternVL for Multiple Choice, followed by normalization or ranking.",
    caption: "NII did not tune the backbones; it tried to win by exposing more moments, closer crops, readable speech, and a task-aware prompt.",
  }),
  imageBlock({
    id: "team_generate_rank_image",
    asset: "team-generate-rank.svg",
    title: "4. Generate first, judge second — HLTCOE and kslab",
    alt: "Video evidence and the question enter a generator. Its candidates or free-text anchor are judged by a second stage: HLTCOE uses a learned listwise reranker, while kslab uses confidence for Answer Generation and cosine similarity for Multiple Choice. The selected items become an ordered submission.",
    caption: "HLT learned a candidate reranker. kslab used confidence for AG and a generated-answer embedding for MC; WHU's MC system is a secondary example of the anchor-and-compare idea.",
  }),
]);

replaceGroupAfter("architecture_takeaway", ["recommended_2026_image"], [
  imageBlock({
    id: "recommended_2026_image",
    asset: "recommended-2026.svg",
    title: "Recommended 2026 architecture",
    alt: "A media manifest enters a reliable decode and normalization stage. A shared evidence cache stores frames, spatial tiles, speech transcripts, optical character recognition, and timestamps. A question router builds an evidence pack. It then splits into an Answer Generation branch that generates, deduplicates, and ranks exactly ten answers, and a Multiple Choice branch that directly scores and ranks four options. Both pass through a schema and provenance validator before the JSON run is written.",
    caption: "Build and cache the expensive evidence layer once, then keep AG and MC as thin, independently testable output heads behind one strict validator.",
  }),
]);

const ids = blocks.map((block) => block.id);
if (new Set(ids).size !== ids.length) throw new Error("Diagram embedding produced duplicate block ids");
if (blocks.some((block) => block.type === "markdown" && /```text/.test(block.body))) {
  throw new Error("A phone-unfriendly text diagram remains in a markdown block");
}

writeFileSync(artifactPath, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
process.stdout.write(`${JSON.stringify({ ok: true, blocks: blocks.length, htmlBlocks: blocks.filter((block) => block.type === "html").length })}\n`);
