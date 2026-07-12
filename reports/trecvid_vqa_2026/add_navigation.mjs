#!/usr/bin/env node

// Add a mobile-safe contents card and stable chapter numbers to the canonical
// report. The current native reader opens Markdown links in a new context, so
// this uses visible chapter markers instead of unreliable fragment links.
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const artifactPath = join(root, "artifact.json");
const artifact = JSON.parse(readFileSync(artifactPath, "utf8"));
const blocks = artifact?.manifest?.blocks;

if (!Array.isArray(blocks)) throw new Error("artifact.json is missing manifest.blocks");

const contentsBlock = {
  id: "table_of_contents",
  type: "markdown",
  layout: "full",
  body: `## Contents and reading paths

This guide is intentionally deep. The chapter numbers below repeat in the large headings throughout the report.

**10-minute orientation:** Executive Summary → Chapter 3 → Chapter 7 → Chapter 8.

**Understand the field from scratch:** Chapters 1 → 2 → 3 → 4.

**Plan and build a 2026 system:** Chapters 1 → 4 → 5 → 6 → 7 → 8.

1. **Task, rules, and outputs** — what the competition asks for and how it is scored.
2. **Models and scoring in plain English** — how video, audio, and language components fit together.
3. **2025 submissions and leaders** — all published runs, strongest results, and scoring traps.
4. **Team architectures and public repositories** — how the seven teams assembled their systems.
5. **Fine-tuning and public weights** — LoRA/QLoRA, data, compute, available checkpoints, and a safe training plan.
6. **Literature and transferable ideas** — six research mechanisms and their competition-ready experiments.
7. **Recommended 2026 system** — one evidence engine, separate AG/MC heads, and strict validation.
8. **Run plan, calendar, and immediate actions** — which submissions to build and what to do next.
9. **Questions, caveats, and glossary** — unresolved organizer questions and newcomer definitions.

> **Navigation tip:** on a phone, scan for “Chapter” in the large headings. On a computer, use Find (Ctrl/Cmd+F) with a chapter number or the bold title above.`,
};

const retained = blocks.filter((block) => block.id !== contentsBlock.id);
const anchorIndex = retained.findIndex((block) => block.id === "headline_metrics");
if (anchorIndex < 0) throw new Error("Expected headline_metrics anchor");
retained.splice(anchorIndex + 1, 0, contentsBlock);
blocks.splice(0, blocks.length, ...retained);

const chapters = [
  ["task_explainer", 1],
  ["model_primer_takeaway", 2],
  ["pilot_field", 3],
  ["patterns", 4],
  ["finetuning_takeaway", 5],
  ["literature_takeaway", 6],
  ["architecture_takeaway", 7],
  ["run_portfolio_takeaway", 8],
  ["further_questions", 9],
];

for (const [id, chapter] of chapters) {
  const block = blocks.find((candidate) => candidate.id === id);
  if (!block || block.type !== "markdown" || typeof block.body !== "string") {
    throw new Error(`Expected markdown chapter anchor ${id}`);
  }
  const lines = block.body.split("\n");
  const headingIndex = lines.findIndex((line) => /^##\s+/.test(line));
  if (headingIndex < 0) throw new Error(`Chapter anchor ${id} has no level-two heading`);
  const title = lines[headingIndex]
    .replace(/^##\s+/, "")
    .replace(/^Chapter\s+\d+\s+·\s+/, "");
  lines[headingIndex] = `## Chapter ${chapter} · ${title}`;
  block.body = lines.join("\n");
}

const ids = blocks.map((block) => block.id);
if (new Set(ids).size !== ids.length) throw new Error("Navigation update produced duplicate block ids");

writeFileSync(artifactPath, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
process.stdout.write(`${JSON.stringify({
  ok: true,
  blocks: blocks.length,
  chapters: chapters.length,
  contentsIndex: blocks.findIndex((block) => block.id === contentsBlock.id),
})}\n`);
