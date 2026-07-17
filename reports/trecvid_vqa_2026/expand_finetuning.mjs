#!/usr/bin/env node

// Add the beginner fine-tuning chapter, public-weight model inventory, and
// responsive fine-tuning diagrams to the canonical report artifact.
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const artifactPath = join(root, "artifact.json");
const sqlPath = join(root, "queries", "finetuning_models.sql");
const artifact = JSON.parse(readFileSync(artifactPath, "utf8"));
const sql = readFileSync(sqlPath, "utf8");
const blocks = artifact?.manifest?.blocks;

if (!Array.isArray(blocks)) throw new Error("artifact.json is missing manifest.blocks");

function parseSqlValueList(tupleText) {
  const values = [];
  let index = 0;
  while (index < tupleText.length) {
    while (/[\s,]/.test(tupleText[index] ?? "")) index += 1;
    if (index >= tupleText.length) break;
    if (tupleText[index] === "'") {
      index += 1;
      let value = "";
      while (index < tupleText.length) {
        if (tupleText[index] === "'" && tupleText[index + 1] === "'") {
          value += "'";
          index += 2;
        } else if (tupleText[index] === "'") {
          index += 1;
          break;
        } else {
          value += tupleText[index];
          index += 1;
        }
      }
      values.push(value);
    } else {
      const start = index;
      while (index < tupleText.length && tupleText[index] !== ",") index += 1;
      const token = tupleText.slice(start, index).trim();
      const number = Number(token);
      values.push(Number.isFinite(number) ? number : token);
    }
  }
  return values;
}

function parseView(viewName) {
  const escaped = viewName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = new RegExp(
    `CREATE OR REPLACE VIEW\\s+${escaped}\\s+AS\\s+SELECT \\*\\s+FROM \\(VALUES([\\s\\S]*?)\\) AS t\\(([^)]+)\\);`,
    "i",
  ).exec(sql);
  if (!match) throw new Error(`Could not find SQL view ${viewName}`);
  const columns = match[2].split(",").map((column) => column.trim());
  const body = match[1];
  const rows = [];
  let index = 0;
  while (index < body.length) {
    while (index < body.length && body[index] !== "(") index += 1;
    if (index >= body.length) break;
    const start = ++index;
    let inString = false;
    while (index < body.length) {
      if (body[index] === "'") {
        if (inString && body[index + 1] === "'") {
          index += 2;
          continue;
        }
        inString = !inString;
      } else if (body[index] === ")" && !inString) {
        break;
      }
      index += 1;
    }
    if (index >= body.length) throw new Error(`Unclosed SQL tuple in ${viewName}`);
    const values = parseSqlValueList(body.slice(start, index));
    if (values.length !== columns.length) {
      throw new Error(`${viewName} tuple has ${values.length} values for ${columns.length} columns`);
    }
    rows.push(Object.fromEntries(columns.map((column, columnIndex) => [column, values[columnIndex]])));
    index += 1;
  }
  return rows;
}

const datasets = {
  finetuning_method_guide: parseView("finetuning_method_guide"),
  finetuning_2025_evidence: parseView("finetuning_2025_evidence"),
  finetuning_data_requirements: parseView("finetuning_data_requirements"),
  finetuning_hardware: parseView("finetuning_hardware"),
  open_model_inventory: parseView("open_model_inventory"),
  public_finetuned_models: parseView("public_finetuned_models"),
};

artifact.snapshot.datasets = { ...artifact.snapshot.datasets, ...datasets };

function upsertById(items, replacement, beforeId) {
  const retained = items.filter((item) => item.id !== replacement.id);
  const index = beforeId ? retained.findIndex((item) => item.id === beforeId) : -1;
  if (index >= 0) retained.splice(index, 0, replacement);
  else retained.push(replacement);
  items.splice(0, items.length, ...retained);
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

const tableDefinitions = [
  {
    id: "finetuning_method_guide",
    title: "Five levels of adaptation",
    subtitle: "SFT, ranking, and preference learning describe the learning objective; LoRA, QLoRA, and full tuning describe which parameters are updated.",
    dataset: "finetuning_method_guide",
    sourceId: "finetuning_guidance",
    defaultSort: { field: "read_order", direction: "asc" },
    density: "spacious",
    layout: "full",
    columns: [
      { field: "read_order", label: "Order", type: "number" },
      { field: "method", label: "Method", type: "text" },
      { field: "weights_updated", label: "What changes", type: "text" },
      { field: "what_it_is_for", label: "What it teaches", type: "text" },
      { field: "use_when", label: "Use when", type: "text" },
      { field: "main_risk", label: "Main risk", type: "text" },
    ],
  },
  {
    id: "finetuning_2025_evidence",
    title: "What the 2025 teams trained—and what they left frozen",
    subtitle: "Published recipes from participant papers; reported ablations are not independent replications.",
    dataset: "finetuning_2025_evidence",
    sourceId: "trec_2025_finetuning_evidence",
    defaultSort: { field: "read_order", direction: "asc" },
    density: "spacious",
    layout: "full",
    columns: [
      { field: "read_order", label: "Order", type: "number" },
      { field: "team", label: "Team", type: "text" },
      { field: "what_was_trained", label: "Training target", type: "text" },
      { field: "published_recipe", label: "Published recipe", type: "text" },
      { field: "reported_lesson", label: "Reported lesson", type: "text" },
      { field: "evidence_limit", label: "Evidence limit", type: "text" },
    ],
  },
  {
    id: "finetuning_data_requirements",
    title: "Training data, labels, and split rules",
    subtitle: "Scale bands are experiment-planning heuristics, not universal thresholds.",
    dataset: "finetuning_data_requirements",
    sourceId: "finetuning_guidance",
    defaultSort: { field: "read_order", direction: "asc" },
    density: "spacious",
    layout: "full",
    columns: [
      { field: "read_order", label: "Order", type: "number" },
      { field: "training_asset", label: "Training asset", type: "text" },
      { field: "practical_scale", label: "Planning scale", type: "text" },
      { field: "example_shape", label: "What one example contains", type: "text" },
      { field: "quality_gate", label: "Quality gate", type: "text" },
      { field: "split_rule", label: "Leakage rule", type: "text" },
    ],
  },
  {
    id: "finetuning_hardware",
    title: "GPU-memory planning bands",
    subtitle: "Broad engineering estimates for bounded video inputs; long clips, high resolution, BF16, larger batches, and unfrozen vision encoders can move a run to the next tier.",
    dataset: "finetuning_hardware",
    sourceId: "finetuning_guidance",
    defaultSort: { field: "read_order", direction: "asc" },
    density: "spacious",
    layout: "full",
    columns: [
      { field: "read_order", label: "Order", type: "number" },
      { field: "model_tier", label: "Model tier", type: "text" },
      { field: "frozen_or_quantized_inference", label: "Inference planning", type: "text" },
      { field: "qlora_planning_band", label: "QLoRA planning", type: "text" },
      { field: "bf16_lora_planning_band", label: "BF16 LoRA planning", type: "text" },
      { field: "operational_note", label: "Operational note", type: "text" },
    ],
  },
  {
    id: "open_model_inventory",
    title: "Official public-weight video model shortlist",
    subtitle: "Exact model repositories checked July 12, 2026; begin with instruction/post-trained checkpoints, not raw base weights.",
    dataset: "open_model_inventory",
    sourceId: "public_model_weights",
    defaultSort: { field: "read_order", direction: "asc" },
    density: "dense",
    layout: "full",
    columns: [
      { field: "read_order", label: "Read", type: "number" },
      { field: "checkpoint", label: "Exact repository", type: "text" },
      { field: "parameter_scale", label: "Scale", type: "text" },
      { field: "license_access", label: "License / access", type: "text" },
      { field: "media_input", label: "Video / audio", type: "text" },
      { field: "checkpoint_state", label: "Weight state", type: "text" },
      { field: "planning_hardware", label: "Planning hardware", type: "text" },
      { field: "competition_role_and_caveat", label: "Best role and caveat", type: "text" },
    ],
  },
  {
    id: "public_finetuned_models",
    title: "Public post-trained and task-specialized checkpoints",
    subtitle: "Originator/lab releases with explicit lineage; relevance does not mean they are drop-in submission models.",
    dataset: "public_finetuned_models",
    sourceId: "public_task_finetunes",
    defaultSort: { field: "read_order", direction: "asc" },
    density: "dense",
    layout: "full",
    columns: [
      { field: "read_order", label: "Read", type: "number" },
      { field: "checkpoint", label: "Exact repository", type: "text" },
      { field: "base_model", label: "Base lineage", type: "text" },
      { field: "post_training", label: "Post-training", type: "text" },
      { field: "license_access", label: "License / access", type: "text" },
      { field: "useful_role", label: "Useful role", type: "text" },
      { field: "main_caveat", label: "Main caveat", type: "text" },
    ],
  },
];

for (const table of tableDefinitions) upsertById(artifact.manifest.tables, table, "literature_review");

const newSources = [
  {
    id: "finetuning_guidance",
    label: "Official TRL, PEFT, and model fine-tuning guidance",
    href: "https://huggingface.co/docs/trl/v0.26.0/en/sft_trainer",
    path: "queries/finetuning_models.sql",
  },
  {
    id: "trec_2025_finetuning_evidence",
    label: "TREC 2025 participant fine-tuning evidence",
    href: "https://trec.nist.gov/pubs/trec34/papers/CERTH-ITI.avs.vqa.pdf",
    path: "queries/finetuning_models.sql",
  },
  {
    id: "public_model_weights",
    label: "Official public-weight video model cards",
    href: "https://huggingface.co/collections/Qwen/qwen3-vl",
    path: "queries/finetuning_models.sql",
  },
  {
    id: "public_task_finetunes",
    label: "Originator-released video task fine-tunes",
    href: "https://huggingface.co/OpenGVLab/VideoChat-R1_5-7B",
    path: "queries/finetuning_models.sql",
  },
];

for (const source of newSources) upsertById(artifact.manifest.sources, source);

const fineTuningBlocks = [
  {
    id: "finetuning_takeaway",
    type: "markdown",
    layout: "full",
    sourceId: "finetuning_guidance",
    body: `## Fine-tuning can help—but it is not the first repair

**Beginner answer:** fine-tuning means continuing training from an existing checkpoint so its behavior shifts toward this task. It is useful when a repeatable error remains **after** the system reliably decodes the video, retrieves the right moments, includes speech or sound when needed, and emits valid AG/MC output.

For this competition, the safest order is **frozen baseline → learned ranker or LoRA/QLoRA adapter → larger adapter experiment → full fine-tune only as a last resort**. A missing causal moment, broken video, omitted alarm sound, or leaked split will not become correct because more weights were updated.

**Recommended first train:** a QLoRA adapter on a 2–4B instruction checkpoint plus a separate evidence-aware AG ranker. Keep the vision/audio encoders frozen initially. Compare against the exact same frozen baseline and evidence pack before escalating to 7–9B.`,
  },
  imageBlock({
    id: "finetuning_decision_image",
    asset: "finetuning-decision.svg",
    title: "Choose the lightest training method that clears validation",
    alt: "A three-step decision ladder. First, improve prompting and evidence retrieval without changing weights. Second, train LoRA or QLoRA adapters if repeatable task-specific errors remain. Third, consider a full fine-tune only after adapters plateau and there is enough clean data, compute, and engineering support. A validation gate applies before every step while the test set stays locked.",
    caption: "The ladder is about evidence, not prestige. Advance only when a controlled validation comparison shows that the lighter method has plateaued.",
  }),
  {
    id: "finetuning_vocabulary",
    type: "markdown",
    layout: "full",
    sourceId: "finetuning_guidance",
    body: `## What “fine-tuning” actually means

- **Checkpoint:** a saved set of model weights. “Base” or “pretrained” checkpoints know broad patterns but are not necessarily trained to follow questions; **Instruct** or **post-trained** checkpoints are the normal competition starting point.
- **SFT — supervised fine-tuning:** show the model an input and the answer or behavior it should produce. SFT is the *learning objective*; it does not say how many weights change.
- **LoRA:** freeze the checkpoint and train small low-rank adapter matrices. The saved adapter is much smaller than another full copy of the model.
- **QLoRA:** LoRA while the frozen checkpoint is loaded in 4-bit form. It reduces base-weight memory, but video activations and context still cost memory.
- **Ranker/head tuning:** train only the final judge that orders AG candidates or scores MC options. The video generator can remain frozen.
- **Preference training — DPO, MPO, or RL variants:** teach that one output is better than another. This can improve judging or reasoning, but it adds data and evaluation complexity.
- **Full fine-tune:** update the whole model. It uses much more memory and risks forgetting general skills; it is rarely the right first move on a 500-video domain set.

Official [TRL SFTTrainer guidance](https://huggingface.co/docs/trl/v0.26.0/en/sft_trainer) supports vision-language training, and [PEFT LoRA](https://huggingface.co/docs/peft/main/en/package_reference/lora) / [quantization guidance](https://huggingface.co/docs/peft/developer_guides/quantization) covers adapter and QLoRA setup. For VLMs, preserve image/video tokens—TRL explicitly warns against text-style truncation that can remove them.`,
  },
  { id: "finetuning_method_guide_block", type: "table", layout: "full", tableId: "finetuning_method_guide" },
  {
    id: "finetuning_2025_takeaway",
    type: "markdown",
    layout: "full",
    sourceId: "trec_2025_finetuning_evidence",
    body: `## What last year’s teams actually fine-tuned

The word covered **three different system choices**.

**CERTH-ITI adapted the multimodal model.** Its [official 2025 paper](https://trec.nist.gov/pubs/trec34/papers/CERTH-ITI.avs.vqa.pdf) reports LoRA SFT of Qwen2.5-Omni-7B on the organizer’s 500 training videos: 10 frames / 1,024 visual tokens, five epochs, rank 16, alpha 64, on one 24 GB RTX 4090. Its participant-reported MC ablation moved Top-1 from 73.2% frozen to 74.4% fine-tuned; adding ten-view aggregation reached 76.1%. That is evidence of a modest adapter gain and a larger combined system gain—not proof that tuning alone caused the final leaderboard position.

**HLTCOE adapted the judge, not the video generator.** Its [participant paper](https://trec.nist.gov/pubs/trec34/papers/HLTCOE.vqa.pdf) describes LoRA on a listwise reranker using 500 examples, a 90/10 split, five epochs, rank 16, alpha 32, and dropout 0.05. The private submission adapter was not released.

**NII_UIT kept both main backbones frozen.** Its [paper](https://trec.nist.gov/pubs/trec34/papers/NII_UIT.vqa.pdf) instead invested in timestamped speech, time-spaced frames, local tiles, and question-type prompts. That is an important baseline: sometimes the best “adaptation” is better evidence.`,
  },
  { id: "finetuning_2025_evidence_block", type: "table", layout: "full", tableId: "finetuning_2025_evidence" },
  {
    id: "finetuning_data_takeaway",
    type: "markdown",
    layout: "full",
    sourceId: "finetuning_guidance",
    body: `## What data a defensible fine-tune needs

One training example should retain **video identity, question, exact visual/audio evidence or timestamps, answer target, output format, and provenance**. For MC, store all four original option strings and the correct option. For AG ranking, store the candidate pool, evidence references, relevance/order labels, and how those labels were created.

The critical rule is **split by video and source before training**, not by QA row. If two questions from the same clip land in train and validation, the model can memorize scenes, people, subtitles, or audio and look more general than it is. Near-duplicate clips, movie segments, channel families, and source collections also stay together.

Use 100–500 checked examples to prove the pipeline; roughly 1,000–5,000 diverse grounded examples is a reasonable first adapter band; 10,000+ can support broader adaptation. These are planning heuristics, not magic thresholds. A small, balanced, timestamped set is often more informative than a large pile of unverified generated answers.`,
  },
  imageBlock({
    id: "finetuning_pipeline_image",
    asset: "finetuning-pipeline.svg",
    title: "A fine-tuning pipeline that keeps the test honest",
    alt: "Five stages: build grounded examples with questions, timestamped frames or audio, and correct answers; split by video into training, validation, and a locked test; train LoRA or QLoRA adapters on a frozen base; select a checkpoint using validation only; then evaluate Answer Generation, Multiple Choice, schema validity, IDs, and timing separately.",
    caption: "Every stage produces an auditable artifact: grounded examples, split manifests, adapter checkpoints, validation decisions, and separate AG/MC outputs.",
  }),
  { id: "finetuning_data_requirements_block", type: "table", layout: "full", tableId: "finetuning_data_requirements" },
  {
    id: "finetuning_hardware_takeaway",
    type: "markdown",
    layout: "full",
    sourceId: "finetuning_guidance",
    body: `## What compute, storage, and software are needed

**GPU memory:** a 2–4B QLoRA experiment is the comfortable one-workstation starting point. A bounded 7–9B adapter run can fit one 24 GB GPU—the CERTH recipe is a real example—but 48–80 GB gives safer room for more frames, higher resolution, and useful batch sizes. Full fine-tuning usually means multiple GPUs.

**Disk and data operations:** keep the original media manifest, normalized videos, frame/audio caches, training JSONL or Parquet, exact split files, base revision, adapter checkpoints, optimizer state for resumable runs, metrics, prompts, and environment lockfile. Media caches can easily require tens to hundreds of gigabytes even when the adapter itself is only hundreds of megabytes.

**Software stack:** use the model’s official processor plus pinned versions of Transformers, TRL, PEFT, bitsandbytes for QLoRA, and Accelerate/FSDP or DeepSpeed only when distribution is required. The official [Qwen3-VL repository includes fine-tuning code](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune). Run a 10-example overfit test and checkpoint-reload test before spending on a long run.

The ranges below are planning bands—not guarantees. Video tokens, resolution, clip length, precision, attention kernels, gradient checkpointing, batch size, and whether the vision/audio towers are frozen can move any recipe to the next tier.`,
  },
  { id: "finetuning_hardware_block", type: "table", layout: "full", tableId: "finetuning_hardware" },
  {
    id: "open_models_takeaway",
    type: "markdown",
    layout: "full",
    sourceId: "public_model_weights",
    body: `## Public weights: a practical 2026 shortlist

**Default path:** start with [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), run [InternVL3.5-4B](https://huggingface.co/OpenGVLab/InternVL3_5-4B) as an architecture-diverse comparison, and use [SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) as the cheap dataset/training sandbox. Escalate to 8B only if a controlled learning curve supports it.

For sound-heavy questions, visual-only models need timestamped ASR plus a non-speech audio path. A frozen native-audio expert such as [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) or [MiniCPM-o-4.5](https://huggingface.co/openbmb/MiniCPM-o-4_5) can handle speech and sound, but omni-model fine-tuning is substantially more complex.

“Public weights” is not one promise. Check whether access is immediate or gated; whether the license covers your use; whether custom code is required; whether the model card names its training data; and whether the checkpoint is **base, Instruct, Thinking, SFT-only, preference-optimized, or task-specialized**. Pin the exact Hub revision and processor.`,
  },
  { id: "open_model_inventory_block", type: "table", layout: "full", tableId: "open_model_inventory" },
  {
    id: "public_finetunes_takeaway",
    type: "markdown",
    layout: "full",
    sourceId: "public_task_finetunes",
    body: `## Public fine-tuned models: use them as specialists, not magic replacements

There are useful originator-released video fine-tunes: [VideoChat-R1.5](https://huggingface.co/OpenGVLab/VideoChat-R1_5-7B) revisits predicted evidence intervals; [Video-R1](https://huggingface.co/Video-R1/Video-R1-7B) exposes public video SFT/RL weights; [SAGE-MM Molmo2](https://huggingface.co/allenai/SAGE-MM-Molmo2-8B-SFT_RL) learns when to call ASR and temporal/high-resolution tools; and [VideoRefer-VideoLLaMA3](https://huggingface.co/DAMO-NLP-SG/VideoRefer-VideoLLaMA3-7B) specializes in spatial-temporal grounding.

Their training objectives often produce long reasoning traces, tool-call JSON, multiple-choice answers, or boxes—not TRECVID’s exact final AG/MC contract. The safest use is as a **retriever, verifier, judge, or grounding specialist** behind a thin competition output head.

Do not treat a Hub “finetunes” count as evidence of quality. Adopt a derivative only if it documents the exact base revision, license, training data, objective, evaluation setup, processor, and reproducible code. The CERTH and HLTCOE competition-specific 2025 adapters themselves were not released as reproducible public checkpoints.`,
  },
  { id: "public_finetuned_models_block", type: "table", layout: "full", tableId: "public_finetuned_models" },
  {
    id: "finetuning_plan",
    type: "markdown",
    layout: "full",
    sourceId: "finetuning_guidance",
    body: `## A safe fine-tuning plan for this year

1. **Freeze the evaluation contract.** Build video/source-exclusive train, validation, and locked test manifests. Define AG top-1/top-5/top-10, MC Top-1/MRR, option-order stability, evidence recall, latency, and schema validity.
2. **Run frozen checkpoints on identical evidence.** Compare SmolVLM2-2.2B, Qwen3-VL-4B, and InternVL3.5-4B. Add one frozen native-audio expert only for questions where sound is diagnostic.
3. **Label the failure, not just the answer.** Mark decode, retrieval, OCR, ASR/audio, temporal ordering, reasoning, candidate generation, ranking, and format failures. Fine-tune only the stage with repeatable residual error.
4. **Train two small experiments first.** (a) a QLoRA SFT adapter on the 2–4B primary model; (b) a separate evidence-aware listwise AG ranker or MC option scorer. Keep the vision/audio encoders frozen.
5. **Use learning curves and ablations.** Train on 25%, 50%, and 100% of the clean set; repeat at least two seeds; compare the same evidence pack; report per-category effects and confidence intervals. Keep an oracle-evidence run to tell retrieval from reasoning.
6. **Escalate only on a validation gate.** Move to a 7–9B adapter when the small-model pipeline is correct and the gain is capacity-limited. Add an ensemble only when errors are complementary. Full fine-tuning is a no-go unless adapters plateau on thousands of diverse examples and the expected gain justifies multi-GPU operations.
7. **Package reproducibility with every run.** Save base revision, adapter, processor, prompts, evidence IDs/timestamps, split hash, environment, seed, training curve, evaluation outputs, and schema-validation receipt.

**Recommended submission portfolio:** frozen evidence-first baseline; 4B QLoRA adapter; ranker-only variant; architecture-diverse frozen or tuned comparison; and an audio-specialist ensemble only if the audio subset gains without harming silent questions.`,
  },
  {
    id: "finetuning_watchouts",
    type: "markdown",
    layout: "full",
    sourceId: "finetuning_guidance",
    body: `## What fine-tuning cannot repair

- **The relevant moment was never sampled.** Fix retrieval and frame coverage.
- **Speech or a non-speech cue was discarded.** Add timestamped ASR and an audio branch.
- **The label is wrong, ambiguous, or generated from the test answer.** Audit or remove it.
- **The same video or near-duplicate appears in train and validation.** Re-split before trusting any gain.
- **MC option order is a shortcut.** Permute options and test no-video controls.
- **AG candidates are repetitive or unsupported.** Improve generation, deduplication, evidence scoring, and listwise ranking.
- **The output violates exact IDs, ranks, counts, option strings, timing, or JSON.** Use a deterministic schema validator after the model.
- **A public checkpoint has unclear license or lineage.** Do not build the submission around it until those terms are resolved.

A fine-tune is successful only when it beats the frozen baseline on a locked, source-separated validation set **and** preserves evidence quality, calibration, runtime, and schema reliability.`,
  },
];

const replacementIds = new Set(fineTuningBlocks.map((block) => block.id));
const retainedBlocks = blocks.filter((block) => !replacementIds.has(block.id));
const anchorIndex = retainedBlocks.findIndex((block) => block.id === "repo_audit_block");
if (anchorIndex < 0) throw new Error("Expected repo_audit_block anchor");
retainedBlocks.splice(anchorIndex + 1, 0, ...fineTuningBlocks);
blocks.splice(0, blocks.length, ...retainedBlocks);

const blockIds = blocks.map((block) => block.id);
const tableIds = artifact.manifest.tables.map((table) => table.id);
const sourceIds = artifact.manifest.sources.map((source) => source.id);
if (new Set(blockIds).size !== blockIds.length) throw new Error("Duplicate block ids after fine-tuning expansion");
if (new Set(tableIds).size !== tableIds.length) throw new Error("Duplicate table ids after fine-tuning expansion");
if (new Set(sourceIds).size !== sourceIds.length) throw new Error("Duplicate source ids after fine-tuning expansion");

writeFileSync(artifactPath, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
process.stdout.write(`${JSON.stringify({
  ok: true,
  blocks: blocks.length,
  htmlBlocks: blocks.filter((block) => block.type === "html").length,
  tables: artifact.manifest.tables.length,
  datasets: Object.keys(artifact.snapshot.datasets).length,
})}\n`);
