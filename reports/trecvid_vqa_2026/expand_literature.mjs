#!/usr/bin/env node

// Expand the competition-adjacent literature section from the canonical SQL
// mirrors, keeping artifact.json and the visible report crosswalk in sync.
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const artifactPath = join(root, "artifact.json");
const sqlPath = join(root, "queries", "literature_review.sql");
const artifact = JSON.parse(readFileSync(artifactPath, "utf8"));
const sql = readFileSync(sqlPath, "utf8");

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
  const tuples = [];
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
    tuples.push(Object.fromEntries(columns.map((column, columnIndex) => [column, values[columnIndex]])));
    index += 1;
  }
  return tuples;
}

const datasets = {
  research_threads: parseView("research_threads"),
  research_concept_guide: parseView("research_concept_guide"),
  team_literature_map: parseView("team_literature_map"),
  literature_review: parseView("literature_review"),
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

const researchThreadsTable = {
  id: "research_threads",
  title: "Six research threads that transfer to TRECVID",
  subtitle: "Field direction, closest 2025 systems, and the lowest-risk 2026 experiment for each thread.",
  dataset: "research_threads",
  sourceId: "team_literature_synthesis",
  defaultSort: { field: "priority", direction: "asc" },
  density: "spacious",
  layout: "full",
  columns: [
    { field: "priority", label: "Priority", type: "number" },
    { field: "thread", label: "Research thread", type: "text" },
    { field: "field_direction", label: "What the field is doing", type: "text" },
    { field: "closest_2025_teams", label: "Closest 2025 teams", type: "text" },
    { field: "recommended_2026_test", label: "Competition-ready test", type: "text" },
  ],
};

const researchConceptGuideTable = {
  id: "research_concept_guide",
  title: "Six ideas in plain language",
  subtitle: "Use the diagnostic question to identify which stage failed before changing the backbone.",
  dataset: "research_concept_guide",
  sourceId: "team_literature_synthesis",
  defaultSort: { field: "priority", direction: "asc" },
  density: "spacious",
  layout: "full",
  columns: [
    { field: "priority", label: "Step", type: "number" },
    { field: "idea", label: "Idea", type: "text" },
    { field: "plain_meaning", label: "Plain meaning", type: "text" },
    { field: "diagnostic_question", label: "Diagnostic question", type: "text" },
    { field: "evidence_to_store", label: "What to log", type: "text" },
    { field: "first_experiment", label: "First experiment", type: "text" },
  ],
};

const teamLiteratureTable = {
  id: "team_literature_map",
  title: "Where each 2025 team sits in the research landscape",
  subtitle: "Interpretive crosswalk, not a causal attribution of leaderboard performance.",
  dataset: "team_literature_map",
  sourceId: "team_literature_synthesis",
  defaultSort: { field: "team", direction: "asc" },
  density: "spacious",
  layout: "full",
  columns: [
    { field: "team", label: "Team", type: "text" },
    { field: "research_lineage", label: "Closest lineage", type: "text" },
    { field: "primary_anchors", label: "Primary anchors", type: "text" },
    { field: "what_literature_supports", label: "What literature supports", type: "text" },
    { field: "frontier_gap", label: "Gap versus frontier", type: "text" },
    { field: "experiment_2026", label: "Concrete 2026 experiment", type: "text" },
  ],
};

const literatureTable = artifact.manifest.tables.find((table) => table.id === "literature_review");
if (!literatureTable) throw new Error("Expected literature_review table");
Object.assign(literatureTable, {
  title: "Competition-adjacent primary literature",
  subtitle: "Twenty-five primary works ordered by competition relevance; proximity describes transfer fit, not paper quality.",
  dataset: "literature_review",
  sourceId: "literature_primary_sources",
  defaultSort: { field: "read_order", direction: "asc" },
  density: "dense",
  layout: "full",
  columns: [
    { field: "read_order", label: "Read", type: "number" },
    { field: "proximity", label: "Transfer fit", type: "text" },
    { field: "evidence_status", label: "Evidence status", type: "text" },
    { field: "thread", label: "Thread", type: "text" },
    { field: "work", label: "Primary work", type: "text" },
    { field: "year", label: "Year", type: "number" },
    { field: "contribution", label: "What it does", type: "text" },
    { field: "closest_2025_teams", label: "Closest 2025 teams", type: "text" },
    { field: "competition_takeaway", label: "TRECVID implication", type: "text" },
  ],
});

upsertById(artifact.manifest.tables, researchThreadsTable, "literature_review");
upsertById(artifact.manifest.tables, researchConceptGuideTable, "literature_review");
upsertById(artifact.manifest.tables, teamLiteratureTable, "literature_review");

const literatureTakeaway = artifact.manifest.blocks.find((block) => block.id === "literature_takeaway");
if (!literatureTakeaway) throw new Error("Expected literature_takeaway block");
literatureTakeaway.body = `## Competition-adjacent research points to one system pattern

This is not one new model architecture. It is a way to divide the job into **six testable problems**: choose what the model sees, check whether it found the right moments, preserve both the timeline and small details, keep speech and other sound, connect evidence into a grounded explanation, and select the final answer without being fooled by wording or option order.

### A running example for the whole chapter

Imagine a 40-second kitchen video. At 8 seconds a pan begins to smoke, at 15 seconds an alarm sounds, and at 27 seconds a person opens a window. The question is: **“Why did the person open the window?”** A credible system must find separated moments, connect them in the right order, and resist easy but unsupported guesses such as “to cool the room.”

### Five terms translated before we use them

- **Visual token:** a unit of model input created from image patches. It is not cryptocurrency. More frames and larger crops consume more of the context and compute budget.
- **Retrieval:** finding the moments that contain useful evidence.
- **Fusion:** combining facts from more than one retrieved moment.
- **Grounding:** keeping a timestamp, frame, crop, transcript span, or sound event that supports a claim.
- **Calibration:** making a score behave like reliability rather than reflecting answer length, repeated wording, or MC option position.

The short version is: **find the evidence, preserve it, combine it, then decide.** The next six sections reuse the kitchen example in prose, add a different concrete example in each image, identify the closest papers and 2025 teams, and end with one controlled TRECVID experiment.`;

const additionalBlocks = [
  {
    id: "research_budget_explainer",
    type: "markdown",
    layout: "full",
    sourceId: "team_literature_synthesis",
    body: `## Idea 1 — Spend the frame budget on coverage and relevance

**Plain meaning:** a video-language model usually does not watch every video frame at full resolution. A sampler chooses a small set of frames, and the vision encoder turns each frame into many visual tokens. A longer or higher-resolution input therefore costs more memory, context, latency, and often money. The real question is not “How many frames can we send?” but “Which combination is least likely to omit the answer?”

**Everyday analogy:** imagine receiving a 200-page security-camera printout but being allowed to keep only 12 pages. Uniform sampling is like keeping every seventeenth page. Question-only selection is like searching for pages that resemble the word “window.” A safer hybrid reserves some pages for the whole story and spends the rest on likely evidence.

**Kitchen example:** a uniform sampler might capture the person opening the window but miss the earlier smoke. A question-relevance sampler might over-focus on frames containing the window and also miss the cause. A coverage-plus-query sampler keeps beginning, middle, and end coverage, then adds smoke-, alarm-, and window-related moments.

**What the literature adds:** [KFS-Bench](https://arxiv.org/abs/2512.14017) makes multi-scene coverage measurable; [BOLT](https://arxiv.org/abs/2503.21483) samples from a question-relevance distribution instead of taking only the top-scoring frames; [VideoTree](https://arxiv.org/abs/2405.19209) searches a coarse-to-fine temporal hierarchy.

**Closest 2025 systems:** NII_UIT's roughly twelve time-spaced frames emphasize coverage. CERTH-ITI's multiple random views increase the chance that at least one view contains the evidence. Neither comparison proves causality; they provide sensible baselines for a controlled selector test.

**TRECVID test:** at fixed budgets of 12, 24, and 32 frames, compare uniform-only, question-only, random multi-view, and half-uniform plus half-question selection. Record selected timestamps, total tokens, latency, answer score, and—on a small labeled slice—required-evidence recall.`,
  },
  imageBlock({
    id: "research_budget_image",
    asset: "research-token-budget.svg",
    title: "A frame budget is an allocation problem",
    alt: "A timeline for the question what does Maya do after the alarm shows many possible moments but a budget of only three frames. The chosen frames preserve the opening context, the alarm ringing, and Maya leaving afterward.",
    caption: "Reserve some frames for the whole story, then spend the remaining budget on likely evidence. Pure relevance can miss the causal setup; pure uniform sampling can miss brief details.",
  }),
  {
    id: "research_retrieval_fusion_explainer",
    type: "markdown",
    layout: "full",
    sourceId: "team_literature_synthesis",
    body: `## Idea 2 — Finding evidence and combining evidence are different skills

**Retrieval** asks whether the system found the right moments. **Fusion** asks whether the model combined those moments correctly. Treating them as one end-to-end score hides the cause of failure and encourages expensive model changes when the sampler may be the real problem.

**Everyday analogy:** retrieval is the detective collecting the correct photographs; fusion is the jury using those photographs to reach the right conclusion. A perfect jury cannot use an unseen photograph, and a complete evidence folder does not guarantee that the jury reasons correctly.

**Kitchen example:** if the sampler retrieves only the window-opening frame, “to cool the room” is a plausible guess caused by retrieval failure. If it retrieves the smoking pan and the open window but still gives that answer, the evidence is present and the failure is fusion. If giving the model the human-selected frames fixes the answer, the backbone already had enough reasoning ability for that example.

**The oracle-frame test:** for a small development set, label the minimum moments needed to answer. Run the model twice—once on its retrieved frames and once on the labeled frames. The gap between end-to-end and oracle-frame performance estimates the retrieval problem; errors that remain with oracle frames point toward fusion, prompting, or answer selection.

**What the literature adds:** [HERBench](https://arxiv.org/abs/2512.14870) explicitly separates missing-evidence and present-but-not-integrated failures. [FALCONEye](https://arxiv.org/abs/2503.19850) couples an open answer with a supporting temporal window. KFS-Bench provides scene-level targets for auditing selectors.

**Closest 2025 systems:** NII_UIT is the clearest evidence-engine baseline; HLTCOE is the clearest answer-selection baseline. Neither paper reports the oracle-frame diagnostic, so their score does not reveal which stage dominated errors.

**TRECVID test:** label required moments for 50–100 representative development questions. Report selector recall, oracle-frame answer quality, and normal end-to-end quality. Change the selector only when retrieval is weak; change reasoning or ranking only when oracle performance is weak.`,
  },
  imageBlock({
    id: "research_retrieval_fusion_image",
    asset: "research-retrieval-fusion.svg",
    title: "Two failures can produce the same wrong answer",
    alt: "For the question why Leo goes back inside, the system first retrieves rain starting and the umbrella remaining indoors, then fuses both clues into the answer that Leo returns for the umbrella because it started raining.",
    caption: "A single final score cannot tell whether evidence was missing or reasoning failed. Oracle frames turn that hidden distinction into a measurable diagnostic.",
  }),
  {
    id: "research_global_local_explainer",
    type: "markdown",
    layout: "full",
    sourceId: "team_literature_synthesis",
    body: `## Idea 3 — Keep the whole timeline, then zoom into a few moments

Video understanding has two competing needs. **Global chronology** preserves which events happened and in what order. **Local detail** preserves small text, objects, facial expressions, hand actions, and screen content. Sending every moment at full resolution is usually wasteful; compressing everything can erase the exact detail the question needs.

**Everyday analogy:** a subway map shows the entire route but not the storefront beside one station. A street photograph shows that storefront but not where it sits in the journey. A useful evidence pack contains both: a map of the story and close-ups of selected stops.

**Kitchen example:** a small timeline panel may show smoke, the alarm reaction, and the open window in order. But the smoke may be only a few gray pixels, and the stove display or product label may be unreadable. After selecting the likely cause and outcome, the system can request native-resolution crops around the pan, alarm, hands, or window.

**What the literature adds:** [Video Panels](https://arxiv.org/abs/2509.23724) packs multiple moments into compact panels for temporal coverage. [Qwen2.5-VL](https://arxiv.org/abs/2502.13923) supports dynamic-resolution visual input, while [TimeChat](https://arxiv.org/abs/2312.02051) emphasizes retaining timestamp identity rather than treating frames as an unordered gallery.

**Closest 2025 system:** NII_UIT explicitly combined time-spaced frames with spatial tiling. That is directionally close to global-plus-local evidence, although the paper does not isolate which part produced the gain.

**TRECVID test:** under one visual-token budget, compare separate global frames, compact panels, tiles for every selected frame, and global frames plus tiles only around question-relevant moments. Break out OCR, small-object, counting, and temporal questions; an aggregate score can hide the tradeoff.`,
  },
  imageBlock({
    id: "research_global_local_image",
    asset: "research-global-local.svg",
    title: "A timeline answers where; a crop answers what",
    alt: "A low-detail global race timeline locates the finish at three minutes forty-two seconds. A high-resolution crop of that chosen moment makes the winner's small bib number 27 readable.",
    caption: "Use cheap global views to preserve chronology and expensive local crops only where small details could change the answer.",
  }),
  {
    id: "research_audio_explainer",
    type: "markdown",
    layout: "full",
    sourceId: "team_literature_synthesis",
    body: `## Idea 4 — A transcript and the soundtrack carry different evidence

**Automatic speech recognition (ASR)** converts spoken words into searchable text with timestamps. That is ideal for dialogue, narration, names, and verbal instructions. **Native audio** preserves signals that are not words: alarms, impacts, applause, music, animal sounds, silence, tone, speaker overlap, and the exact timing between sound and action.

**Everyday analogy:** a transcript is closed captions; native audio is the soundtrack. Reading “I am fine” does not reveal whether it was whispered, shouted, or sarcastic, and many systems omit a sound such as a beep entirely.

**Kitchen example:** Whisper may transcribe the person's speech but represent the alarm only as an uncertain tag—or omit it. Native audio can preserve that a sharp alarm began after the smoke and before the window opened. Conversely, native audio alone may make exact spoken words harder for a general video model to recover than a clean timestamped transcript.

**What the literature adds:** [Whisper](https://proceedings.mlr.press/v202/radford23a.html) provides a strong timestamped speech path. [Qwen2.5-Omni](https://arxiv.org/abs/2503.20215) is designed for time-aligned native audio and video. [Video-MME](https://arxiv.org/abs/2405.21075) demonstrates the value of matched visual-, subtitle-, and audio-condition evaluation rather than assuming one modality always helps.

**Closest 2025 systems:** NII_UIT and HLTCOE exposed transcript evidence; CERTH-ITI used a native audio-video model. These systems were not evaluated in a shared modality ablation, so the leaderboard cannot tell which audio representation was best.

**TRECVID test:** run the same frozen model and evidence sampler in four conditions: video only, video plus ASR, video plus native audio, and video plus both. Store transcript confidence and sound-event timestamps, then inspect dialogue, sound-event, causal, and temporal slices separately.`,
  },
  imageBlock({
    id: "research_audio_image",
    asset: "research-audio.svg",
    title: "Speech text and native sound are complementary",
    alt: "For a question asking which gate the announcer says is closing, original audio preserves voice, timing, and emphasis while speech-to-text provides the searchable words Gate 12 is now closing. Together they support Gate 12.",
    caption: "ASR makes words easy to search; native audio keeps non-speech events and timing. Test both branches instead of treating one as a substitute for the other.",
  }),
  {
    id: "research_causal_explainer",
    type: "markdown",
    layout: "full",
    sourceId: "team_literature_synthesis",
    body: `## Idea 5 — A causal answer needs an inspectable evidence chain

A model can produce a fluent explanation from general world knowledge without using the video. **Grounded causal reasoning** requires the answer to connect specific observed events and preserve where each claim came from. The goal is not a long hidden chain-of-thought; it is a short external receipt that another person or program can check.

**Kitchen example:** the evidence chain might be “pan begins smoking at 8 seconds → alarm sounds at 15 seconds → person opens the window at 27 seconds.” The answer “because the pan was smoking and triggered an alarm” is supported. “To cool the room” may be plausible in ordinary life, but the video evidence does not support it.

**Why order matters:** showing the smoke and window frames is not enough if their timestamps are lost or reversed. Cause should precede the response, and intervening evidence can distinguish a direct cause from a coincidence. A system should therefore keep frame IDs, timestamps, transcript spans, and audio-event times through every stage.

**What the literature adds:** [NExT-QA](https://arxiv.org/abs/2105.08276) highlights the gap between describing actions and explaining temporal or causal relations. [CaST-Bench](https://arxiv.org/abs/2605.23216) makes cause-to-effect evidence chains explicit. [HERBench](https://arxiv.org/abs/2512.14870) makes several separated cues unavoidable.

**Closest 2025 systems:** every team could benefit from evidence traces, but the public papers mainly describe inputs and outputs rather than a scored causal chain. This is a frontier gap, not a claim that their answers were ungrounded.

**TRECVID test:** for a small causal slice, label two- or three-step evidence chains. Require each candidate answer to cite timestamps, then score answer correctness, evidence recall, order consistency, and performance when cited evidence is removed.`,
  },
  imageBlock({
    id: "research_causal_image",
    asset: "research-causal.svg",
    title: "A grounded answer carries its receipts",
    alt: "For the question why the cyclist stops, a timestamped chain shows the traffic signal turning red at twelve seconds followed by the cyclist braking and waiting at fourteen seconds, supporting the answer that the red light caused the stop.",
    caption: "Grounding keeps the observed cause, intermediate cue, and outcome attached to the answer. Plausibility alone is not video understanding.",
  }),
  {
    id: "research_election_explainer",
    type: "markdown",
    layout: "full",
    sourceId: "team_literature_synthesis",
    body: `## Idea 6 — Brainstorming, judging, and confidence should be separate stages

**Generation** produces plausible candidates. **Election** chooses among them using evidence and agreement. **Calibration** checks whether the numerical confidence behaves like reliability rather than rewarding repetition, short answers, familiar phrasing, or a preferred MC option position.

**Everyday analogy:** brainstorming and judging are different meetings. The first benefits from diversity; the second needs a rubric. Counting identical wording is a weak vote because “because of smoke,” “the pan was smoking,” and “smoke triggered the alarm” may express the same underlying answer.

**Kitchen example:** suppose the model generates “to cool the room,” “because smoke filled the kitchen,” and “because the smoke alarm sounded.” Semantic clustering prevents paraphrases from occupying several AG ranks. Evidence-aware election prefers candidates that cite the smoking pan and alarm. For MC, shuffling A/B/C/D and mapping scores back reveals whether the model favors an answer position rather than the content.

**What the literature adds:** [VCSS](https://arxiv.org/abs/2503.20472) and [Video Parallel Scaling](https://arxiv.org/abs/2509.08016) aggregate complementary visual views. [RADI](https://arxiv.org/abs/2403.14430) studies open-answer ordering with incomplete labels. [RLER](https://arxiv.org/abs/2604.04379) elects candidates using explicit evidence properties. [BOLD](https://aclanthology.org/2025.acl-long.162/) targets MC option-position bias.

**Closest 2025 systems:** CERTH-ITI used multi-view voting and score pooling; HLTCOE learned a listwise reranker; kslab used confidence and embedding similarity; WHU-NERCMS used a generated free-text anchor for MC matching. They are different approximations of an election layer.

**TRECVID test:** AG should generate more than ten candidates, cluster semantic duplicates, and compare confidence-only with evidence-aware ranking while reporting top-1, oracle-at-k, official top-10, and cost. MC should score options directly, permute their order, map scores back, and pool calibrated logits across evidence views.`,
  },
  imageBlock({
    id: "research_election_image",
    asset: "research-election.svg",
    title: "Generate several answers, then elect with evidence",
    alt: "Video evidence shows Ana pulling a red suitcase aboard a train. The system proposes bag, red suitcase, and backpack, then ranks red suitcase first because it is exact and evidence-backed, bag second as too vague, and backpack last as contradicted.",
    caption: "Diversity belongs in generation; evidence and calibration belong in selection. Exact-string voting and raw option-letter confidence are fragile shortcuts.",
  }),
  {
    id: "research_concept_guide_takeaway",
    type: "markdown",
    layout: "full",
    body: `## Use the six ideas as a diagnostic checklist

The table below compresses the chapter into questions a team can answer from logs. It is deliberately stage-based: before replacing the backbone, ask whether the video decoded, whether the needed moments were selected, whether detail or sound was lost, whether the model could answer from oracle evidence, and whether a later election rule overrode a good candidate.`,
  },
  { id: "research_concept_guide_block", type: "table", layout: "full", tableId: "research_concept_guide" },
  {
    id: "research_threads_takeaway",
    type: "markdown",
    layout: "full",
    body: `## The research threads translate into six controlled experiments

Each row below changes one major mechanism while holding the rest of the pipeline as stable as possible. The point is not to reproduce every paper. It is to borrow the diagnostic idea, test it under the TRECVID input and output contract, and retain it only if it improves the right slice at an acceptable cost.`,
  },
  { id: "research_threads_block", type: "table", layout: "full", tableId: "research_threads" },
  {
    id: "research_priority_ladder",
    type: "markdown",
    layout: "full",
    sourceId: "team_literature_synthesis",
    body: `## What to implement first

1. **First make evidence observable.** Cache global frames, timestamped ASR, selected-frame IDs, option scores, candidate answers, and a strict run manifest. A weak but inspectable baseline is more useful than a strong black box.
2. **Then improve retrieval at fixed cost.** Add the half-uniform plus half-question sampler and a small required-frame development set. Do not increase frames and change the selector in the same experiment.
3. **Add detail and audio only where diagnostics justify them.** Selective tiles target OCR and small objects; native audio targets sound-event and timing questions. Keep matched ablations so added modalities cannot receive credit merely for added compute.
4. **Add evidence traces before learned election.** Store candidate-to-evidence links and semantic clusters. Simple rules become strong baselines and create training targets for a later reranker.
5. **Train the complicated selector or ranker last.** Reinforcement-learning selectors and listwise rerankers are valuable only after a training-free alternative wins locally and the evaluation path is stable.

This order minimizes confounding: every added component has a logged input, a measurable job, and a rollback path.`,
  },
  {
    id: "team_literature_takeaway",
    type: "markdown",
    layout: "full",
    sourceId: "team_literature_synthesis",
    body: `## The 2025 teams sampled different pieces of this frontier

**CERTH-ITI** is closest to multi-view test-time scaling; **NII_UIT** is closest to question-conditioned evidence selection; **HLTCOE** is closest to answer-generation plus listwise ranking. **WHU-NERCMS** and **kslab** expose the risks of generate-then-embed MC ranking, while **tcna** is a useful proprietary ceiling with a reproducibility gap and **TJUMI** shows why fine-tuning cannot rescue an unreliable submission path.

The crosswalk below states what each lineage supports, what the published system still lacks, and one experiment that changes a single major variable for 2026. These are evidence-backed design hypotheses, not claims that a paper mechanism caused a team's leaderboard position.`,
  },
  { id: "team_literature_map_block", type: "table", layout: "full", tableId: "team_literature_map" },
  {
    id: "literature_transfer_caveat",
    type: "markdown",
    layout: "full",
    sourceId: "team_literature_synthesis",
    body: `## What does not transfer cleanly

- **Most VideoQA papers optimize one best MC answer.** TRECVID AG requires an ordered set of exactly ten open answers, so answer diversity and weak-tail control matter in a way most benchmarks do not measure.
- **Long-video gains are not automatically relevant to six-minute clips.** Retrieval and memory methods should first beat a repaired fixed-frame baseline at the same frame, runtime, and API-call budget.
- **Cross-paper score deltas are not comparable.** Datasets, option counts, video duration, modality access, model scale, and evaluation scripts differ; the table is a mechanism map, not a meta-analysis.
- **A fluent rationale is not evidence.** Store the timestamps, crops, transcript spans, and option scores that actually supported each answer; evaluate evidence recall separately whenever local labels permit.
- **Publication does not guarantee local transfer.** VCSS and Video Parallel Scaling are now peer-reviewed, but their sample-scaling mechanisms still need a controlled local ablation under TRECVID's frame, latency, and output constraints.

The deepest table is therefore ordered by **transfer fit** and practical learning value rather than by a synthetic paper ranking.`,
  },
  { id: "literature_review_block", type: "table", layout: "full", tableId: "literature_review" },
];

const replacementIds = new Set(additionalBlocks.map((block) => block.id));
const retainedBlocks = artifact.manifest.blocks.filter((block) => !replacementIds.has(block.id));
const anchorIndex = retainedBlocks.findIndex((block) => block.id === "literature_takeaway");
if (anchorIndex < 0) throw new Error("Expected literature_takeaway anchor");
retainedBlocks.splice(anchorIndex + 1, 0, ...additionalBlocks);
artifact.manifest.blocks = retainedBlocks;

const sourceNames = datasets.literature_review.map((row) => `${row.work} (${row.year})`);
const literatureSource = artifact.sources.find((source) => source.id === "literature_primary_sources");
if (!literatureSource) throw new Error("Expected literature_primary_sources source");
Object.assign(literatureSource, {
  label: "Competition-adjacent primary VideoQA literature",
  href: "https://arxiv.org/abs/2605.23216",
  path: "queries/literature_review.sql",
  query: {
    engine: "analysis",
    description: "Primary-source review of VideoQA benchmarks and methods selected for transfer relevance to TRECVID evidence extraction, reasoning, AG ranking, and MC calibration.",
    executed_at: "2026-07-12T05:20:00-04:00",
    language: "sql",
    tables_used: sourceNames,
  },
});

const synthesisSource = {
  id: "team_literature_synthesis",
  label: "TREC team-to-literature synthesis",
  href: "https://trec.nist.gov/pubs/trec34/xref.html",
  path: "queries/literature_review.sql",
  query: {
    engine: "analysis",
    description: "Analyst crosswalk joining the official TREC 2025 overview and participant papers to the reviewed primary VideoQA literature; mappings are interpretive, not causal performance attribution.",
    executed_at: "2026-07-12T05:20:00-04:00",
    language: "sql",
    tables_used: [
      "Overview_vqa.pdf",
      "CERTH-ITI.avs.vqa.pdf",
      "NII_UIT.vqa.pdf",
      "HLTCOE.vqa.pdf",
      "WHU-NERCMS.avs.vqa.pdf",
      "kslab.vqa.pdf",
      ...sourceNames,
    ],
  },
};
upsertById(artifact.sources, synthesisSource, "starter_repositories");

const manifestLiteratureSource = artifact.manifest.sources.find((source) => source.id === "literature_primary_sources");
if (!manifestLiteratureSource) throw new Error("Expected manifest literature source");
Object.assign(manifestLiteratureSource, {
  label: literatureSource.label,
  href: literatureSource.href,
  path: literatureSource.path,
});
upsertById(artifact.manifest.sources, {
  id: synthesisSource.id,
  label: synthesisSource.label,
  href: synthesisSource.href,
  path: synthesisSource.path,
}, "starter_repositories");

const ids = artifact.manifest.blocks.map((block) => block.id);
if (ids.length !== new Set(ids).size) throw new Error("Duplicate report block ids after literature expansion");

writeFileSync(artifactPath, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
process.stdout.write(`${JSON.stringify({
  ok: true,
  literatureRows: datasets.literature_review.length,
  researchThreads: datasets.research_threads.length,
  researchConcepts: datasets.research_concept_guide.length,
  teamMappings: datasets.team_literature_map.length,
  topLevelBlocks: artifact.manifest.blocks.length,
})}\n`);
