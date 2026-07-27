# TREC 2026 VQA report — source and analysis notes

> **Background only:** This July 12 field guide supports historical context.
> The current [Project Status](../../docs/project-status/index.md) supersedes
> it for local experiment decisions.

Competition evidence snapshot: July 11, 2026 (America/New_York). The model-primer, team-architecture, literature, fine-tuning, and public-weight model supplements were researched July 12. This file records the evidence boundary, reconciliation decisions, calculations, report mapping, and visual contracts behind the canonical report artifact and its delivery surfaces. It is supporting material, not part of the main reading path.

## Reporting job

- Question: What is the TREC 2026 Video Question Answering task, what was submitted in the 2025 pilot, which published runs scored highest, and what should a new team do now?
- Audience: product stakeholders / general technical readers who are new to video QA.
- Decision supported: whether and how to enter both AG and MC, what to build first, and which 2025 ideas are safe to reuse.
- Scope: official public material available through July 11, 2026; no protected active-participant files were opened.
- Comparison baseline: the 2025 pilot versus the published 2026 plan.
- Success criteria: exact task and calendar facts; current appendix scores distinguished from workshop-era scores; newcomer-safe model background; visual team-architecture comparisons; a beginner-first, image-led literature chapter that explains six research mechanisms, ties them to every 2025 team, and converts them into controlled 2026 experiments; a fine-tuning chapter that separates model adaptation from ranker training, gives data/compute/leakage requirements, inventories official public weights and originator fine-tunes, and converts them into a gated 2026 plan; an early mobile-safe contents card, task-specific reading paths, and stable numbered chapter headings; an actionable architecture and submission plan; accessible native artifact rendering and a self-contained HTML fallback.

## Evidence inventory

Primary sources:

1. NIST 2026 VQA guidelines: https://www-nlpir.nist.gov/projects/tv2026/vqa.html
2. NIST 2026 schedule: https://www-nlpir.nist.gov/projects/tv2026/schedule.html
3. NIST 2026 data page: https://www-nlpir.nist.gov/projects/tv2026/data.html
4. TREC 2026 call for participation: https://trec.nist.gov/cfp.html
5. MCOLL agreement: https://www-nlpir.nist.gov/projects/tv2026/DVU_Org.Form.txt
6. NIST 2025 VQA overview: https://trec.nist.gov/pubs/trec34/papers/Overview_vqa.pdf
7. Current 2025 AG appendix: https://trec.nist.gov/pubs/trec34/appendices/trec2025-vqa-gen.html
8. Current 2025 MC appendix: https://trec.nist.gov/pubs/trec34/appendices/trec2025-vqa-mc.html
9. NIST VQA scorer repository, commit c3cf9b701a1e9d8204ef6effdc563c7960523de0 observed July 11: https://github.com/gawad-nist/VQA
10. CERTH-ITI 2025 paper: https://trec.nist.gov/pubs/trec34/papers/CERTH-ITI.avs.vqa.pdf
11. NII_UIT 2025 paper: https://trec.nist.gov/pubs/trec34/papers/NII_UIT.vqa.pdf
12. HLTCOE 2025 paper: https://trec.nist.gov/pubs/trec34/papers/HLTCOE.vqa.pdf
13. WHU-NERCMS 2025 paper: https://trec.nist.gov/pubs/trec34/papers/WHU-NERCMS.avs.vqa.pdf
14. NII_UIT organizer-linked starter repository: https://github.com/aiclub-uit/TrecVID2025_VQA
15. HLTCOE organizer-linked starter repository: https://github.com/adoptedirelia/TREC2025-HLTCOE
16. NII AG per-response file: https://trec.nist.gov/pubs/trec34/appendices/trec2025-vqa-gen/Aria8x3.5B_VidLLaMa.per_response.csv
17. HLTCOE AG per-response file: https://trec.nist.gov/pubs/trec34/appendices/trec2025-vqa-gen/MARS-GenR_1.per_response.csv

Primary model and literature sources:

18. CLIP: https://arxiv.org/abs/2103.00020
19. LoRA: https://arxiv.org/abs/2106.09685
20. NExT-QA: https://openaccess.thecvf.com/content/CVPR2021/html/Xiao_NExT-QA_Next_Phase_of_Question-Answering_to_Explaining_Temporal_Actions_CVPR_2021_paper.html
21. BLIP-2: https://arxiv.org/abs/2301.12597
22. Whisper: https://arxiv.org/abs/2212.04356
23. Sentence-BERT: https://arxiv.org/abs/1908.10084
24. Video-LLaMA: https://arxiv.org/abs/2306.02858
25. EgoSchema: https://arxiv.org/abs/2308.09126
26. MVBench: https://arxiv.org/abs/2311.17005
27. Gemini 1.5: https://arxiv.org/abs/2403.05530
28. Video-MME: https://arxiv.org/abs/2405.21075
29. VideoLLaMA 2: https://arxiv.org/abs/2406.07476
30. Aria: https://arxiv.org/abs/2410.05993
31. VideoLLaMA 3: https://arxiv.org/abs/2501.13106
32. Qwen2.5-VL: https://arxiv.org/abs/2502.13923
33. Qwen2.5-Omni: https://arxiv.org/abs/2503.20215
34. InternVL3.5: https://arxiv.org/abs/2508.18265
35. CaST-Bench: https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_CaST-Bench_Benchmarking_Causal_Chain-Grounded_Spatio-Temporal_Reasoning_for_Video_Question_Answering_CVPR_2026_paper.html

Fine-tuning and current public-weight sources checked July 12, 2026:

1. TRL SFTTrainer VLM guidance: https://huggingface.co/docs/trl/v0.26.0/en/sft_trainer
2. PEFT LoRA reference: https://huggingface.co/docs/peft/main/en/package_reference/lora
3. PEFT quantization / QLoRA guidance: https://huggingface.co/docs/peft/developer_guides/quantization
4. Official Qwen3-VL fine-tuning code: https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune
5. Qwen3-VL official collection and 4B/8B Instruct cards: https://huggingface.co/collections/Qwen/qwen3-vl
6. Qwen2.5-VL-7B-Instruct: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
7. Qwen2.5-Omni-7B: https://huggingface.co/Qwen/Qwen2.5-Omni-7B
8. InternVL3.5-4B/8B and MPO checkpoints: https://huggingface.co/OpenGVLab/InternVL3_5-4B
9. VideoLLaMA3 2B/7B: https://huggingface.co/DAMO-NLP-SG/VideoLLaMA3-2B
10. LLaVA-Video-7B-Qwen2: https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2
11. SmolVLM2 500M/2.2B: https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct
12. MiniCPM-V-4.5 and MiniCPM-o-4.5: https://huggingface.co/openbmb/MiniCPM-V-4_5
13. Molmo2 4B/8B: https://huggingface.co/allenai/Molmo2-8B
14. InternVideo3-8B-Instruct: https://huggingface.co/yanziang/InternVideo3-8B-Instruct
15. VideoChat-R1.5-7B: https://huggingface.co/OpenGVLab/VideoChat-R1_5-7B
16. Video-R1-7B: https://huggingface.co/Video-R1/Video-R1-7B
17. SAGE-MM-Molmo2-8B-SFT_RL: https://huggingface.co/allenai/SAGE-MM-Molmo2-8B-SFT_RL
18. VideoRefer-VideoLLaMA3-7B: https://huggingface.co/DAMO-NLP-SG/VideoRefer-VideoLLaMA3-7B
19. LLaVA-NeXT-Video-7B-DPO: https://huggingface.co/lmms-lab/LLaVA-NeXT-Video-7B-DPO
20. Cosmos-Reason2-8B: https://huggingface.co/nvidia/Cosmos-Reason2-8B

Hub access, license tags, parameter metadata, and checkpoint names were verified with the Hugging Face CLI where available and reconciled against the current official model card. “Public” in the report distinguishes ungated downloads from gated public pages; it does not imply that every dataset or downstream use has identical terms.

Competition-adjacent deep-review inventory (25 primary works; overlap with the model list above is intentional):

1. NExT-QA: https://arxiv.org/abs/2105.08276
2. EgoSchema: https://arxiv.org/abs/2308.09126
3. LongVideoBench: https://arxiv.org/abs/2407.15754
4. Video-MME: https://arxiv.org/abs/2405.21075
5. MLVU: https://arxiv.org/abs/2406.04264
6. HERBench: https://arxiv.org/abs/2512.14870
7. CaST-Bench: https://arxiv.org/abs/2605.23216
8. MovieRecapsQA: https://arxiv.org/abs/2601.02536
9. VideoAgent: https://arxiv.org/abs/2403.10517
10. VideoTree: https://arxiv.org/abs/2405.19209
11. BOLT: https://arxiv.org/abs/2503.21483
12. KFS-Bench: https://arxiv.org/abs/2512.14017
13. Video Panels: https://arxiv.org/abs/2509.23724
14. Efficient Frame Selection via Reinforcement Learning: https://openaccess.thecvf.com/content/CVPR2026/html/Qin_Efficient_Frame_Selection_for_Long_Video_Understanding_via_Reinforcement_Learning_CVPR_2026_paper.html
15. TimeChat: https://arxiv.org/abs/2312.02051
16. QGAC-TR: https://aclanthology.org/2024.findings-emnlp.176/
17. FALCONEye: https://arxiv.org/abs/2503.19850
18. Qwen2.5-VL: https://arxiv.org/abs/2502.13923
19. Qwen2.5-Omni: https://arxiv.org/abs/2503.20215
20. Whisper: https://proceedings.mlr.press/v202/radford23a.html
21. Visual Context Sample Scaling: https://arxiv.org/abs/2503.20472
22. Video Parallel Scaling: https://arxiv.org/abs/2509.08016
23. BOLD / Addressing Blind Guessing: https://aclanthology.org/2025.acl-long.162/
24. RADI: https://arxiv.org/abs/2403.14430
25. RLER: https://arxiv.org/abs/2604.04379

Evidence-status split: 23 peer-reviewed papers and 2 official technical reports. VCSS is peer-reviewed at ICCV 2025, Video Parallel Scaling is peer-reviewed in CVPR Findings 2026, and Whisper is the ICML 2023 publication. Selection favored transfer proximity to TRECVID over citation count or general model popularity.

## Evidence reconciliation and data-quality ledger

Intended grain for historical score claims: one submitted run, one published metric aggregate. The current TREC proceedings appendices are the canonical lookup surface; workshop papers are used for methods and to explain score-version differences.

### Finding 1 — publication counts do not fully reconcile

- Evidence: the track overview reports 20 AG runs from 7 teams and 9 MC runs from 5 teams. The general TREC overview says 20 AG and 8 MC. The current appendices expose 12 AG rows and 7 MC rows.
- Likely cause: TJUMI submitted buggy runs and supplied corrected runs after the official release; some run artifacts were not carried into the current appendices.
- Impact: the public record is not complete enough to present an exact score table for every submitted file.
- Severity: medium for exhaustiveness, low for identifying the highest current appendix scores.
- Treatment: report the overview counts, show every row in the current appendices, and label the table scope explicitly.

### Finding 2 — MC has two published denominator regimes

- Evidence: participant papers use 2,000 ground-truth rows while skipping 18 unavailable videos in the numerator, yielding NII_UIT 0.774 Top-1 / 0.859 MRR and CERTH run 2 0.761 / 0.855. The current appendix aggregates the 1,982 emitted evaluation records, yielding NII_UIT 0.7810 / 0.8671 and CERTH run 2 0.7679 / 0.8629.
- Cause: the public 2025 MC script sets `total_queries` before skipping 18 unavailable IDs; the appendix row aggregation uses the 1,982 evaluated records.
- Impact: decimals differ by source, but ordering does not.
- Severity: medium.
- Treatment: use current appendix values in ranking tables, state the 1,982-record basis, and explain the paper-era alternative once.

### Finding 3 — perfect 2025 AG NDCG can be structurally trivial

- Evidence: the public scorer returns NDCG = 1.0 for a singleton list. WHU runs returned one answer per query and therefore show 1.0000 NDCG despite lower content-similarity scores.
- Impact: a 1.0000 NDCG value is not evidence that a 10-answer ranking system was best.
- Severity: high for interpretation.
- Treatment: do not crown an overall AG winner. Identify leaders separately for METEOR, BERTScore, STS, and ranking, and explain the singleton effect. The 2026 exactly-10 rule removes this shortcut.

### Finding 4 — lower-ranked AG candidates diluted 2025 quality

- Evidence from current per-response CSVs: NII submitted 10 answers for each of 1,982 evaluated queries (19,820 rows). Its rank-1 METEOR was 0.2118 versus 0.1728 across all answers; rank-1 STS was 0.3098 versus about 0.2697 across all answers. HLTCOE covered 1,965 queries with 14,227 answer rows; only 710 queries had all 10 answers. Its rank-1 METEOR was 0.2284 versus a raw all-row mean of 0.2124.
- Impact: candidate count and tail quality materially affected 2025 aggregate content metrics.
- Severity: high for 2026 design.
- Treatment: generate a larger candidate pool, remove near-duplicates and weak tails, then learn or calibrate a top-10 ranker. Evaluate at top-1, top-5, and top-10 as soon as the 2026 scorer is released.

### Finding 5 — the public scorer is provisional for 2026

- Evidence: the organizer repository README says it is based on 2025 and must be updated for the 2026 JSON format and cutoff reporting. It still reads CSV and does not implement top-1/5/10 AG reports.
- Impact: optimizing against the current code may optimize the wrong aggregation.
- Severity: high.
- Treatment: use it only as a regression harness, pin the final organizer commit once announced, and preserve raw per-query outputs so scores can be recomputed.

### Finding 6 — organizer-linked repositories are reference code, not frozen reproductions

- Evidence: both public repositories were created after the 2025 evaluation and omit submission-time pieces. NII_UIT exposes useful MC preprocessing/inference ideas but does not fully encode the paper pipeline. HLTCOE has missing imports/variables, a training/inference output mismatch, and no released private LoRA weights.
- Impact: cloning either repository is not sufficient for a reproducible 2026 baseline.
- Severity: high for schedule risk.
- Treatment: pin commits, write smoke tests, reconstruct missing modules, and treat reported scores as targets rather than expected reproduction results.

### Finding 7 — team architecture evidence is uneven

- Evidence: five teams published detailed participant papers; tcna and TJUMI have much thinner public implementation descriptions. Even the detailed papers omit some operational choices.
- Impact: a blank component cannot safely be interpreted as proof that the component was absent or unsupported by the underlying model.
- Severity: high for fair architectural comparison.
- Treatment: the architecture heatmap encodes only “explicitly documented” versus “not documented.” Its adjacent prose and semantic data table repeat that rule; it is not presented as a quality score.

### Finding 8 — model names do not isolate the reason a run performed well

- Evidence: teams using related multimodal backbones differed in frame sampling, spatial tiling, audio treatment, adaptation, answer generation, and ranking. Published results are submissions, not controlled architecture ablations.
- Impact: leaderboard differences cannot be causally attributed to a backbone family alone.
- Severity: high for system selection.
- Treatment: the report separates model anatomy, family background, team pipeline, and measured run performance. Recommendations are framed as testable design hypotheses rather than model rankings.

### Finding 9 — the literature is a mechanism map, not a score meta-analysis

- Evidence: reviewed papers use different datasets, video lengths, option counts, modality access, model scales, runtime budgets, training labels, and evaluation scripts. Most optimize one MC answer; TRECVID additionally requires an ordered exact-10 open-answer list.
- Impact: cross-paper score deltas cannot be pooled or used to predict TRECVID leaderboard position.
- Severity: high for research transfer.
- Treatment: organize 25 works by transfer fit, evidence status, and mechanism. Tie each team to a controlled 2026 experiment, keep every team-to-paper mapping explicitly interpretive rather than causal, and require local ablations even for peer-reviewed methods.

### Finding 10 — fine-tuning evidence is stage-specific and not fully reproducible

- Evidence: CERTH-ITI reports LoRA SFT of Qwen2.5-Omni-7B on the organizer 500-video set and a participant-reported MC Top-1 ablation of 73.2% frozen, 74.4% fine-tuned, and 76.1% with fine-tuning plus ten-view aggregation. HLTCOE reports LoRA on a listwise answer reranker rather than on the video generator. NII_UIT reports frozen main backbones with evidence-engineering improvements. The CERTH and HLT submission adapters are not available as complete reproducible public checkpoints.
- Impact: “fine-tuning helped” cannot be interpreted as one causal intervention. Model adaptation, ranker adaptation, and evidence-pipeline changes have different costs, labels, and failure modes.
- Severity: high for experiment design and compute planning.
- Treatment: compare every adapter against the same frozen checkpoint and evidence pack; split by video/source; train the smallest stage with repeatable residual error; keep the test locked; treat hardware figures as broad planning bands; prefer official public Instruct checkpoints with explicit access and license metadata.

## Calculations used in the report

- Known 2026 carryover share, causal: 794 / 1,457 = 54.4955%, displayed as 54.5%.
- Other carryover shares: temporal 121 / 1,457 = 8.3%; audio + multi-hop 180 / 1,457 = 12.4%; objects and people 164 / 1,457 = 11.3%; counting + commonsense 198 / 1,457 = 13.6%.
- Current MC lead over CERTH run 2: 0.7810 - 0.7679 = 0.0131, or 1.31 percentage points Top-1; MRR gap 0.8671 - 0.8629 = 0.0042.
- Days from July 11 snapshot: AG release July 17 = 6; AG page opens August 10 = 30; AG closes August 17 = 37; MC release August 21 = 41; MC page opens September 11 = 62; MC closes September 18 = 69; results September 28 = 79.

## Report spine

- Decision-useful answer: enter both tasks if registration is still accepted, share one evidence-extraction layer, prioritize MC as the more predictable score target, and use AG to develop an exact-10 candidate generation/ranking pipeline.
- Evidence: official task changes, causal-heavy holdout, current appendix results, participant-paper methods, source-code audit, 25 competition-adjacent primary works, and schedule.
- Caveats that could change the plan: final registration status, corpus count/provenance, run cap, exact scorer, runtime definition, duplicate-answer policy, and workshop dates.
- Recommended next step: clear administrative access immediately, establish repaired reference baselines before July 17, and pin a schema-valid end-to-end run by July 20.

## Executive-report structure mapping

1. Title — first markdown block and manifest title.
2. Executive Summary — second markdown block, immediately after title; the headline metric strip remains next, followed by the contents and reading-path card.
3. Key findings with visual evidence — nine stable chapter markers organize submission-contract and task-change tables, a beginner model primer, model-family guide, team architecture matrix and pipeline flows, a two-diagram fine-tuning tutorial with method, 2025 evidence, data, hardware, public-weight, and public-fine-tune tables, a six-part beginner literature chapter with one visual per concept, a diagnostic concept table, six controlled research threads, a seven-team literature crosswalk, a 25-work primary-literature path, known holdout chart, AG chart/table, MC chart/table, and approach/repository tables.
4. Recommended next steps — evidence-first architecture, run portfolio, and dated execution plan.
5. Further questions — explicit organizer-question section.
6. Caveats and assumptions — final evidence-limit section.

## Chart map and contracts

### Known 2026 carryover distribution

- Analytical question: where is the known 1,457-video carryover concentrated?
- Takeaway: causal questions are 54.5% of the carryover while the original 500-video training set had no causal examples.
- Family/type: comparison and ranking; horizontal bar.
- Data sufficiency: 5 complete grouped categories, counts sum to 1,457; no trend implied.
- Fields: grouped category, video count, share.
- Palette: single-root sequential; labels and ordering carry meaning without color.
- Caveat: excludes approximately 400 movie scenes, new Shorts, and auto-generated QA whose final counts were not published.

### 2025 AG current appendix ranking

- Analytical question: which currently published run has the highest STS content score?
- Takeaway: CERTH run 2 leads STS and METEOR; CERTH run 4 narrowly leads BERTScore.
- Family/type: comparison and ranking; horizontal bar sorted by STS.
- Data sufficiency: all 12 rows in the current AG appendix.
- Fields: run, team, STS; exact METEOR/BERTScore/NDCG values remain in the adjacent table.
- Palette: single-root; no redundant legend.
- Caveat: appendix is not the complete 20-run overview population.

### 2025 MC current appendix comparison

- Analytical question: how do current Top-1 and MRR scores compare across published runs?
- Takeaway: NII_UIT is highest on both, with CERTH run 2 close behind.
- Family/type: comparison; grouped horizontal bars.
- Data sufficiency: all 7 rows in the current MC appendix, two same-scale metrics per run.
- Fields: run, Top-1, MRR.
- Palette: hard two-root cap; legend plus bar grouping provide non-color distinction.
- Caveat: values use the current 1,982-record appendix regime; papers show slightly lower 2,000-denominator values.

### 2025 team architecture component matrix

- Analytical question: which system components were explicitly documented for each of the seven 2025 teams?
- Takeaway: the submissions differ at least as much in evidence collection and output ranking as they do in backbone name; several apparent gaps are documentation gaps.
- Family/type: matrix; heatmap.
- Data sufficiency: 56 complete team-component cells, covering 7 teams × 8 components. Each binary value is tied to an evidence note in the reproducible source mirror.
- Fields: team, component, documented indicator, evidence note.
- Palette: single-root sequential; row/column labels, keyboard-accessible details, adjacent explanation, and the semantic data fallback carry the meaning without color.
- Caveat: 1 means explicitly reported; 0 means not documented, not proven absent. Cell totals are not performance scores and must not be compared with the leaderboards.

## Conceptual architecture image map

These are explanatory diagrams rather than quantitative charts, so they use deterministic SVG artwork with exact labels instead of a chart geometry that would imply magnitude.

1. `assets/model-primer.svg` — visual frames, audio/text, and the question converge on a multimodal model, then split into AG and MC output heads.
2. `assets/team-direct.svg` — the direct-backbone pattern used to describe WHU, tcna, and TJUMI, with unpublished stages labeled rather than inferred.
3. `assets/team-multiview.svg` — CERTH's ten-view Qwen2.5-Omni ensemble and its separate AG-vote and MC-logit-pooling exits.
4. `assets/team-evidence.svg` — NII's frozen-backbone design, where time-spaced frames, spatial tiles, Whisper, and question routing improve the evidence package.
5. `assets/team-generate-rank.svg` — HLTCOE and kslab separate generation from a learned, confidence-based, or embedding-based judging stage.
6. `assets/recommended-2026.svg` — the recommended shared evidence engine, two thin task heads, and one schema/provenance validator.
7. `assets/research-token-budget.svg` — a fixed frame budget split between whole-timeline coverage and question-relevant evidence.
8. `assets/research-retrieval-fusion.svg` — missing evidence versus present-but-miscombined evidence, separated by an oracle-frame test.
9. `assets/research-global-local.svg` — a cheap global timeline plus high-resolution crops around selected moments.
10. `assets/research-audio.svg` — timestamped ASR and native non-speech audio as complementary evidence branches.
11. `assets/research-causal.svg` — an inspectable cause-to-effect chain contrasted with an unsupported plausible guess.
12. `assets/research-election.svg` — semantic deduplication, evidence-aware answer election, and MC option-order calibration.
13. `assets/finetuning-decision.svg` — a prompt/retrieval → LoRA/QLoRA → full-tune decision ladder with a locked-test validation gate.
14. `assets/finetuning-pipeline.svg` — grounded examples, video-exclusive splits, adapter training, validation-only checkpoint selection, and separate AG/MC/schema evaluation.

Image delivery contract: every SVG uses a 640 × 440 viewBox, large text, high-contrast blue/gold styling, labeled boxes, and explicit arrows. Following mobile review, arrowheads are 35% smaller (`8` → `5.2`); original architecture connectors use 2.2-pixel strokes, the six research explainers use 2.1-pixel strokes, and the two fine-tuning diagrams use 2-pixel strokes. `embed_diagrams.mjs` embeds the original architecture set, `expand_literature.mjs` embeds the beginner research set, and `expand_finetuning.mjs` embeds the fine-tuning set; every asset becomes a self-contained data image inside the canonical artifact. Every image has SVG title/description metadata, an HTML `alt` description, and a plain-language `figcaption`; the explanatory prose remains adjacent. CSS scales images to 100% of the available width without horizontal scrolling, while vector rendering remains sharp under phone zoom.

## Omitted quantitative visuals

- No trend chart: the task has only one prior year, so a time series would be underpowered.
- No final 2026 corpus composition chart: the final total and the new-Shorts/auto-QA counts are not published.
- No question-category performance chart: the 2025 overview provides only plotted aggregates, not an exact downloadable category table; qualitative findings are reported instead of transcribing approximate bar heights.
- No cross-paper performance chart: benchmark scores are not comparable across datasets, duration bands, modality access, model scale, option counts, and scoring rules. Tables map mechanisms and transfer fit without inventing a common scale.
- No funnel/Sankey for team pipelines: architecture stages are not attrition or conserved-flow quantities. Responsive vector images preserve the reported ordering and branch logic without implying numeric magnitude; alt descriptions and captions provide the equivalent non-visual reading path.

## Report interaction and accessibility plan

- The portable reader supplies keyboard-accessible source actions, chart tooltips/detail views, sortable tables, light/dark system appearance, responsive layout, and a semantic no-script/print fallback.
- Navigation is structural and mobile-safe: an early contents card offers a 10-minute orientation, a field-from-scratch path, and a build-for-2026 path; the nine entries repeat as visible `Chapter 1`–`Chapter 9` prefixes on the corresponding large headings. Current native Markdown fragment links open in a new context and custom HTML blocks are sandboxed, so the report deliberately avoids brittle pseudo-clickable anchors and provides exact Find/search terms instead.
- Progressive disclosure is structural: executive answer and contents first; plain-language task and model anatomy next; model-family and team-flow detail; then a fine-tuning decision image, vocabulary, 2025 evidence, data/compute requirements, public model inventories, and a gated experiment plan; then a six-part literature tutorial with one running example and one image per idea, a compact diagnostic table, six controlled research threads, an implementation priority ladder, a team crosswalk, and the 25-paper deep table; followed by system implementation detail, open questions, caveats, and glossary.
- Every chart has an adjacent takeaway paragraph and a chart-data fallback.
- Conceptual architecture visuals are embedded, responsive SVG images rather than Markdown code fences. Each carries a meaningful `alt` description and an adjacent caption; no conclusion depends on color or arrow direction alone.
- No conclusion depends on color alone; exact values are available in tables.
- Links use descriptive labels. Abbreviations are expanded before use.

## Delivery QA receipt

- Canonical input: `artifact.json`; reproducible source mirrors: `queries/*.sql`; deterministic diagram sources: `assets/*.svg`; artifact updaters: `embed_diagrams.mjs`, `expand_literature.mjs`, `expand_finetuning.mjs`, and `add_navigation.mjs` (run last so chapter labels follow any content refresh).
- Current selected delivery surface: native Data Analytics MCP report artifact; the full manifest and bounded snapshot passed validation and the single visible render call returned successfully.
- Current canonical shape before rendering: 86 top-level blocks, including 4 native charts, 14 responsive image blocks, and 23 tables; 27 bounded datasets support the report.
- The existing `report.html` is an earlier portable delivery snapshot. It is not the selected surface for this revision and should be regenerated from the updated canonical artifact before any future file-based handoff.
