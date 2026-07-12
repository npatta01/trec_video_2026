-- Analyst-authored synthesis views used by report tables.
-- Derived from the official 2026 guidelines/schedule/data pages, the 2025 overview,
-- current NIST result appendices, participant papers, public scorer, and repository
-- audit. Full claim-level provenance and caveats are in ../source_notes.md.
CREATE OR REPLACE VIEW task_change AS
SELECT *
FROM (VALUES
  ('AG answer count','Up to 10','Exactly 10; fewer are rejected','Tail-answer quality and ranking are now mandatory rather than optional.'),
  ('Annotation source','Human-written QA','Human-written plus automatically generated QA','Measure performance by annotation source and watch for synthetic-reference artifacts.'),
  ('Evaluation views','Aggregate metrics and category analysis','AG metrics at top 1, 5, and 10; intended short/long and human/automatic axes','Preserve raw ranked outputs and segment every local evaluation.'),
  ('Media','About 2,000 YouTube Shorts, roughly 30 seconds','1,457 held-out Shorts, about 400 movie scenes, new Shorts, and caption-derived QA','A Shorts-only fixed-frame pipeline will miss longer narratives and local movie files.'),
  ('Submission serialization','Headerless CSV','JSON arrays for topics and runs','Do not reuse the old writers; add JSON Schema and semantic validation.'),
  ('Training material','500 videos; no causal examples','At least the same 500 plus released 2025 test data and allowed public external data','Create a clean development protocol and close the causal-domain gap without test leakage.'),
  ('Video duration','About 30 seconds','Roughly 30 seconds to 2 minutes on average; movie clips may reach about 6 minutes','Use shot-aware retrieval and adaptive evidence budgets.')
) AS t(aspect, y2025, y2026, implication);

CREATE OR REPLACE VIEW metric_guide AS
SELECT *
FROM (VALUES
  ('AG','BERTScore','Contextual token similarity between a generated answer and the reference.','Scores have a high baseline; small decimal gaps are not automatically meaningful.'),
  ('AG','METEOR','Word-level alignment with stemming and synonym awareness, balancing precision and recall.','Wording and answer count can change the mean even when the first answer is correct.'),
  ('AG','NDCG','Whether higher-quality candidate answers appear earlier in the submitted ranking.','In the 2025 code, a one-answer list automatically receives 1.0.'),
  ('AG','STS','Sentence-level semantic equivalence to the reference answer.','The 2025 implementation used an external service and unpinned dependencies.'),
  ('MC','MRR','Average reciprocal rank: 1 for first, 1/2 for second, 1/3 for third, 1/4 for fourth.','A system can improve MRR without improving how often it is correct at rank 1.'),
  ('MC','Top-1 accuracy','Share of questions whose correct option is ranked first.','The 2025 scorer used exact option-string equality, making formatting part of correctness.')
) AS t(task, metric, plain_english, watchout);

CREATE OR REPLACE VIEW repo_audit AS
SELECT *
FROM (VALUES
  ('NII_UIT TrecVID2025_VQA','0f270ed11fe51fc11bc46dd7b5d63fa241970637','Codec normalization, cached Whisper, uniform frames, dynamic tiles, direct full-option ranking','No exact HIGHEST_PIPELINE config/output; paper''s question router absent; AG combination recipe missing; some fallbacks silently return A','README says MIT, but no LICENSE file and no detected license'),
  ('HLTCOE TREC2025-HLTCOE','618f31eaf549e022385c25040dccefb12e5e6ef9','Efficient multi-sample generation and the candidate-pointer reranking objective','Missing import; dataset NameError; train/inference format mismatch; private LoRA paths; no valid submission writer','No license file or declaration')
) AS t(repository, commit, reuse, gap, license);

CREATE OR REPLACE VIEW architecture AS
SELECT *
FROM (VALUES
  (1,'Media manifest','Resolve YouTube ID or local clip; record URL/path token, SHA-256, duration, codec, audio, availability, and permission class.','Every query maps to one decodable asset or a documented unavailable status; no hard-coded dataset size.'),
  (2,'Normalized media cache','Transcode unsupported codecs deterministically; extract audio once; preserve original and processing command.','Re-running produces identical checksums and timestamps; failed decodes stop the job.'),
  (3,'Evidence extraction','Timestamped Whisper automatic speech recognition (ASR), optical character recognition (OCR), global uniform frames, shot-aware keyframes, and high-resolution tiles; cache by media and model revision.','Short and six-minute clips fit bounded budgets; sampled times cover beginning, middle, end, and detected shots.'),
  (4,'Question router','Route temporal/counting to dense event evidence, audio/text to ASR+OCR, people-object to tiles/tracking, causal/commonsense to multi-view reasoning.','Category-level validation beats one fixed evidence policy and never removes global coverage.'),
  (5,'AG head','Generate 20–30 candidates in batched multi-sample calls; normalize, cluster, deduplicate, and rank exactly ten with a held-out metric-aware model.','Top-1 does not regress, top-5/10 improve, every query has ranks 1–10 and measured times, no filler/error strings.'),
  (6,'MC head','Copy option strings exactly; score A/B/C/D directly across evidence views; mean/robust-pool logits; optionally test option-order test-time augmentation (TTA).','Each option appears exactly once; permutation tests expose position bias; Top-1 and MRR improve together.'),
  (7,'Validation and provenance gate','JSON Schema plus semantic checks for IDs, counts, ranks, strings, times, duplicates, finite values, and full coverage; save immutable run manifest.','Malformed fixtures fail; a clean rerun reproduces output checksum and scorer results.')
) AS t(stage, component, implementation, acceptance_test);

CREATE OR REPLACE VIEW run_portfolio AS
SELECT *
FROM (VALUES
  (1,'B0 — repaired open baseline','AG + MC','One validated 7–8B video VLM, Whisper, 8–12 global frames, direct outputs','Does the full ingestion, scoring, and submission path work end to end?'),
  (2,'B1 — evidence-dense','AG + MC','Add shot-aware frames, tiles, OCR, and question routing','How much comes from presenting better evidence rather than changing models?'),
  (3,'B2 — multi-view aggregation','AG + MC','Aggregate multiple frame/shot views; AG candidate clustering, MC option-logit pooling','Does inference-time scaling improve robustness on long and causal items?'),
  (4,'B3 — learned ranker','AG','Low-Rank Adaptation (LoRA) listwise or cross-encoder ranker trained without injecting the gold answer','Can ranks 2–10 improve NDCG without diluting content quality?'),
  (5,'B4 — option-order stress test','MC','Permute option order, map back, and aggregate','Is the model reasoning from video evidence or exploiting answer position?')
) AS t(priority, run, task, change, question);
