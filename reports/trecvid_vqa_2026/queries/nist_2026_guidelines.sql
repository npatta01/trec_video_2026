-- Reproducible report input transcribed from the official 2026 VQA guidelines.
-- Source: https://www-nlpir.nist.gov/projects/tv2026/vqa.html
-- Evidence snapshot: 2026-07-11 (America/New_York).
CREATE OR REPLACE VIEW headline AS
SELECT *
FROM (VALUES
  (6, 1457, 400, 10)
) AS t(days_to_ag_release, known_carryover, planned_movie_scenes, required_ag_answers);

CREATE OR REPLACE VIEW vqa_2026_task_rules AS
SELECT *
FROM (VALUES
  ('AG', 'Return exactly ten ranked natural-language answers per query plus the time, in seconds, used for each answer.'),
  ('MC', 'Rank all four supplied answer options per query.'),
  ('Evaluation', 'Report AG content metrics and NDCG at cutoffs 1, 5, and 10; report MC Top-1 accuracy and MRR.'),
  ('Test design', 'Combine 1,457 held-out 2025 Shorts, approximately 400 movie scenes, new Shorts, and caption-derived automatic QA.'),
  ('Access', 'Complete the MCOLL/KinoLorber movie-data agreement; runs without the agreement are not accepted or scored.')
) AS t(rule_group, published_rule);

CREATE OR REPLACE VIEW submission_contract AS
SELECT *
FROM (VALUES
  ('AG', 'Q_ID, Video_ID, Question', 'Q_ID, Video_ID, Answers[{Rank, Answer, Time}]', 'Exactly 10 answers; ranks 1–10; Time is real-time seconds per answer'),
  ('MC', 'Q_ID, Video_ID, Question, Options[4 × {option}]', 'Q_ID, Video_ID, Answers[{Rank, Answer}]', 'Rank all 4 supplied option strings from 1–4; copy answer text exactly')
) AS t(task, topic_record, run_record, validator_gate);
