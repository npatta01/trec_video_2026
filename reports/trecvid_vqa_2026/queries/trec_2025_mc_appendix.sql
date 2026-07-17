-- Exact current MC appendix scores, transcribed from NIST.
-- Source: https://trec.nist.gov/pubs/trec34/appendices/trec2025-vqa-mc.html
-- The current appendix evaluates 1,982 records; paper/overview values used a
-- denominator of 2,000 and are retained for reconciliation, not leaderboard order.
CREATE OR REPLACE VIEW mc_results AS
SELECT *
FROM (VALUES
  ('HIGHEST_PIPELINE',      'NII_UIT', 0.7810, 0.8671, 0.774, 0.859, 0.7810, 0.8671),
  ('certh.vqa.mc.run_2',    'CERTH-ITI', 0.7679, 0.8629, 0.761, 0.855, 0.7679, 0.8629),
  ('certh.vqa.mc.run_1',    'CERTH-ITI', 0.7487, 0.8510, 0.742, 0.843, 0.7487, 0.8510),
  ('certh.vqa.mc.run_4',    'CERTH-ITI', 0.5641, 0.7337, 0.559, 0.727, 0.5641, 0.7337),
  ('certh.vqa.mc.run_3',    'CERTH-ITI', 0.5353, 0.7141, 0.530, 0.708, 0.5353, 0.7141),
  ('videollama-7B-vqa',     'WHU-NERCMS', 0.5272, 0.7063, 0.522, 0.700, 0.5272, 0.7063),
  ('tv25-kslab-mc',         'kslab', 0.5040, 0.6921, 0.499, 0.686, 0.5040, 0.6921)
) AS t(run, team, top1, mrr, workshop_top1, workshop_mrr, top1_value, mrr_value);

CREATE OR REPLACE VIEW mc_chart AS
SELECT run, team, 'Top-1 accuracy' AS metric, top1_value AS score
FROM mc_results
UNION ALL
SELECT run, team, 'MRR' AS metric, mrr_value AS score
FROM mc_results;
