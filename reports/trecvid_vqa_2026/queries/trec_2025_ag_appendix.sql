-- Exact current AG appendix scores, transcribed from NIST.
-- Source: https://trec.nist.gov/pubs/trec34/appendices/trec2025-vqa-gen.html
-- NIST exposes 12 rows in the current appendix, fewer than the 20 official AG runs
-- recorded in the overview. This view intentionally contains only exact appendix rows.
CREATE OR REPLACE VIEW ag_results AS
SELECT *
FROM (VALUES
  ('certh.vqa.ag.run_2',       'CERTH-ITI', 0.2604, 0.8982, 0.3510, 0.9879, 0.9879, 0.3510),
  ('certh.vqa.ag.run_4',       'CERTH-ITI', 0.2596, 0.9034, 0.3485, 0.9152, 0.9924, 0.3485),
  ('certh.vqa.ag.run_1',       'CERTH-ITI', 0.2596, 0.9033, 0.3483, 0.9152, 0.9924, 0.3483),
  ('videollama-2B',            'WHU-NERCMS', 0.2345, 0.8657, 0.3023, 1.0000, 1.0000, 0.3023),
  ('videollama-2B-prompt',     'WHU-NERCMS', 0.2311, 0.8667, 0.3006, 1.0000, 1.0000, 0.3006),
  ('videollama-7B-prompt',     'WHU-NERCMS', 0.2384, 0.8649, 0.2987, 1.0000, 1.0000, 0.2987),
  ('certh.vqa.ag.run_3',       'CERTH-ITI', 0.2031, 0.8906, 0.2852, 0.7823, 0.9906, 0.2852),
  ('nut-kslab-2025',           'kslab', 0.2265, 0.8931, 0.2837, 0.8318, 0.9927, 0.2837),
  ('MARS-GenR_1',              'HLTCOE', 0.2191, 0.8893, 0.2817, 0.8181, 0.9867, 0.2817),
  ('Aria8x3.5B_VidLLaMa',      'NII_UIT', 0.1728, 0.8866, 0.2696, 0.8254, 0.9955, 0.2696),
  ('gemini_flash_25',          'tcna', 0.2128, 0.8617, 0.2567, 0.8376, 0.9870, 0.2567),
  ('gemini_flash_lite',        'tcna', 0.1886, 0.8560, 0.2162, 0.8278, 0.9846, 0.2162)
) AS t(run, team, meteor, bertscore, sts, ndcg_meteor, ndcg_bert, sts_value);
