-- Reproducible report input transcribed from the tentative official schedule.
-- Source: https://www-nlpir.nist.gov/projects/tv2026/schedule.html
-- Evidence snapshot: 2026-07-11 (America/New_York).
CREATE OR REPLACE VIEW headline AS
SELECT *
FROM (VALUES
  (6, 1457, 400, 10)
) AS t(days_to_ag_release, known_carryover, planned_movie_scenes, required_ag_answers);

CREATE OR REPLACE VIEW schedule AS
SELECT *
FROM (VALUES
  (DATE '2026-07-17', 6,   'AG test data available',       'Within 24 hours: complete media manifest and cheap schema-valid sanity run'),
  (DATE '2026-08-10', 30,  'AG submission page opens',    'All candidate runs frozen, rescored, and reproducible from clean manifests'),
  (DATE '2026-08-17', 37,  'AG submissions close',        'Submit early; retain validator receipts and exact uploaded checksums'),
  (DATE '2026-08-21', 41,  'MC test data available',       'Reuse cached evidence; run direct option-ranking baseline immediately'),
  (DATE '2026-09-11', 62,  'MC submission page opens',     'Final MC portfolio frozen; option-order and exact-string tests passed'),
  (DATE '2026-09-18', 69,  'MC submissions close',         'Submit early; archive run manifests and receipts'),
  (DATE '2026-09-28', 79,  'AG and MC results released',   'Begin category/duration/source error analysis and notebook-paper outline'),
  (DATE '2026-11-09', 121, 'Notebook papers due',          'Methods, ablations, limitations, and reproducibility package complete')
) AS t(date, days_from_snapshot, event, team_gate);
