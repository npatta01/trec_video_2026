-- Team architecture component matrix transcribed from the official 2025 VQA
-- overview and the CERTH-ITI, NII_UIT, HLTCOE, WHU-NERCMS, and kslab papers.
-- A value of 1 means the component was explicitly documented somewhere in the
-- team's reviewed submission material; 0 means it was not documented, not that
-- the underlying base model was incapable of it.
CREATE OR REPLACE VIEW team_architecture_matrix AS
SELECT *
FROM (VALUES
  ('CERTH-ITI','Fine-tuned',1,'LoRA on the 500 organizer examples'),
  ('CERTH-ITI','Audio path',1,'Full native audio embeddings accompanied every frame view'),
  ('CERTH-ITI','Spatial tiles',0,'No spatial tiling reported'),
  ('CERTH-ITI','Multi-view',1,'Ten independently sampled frame subsets'),
  ('CERTH-ITI','Multi-sample AG',1,'A multinomial-sampling AG variant was submitted'),
  ('CERTH-ITI','Learned reranker',0,'Answers were combined by vote count, not a learned reranker'),
  ('CERTH-ITI','Direct MC',1,'Option-letter log probabilities were pooled across views'),
  ('CERTH-ITI','Embedding MC',0,'No generated-anchor embedding stage'),

  ('NII_UIT','Fine-tuned',0,'Submission was training-free'),
  ('NII_UIT','Audio path',1,'Whisper transcript entered as text evidence'),
  ('NII_UIT','Spatial tiles',1,'Frames were divided into zoomed local tiles'),
  ('NII_UIT','Multi-view',0,'One densified evidence package was described, not view ensembling'),
  ('NII_UIT','Multi-sample AG',0,'Exact ten-answer construction was not fully documented'),
  ('NII_UIT','Learned reranker',0,'No learned answer reranker documented'),
  ('NII_UIT','Direct MC',1,'InternVL directly produced an ordered option list'),
  ('NII_UIT','Embedding MC',0,'No generated-anchor embedding stage'),

  ('HLTCOE','Fine-tuned',1,'LoRA trained the listwise Qwen reranker'),
  ('HLTCOE','Audio path',1,'ASR transcript was included in prompts'),
  ('HLTCOE','Spatial tiles',0,'No spatial tiling reported'),
  ('HLTCOE','Multi-view',0,'No view-level ensemble reported'),
  ('HLTCOE','Multi-sample AG',1,'Generator produced multiple candidates with prompt/decoding diversity'),
  ('HLTCOE','Learned reranker',1,'LoRA listwise pointer reranker ordered candidates'),
  ('HLTCOE','Direct MC',0,'AG-only submission'),
  ('HLTCOE','Embedding MC',0,'AG-only submission'),

  ('WHU-NERCMS','Fine-tuned',0,'Off-the-shelf VideoLLaMA'),
  ('WHU-NERCMS','Audio path',0,'Team audio treatment was not reported'),
  ('WHU-NERCMS','Spatial tiles',0,'No spatial tiling reported'),
  ('WHU-NERCMS','Multi-view',0,'No view-level ensemble reported'),
  ('WHU-NERCMS','Multi-sample AG',0,'Published current runs returned singleton answers'),
  ('WHU-NERCMS','Learned reranker',0,'No learned reranker reported'),
  ('WHU-NERCMS','Direct MC',0,'MC first generated a free-text anchor'),
  ('WHU-NERCMS','Embedding MC',1,'Anchor and options were ranked by cosine similarity'),

  ('kslab','Fine-tuned',1,'LoRA tuned VideoLLaMA3-2B'),
  ('kslab','Audio path',0,'Audio was explicitly omitted'),
  ('kslab','Spatial tiles',0,'No spatial tiling reported'),
  ('kslab','Multi-view',0,'No view-level ensemble reported'),
  ('kslab','Multi-sample AG',1,'Stochastic decoding produced answer candidates'),
  ('kslab','Learned reranker',0,'Candidates used model confidence, not a learned second stage'),
  ('kslab','Direct MC',0,'MC first generated a free-text anchor'),
  ('kslab','Embedding MC',1,'Sentence-Transformer cosine similarity ranked options'),

  ('tcna','Fine-tuned',0,'Run forms report off-the-shelf Gemini models'),
  ('tcna','Audio path',0,'Team use of Gemini audio was not published'),
  ('tcna','Spatial tiles',0,'Not published'),
  ('tcna','Multi-view',0,'Not published'),
  ('tcna','Multi-sample AG',0,'Answer-list construction was not published'),
  ('tcna','Learned reranker',0,'Not published'),
  ('tcna','Direct MC',0,'AG-only submission'),
  ('tcna','Embedding MC',0,'AG-only submission'),

  ('TJUMI','Fine-tuned',1,'VideoLLaMA2 was fine-tuned on several video datasets'),
  ('TJUMI','Audio path',0,'Audio treatment was not reported'),
  ('TJUMI','Spatial tiles',0,'Not reported'),
  ('TJUMI','Multi-view',0,'Not reported'),
  ('TJUMI','Multi-sample AG',0,'Answer-list construction was not reported'),
  ('TJUMI','Learned reranker',0,'Not reported'),
  ('TJUMI','Direct MC',0,'MC ranking path was not reported'),
  ('TJUMI','Embedding MC',0,'MC ranking path was not reported')
) AS t(team, component, reported, evidence);
