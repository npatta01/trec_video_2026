-- Participation, category counts, and system descriptions transcribed from the
-- official TREC 2025 VQA overview and reconciled against the current appendices.
-- Source: https://trec.nist.gov/pubs/trec34/papers/Overview_vqa.pdf
-- Evidence snapshot: 2026-07-11.
CREATE OR REPLACE VIEW holdout_distribution AS
SELECT *
FROM (VALUES
  ('Causal', 794, 0.544955),
  ('Counting + commonsense', 198, 0.135896),
  ('Audio + multi-hop', 180, 0.123541),
  ('Objects and people', 164, 0.112560),
  ('Temporal', 121, 0.083048)
) AS t(category, videos, share);

CREATE OR REPLACE VIEW submission_inventory AS
SELECT *
FROM (VALUES
  ('AG','CERTH-ITI','certh.vqa.ag.run_1','Yes','Main multi-view greedy ensemble'),
  ('AG','CERTH-ITI','certh.vqa.ag.run_2','Yes','Single-generation baseline'),
  ('AG','CERTH-ITI','certh.vqa.ag.run_3','Yes','Multinomial sampling variant'),
  ('AG','CERTH-ITI','certh.vqa.ag.run_4','Yes','Run 1 with quote removal; parsing check'),
  ('AG','WHU-NERCMS','videollama-2B','Yes','Off-the-shelf VideoLLaMA'),
  ('AG','WHU-NERCMS','2B-without-prompt','No','Shown in overview figure only'),
  ('AG','WHU-NERCMS','videollama-2B-prompt','Yes','Prompted 2B variant'),
  ('AG','WHU-NERCMS','videollama-7B-prompt','Yes','Prompted 7B variant'),
  ('AG','kslab','nut-kslab-2025','Yes','Fine-tuned VideoLLaMA3-2B'),
  ('AG','HLTCOE','MARS-GenR_1','Yes','Qwen generator and listwise reranker'),
  ('AG','NII_UIT','ag_nii_uit','Yes, under likely alias','Appears to correspond to Aria8x3.5B_VidLLaMa; NIST does not explicitly confirm the alias'),
  ('AG','tcna','gemini_flash','No','Shown in overview figure only'),
  ('AG','tcna','gemini_flash_2.5','No','Shown in overview figure only'),
  ('AG','tcna','gemini_flash_25','Yes','Gemini Flash 2.5'),
  ('AG','tcna','gemini_flash_lite','Yes','Gemini Flash Lite'),
  ('AG','TJUMI','TJUMI.V2','No','Original submission; team later reported a bug'),
  ('AG','TJUMI','TJUMIV3','No','Original submission; corrected after official release'),
  ('AG','TJUMI','TJUMIV4','No','Original submission; corrected after official release'),
  ('AG','TJUMI','TJUMIV5','No','Original submission; corrected after official release'),
  ('AG','TJUMI','TJUMIV6','No','Original submission; corrected after official release'),
  ('MC','NII_UIT','HIGHEST_PIPELINE','Yes','Highest current Top-1 and MRR'),
  ('MC','CERTH-ITI','certh.vqa.mc.run_1','Yes','Zero-shot logit pooling'),
  ('MC','CERTH-ITI','certh.vqa.mc.run_2','Yes','Fine-tuned logit pooling'),
  ('MC','CERTH-ITI','certh.vqa.mc.run_3','Yes','Zero-shot confidence scoring'),
  ('MC','CERTH-ITI','certh.vqa.mc.run_4','Yes','Fine-tuned confidence scoring'),
  ('MC','WHU-NERCMS','videollama-7B-vqa','Yes','Generated answer embedded against options'),
  ('MC','kslab','tv25-kslab-mc','Yes','Generated answer embedded against options'),
  ('MC','TJUMI','TjuMI_MC_VQA_V4.csv','No','Original bugged submission'),
  ('MC','TJUMI','tjumi-v1.csv','No','Original bugged submission')
) AS t(task, team, run, exact_appendix_score, note);

CREATE OR REPLACE VIEW team_approaches AS
SELECT *
FROM (VALUES
  ('CERTH-ITI','AG + MC','Qwen2.5-Omni-7B','Low-Rank Adaptation (LoRA) on 500 examples; ten random frame subsets; AG majority vote; MC mean option-logit pooling','2 weeks; one RTX 4090 reported for fine-tuning'),
  ('HLTCOE','AG','Qwen2.5-VL-7B + Whisper','Candidate generation plus LoRA listwise pointer reranker with rank-weighted loss','3 weeks'),
  ('kslab','AG + MC','VideoLLaMA3-2B','LoRA and stochastic AG; MC ranks options by similarity to a generated anchor answer','About 1 month'),
  ('NII_UIT','AG + MC','Aria 8x3.5B / VideoLLaMA-7B; InternVL3.5-38B + Whisper','Training-free dense frames, spatial tiling, textualized audio, question-aware prompts, normalization','About 1 month'),
  ('tcna','AG','Gemini Flash 1.5 / 2.5 family','Off-the-shelf proprietary models; no participant paper','1 month'),
  ('TJUMI','AG + MC','VideoLLaMA2','Fine-tuned on ActivityNet-200, TRECVID VTT/VQA, InternVid-10M, and other data; original runs had a submission bug','Since March 2024'),
  ('WHU-NERCMS','AG + MC','VideoLLaMA 2B / 7B','Off-the-shelf AG; MC embeds a generated answer and options, then ranks by cosine similarity','3 weeks')
) AS t(team, tasks, models, method, development);
