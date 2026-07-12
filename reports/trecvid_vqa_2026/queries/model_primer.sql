-- Beginner model-primer views synthesized from primary model papers/model cards.
-- Exact links and claim-level notes are recorded in source_notes.md.
CREATE OR REPLACE VIEW model_anatomy AS
SELECT *
FROM (VALUES
  (1,'Media sampler','Chooses frames, shots, crops, and audio segments the model is allowed to inspect.','Uniform frames, random views, shot retrieval','The decisive moment is never selected.'),
  (2,'Vision encoder','Turns image patches into numeric visual tokens.','ViT, SigLIP, CLIP, InternViT','Small text or objects vanish at the selected resolution.'),
  (3,'Projector / connector','Translates and compresses visual tokens into the language model''s input space.','MLP, Q-Former, spatial-temporal connector','Compression discards brief motion or fine detail.'),
  (4,'Audio path','Supplies speech as a transcript or native sound features.','Whisper ASR, native audio encoder','Names, overlapping speech, music, or effects are lost.'),
  (5,'Language model','Combines the question and multimodal evidence to produce or score answers.','Qwen, InternVL LLM, Aria MoE, Gemini','Fluent output is unsupported or hallucinates missing evidence.'),
  (6,'Adaptation','Changes behavior with prompts or task-specific weight updates.','Prompt routing, LoRA fine-tuning','Small training data causes overfitting or leakage.'),
  (7,'Output head / ranker','Converts raw model output into the exact AG or MC ranked contract.','Vote pooling, option scoring, listwise reranker','Good evidence is lost to parsing, duplicates, or a weak answer tail.')
) AS t(stage, component, plain_meaning, examples, common_failure);

CREATE OR REPLACE VIEW model_family_guide AS
SELECT *
FROM (VALUES
  ('Qwen2.5-Omni','Visual and audio encoders feed a text-generating Thinker; an unused Talker can generate speech.','Native','CERTH-ITI AG + MC','Only the Thinker matters for VQA; time-aligned audio/video is its distinguishing feature.'),
  ('Qwen2.5-VL','Dynamic-resolution vision transformer and merger feed a Qwen language decoder with explicit time encoding.','External transcript','HLTCOE generator + reranker','Strong visual/video backbone; speech must arrive through ASR or another audio path.'),
  ('InternVL3.5-38B','InternViT vision encoder → MLP bridge → language model, with dynamic high-resolution tiling.','External transcript','NII_UIT MC','Tiling preserves local detail but multiplies the visual-token budget.'),
  ('Aria','Vision encoder and projector feed a sparse mixture-of-experts language decoder.','External transcript','NII_UIT AG','Only a subset of experts activates per token; activated size is not total storage.'),
  ('VideoLLaMA2','Frame encoder and spatial-temporal connector feed an LLM; an audio-video variant adds an audio branch.','Variant-dependent','TJUMI; WHU lineage','Connector mixes space and time, but omitted frames cannot be recovered later.'),
  ('VideoLLaMA3','SigLIP vision encoder, any-resolution tokenization, and differential frame pruning feed Qwen.','No native audio in reported design','kslab; WHU lineage','Efficient video tokens; subtle changes need stress testing after pruning.'),
  ('Gemini Flash','Closed multimodal service; component-level internals are not sufficiently published for reproduction.','Native service capability','tcna AG','Treat as a black box and record exact endpoint, date, prompt, and settings.'),
  ('Whisper','Speech encoder-decoder that converts audio into timestamped text.','ASR only','NII_UIT and HLTCOE','A transcript exposes dialogue but not every sound, speaker cue, or acoustic event.'),
  ('Sentence-BERT style encoder','Maps an answer and each option into comparable text vectors.','None','WHU-NERCMS and kslab MC','Convenient fallback, but generator errors and semantic distractors propagate into ranking.')
) AS t(family, blueprint, audio, use_2025, beginner_takeaway);
