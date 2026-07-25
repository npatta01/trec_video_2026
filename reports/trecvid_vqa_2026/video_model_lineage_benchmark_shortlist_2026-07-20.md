# Video-model lineage and a non-redundant TRECVID pilot shortlist

**Research date:** 2026-07-20
**Scope:** VideoLLaMA3-7B, MiniCPM-o 4.5, InternVL3.5 4B/8B, Molmo2 4B/8B, and Qwen3.5 4B/9B. Claims below use first-party model cards, repositories, papers, and AMD documentation.

## Answer first

Yes: **VideoLLaMA3, InternVL3.5, Molmo2, and MiniCPM-o 4.5 were all released before the small Qwen3.5-4B/9B checkpoints.** The Qwen3.5 family began with a 397B-A17B model on 2026-02-16, but the downloadable 4B and 9B checkpoints did not arrive until **2026-03-02** ([Qwen repository news](https://github.com/QwenLM/Qwen3.5#news)). MiniCPM-o 4.5 preceded even the first Qwen3.5 release by 13 days.

They can be worth benchmarking despite their Qwen language backbones. A VLM is more than its LLM: its vision/audio encoders, connector, multimodal training data, temporal representation, token compression, tiling, and inference processor can materially change video QA. But running every size of every family would be expensive and highly redundant.

The highest-value initial additions are:

1. **Molmo2-4B** — best clean test of whether video-specific training and temporal grounding improve on a similarly sized Qwen-based baseline.
2. **MiniCPM-o 4.5** — unique test of end-to-end audio+video versus Qwen3.5 plus a Whisper transcript.
3. **InternVL3.5-4B**, initially only on a visual-detail subset — tests dynamic high-resolution tiling, the idea behind the prior winning InternVL pipeline.

Do **not** initially run VideoLLaMA3-7B, Molmo2-8B, and InternVL3.5-8B across all 100 examples. They add less information per unit of setup and compute.

## Release chronology and exact lineage

| Checkpoint | First official release | Exact language backbone | Vision/audio stack | License signal |
|---|---:|---|---|---|
| `DAMO-NLP-SG/VideoLLaMA3-7B` | 2025-01-21 weights/repo; paper 2025-01-22 (its card has an apparent `2024` typo) | `Qwen2.5-7B-Instruct` | tuned SigLIP SO400M/14 NaViT; no audio path in this checkpoint | Apache-2.0 metadata, but README adds a non-commercial research-preview restriction |
| `OpenGVLab/InternVL3_5-4B` | 2025-08-26 | `Qwen3-4B` | InternViT-300M, dynamic 448px tiles; no audio | Apache-2.0 |
| `OpenGVLab/InternVL3_5-8B` | 2025-08-26 | `Qwen3-8B` | InternViT-300M, dynamic 448px tiles; no audio | Apache-2.0 |
| `allenai/Molmo2-4B` | 2025-12-11 announcement | `Qwen3-4B-Instruct-2507` | SigLIP 2; timestamp-aware image/video connector; no audio | Apache-2.0, with training-data use caveat |
| `allenai/Molmo2-8B` | 2025-12-11 announcement | `Qwen3-8B` | SigLIP 2; same video-native design; no audio | Apache-2.0, with training-data use caveat |
| `openbmb/MiniCPM-o-4_5` | 2026-02-03 | `Qwen3-8B` | SigLIP2 + Whisper-medium + CosyVoice2 | Apache-2.0 |
| Qwen3.5 family | 2026-02-16 | new Qwen3.5 hybrid Gated DeltaNet/attention architecture | unified early-fusion vision-language model | Apache-2.0 |
| `Qwen/Qwen3.5-4B`, `-9B` | **2026-03-02** | native Qwen3.5 4B or 9B (not a Qwen3-VL adapter) | Qwen3VL video processor; no audio | Apache-2.0 |

Primary evidence:

- VideoLLaMA3's [official model card](https://huggingface.co/DAMO-NLP-SG/VideoLLaMA3-7B) identifies Qwen2.5-7B and a direct video processor. Hugging Face's [official model API metadata](https://huggingface.co/api/models/DAMO-NLP-SG/VideoLLaMA3-7B) records repository creation on 2025-01-21; the [paper record](https://arxiv.org/abs/2501.13106) is dated 2025-01-22, resolving the card's obvious year typo.
- InternVL's [official repository news](https://github.com/OpenGVLab/InternVL#news-) dates InternVL3.5 to 2025-08-26. The [4B config](https://huggingface.co/OpenGVLab/InternVL3_5-4B/blob/main/config.json) names `Qwen3ForCausalLM` and `/Qwen3-4B`; the [4B pretrained model tree](https://huggingface.co/OpenGVLab/InternVL3_5-4B-Pretrained) and [8B card](https://huggingface.co/OpenGVLab/InternVL3_5-8B) confirm the exact sizes and Apache license.
- Ai2's [Molmo2 announcement](https://allenai.org/blog/molmo2) is dated 2025-12-11. The [4B card](https://huggingface.co/allenai/Molmo2-4B) identifies `Qwen3-4B-Instruct-2507`; the [8B card](https://huggingface.co/allenai/Molmo2-8B) identifies `Qwen3-8B`.
- OpenBMB's [official repository](https://github.com/OpenBMB/MiniCPM-V#minicpm-o-45) dates MiniCPM-o 4.5 to 2026-02-03 and identifies Qwen3-8B, SigLIP2, Whisper-medium, and CosyVoice2.
- Qwen's [official repository news](https://github.com/QwenLM/Qwen3.5#news) distinguishes the 2026-02-16 family launch from the 2026-03-02 release of 4B/9B. The [4B model card](https://huggingface.co/Qwen/Qwen3.5-4B) documents the new architecture, video API, and Apache-2.0 license.

## What each model actually adds

### Input-path reality check

| Checkpoint | Video input in the official example | Audio from the video |
|---|---|---|
| Qwen3.5-4B/9B | Direct video URL/path handled by its Qwen3VL video processor; configurable FPS in vLLM | No |
| Molmo2-4B/8B | Direct `type="video"` processor path | No |
| InternVL3.5-4B/8B | Decord samples frames, tiles each as images, then constructs `Frame1: <image> ...` | No |
| MiniCPM-o 4.5 | Structured `video_url` helper produces synchronized interleaved image/audio inputs | **Yes** |
| VideoLLaMA3-7B | Direct `video_path` with FPS/max-frame controls | No |

“Native video” therefore has two meanings in model marketing: accepting a video file and decoding it for the user, or actually training/representing temporal video relationships. Neither implies that audio is used. MiniCPM-o 4.5 is the only checkpoint in this table with an integrated audio path.

### Qwen3.5-4B and 9B

These are the current reference models, not merely later versions of the Qwen3 backbones embedded in the specialist VLMs. Qwen3.5 uses unified early-fusion multimodal training and a hybrid architecture of Gated DeltaNet and periodic full attention. Its processor accepts a video URL/path and samples frames (the vLLM example defaults to 2 FPS and permits an FPS override). It does **not** consume the video's audio track; audio must be represented separately, such as a Whisper transcript.

Qwen's same-table results are unusually strong: Qwen3.5-4B beats Qwen3-VL-30B-A3B on all seven listed video benchmarks, including Video-MME without subtitles (76.9 vs 73.3), MLVU (82.8 vs 78.9), and LVBench (66.4 vs 59.2). Qwen3.5-9B is stronger again on all seven (for example, 78.4 Video-MME without subtitles and 70.0 LVBench). These are publisher-reported—not TRECVID—results, but they justify using 4B as the baseline and 9B as the scale rung. See the [official 4B benchmark table](https://huggingface.co/Qwen/Qwen3.5-4B#vision-language).

### Molmo2-4B and 8B

Molmo2 is genuinely video-native even though its LLM is Qwen3. The processor accepts `type="video"` directly. Its lightweight connector interleaves visual tokens with timestamps and image indices; training adds dense video captioning, temporal/spatial pointing, tracking, counting-by-pointing, multi-image QA, and subtitle-aware QA. Ai2 also describes a SlowFast-style option: high resolution on key frames and lower resolution elsewhere. It has no audio encoder, so use the same Whisper transcript as Qwen for a fair audio-aware comparison.

Ai2's common 15-video-benchmark average reports Molmo2-4B at **62.8** and Molmo2-8B at **63.1**, versus Qwen3-VL-4B at 58.1, Qwen3-VL-8B at 59.5, InternVL3.5-4B at 53.4, and InternVL3.5-8B at 54.1 ([Molmo2-4B evaluation table](https://huggingface.co/allenai/Molmo2-4B#evaluations)). This is strong evidence that the video specialization matters, but not proof against Qwen3.5, which was released later and is absent from that table.

The 0.3-point 4B-to-8B difference makes the **4B model the informative first run**. Run 8B only if the 4B result is competitive on TRECVID or if a targeted failure analysis suggests language/reasoning capacity is the bottleneck.

License nuance: the checkpoints are Apache-2.0, but Ai2 says some third-party training datasets are limited to academic/non-commercial research. That does not prevent this research benchmark, but it matters for later commercial reuse ([8B license section](https://huggingface.co/allenai/Molmo2-8B#license-and-use)).

### InternVL3.5-4B and 8B

InternVL3.5 combines a Qwen3 LLM with InternViT-300M. Its differentiator for this project is dynamic spatial preprocessing: an image can be broken into as many as 12 448×448 patches, plus a thumbnail. A video is decoded into frames and each frame is processed through this image/tiling path. Thus it is better described as a frame-sequence VLM with explicit temporal frame prefixes than an end-to-end audio-video model. It has no audio encoder.

The 3.5 generation adds native multimodal pretraining, Cascade RL (offline MPO plus online RL), Visual Resolution Router, and Decoupled Vision-Language Deployment. The dynamic patch path directly tests whether zoomed spatial evidence helps TRECVID questions about small objects, text, or background details.

Why not run both sizes immediately? They share the same vision encoder, tiling behavior, and Qwen3 lineage. Ai2's later common-eval table puts 4B and 8B close together (53.4 vs 54.1) and below both corresponding Qwen3-VL and Molmo2 checkpoints. OpenBMB's own current video table reports InternVL3.5-8B at 66.0 on Video-MME without subtitles and 70.2 on MLVU, below MiniCPM-o 4.5 on those rows ([OpenBMB evaluation table](https://github.com/OpenBMB/MiniCPM-V#evaluation)). These cross-publisher results are not perfectly neutral, but they argue for a **targeted 4B tiling test**, not two full runs.

### MiniCPM-o 4.5

MiniCPM-o is the one candidate here that directly changes the audio experiment. It is an end-to-end 9B omnimodal model built from Qwen3-8B, SigLIP2, Whisper-medium, and CosyVoice2. It can jointly process streaming video and audio and supports general audio description, speech recognition, and speaker analysis—not merely append a text transcript. Its video path supports up to 10 FPS, and the architecture synchronizes input/output streams on a millisecond timeline.

On OpenBMB's same table, MiniCPM-o 4.5 is essentially tied with the much larger Qwen3-Omni-30B-A3B-Instruct on Video-MME without subtitles (70.4 vs 70.5), leads it on MLVU (76.5 vs 75.2), and trails slightly on LongVideoBench and MotionBench. On a seven-benchmark audio-video suite, OpenBMB reports a 68.5 average versus 63.7 for Qwen3-Omni and 63.6 for Gemini 2.5 Flash. Some competitor entries are marked as reproduced by OpenBMB; treat this as selection evidence, not a final verdict ([official table and notation](https://github.com/OpenBMB/MiniCPM-V#evaluation)).

It therefore deserves a test, but specifically as **native video+audio versus Qwen3.5+Whisper**, not as another generic 9B visual model.

### VideoLLaMA3-7B

VideoLLaMA3-7B is the oldest candidate. It uses Qwen2.5-7B-Instruct and an adapted SigLIP SO400M/14 encoder. The processor accepts a video path, FPS, and maximum frame count directly. Its vision-centric design supports variable-resolution images and prunes similar/redundant visual tokens across video frames to form a compact representation. It was trained in four stages culminating in video-centric fine-tuning ([official paper](https://arxiv.org/abs/2501.13106)).

Despite the family's historic “audio-visual” branding, the exact VideoLLaMA3-7B checkpoint documented here is **video+text, not audio+video**. The official example passes only video and text, and the architecture/model card exposes no audio encoder. It is also an older Qwen2.5 model whose official example requires `flash_attention_2` and CUDA-oriented calls. It is useful as a historical control, but it is a low-priority use of the first 100 examples.

The release paper did show that the specialist treatment improved over its contemporary Qwen baseline: 7B beat Qwen2-VL-7B on Video-MME without subtitles (66.2 vs 63.3), MVBench (69.7 vs 67.0), and LongVideoBench (59.8 vs 55.6), while losing on EgoSchema (63.3 vs 66.7). That establishes historical value, not competitiveness with Qwen3.5 ([official paper, Table 8](https://arxiv.org/pdf/2501.13106)).

License caution: while the repository/model metadata says Apache-2.0, the [official README license section](https://github.com/DAMO-NLP-SG/VideoLLaMA3#license) also calls it a research preview for non-commercial use only and subjects use to upstream/data terms. Treat the checkpoint's commercial reuse status as ambiguous until clarified; this does not prevent the proposed research evaluation.

## AMD Ryzen AI MAX+ 395 / ROCm risk

The host itself is now a supported target: AMD's ROCm 7.2.1 Linux matrix lists the Ryzen AI MAX+ 395 (`gfx1151`) and PyTorch 2.9.1, but says only **FP16** has been officially validated. AMD also warns of lower-than-expected performance on some LLM workloads and recommends 64 GB or more system memory ([support matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html), [limitations](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/limitations/limitationsryz.html), [memory guidance](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/prerequisites/prerequisitesryz.html)). Model support is nevertheless not guaranteed just because PyTorch supports the GPU.

| Model | ROCm integration risk | Why / first fallback |
|---|---|---|
| Qwen3.5-4B/9B | **Medium** | Official card currently asks for latest/main Transformers or nightly vLLM; hybrid Gated DeltaNet kernels are newer. Start in PyTorch/Transformers with FP16, SDPA/eager attention, short context, batch 1. |
| Molmo2-4B/8B | **Medium** | Standard PyTorch path but uses trusted remote code plus `decord2`/`molmo_utils`; avoid assuming vLLM ROCm compatibility. Disable optional fused attention and smoke-test decode separately. |
| InternVL3.5-4B/8B | **Medium-high** | Config enables FlashAttention and custom code by default; dynamic tiling can sharply increase memory. Force eager/SDPA, FP16, and cap tiles. |
| MiniCPM-o 4.5 | **High** | Official full demo requires an Nvidia GPU with at least 28 GB and uses `.cuda()`; audio/video streaming adds custom dependencies. First test offline non-streaming inference; if ROCm fails, use its GGUF/CPU path only if it preserves the needed video+audio mode. |
| VideoLLaMA3-7B | **High** | Official installation specifies CUDA ≥11.8, FlashAttention 2, and `.cuda()`; custom older code is likely to need edits or eager-attention fallback on ROCm. |

Before downloading all weights, run one local-video smoke test per chosen architecture in a rootless container, with no service exposed. Record processor success, device placement, peak shared memory, decode time, and whether any tensor silently falls back to CPU.

## Recommended 100-example benchmark design

### Keep two comparisons separate

1. **Controlled evidence track:** every model receives the same 12 timestamped frames, question, prompt, transcript, and decoding settings. This isolates model/vision-stack differences as much as practical.
2. **Preferred native track:** a model may use its own direct-video processor or native audio path. This measures the best usable system, but it no longer isolates the model from preprocessing.

Do not compare a 12-frame Qwen run with a 10-FPS MiniCPM run and call the difference a model-family effect.

### Stage 0 — portability gate (3 examples, no metric decision)

Run three deliberately different clips: one short visual question, one small-detail/text question, and one audio-dependent question. Test only:

- Qwen3.5-4B
- Molmo2-4B
- InternVL3.5-4B
- MiniCPM-o 4.5

Stop a model's expansion if it cannot run reproducibly on ROCm/CPU fallback or its preprocessing timestamps cannot be logged. This avoids four large downloads turning into four debugging projects.

### Stage 1 — core 100-example matrix

| Run | Input | What it answers |
|---|---|---|
| A | Qwen3.5-4B + 12 frames | visual baseline |
| B | Qwen3.5-4B + same frames + Whisper | transcript value |
| C | Molmo2-4B + same frames/timestamps + same Whisper text | benefit of video/grounding specialization at roughly 4B |
| D | Qwen3.5-9B + same frames + same Whisper | scale effect within one family |

These four full runs give the most interpretable first result.

### Stage 2 — targeted specialist tests

- **InternVL3.5-4B:** run first on the 20–30 pilot questions tagged as OCR, small object, counting, or background detail, using its native dynamic tiles. Expand to 100 only if it clearly helps this subgroup.
- **MiniCPM-o 4.5:** run its native audio+video path first on all audio/speech/sound-dependent pilot questions, plus a matched sample of visual-only questions. Expand to 100 if native audio helps or overall performance is competitive.
- **One truly non-Qwen language family:** add Gemma 4 12B (or Molmo2-O-7B if full openness is the priority) after the Qwen-derived specialist results. This is the family-diversity test; InternVL/Molmo/MiniCPM do not provide it.

### Deferred

- **Molmo2-8B:** only if Molmo2-4B is competitive; its publisher's 15-benchmark gain over 4B is only 0.3 points.
- **InternVL3.5-8B:** only if 4B tiling helps and failures look like reasoning capacity rather than missing evidence.
- **VideoLLaMA3-7B:** historical ablation only, after the current models; it uses an older Qwen2.5 backbone, lacks audio, and has the highest legacy CUDA integration cost.

## Metrics and logging needed to make the benchmark useful

For every example and run, retain:

- exact checkpoint revision and license;
- video ID, question ID, question, reference, and prediction;
- input mode (`frames`, `video`, `video_audio`), sampled timestamps/FPS, resolution/tiles, and transcript;
- deterministic decoding configuration and prompt;
- official-style METEOR/BERTScore/STS outputs when available, plus fixed human `correct / partial / wrong` judgments;
- latency, peak memory, failure/fallback state, and output token count;
- question tags: audio, temporal order, causal, counting, OCR/small detail, and general visual.

Primary model selection should use the same 100 examples and human factual correctness alongside the text metrics. Report subgroup results even when small: the purpose of specialist models is precisely to reveal whether audio fusion, temporal grounding, or high-resolution tiling solves a particular failure mode.

## Decision rule

The specialist benchmark is worthwhile, but it should be a **funnel rather than a model zoo**:

```text
Qwen3.5-4B baseline
   ├── + Whisper                 (audio represented as text)
   ├── Molmo2-4B                 (video grounding/training)
   ├── InternVL3.5-4B subset     (spatial tiling)
   └── MiniCPM-o 4.5 subset      (native audio+video)
            ↓
promote only winners to all 100
            ↓
Qwen3.5-9B + one truly non-Qwen family
```

This tests the useful innovations without paying for several nearly duplicate Qwen3-based 8B runs.
