# Current open video-model options for TRECVID VQA (2026-07-20)

## Bottom line

Yes, there are now **credible open-weight upgrades and alternatives** to the models used by the leading TRECVID 2025 teams. The clearest current candidates for this repository are:

1. **Baseline:** `Qwen/Qwen3.5-4B` with sampled frames plus a Whisper transcript.
2. **Stronger visual run:** `Qwen/Qwen3.5-9B` with exactly the same evidence and prompt.
3. **Audio-native specialist:** `openbmb/MiniCPM-o-4_5`, passing both video and original audio.
4. **Conservative/fallback baseline:** `Qwen/Qwen3-VL-4B-Instruct` or `Qwen/Qwen3-VL-8B-Instruct` plus Whisper.

These are **recommendations to evaluate**, not claims about TRECVID accuracy. None of the official sources below reports a TRECVID 2025/2026 result for these newer checkpoints. Benchmark gains do not guarantee gains on short-answer TRECVID VQA, and model-publisher tables are not independent evaluations.

## What “open” means here

Qwen3-VL, Qwen3.5, Qwen3-Omni, MiniCPM-V/o, and the released InternVL3.5 weights are distributed under Apache-2.0 according to their official model cards. InternVL's repository code is MIT while released model weights/cards are Apache-2.0; third-party components retain their licenses. This is permissive and practical for experimentation, but “open-weight” is the safest general term: releasing weights and inference code does not necessarily expose the full pretraining corpus or make the entire training process reproducible.

Sources: [Qwen3-VL-8B card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), [Qwen3.5-4B card](https://huggingface.co/Qwen/Qwen3.5-4B), [Qwen3.5-9B card](https://huggingface.co/Qwen/Qwen3.5-9B), [Qwen3-Omni repository](https://github.com/QwenLM/Qwen3-Omni), [MiniCPM-o 4.5 card](https://huggingface.co/openbmb/MiniCPM-o-4_5), [InternVL3.5-8B card](https://huggingface.co/OpenGVLab/InternVL3_5-8B).

## Ranked recommendation

### 1. Baseline: Qwen3.5-4B + 12 frames + Whisper

Why it is the most interesting current starting point:

- It is a 4B-parameter unified vision-language model with native video input and Apache-2.0 weights.
- The official card reports 83.5 VideoMME with subtitles, 76.9 without subtitles, 74.1 VideoMMMU, 82.8 MLVU, 71.2 MVBench, 66.4 LVBench, and 64.9 MMVU. In the same publisher table, it exceeds Qwen3-VL-30B-A3B on six of seven listed video benchmarks (MVBench is the exception: 71.2 versus 72.0). This is unusually strong evidence for a small model, but remains a publisher-reported cross-benchmark comparison—not a TRECVID result.
- The official video example supports adjustable FPS sampling; video is visual input only, so a Whisper transcript still supplies speech.
- The BF16 repository is about 9.3 GB of weights. On this 122 GiB unified-memory host, the weight footprint is comfortable; visual tokens, KV cache, activations, and runtime add memory. Actual Radeon 8060S/ROCm compatibility and throughput are not established by the model card and require a smoke test.

Caveats:

- Qwen3.5 is newer and currently asks for very recent/nightly vLLM, SGLang, or Transformers in its official instructions. That may be less frictionless than Qwen3-VL.
- It thinks by default; disable thinking for a cheap direct-answer baseline, or evaluate thinking as a separate controlled factor.
- The model card recommends very large context for general reasoning, but TRECVID's short clips/questions do not require serving the full 262K context. Limit model length to avoid wasting memory.

Sources: [official Qwen3.5-4B model card and benchmark table](https://huggingface.co/Qwen/Qwen3.5-4B), including its [video-input example](https://huggingface.co/Qwen/Qwen3.5-4B#video-input).

### 2. Stronger visual run: Qwen3.5-9B + the identical evidence pipeline

- Same Apache-2.0 family and video interface as the 4B model.
- The official table reports higher results than the 4B on every listed video benchmark: VideoMME 84.5/78.4 (with/without subtitles), VideoMMMU 78.9, MLVU 84.4, MVBench 74.4, LVBench 70.0, and MMVU 67.8.
- Its roughly 9B language-model scale implies approximately 18 GB for BF16 language weights, before the vision encoder and runtime state. That should fit the host's unified memory, but speed and ROCm kernel compatibility still need measurement.

This should be the first scale ablation because it changes the model while keeping frame timestamps, transcript, prompt, decoding, and scoring fixed.

Source: [official Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B).

### 3. Audio-native specialist: MiniCPM-o 4.5

- A 9B end-to-end model built from SigLIP2, Whisper-medium, CosyVoice2, and Qwen3-8B. It accepts simultaneous video and audio and can therefore use dialogue plus non-speech sound without converting everything into prompt text.
- It supports up to 10 FPS video, instruct/thinking modes, and is Apache-2.0.
- OpenBMB's model zoo lists approximately 19 GB GPU memory for the full model, 11 GB for AWQ, and 10 GB for GGUF. That makes it much more practical locally than Qwen3-Omni-30B-A3B.
- The official implementation recommends NVIDIA GPU inference for full precision but also lists llama.cpp/Ollama, quantized formats, vLLM, and SGLang. The Radeon/ROCm path is not verified in the official materials; GGUF/CPU is the safer compatibility fallback.

Use this as an **audio ablation**, not an automatic replacement for Whisper: compare the same examples under Qwen3.5 + transcript versus MiniCPM-o with native audio. Its integrated Whisper-medium encoder is not evidence that it will transcribe speech better than a separately chosen Whisper checkpoint, and native audio may add value mainly on non-speech cues.

Source: [official MiniCPM-o 4.5 model card](https://huggingface.co/openbmb/MiniCPM-o-4_5) and [official MiniCPM repository/model zoo](https://github.com/OpenBMB/MiniCPM-V).

### 4. Conservative baseline: Qwen3-VL-4B/8B + Whisper

- Qwen3-VL is a direct successor to Qwen2.5-VL and explicitly added improved spatiotemporal position encoding, multi-level visual features, and text/timestamp alignment for temporal grounding. It accepts videos or explicit image lists as a video and exposes FPS/pixel/token limits.
- The dense family includes 2B, 4B, 8B, and 32B variants; 4B/8B Instruct weights are Apache-2.0.
- The official report describes native 256K interleaved text/image/video context. Qwen3-VL is visual-only, so Whisper remains necessary for audio.
- Approximate BF16 weight floors are about 8–9 GB for 4B and 16–17 GB for 8B, before the visual encoder and runtime state. Both are plausible on this host; actual ROCm support must still be tested.

Qwen3-VL is still a strong choice if Qwen3.5's newer runtime causes trouble. It is a more defensible “one generation newer” replacement for the Qwen2.5-VL-7B used by HLTCOE. However, current official Qwen3.5 results make Qwen3.5-4B/9B the more interesting primary experiment today.

Sources: [Qwen3-VL technical report](https://arxiv.org/abs/2511.21631), [official repository and video-processing examples](https://github.com/QwenLM/Qwen3-VL), and [Qwen3-VL-8B-Instruct card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).

## Other relevant options

### Qwen3-Omni-30B-A3B: strongest ambitious audio/video experiment, not the first local baseline

Qwen3-Omni is the direct omni-modal successor to Qwen2.5-Omni-7B used by CERTH-ITI. Its 30B-A3B Instruct and Thinking checkpoints accept text, images, video, and audio; `use_audio_in_video=True` explicitly extracts and passes a video's sound. It is Apache-2.0 and the official report/repository claims broad gains over prior omni-modal models.

The deployment cost is the problem. Qwen's own minimum-memory table gives approximately 78.85 GB for Instruct BF16 on a 15-second video and 68.74 GB for Thinking, rising steeply with duration (for example, 107.74/95.76 GB at 60 seconds). The host has 122 GiB unified memory, so the raw totals are not impossible, but the official stack is CUDA/FlashAttention/vLLM-oriented and does not verify this Radeon 8060S. A full-precision local run is therefore high-risk and should follow smaller end-to-end checks, not precede them.

Sources: [official Qwen3-Omni repository](https://github.com/QwenLM/Qwen3-Omni), [Qwen3-Omni-30B-A3B-Thinking card](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking), and [technical report](https://arxiv.org/abs/2509.17765).

### InternVL3.5: still current; no clear general successor to NII_UIT's 38B model

As of this date, InternVL3.5 remains OpenGVLab's latest general-purpose video-capable InternVL family. It spans small dense checkpoints through 38B and MoE variants, and has later “Flash” releases that use a visual-resolution router to cut visual tokens by about half while retaining nearly all aggregate image-benchmark performance.

Important nuance: InternVL's video path uniformly samples frames and treats them as a multi-image sequence. The official benchmark protocol tries 16/32/48/64 frames for most video benchmarks and reports the best result; MVBench uses 16. Official overall video scores scale from 64.9 (4B) to 65.8 (8B), 67.6 (30B-A3B), 71.0 (38B), and 71.4 (241B-A28B). InternVL3.5 is not uniformly better than InternVL3 at matched/nearby sizes in that table, so it should not be described as a blanket video upgrade.

For this host, 4B/8B are plausible experiments. The official deployment guidance is NVIDIA A100-specific and examples call CUDA/FlashAttention/bitsandbytes; Radeon ROCm compatibility is unverified. The 38B used by NII_UIT remains expensive (rough BF16 weight floor about 77 GB before runtime state), even though this machine's unified memory is large.

Sources: [InternVL3.5 technical report, especially video Table 9](https://arxiv.org/html/2508.18265v1), [official InternVL3.5-8B card](https://huggingface.co/OpenGVLab/InternVL3_5-8B), [official repository](https://github.com/OpenGVLab/InternVL), and [official Flash collection](https://huggingface.co/collections/OpenGVLab/internvl35-flash).

### MiniCPM-V 4.5/4.6: efficiency controls, not the leading quality bet

- MiniCPM-V 4.5 is an 8–9B visual-only video model based on Qwen3-8B and supports compressed high-FPS video. It is a credible efficient alternative to Qwen3-VL-8B.
- MiniCPM-V 4.6 is a 1.3B visual-only edge model requiring about 4 GB in the official model zoo (3 GB quantized). It is valuable as a very cheap pipeline smoke test, not the most likely quality winner.
- Both are Apache-2.0 and still need Whisper for speech.

Sources: [MiniCPM-V 4.5 card](https://huggingface.co/openbmb/MiniCPM-V-4_5), [MiniCPM-V 4.6 card](https://huggingface.co/openbmb/MiniCPM-V-4.6), and [official repository](https://github.com/OpenBMB/MiniCPM-V).

## Comparison with the 2025 leading-team models

| 2025 reference | Current candidate | Verified difference | What is *not* verified |
|---|---|---|---|
| Qwen2.5-VL-7B | Qwen3-VL-4B/8B | Newer Qwen video architecture; explicit temporal/timestamp upgrades; newer family | Higher TRECVID score |
| Qwen2.5-VL-7B | Qwen3.5-4B/9B | Newer unified VLM; publisher reports very strong video benchmarks, including 4B beating its own Qwen3-VL-30B-A3B on 6/7 listed video tests | Direct same-prompt comparison with Qwen2.5-VL-7B on TRECVID |
| Qwen2.5-Omni-7B | MiniCPM-o 4.5 | Current 9B Apache model with native simultaneous video/audio | Better TRECVID audio reasoning |
| Qwen2.5-Omni-7B | Qwen3-Omni-30B-A3B | Direct newer Qwen omni family; native audio/video; much larger total model | Practical ROCm deployment or higher TRECVID score |
| InternVL3.5-38B | InternVL3.5 Flash / smaller 30B-A3B | Same current family with cheaper visual-token routing or fewer active parameters | A newer OpenGVLab generation that clearly replaces the winning 38B model |
| Aria-8×3.5B | Qwen3.5-9B or Qwen3-Omni | Newer actively supported multimodal families with video examples | Superior AG ranking/semantic score on TRECVID |

## The experiment that would answer the question honestly

Use one fixed validation split and one frozen evidence bundle per example:

1. Extract the same 12 global frames and timestamps once.
2. Generate one Whisper transcript once.
3. Run `Qwen3-VL-4B-Instruct`, `Qwen3.5-4B`, and `Qwen3.5-9B` with identical frame files, transcript, prompt, decoding, and answer normalization.
4. Run MiniCPM-o 4.5 separately with the same visual timestamps plus native audio.
5. Record accuracy/similarity, decoding failures, peak memory, wall time, and failure category.

This cleanly separates **model quality** from frame selection, transcript quality, prompting, and preprocessing. Only those local results can establish which current model is actually better for TRECVID.

## Host-specific practicality notes

Verified local hardware: AMD Ryzen AI MAX+ 395 / Radeon 8060S (`gfx1151`) with 122 GiB system/unified memory available to the host. Memory capacity is unusually generous; software support is the larger unknown.

- Prefer a rootless Podman ROCm smoke test before a dataset-scale run.
- Start with one short decoded video and 12 frames; then a batch of 5–10 mixed-codec videos.
- Do not infer GPU compatibility merely from a model fitting in RAM. Official examples for several candidates assume CUDA, FlashAttention, bitsandbytes, or NVIDIA containers.
- If ROCm kernels fail, Transformers eager/SDPA or CPU/GGUF can establish correctness, but performance may be much lower.
- Do not pull Qwen3-Omni or a 38B model before the 4B/9B candidates pass an end-to-end test; their downloads and runtime costs are much larger.
