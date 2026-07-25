# Non-Qwen model options for TRECVID-style video QA (2026-07-20)

## Bottom line

Qwen is a sensible baseline, but it is **not established as the best model for this task** until the candidates are scored on the same TRECVID pilot. Google now has a particularly relevant local alternative:

- **Recommended genuinely different-family local run:** `google/gemma-4-12B-it`. It is a 11.95B-parameter, Apache-2.0 Google model that accepts video as a sequence of frames and also accepts audio. Its official card supports up to 60 seconds of video at 1 FPS and up to 30 seconds of audio. This makes it a clean architectural and vendor contrast with Qwen3.5-4B + 12 frames + Whisper.
- **Cheaper genuinely different-family fallback:** `microsoft/Phi-4-multimodal-instruct` (5.6B, MIT). It accepts up to 64 images/frames plus audio, but has no direct video-container interface or publisher video-benchmark table.
- **Optional proprietary ceiling:** `gemini-3.1-pro-preview`, using the original video with its audio. It is a hosted closed model, not a downloadable Google model. Do this only if the dataset terms permit third-party cloud upload.

The most important naming trap is that several apparently non-Qwen projects are **Qwen underneath**: Molmo 2 8B is based on Qwen3-8B; InternVL3.5-8B uses Qwen3; VideoLLaMA3-7B uses Qwen2.5-7B; and MiniCPM-o 4.5 uses Qwen3-8B. They can still be useful video systems, but they are not independent language-backbone families.

## Google: Gemini is not Gemma

### Gemini: hosted, closed, native video and audio

Gemini models are API services. `gemini-3.1-pro-preview` accepts text, images, video, audio, and PDFs, has a 1,048,576-token input limit, and returns text. Google's video guide says Gemini processes both visual and audio streams; the Files API normally samples video at 1 FPS, processes audio, and attaches timestamps. This is the strongest sensible **ceiling/control** because it can consume the original clip rather than our 12-frame/Whisper approximation.

Sources: [Gemini 3.1 Pro Preview model specification](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview), [Gemini video-understanding guide](https://ai.google.dev/gemini-api/docs/video-understanding).

This is not local or open-weight. Uploading TRECVID media is an external disclosure. Before any run, verify the collection's participant agreement and each media source permits cloud processing. Google's current terms say unpaid-service inputs and outputs may be used to improve products and may be human reviewed; paid Gemini API content is not used for product improvement, although limited logging and processing still occur. Files API uploads persist until deletion or automatic expiry (normally 48 hours). If authorized, use a paid Cloud project, avoid the unpaid tier, disable state storage where supported, delete uploaded files immediately after scoring, and keep query/answer artifacts private.

Sources: [Gemini API terms: unpaid versus paid data use](https://ai.google.dev/gemini-api/terms), [Files API storage and deletion](https://ai.google.dev/gemini-api/docs/files), [zero-data-retention guidance](https://ai.google.dev/gemini-api/docs/zdr).

### Gemma: downloadable Google weights

Google's current open-weight family is Gemma. The most relevant release is **Gemma 4**, not Gemma 3n:

| Checkpoint | Size | Video path | Audio | License | TRECVID relevance |
|---|---:|---|---|---|---|
| Gemma 4 E4B | 4.5B effective, 8B total with embeddings | Video processed as frames | Yes | Apache-2.0 | Small Google-family comparison, though its actual weight footprint is larger than “4B” suggests |
| **Gemma 4 12B Unified** | **11.95B** | **Video processed as frames; max 60 s at 1 FPS in the official recipe** | **Yes; max 30 s** | **Apache-2.0** | Best genuinely independent local comparison in the requested 4B–12B range |
| Gemma 4 31B / 26B-A4B | 30.7B / 25.2B total | Frames | No audio | Apache-2.0 | Larger and less relevant to the first pilot |
| Gemma 3n E2B/E4B | 5.1B/8B total, lower effective compute | Images and audio | Yes | Gemma open-weight terms | Superseded for this experiment by Gemma 4; video must be represented as images/frames |

Gemma 4 12B is unusually clean for this comparison: raw image patches and audio waveforms are projected directly into a single decoder-only transformer rather than relying on a Qwen language model. Its model card exposes both a video input and an audio input. It recommends lower visual-token budgets when processing many video frames and reports a 256K context window.

The official card reports strong image and audio results (for example, MMMU-Pro 69.1, MATH-Vision 79.7, and FLEURS WER 0.069 excluding Chinese), but **does not publish a video benchmark in its main table**. Do not infer TRECVID superiority from its image scores.

Sources: [Gemma 4 12B official model card, architecture, inputs, limits, benchmarks, and license](https://huggingface.co/google/gemma-4-12B), [Google's Gemma family overview](https://ai.google.dev/gemma/docs/get_started), [Gemma 3n overview](https://ai.google.dev/gemma/docs/gemma-3n).

## Candidate map

“Video input” below means the software accepts a video and samples/represents it as frames. It does not imply that raw compressed video is modeled without frame extraction.

| Model | Truly non-Qwen? | Size | Visual/video input | Audio | License | Local AMD/ROCm assessment |
|---|---|---:|---|---|---|---|
| **Gemma 4 12B Unified** | **Yes** | 11.95B | Direct video interface; frames, up to 60 at 1 FPS | Native, up to 30 s | Apache-2.0 | Best first alternative; BF16 weights are roughly 24 GB before runtime state, but ROCm is not claimed by Google |
| Gemma 4 E4B | Yes | 4.5B effective / 8B total | Frames | Native | Apache-2.0 | Smaller compute target, but newer runtime support still needs a smoke test |
| Phi-4-multimodal-instruct | Yes, Phi-4-mini backbone | 5.6B | Multi-image, up to 64 frames; no direct video-container input documented | Native; joint image+audio prompts supported | MIT | Weight size is easy; custom code, modality LoRAs, and CUDA-centric examples add integration risk |
| Ministral 3 8B Instruct | Yes, Mistral | 8.4B LM + 0.4B vision | Images; pass 12 frames as multi-image if the runtime supports the count | No | Apache-2.0 | Attractive portability: official GGUF variants include Q4/Q5/Q8/BF16; no video specialization |
| Pixtral 12B | Yes, Mistral | 12B + 0.4B vision | Interleaved/multiple images; use sampled frames | No | Apache-2.0 | Fits memory, but older and image-first; no reason to prioritize over Ministral 3 or Gemma 4 |
| Aria | Yes, custom MoE | 25.3B total / 3.9B active | Up to 256 frames; publisher reports LongVideoBench 65.3 | No | Apache-2.0 | Official BF16 recipe targets one 80 GB A100 and recommends FlashAttention/grouped GEMM; poor first ROCm choice |
| Molmo 2 4B/8B | **No: Qwen3 backbone** | about 5B/9B total | Direct video processor, frame sampling; video grounding/tracking | No | Apache-2.0, with training-data-use caveat | Interesting later; official examples use PyTorch and video utilities, but CUDA-centric serving remains unverified on ROCm |
| InternVL3.5-8B | **No: Qwen3 component** | about 9B total | Uniformly sampled frames / multi-image sequence | No | Apache-2.0 | Official stack uses custom code and commonly CUDA/FlashAttention; not an independent family test |
| VideoLLaMA3-7B | **No: Qwen2.5-7B backbone** | about 8B total | Direct video interface sampling frames (`fps`, `max_frames`) | No in VideoLLaMA3 | Apache-2.0 | Official install explicitly pins CUDA 11.8 and FlashAttention; weak ROCm fit |
| Llama 3.2 Vision 11B | Yes, Llama | 11B | Image input; sampled-frame workaround | No | Custom Llama 3.2 Community License | Fits memory but is old and image-first; low priority |
| Llama 4 Scout | Yes, Llama | 109B total / 17B active | Text and image, no audio | No | Custom Llama 4 Community License | BF16 checkpoint is far too large for a convenient first local experiment despite MoE activation sparsity |

### Microsoft Phi-4 multimodal

Phi-4-multimodal-instruct is a legitimate different-family option: Microsoft documents 5.6B parameters, a Phi-4-mini language backbone, text/image/audio input, 128K context, and an MIT license. Its processor supports multiple images—up to 64 frames with a resolution tradeoff—and can combine images and audio in one prompt. The model was trained with video data, but its public interface is multi-image rather than a video-container input and its official card gives no video QA benchmark. It is most useful as a cheaper second local alternative if Gemma 4 support is immature.

Source: [Microsoft Phi-4-multimodal-instruct official model card](https://huggingface.co/microsoft/Phi-4-multimodal-instruct).

### Mistral/Pixtral

The current compact Mistral option is Ministral 3 8B: an 8.4B Mistral language model plus a 0.4B vision encoder, Apache-2.0, with official GGUF variants from about 5.2 GB (Q4) to 17 GB (BF16). It is image-capable rather than video/audio-native, so the experiment would pass the same 12 chronological frames and Whisper text used for Qwen. That makes it a good backbone-control experiment, but it lacks evidence of video-specific training or official video scores.

Pixtral 12B is also Apache-2.0 and supports interleaved images/text, but it is a September 2024 image-first model. It should be below Ministral 3 8B in the queue.

Sources: [Ministral 3 8B official GGUF card](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-GGUF), [Pixtral 12B official card](https://huggingface.co/mistralai/Pixtral-12B-2409).

### Aria

Aria is genuinely independent and video-capable, with 25.3B total parameters but only 3.9B active per token. Rhymes reports LongVideoBench 65.3 and support for up to 256 frames. However, the official BF16 recipe targets an A100 80 GB and recommends FlashAttention and grouped GEMM. It has no audio input and is now an older model. It is scientifically interesting but not the simplest run on this AMD host.

Source: [Rhymes Aria official card](https://huggingface.co/rhymes-ai/Aria).

### Models whose branding hides a Qwen backbone

- **Molmo 2:** Ai2's official card says Molmo2-8B is based on Qwen3-8B plus SigLIP 2. It is a strong video specialist: Ai2 reports a 63.1 average over 15 video benchmarks for 8B and 62.8 for 4B, versus 59.5 for Qwen3-VL-8B in the same publisher evaluation. Those results make Molmo 2 worth a later video-specialist run, but it is not independent of Qwen. Its weights are Apache-2.0; the card separately warns that some training datasets are subject to academic/noncommercial-use restrictions, so commercial reuse requires dataset-level review. [Official Molmo2-8B card](https://huggingface.co/allenai/Molmo2-8B)
- **InternVL3.5:** the 8B official card explicitly says it uses pretrained Qwen3. It remains a useful vision stack, but comparing it with Qwen does not fully diversify the language backbone. [Official InternVL3.5-8B card](https://huggingface.co/OpenGVLab/InternVL3_5-8B)
- **VideoLLaMA3:** the official model zoo lists Qwen2.5-7B and Qwen2.5-1.5B as its base models. It handles videos through sampled frames, and the prescribed runtime is explicitly CUDA/FlashAttention-oriented. [Official VideoLLaMA3 repository](https://github.com/DAMO-NLP-SG/VideoLLaMA3)
- **MiniCPM-V/o:** MiniCPM-o 4.5 is attractive for native audio/video, but its official composition includes Qwen3-8B. It is a different multimodal system, not a different language-backbone family. [Official MiniCPM-o 4.5 card](https://huggingface.co/openbmb/MiniCPM-o-4_5)

## Recommended 100-example experiment ladder

Keep the exact same deterministic pilot examples and report official-style STS, BERTScore, METEOR, runtime, memory, and failures.

1. `Qwen3.5-4B` + the same 12 chronological frames, no transcript.
2. `Qwen3.5-4B` + the same 12 frames + the frozen Whisper transcript.
3. `Qwen3.5-4B` + Whisper + one controlled addition (for example, selective tiles or more frames).
4. Larger Qwen model with identical evidence and decoding.
5. **`google/gemma-4-12B-it`** using the same 12 frames + the same Whisper transcript. This isolates model-family quality.
6. Separately run Gemma 4 12B with its video/audio interface. This compares our explicit evidence pipeline against Google's native multimodal path, but is not a pure backbone comparison.
7. If authorized, run **Gemini 3.1 Pro Preview** on the original video+audio as a cloud ceiling.

Do not compare Qwen+12 frames against Gemma/Gemini+full video and call the score difference a model difference. Record those as two separate experiments: **fixed evidence** and **best supported input path**.

## AMD/ROCm execution caution

The host's 122 GiB unified memory makes 6B–12B BF16 weight capacity plausible. Compatibility, not capacity, is the main uncertainty:

- None of the cited local model cards certifies Radeon 8060S / `gfx1151` ROCm inference.
- Start with a rootless Podman ROCm smoke test on one short clip and one question before downloading multiple large checkpoints or running the 100-example pilot.
- Prefer plain Transformers eager/SDPA before CUDA-specific FlashAttention, bitsandbytes, or vLLM paths.
- Gemma 4 is very new and requires a recent Transformers multimodal class; pin the exact working image and model revision.
- Ministral's official GGUF files offer the safest CPU/portable fallback. Phi-4's custom code and modality LoRAs are a larger integration risk. Aria and VideoLLaMA3 are the least attractive first attempts because their official recipes are explicitly NVIDIA-oriented.

## Answer to “are Qwen models the best?”

Not yet knowable. Qwen has three practical advantages: strong publisher video benchmarks, mature video preprocessing, and multiple usable sizes. But Gemma 4 12B is new enough that its official card does not publish a comparable video table, and none of these sources reports TRECVID results. The fixed 100-example pilot is the evidence needed. The best current experiment design is therefore **Qwen3.5-4B as the economical baseline, Gemma 4 12B as the genuinely independent local challenger, and Gemini 3.1 Pro as an optional authorized cloud ceiling**.
