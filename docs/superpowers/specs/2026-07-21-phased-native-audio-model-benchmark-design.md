# Phased Native-Audio Model Benchmark Design

## Objective

Select a practical open model and evidence recipe for TRECVID VQA by running a
paired, reproducible benchmark on labeled 2025 training examples. The benchmark
must distinguish four questions that the earlier three-video gate could not:

1. Does additional temporal sampling improve a small visual model?
2. Does a stronger or different visual family improve answer quality?
3. Does a better transcript improve speech-dependent answers?
4. Does native audio improve speech and sound questions when the visual input,
   prompt, decoding, and model family are held fixed?

The benchmark is a model-selection experiment, not a statistically precise
leaderboard. It uses a ten-video compatibility gate before promoting viable
conditions to a frozen 30-video panel. It does not use 2025 test answers, train
models, or launch the eventual full-dataset inference run.

## Dataset audit and frozen sample

The README's stated 500 annotations is not the usable local count. The audited
inventory contains 492 logical CSV records, 481 individual labeled JSON files,
and 475 downloaded media files. There are 467 IDs common to all three sources.
After excluding the original gate videos `BDKkVYmWqdM`, `O60MtMuSpf8`, and
`Fk_j6re_Uiw`, the conservative eligible pool is 464 examples.

Freeze the following 30 examples as manifest version
`trecvid-native-audio-2026-07-21-v1`. Strata are assigned from question wording
before inference, and labels are available only to evaluation.

| Stratum | Six frozen IDs | Ten-video gate IDs |
|---|---|---|
| Dialogue and speech | `LKiaCf1QZFw`, `Ie-KCtcZ7uc`, `J1CS47pBX_M`, `Mi-NpnCBegw`, `NN7G28sBOlY`, `NUmb2P3Bzu8` | `Mi-NpnCBegw`, `NN7G28sBOlY` |
| Non-speech sound and music | `IRTTtnSRTyY`, `IWSPAfCKtV8`, `Jc_vZjKwtvg`, `MdMKHhFFqJQ`, `NkXPPJoZ-h8`, `OzNdzgQi1CA` | `Jc_vZjKwtvg`, `NkXPPJoZ-h8` |
| Temporal and action | `CH8Dp7EgW4I`, `EtNOkUJd5w8`, `Gp4LVTn3YbY`, `KgOvCPOX-Ms`, `PcDuz7j9LL8`, `C4j3Pp79KSg` | `EtNOkUJd5w8`, `PcDuz7j9LL8` |
| Spatial, OCR, and color | `Hw0bRvn1ZIo`, `J-sG9vePDmI`, `Jc2BdtyAF48`, `HyOKrz1C4zU`, `Msbpqv5aOxk`, `JOClNGH6yL8` | `Jc2BdtyAF48`, `HyOKrz1C4zU` |
| Visual controls | `EMI_anpHII4`, `IGL7j9-JRG4`, `JSOh_pKFq84`, `QFljXc4WpFA`, `PiIrinu42DQ`, `LusRurRrmf0` | `EMI_anpHII4`, `IGL7j9-JRG4` |

The ten-video subset is selected deterministically by hashing manifest version,
the compatibility marker, stratum, and video ID, selecting two per stratum.
The dialogue selection additionally retains one English and one non-English
metadata-language example. All 30 media files passed `ffprobe`, contain audio,
and total 587 seconds (median 17 seconds, range 9–30 seconds). One temporal gate
clip has six-channel AAC audio and intentionally tests decoder robustness.

## Experimental funnel

The funnel prevents large downloads and long runs from obscuring basic
compatibility failures.

### Phase 0: repair and one-video probes

- Diagnose the cached MiniCPM-o 4.5 audio tensor mismatch. Compare the current
  hand-interleaved evidence path against the model's processor-native video
  path, and test its documented same-content audio merge behavior.
- Add a true audio-off/audio-on switch with identical selected frames and
  decoding.
- Probe every newly integrated model on one representative example before it
  can enter the ten-video gate.
- Probe heavyweight or uncertain ROCm paths on one example only unless they
  load, complete, and remain within practical memory and latency bounds.

### Phase 1: frozen ten-video gate

Run every technically viable condition on the same ten examples and exact
question prompts. This phase detects failures and directional signal; its small
sample is not presented as an accuracy estimate.

### Phase 2: frozen 30-video benchmark

Promote conditions that complete reliably, lead the primary metrics, provide a
unique correct-answer pattern, or demonstrate useful paired audio benefit.
Run promoted conditions on all 30 frozen examples and report per-stratum as
well as aggregate results.

## Candidate conditions

### Controlled visual baselines

- `Qwen/Qwen3.5-4B`, 12 chronological bin-midpoint frames, no audio text.
- The same Qwen3.5-4B revision with 24 frames. This directly measures whether
  more temporal samples help.
- `Qwen/Qwen3.5-9B`, 12 frames. This isolates the approximate capacity effect
  within the family.
- Existing distinct visual-family candidates on the ten-video gate:
  `OpenGVLab/InternVL3_5-4B`, `google/gemma-4-12b-it`, and
  `allenai/Molmo2-4B`. Their processor-specific tiling or frame handling is
  recorded, so cross-family results are practical comparisons rather than
  claims of a perfectly controlled architecture ablation.

The 12-versus-24 comparison addresses temporal coverage. Spatial detail is a
separate issue: InternVL-style tiling preserves small text and objects by also
encoding higher-resolution image tiles. It does not mean sampling four times
as many moments; it means examining selected moments at more spatial detail.

### Transcript evidence

Generate transcript evidence independently of the reference answer and test it
with a frozen downstream visual model/prompt. Candidates are:

- Whisper large-v3 translation, retained as the known baseline.
- SeamlessM4T-v2 translation, retained despite the earlier partial transcript
  so the broader panel can determine whether that clip was atypical.
- `mistralai/Voxtral-Mini-3B-2507` for multilingual audio transcription,
  translation, and semantic audio understanding.
- `Qwen/Qwen3-ASR-1.7B` as a small alternative transcription family, with a
  separate deterministic translation step when its output is not English.

There are no gold transcripts, so ASR is judged by downstream VQA changes and
a bounded semantic audit of transcript evidence, not word error rate. The hard
Hindi/Urdu clip from the earlier gate remains a qualitative diagnostic but is
not part of the frozen 30.

### Native audio and video

Every native multimodal candidate receives a matched within-family ablation:
same frames or video, same question, same revision and decoding, with audio
disabled versus enabled. This is the only comparison used to claim an audio
effect.

Local order:

1. `openbmb/MiniCPM-o-4_5`, already cached. Repair and run matched audio off/on.
2. `Qwen/Qwen2.5-Omni-3B`, the most practical additional native-AV checkpoint;
   use its documented `use_audio_in_video` switch and disable speech output.
3. `Qwen/Qwen2.5-Omni-7B`, first as a compatibility probe and then as a
   capacity comparison if runtime is practical or the 3B condition is
   promising.
4. `baichuan-inc/Baichuan-Omni-1d5`, at most a one-to-three-video family
   diversity probe because of its custom runtime and additional license terms.

`Qwen/Qwen3-Omni-30B-A3B-Thinking` is recorded as a larger-machine follow-up,
not a required local 30-video condition. Its official 15-second BF16 minimum is
68.74 GB before runtime headroom. The text-oriented Thinking checkpoint is more
appropriate than carrying the Instruct model's speech-output stack.

Official implementation and memory references are the
[MiniCPM-o 4.5 model card](https://huggingface.co/openbmb/MiniCPM-o-4_5),
[Qwen2.5-Omni 3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B),
[Qwen2.5-Omni 7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B), and
[Qwen3-Omni Thinking](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking).

## Promotion and stopping rules

A condition advances from ten to 30 when all applicable rules hold:

- At least 90% of examples reach a terminal, scoreable result; failures are
  understood and do not indicate silent evidence loss or CPU fallback.
- It is among the leading conditions on mean token F1, normalized exact match,
  and METEOR, or it provides correct answers that the leaders systematically
  miss.
- A native-audio condition shows a positive paired change on the four audio
  gate examples without a material regression on visual controls. Audio benefit
  is reported as paired wins, losses, and metric delta rather than only an
  aggregate score.
- Its measured latency and peak memory make a 30-example run practical. A
  useful but slow condition may be designated larger-machine-only instead.

The gate can also promote a condition for diagnostic value, but the reason is
recorded before the 30-video run. Failed or redundant conditions stop early.
No unbounded Cartesian product of models, frame counts, transcripts, and
prompts is run.

## Metrics and interpretation

For each answer, calculate normalized exact match, token F1, and METEOR using
the existing evaluation code and pinned NIST parity path. BERTScore may be
reported as a secondary semantic diagnostic when its pinned runtime is
available. Report completion rate, latency, peak accelerator memory, and
failure type beside answer quality.

For 30 examples, report:

- aggregate and per-stratum means;
- audio-on minus audio-off paired deltas;
- paired correct/incorrect wins and losses;
- bootstrap uncertainty intervals, explicitly labeled exploratory because 30
  examples remain small;
- per-example predictions and references so metric artifacts remain visible.

No single metric is treated as ground truth. Short-answer paraphrases make
exact match conservative, while semantic metrics can reward unsupported
answers. Final selection considers metric agreement and a label-blind evidence
audit of disagreements.

## Reproducibility and leakage controls

- Freeze manifest, immutable model revision, runtime image digest, prompt,
  frame policy, decoding, and audio switch before each gate starts.
- Never include the correct answer, incorrect alternatives, or scoring fields
  in a runner request.
- Store media-derived evidence, transcripts, predictions, and raw logs only
  under ignored `data/vqa/benchmarks/` paths.
- Keep append-only attempts and resumable successful outputs. Do not overwrite
  a success with a later failure.
- Record source fingerprints, evidence checksums, dependency versions, device,
  precision, elapsed time, and peak memory.
- Do not inspect or use 2025 test labels.

## Runtime policy

The host has 122 GiB unified memory and sufficient disk, but ROCm portability
is not assumed from CUDA examples. Use rootless Podman, PyTorch SDPA where
possible, immutable Hub revisions, and digest-pinned images. A model that
requires CUDA-only packaged kernels is reported unsupported rather than
silently moved to CPU. Downloads are retained; caches are not pruned.

The benchmark runs serially to avoid unified-memory contention. Each condition
has a timeout and produces progress evidence frequently enough to distinguish a
long generation from a hang. Expensive downloads and runs occur only after the
cheapest representative preflight passes.

## Verification and independent review

Implementation is test-first. Before selecting a winner:

1. Validate the frozen manifest against annotations, media, streams, and the
   shared-data boundary.
2. Unit-test sample determinism, label separation, audio toggling, evidence
   association, resumption, and paired metric calculations.
3. Run one-video probes, the ten-video gate, and promoted 30-video conditions.
4. Regenerate all scores and reports from saved predictions.
5. Ask a fresh advisor to review code and findings for leakage, mismatched
   evidence, unfair comparisons, scoring defects, ROCm fallback, and
   overclaiming.
6. Address findings and rerun affected checks before handoff.

## Deliverable

The final report recommends:

- a visual-only base model and frame/spatial policy;
- the best transcript path, if transcript evidence helps;
- the best native-audio family, if its matched ablation helps;
- expected local latency and memory for 10 and 30 examples;
- the small shortlist and configuration to move to a larger machine.
