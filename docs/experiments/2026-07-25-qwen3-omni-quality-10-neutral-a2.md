# Qwen3-Omni quality-10 neutral a2 pre-run record

## Scope

This is local scaffolding for a paired rerun, not a submitted or previewed
workload. The frozen a1 configuration, scripts, artifacts, and report remain
unchanged.

- Run ID: `qwen3-omni-quality-10-neutral-20260725-a2`
- Private prefix:
  `hf://buckets/Npatta01/trec_mlm_2026/trec_video_2026/experiments/qwen3-omni-quality-10-neutral-20260725-a2`
- New worst-case spend: `$4.50`; cumulative worst case: `$9.0878`, below the
  approved `$10` cap.

## Frozen controls

The a2 task maps the same label-free contract, 120 uniform frames, ten WAV
files, and three oversized-frame transport chunks as a1. It retains the same
model revision, pinned runtime image, seeded decoding, EOS/padding checks,
first-pair canary, private synchronization, atomic local collector, one-hour
duration, `$4.50/hour` cap, GPU envelope, and on-demand providers.

The only request-content difference between the paired conditions is the
native audio element and its resulting tensors. Both use the exact same
neutral instruction:

> Answer the question using the provided evidence. Return only a JSON object
> with non-empty string fields "answer" and "evidence". The evidence field must
> briefly state the cues supporting the answer.

The instruction does not mention audio, modalities, or a requirement for
auditory evidence.

## Authorized next steps (not performed)

After the usual local boundary checks and an approved preview with at least two
eligible on-demand offers when practical, submit exactly once without sending
the repository archive:

```bash
dstack apply --no-repo \
  -f dstack/qwen3-omni-quality-10-neutral.dstack.yml -y -d
```

After a terminal result, collect only the immutable private prefix:

```bash
bash scripts/collect-qwen3-omni-quality-10-neutral-dstack.sh \
  qwen3-omni-quality-10-neutral-20260725-a2
```

No dstack submission, Hugging Face synchronization, preview, or collection was
performed while creating this record.
