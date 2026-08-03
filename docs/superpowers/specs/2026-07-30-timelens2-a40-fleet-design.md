# TimeLens2 A40-Class Fleet Design

## Objective

Unblock the TimeLens2 CUDA compatibility gate by allowing the shared
`gpu-marketplace` fleet to provision up to four concurrent A40-class or newer
professional/data-center GPUs, while requiring at least 24 GB of VRAM and
retaining the existing task-level price and duration caps.

## Approved resource policy

- Fleet size: `0..4`; zero nodes are prepaid or kept alive as a minimum.
- GPU pool: `A40`, `A6000`, `L40`, `L40S`, `RTX6000Ada`, `A100`, `H100`, and
  `H200`.
- Per-GPU VRAM: at least 24 GB.
- Providers: RunPod and Vast.ai.
- Provisioning: on-demand only.
- Fleet idle timeout: five minutes.
- Canary cap: `$0.55/hour` for at most 15 minutes.
- Compatibility cap: `$0.65/hour` for at most one hour.
- The task price caps remain authoritative; expensive allowed GPU types are not
  eligible above those caps.

## Immutable corrective attempt

Use `timelens2-runner-canary-20260730-a2` and
`timelens2-selector-pilot-20260730-a2`. The `-a1` planner attempts remain
historical and are not overwritten or submitted.

## Execution boundary

1. Test the fleet and task configuration contracts before changing them.
2. Preview and apply the fleet update.
3. Require multiple physically distinct compatible offers when practical.
4. Preview the exact-image canary and submit it once, detached.
5. Monitor the foreground workload and verify its Hugging Face Bucket artifacts
   by listing, downloading, and checking `SHA256SUMS`.
6. Run the private three-video compatibility gate only if the canary passes.
7. Do not run the focused selector comparison: its authorized oracle remains
   absent.

## Failure and privacy policy

- Stop after one corrective technical attempt if an infrastructure or runtime
  gate remains unavailable.
- Do not raise price caps without new authorization.
- Keep `HF_TOKEN` in dstack secrets and never use diagnostic output on
  secret-bearing runs.
- Keep the label-free request manifest and all private results under the ignored
  `.timelens2-private/` boundary locally and immutable run prefixes remotely.
- Preserve incomplete evidence and its three-row completion denominator.
