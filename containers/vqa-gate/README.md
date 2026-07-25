# VQA gate container lineage

`Containerfile` is the reproducible base definition. It starts from a public,
digest-pinned ROCm/PyTorch image and pins the Python and NLTK dependencies used
by the original compatibility gate.

The model-specific Containerfiles are historical replay definitions. They pin
the exact host-local base image used by the recorded experiments:

```text
localhost/trec-vqa-gate-rocm@sha256:510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5
```

Before archiving the original experiment host, preserve that image outside any
Git worktree if exact replay is required:

```bash
podman image exists localhost/trec-vqa-gate-rocm@sha256:510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5
podman save --format oci-archive \
  -o trec-vqa-gate-rocm-510e4ca.oci.tar \
  localhost/trec-vqa-gate-rocm@sha256:510e4ca0a165b60a50abf3a1f24545b8f4bd561f9643f0295e7d82212fe801d5
```

The archive can later be restored with `podman load -i`. Keep it private; do
not commit it to this repository.

If the exact image is unavailable, rebuild `Containerfile` from its public
base and treat the result as a new runtime. Update the child `FROM` pin, rebuild
the child image, rerun the three-video compatibility gate, and record the new
image digest with the predictions. A rebuilt image must not be represented as
the historical runtime merely because it used the same Containerfile.
