# dstack Backend Recovery Design

## Objective

Reconstruct the deleted local dstack backend configuration for RunPod, Vast.ai,
DigitalOcean, and GCP while retaining the existing `default-byoc` dstack Sky
client project. Keep provider credentials out of Git, shell history, command
output, and chat.

## Verified State

- `~/.dstack/config.yml` exists and points the CLI at the `default-byoc` project
  on `https://sky.dstack.ai`.
- `~/.dstack/server/config.yml` previously existed but has been deleted.
- No recoverable trash copy, editor backup, open deleted inode, or configured
  backend entry remains.
- RunPod, Vast.ai, DigitalOcean, and GCP are supported dstack backends.
- GCP will authenticate with a dedicated service account.

## Local Credential Configuration

Create `~/.dstack/server/config.yml` with mode `0600`. The file will define one
project and four backends:

- RunPod with a Read & Write API key.
- Vast.ai with an API key.
- DigitalOcean with a scoped API token and optional project name.
- GCP with a service-account JSON credential and explicit project and region
  selection.

The initial file will contain unambiguous placeholders rather than credentials.
The user will replace those placeholders directly on the host. Commands and
verification output must report only whether each required value is present;
they must never print credential values.

The existing `~/.dstack/config.yml` will not be overwritten. No credential file
will be created inside the repository.

## Sky and Local-Server Boundary

`~/.dstack/server/config.yml` is consumed by a dstack server process, whereas
`~/.dstack/config.yml` selects the server used by the CLI. The implementation
will first reconstruct and validate the local server configuration. It will
then determine, from actual dstack behavior, whether the prior workflow used a
local server alongside Sky or imported backend definitions into the
`default-byoc` Sky project. It will not claim that Sky can read a local file
directly.

If the credentials must be uploaded to Sky for provisioning, that external
credential-storage consequence will be reported before any upload. The user
must authorize that step separately.

## Workload Configuration

After backend validation, add repository-local dstack configurations that
contain no secrets:

- A GPU fleet restricted to RunPod and Vast.ai.
- A CPU fleet restricted to DigitalOcean and GCP.
- Both fleets use `nodes: 0..1`, a short idle timeout, and no provisioned
  minimum. The CPU fleet has an explicit hourly limit; GPU price limits belong
  on each consuming task so workloads can choose an appropriate accelerator
  without silently inheriting an unsuitable fleet-wide ceiling.
- A CPU network test checks one public YouTube URL without downloading media.
- The full downloader remains a separate, explicitly authorized task.

Creating a zero-minimum fleet template must not launch an instance. All run
plans will be previewed first, including their unmodified offer tables and cost
estimates. No billable apply will run without explicit approval.

## Validation

1. Validate YAML syntax and restrictive file permissions locally.
2. Confirm all four credential placeholders have been replaced without
   displaying their values.
3. Start or validate the appropriate dstack control-plane path.
4. Query offers independently for RunPod, Vast.ai, DigitalOcean, and GCP.
5. Preview fleet and test-task plans without accepting them.
6. Confirm the repository contains no provider keys or GCP service-account
   material.

## Failure Handling

- Invalid credentials: report the backend and provider error without exposing
  the credential.
- No offers: distinguish missing backend configuration, quota limitations,
  region filters, resource mismatch, and provider capacity.
- GCP permission failure: identify the missing permission or API and stop;
  do not broaden the service-account role indiscriminately.
- YouTube rejection: terminate the CPU test instance and report that the
  datacenter IP is unsuitable. Do not launch the full downloader.
- Any unexpected charge-bearing plan: stop before confirmation.
