# Task 4 report: repository YouTube dataset skill

## Outcome

Created `.agents/skills/downloading-youtube-datasets` with the required generated UI metadata and a concise workflow-only `SKILL.md`. The skill always delegates to this repository's `youtube-dataset-download` and explicitly forbids raw `yt-dlp`, `vqa-download`, or custom downloader implementations. It contains no scripts, references, or assets.

## Baseline coverage

The skill addresses private mode-`0600` credentials, Git-ignore/tracking checks, API-first preparation in batches of at most 50 with an atomic snapshot and fail-closed behavior, independent neutral-HTTPS and YouTube proxy preflight, a single detached worker with one-video-at-a-time pacing, archive/partial/no-overwrite resume, 429/bot-check one-/two-hour cooldowns, prompt stop and notification after repeated proxy 502 failures, PID/progress/cooldown monitoring, and categorized final reporting with an exact resume command.

## Validation

- `quick_validate.py .agents/skills/downloading-youtube-datasets`: **passed** (`Skill is valid!`).
- `wc -w .agents/skills/downloading-youtube-datasets/SKILL.md`: **377 words** (under 500).
- Metadata consistency assertions: **passed**. Name matches the directory; description starts with `Use when` and contains triggers only; display name is `Download YouTube Datasets`; short description is 42 characters; default prompt is one sentence and explicitly contains `$downloading-youtube-datasets`.
- Resource check: **passed**. Only `SKILL.md` and `agents/openai.yaml` exist.
- Fresh-agent forward scenario: **passed all 10 gates**. The agent used only the generic CLI, checked credential/Git safety, prepared through the API, required both proxy preflights, started one detached worker, delegated pacing/resume/cooldowns, stopped/notified on repeated 502s, monitored state, and produced categorized reporting plus an exact resumable command.

## Concern

The current generic CLI's `status` output reports running/stopped state, completion/media/pending/API-omitted counts, and the log path, but does not yet print the approved design's last-progress time, cooldown deadline, PID, or categorized failure inventory. The skill requires those fields and log-based monitoring as designed; the CLI implementation should be brought into alignment in a separate implementation task rather than worked around in the skill.
