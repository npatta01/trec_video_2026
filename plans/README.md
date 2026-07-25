# GitHub-issue execution-plan archive

These five plans document merged GitHub-issue work. They are preserved as
historical execution records, not live task queues. Current decisions belong
in [Project Status](../docs/project-status/index.md).

| Plan | Merged/completed outcome | Current decision |
| --- | --- | --- |
| [Issue #5: Qwen3-Omni dstack compatibility](issue-5-qwen3-omni-dstack.md) | [Issue #5](https://github.com/npatta01/trec_video_2026/issues/5) and [PR #14](https://github.com/npatta01/trec_video_2026/pull/14) merged; followed by [Issue #17](https://github.com/npatta01/trec_video_2026/issues/17), [PR #18](https://github.com/npatta01/trec_video_2026/pull/18), and [PR #20](https://github.com/npatta01/trec_video_2026/pull/20). | Do not promote from compatibility alone; the later strict quality run remains ineligible. |
| [Issue #6: non-Qwen omni compatibility](issue-6-nonqwen-omni.md) | [Issue #6](https://github.com/npatta01/trec_video_2026/issues/6) and [PR #11](https://github.com/npatta01/trec_video_2026/pull/11) merged; later corrected in [Issue #12](https://github.com/npatta01/trec_video_2026/issues/12) / [PR #15](https://github.com/npatta01/trec_video_2026/pull/15). | Semantic-quality evidence stopped both Gemma and Phi. |
| [Issue #7: automatic-transcript ablation](issue-7-automatic-transcript-ablation.md) | [Issue #7](https://github.com/npatta01/trec_video_2026/issues/7) and [PR #9](https://github.com/npatta01/trec_video_2026/pull/9) merged. | Route Whisper; do not use it globally. The strict human-oracle tier is deferred. |
| [Issue #12: omni semantic quality correction](issue-12-omni-quality.md) | [Issue #12](https://github.com/npatta01/trec_video_2026/issues/12) and [PR #15](https://github.com/npatta01/trec_video_2026/pull/15) merged. | Both Gemma and Phi fail the frozen semantic gate; no expansion. |
| [Issue #17: Qwen3-Omni quality-10](issue-17-qwen3-omni-quality-10.md) | [Issue #17](https://github.com/npatta01/trec_video_2026/issues/17), [PR #18](https://github.com/npatta01/trec_video_2026/pull/18), and [PR #20](https://github.com/npatta01/trec_video_2026/pull/20) merged. | Strict output-contract failure makes Qwen3-Omni ineligible for promotion. |
