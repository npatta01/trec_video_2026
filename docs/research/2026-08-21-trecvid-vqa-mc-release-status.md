# TRECVID 2026 VQA multiple-choice release status

**Snapshot.** Checked 21 August 2026, 05:51 America/New_York (the NIST
responses carried `Date: Fri, 21 Aug 2026 09:51:45 GMT`). Sources below are
official NIST/TRECVID pages or NIST-hosted data files.

## Answers

1. **Same videos: yes.** The TRECVID 2026 VQA guidelines say explicitly:
   “Both AG and MC tasks will share the same dataset of testing videos.” The
   tasks have different submission deadlines, but the video collection is
   shared.

2. **Question identity: not fully specified for 2026.** The published 2026
   schemas make MC a question record with answer options: AG topics contain
   `Q_ID`, `Video_ID`, and `Question`; MC topics contain those same fields plus
   four `Options`. The page's examples use the same example question/video
   records and add options, so **“same question records plus options” is the
   best-supported interpretation**. However, NIST only guarantees the shared
   *videos* in prose; it does not state that the 2026 AG and MC question text
   or Q_ID set is byte-identical. That exact claim remains unverified until
   the MC file can be read.

   As a historical format check (not proof about the unreleased 2026 file),
   NIST's public 2025 AG and MC files show the pattern directly: the first AG
   rows contain `Q_ID, Video_ID, Question`, while the corresponding MC rows
   contain those fields followed by four choices. The 2025 readme describes
   the MC file as `QueryID, youTube video ID, question, choice 1, ... choice
   4`.

3. **The Aug 21 date is officially scheduled, but public availability is not
   verified.** NIST's tentative schedule says “21 Aug — VQA: Testing data
   available for multiple choice (MC) task.” At the snapshot time:

   - the public 2026 VQA page listed the training data and the **2025** test
     data, but no live 2026 MC download;
   - the public 2026 data page said “Testing dataset (task 1) is now
     available” (AG) and gave no task-2/MC download;
   - the VQA page's HTML contains an MC URL only inside a commented-out block:
     `https://trec.nist.gov/act_part/trecvid.active/VQA/testing.data/vqa.testing.dataset.task2.MC.csv`;
   - an unauthenticated `HEAD` request to that official endpoint returned
     `401 Unauthorized` with `WWW-Authenticate: Basic realm="trec2026"`.

   Therefore, the defensible status is **scheduled for today, but not
   publicly accessible/confirmed from the public NIST pages**. It may be
   available to an authenticated active participant, but that cannot be
   distinguished from “not yet posted” without authorized credentials. The
   direct endpoint above should not be treated as a confirmed 2026 file: in
   the public 2026 markup it is commented out and follows the 2025-era path.

## Official sources

- [TRECVID 2026 VQA guidelines (NIST)](https://www-nlpir.nist.gov/projects/tv2026/vqa.html)
- [TRECVID 2026 schedule (NIST)](https://www-nlpir.nist.gov/projects/tv2026/schedule.html)
- [TRECVID 2026 data/distribution page (NIST)](https://www-nlpir.nist.gov/projects/tv2026/data.html)
- [Protected active-participant MC endpoint shown in the page markup](https://trec.nist.gov/act_part/trecvid.active/VQA/testing.data/vqa.testing.dataset.task2.MC.csv)
- [NIST's public 2025 AG file (format precedent)](https://www-nlpir.nist.gov/projects/tv2025/pastdata/vqa/testing.dataset.vqa.2025.csv)
- [NIST's public 2025 MC file (format precedent)](https://www-nlpir.nist.gov/projects/tv2025/pastdata/vqa/vqa.testing.dataset.task2.MC.csv)
- [NIST's 2025 MC readme](https://www-nlpir.nist.gov/projects/tv2025/pastdata/vqa/testing.data.task2.readme.txt)
