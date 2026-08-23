# TRECVID 2026 VQA multiple-choice release status

**Snapshot.** Checked 21 August 2026, 05:51 America/New_York (the NIST
responses carried `Date: Fri, 21 Aug 2026 09:51:45 GMT`). Sources below are
official NIST/TRECVID pages or NIST-hosted data files.

## Resolution update — 23 August 2026

The official 2026 Task 1 and Task 2 files are now available in this repository
under `datasets/trecvid-2026/vqa/`.

Direct comparison confirms:

- both files contain 2,443 rows;
- `Q_ID` values and row order are identical;
- `Video_ID` is identical for every row;
- 2,423 question strings are byte-identical;
- the remaining 20 differ only by leading or trailing whitespace; and
- Task 2 adds the released options, with 4 to 11 options per question.

The Multiple-Choice Evalbase submission window opens **11 September 2026** and
closes **18 September 2026**.

The signed-in Evalbase task page was checked on 23 August and correctly showed
“Multiple Choice Task (submissions not open).”

## Answers

1. **Same videos: yes.** The TRECVID 2026 VQA guidelines say explicitly:
   “Both AG and MC tasks will share the same dataset of testing videos.” The
   tasks have different submission deadlines, but the video collection is
   shared.

2. **Question identity: verified.** Direct comparison of the released 2026
   files confirms identical `Q_ID`, order, and `Video_ID` for all 2,443 rows.
   The question text is byte-identical for 2,423 rows; the other 20 differ only
   by leading or trailing whitespace. Task 2 adds the released `Options`.

3. **At the original Aug. 21 snapshot, public availability was not yet
   verified.** NIST's tentative schedule said “21 Aug — VQA: Testing data
   available for multiple choice (MC) task.” At 05:51 America/New_York that
   morning:

   - the public 2026 VQA page listed the training data and the **2025** test
     data, but no live 2026 MC download;
   - the public 2026 data page said “Testing dataset (task 1) is now
     available” (AG) and gave no task-2/MC download;
   - the VQA page's HTML contains an MC URL only inside a commented-out block:
     `https://trec.nist.gov/act_part/trecvid.active/VQA/testing.data/vqa.testing.dataset.task2.MC.csv`;
   - an unauthenticated `HEAD` request to that official endpoint returned
     `401 Unauthorized` with `WWW-Authenticate: Basic realm="trec2026"`.

   That was a time-bounded observation, not the final release state. The Aug.
   23 comparison above confirms that the released files were subsequently
   obtained and validated.

## Official sources

- [TRECVID 2026 VQA guidelines (NIST)](https://www-nlpir.nist.gov/projects/tv2026/vqa.html)
- [TRECVID 2026 schedule (NIST)](https://www-nlpir.nist.gov/projects/tv2026/schedule.html)
- [TRECVID 2026 data/distribution page (NIST)](https://www-nlpir.nist.gov/projects/tv2026/data.html)
- [Protected active-participant MC endpoint shown in the page markup](https://trec.nist.gov/act_part/trecvid.active/VQA/testing.data/vqa.testing.dataset.task2.MC.csv)
- [NIST's public 2025 AG file (format precedent)](https://www-nlpir.nist.gov/projects/tv2025/pastdata/vqa/testing.dataset.vqa.2025.csv)
- [NIST's public 2025 MC file (format precedent)](https://www-nlpir.nist.gov/projects/tv2025/pastdata/vqa/vqa.testing.dataset.task2.MC.csv)
- [NIST's 2025 MC readme](https://www-nlpir.nist.gov/projects/tv2025/pastdata/vqa/testing.data.task2.readme.txt)
