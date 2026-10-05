# TRECVID 2026 paper draft: canonical working record

## Objective and scope

Create a private, reviewable notebook-paper draft describing NP Labs' actual submitted AG and revised-option MC approach. The editable paper is `trecvid2026-draft.tex`, a standalone manuscript, not a generated sibling of an existing QMD report. Existing research reports and submission files remain unchanged.

## Decisions

- Apply the personal `technical-writing` paper profile.
- Use the 2026 organizer instructions: US Letter, fonts at least 9 pt, and a structured abstract separately covering each task's training data, submitted runs, differences, component contributions, and lessons.
- Use 10 pt two-column article formatting; no mandatory LaTeX class is specified on the organizer page. The 2025 HLTCOE VQA paper informs layout and section organization, not our claims or results.
- Lead with all eight final submitted runs. Exclude the superseded first-release MC results from final-run tables.
- Use the final submission ledger for upload status. The Gemini 3.8 sanitized summary is a historical pre-upload snapshot and does not supersede the later ledger's acceptance record.
- Keep measured 2025 development quality separate from 2026 submission mechanics and pending official effectiveness.
- Preserve the distinction between the recommended experimental speech router and the fixed transcript-first submitted Qwen dual view.
- Do not infer Qwen native audio support from its prompt's mention of audio; the inspected request sends a video payload.
- Use author placeholders; user was asked for names and affiliations.
- Do not publish or submit the paper.

## Source map

| Paper content | Evidence |
|---|---|
| Eight runs and acceptance | `docs/submissions/README.md` and linked run records |
| 12-frame baseline, 1,956 videos, answer-slot sources | `docs/submissions/2026-q35-4b-u12.md` |
| Qwen dual view | `docs/submissions/2026-q36-dualview.md`; `scripts/run_vqa_deadline_qwen36_full.py` |
| Gemini fusion | `docs/submissions/2026-g37-q36-fusion.md`; `scripts/run_vqa_deadline_gemini_full.py` |
| Grouped rewrite, repairs, retained rank one, redundancy | `docs/submissions/2026-g37-grouped-unique.md` |
| Final MC coverage, options, disagreement, costs | `docs/submissions/2026-vqa-mc-task2-v2-final-candidates.md` |
| MC audit queue decision | `scripts/run_vqa_mc_text_audit_full.py`, `organize_audit` |
| Direct MC prompt | `scripts/run_vqa_mc_gemini_direct_full.py`, `_prompt` |
| Gemini 3.8 completion and costs | `docs/experiments/artifacts/vqa-mc-gemini38-static-20260913/summary.json`; later submission record for upload state |
| Development table and intervals | `reports/trecvid_vqa_2026/native_audio_model_benchmark_results_2026-07-21.md` |

Inspected repository revision: `03678ecf1829be3d265cc4bba43dfa968b8594f3`.

## External format and literature sources

- [2026 notebook instructions](https://www-nlpir.nist.gov/projects/tv2026/notebookpapers.html)
- [2026 VQA task and evaluation guidelines](https://www-nlpir.nist.gov/projects/tv2026/vqa.html)
- [2025 HLTCOE example](https://trec.nist.gov/pubs/trec34/papers/HLTCOE.vqa.pdf)
- [NIST citation guidance](https://trecvid.nist.gov/trecvid.citation.html)
- [General TRECVID citation guidance linked by organizers](https://www-nlpir.nist.gov/projects/trecvid/trecvid.citation.html)

## Current status and verification

Draft authored (approximately 3,400 whitespace-delimited source words, including LaTeX markup). Opening in the native editor was requested and returned `queued`; no other editor was opened.

Verification completed:

- All eight archived JSON files parse, contain 2,443 unique question IDs, and have consecutive ranks.
- All four AG files have ten answers per row; all four revised MC files have 472 four-option rows and 1,971 eight-option rows.
- Baseline contains 1,956 unique video identities.
- Every grouped rank-one answer exactly matches its fusion source.
- MC Top-1/full-ranking disagreement counts reproduce 229/654 for targeted and 363/1,976 for direct against text-only.
- Paper count arithmetic reconciles; citation keys resolve; source has no external TeX inputs, media dependencies, local host paths, or obvious credential patterns.
- `git diff --check` passed. No existing report, code, submission, or dataset files were edited.
- Native compilation was requested once and retried three times, with brief waits. Every attempt returned `busy` (another LaTeX document compiling). Compilation and rendered-layout inspection therefore remain **unverified**. The source is preserved, and the queued editor request was not closed.

## Next action

Retry native compilation when the compiler becomes available and inspect the resulting layout. Incorporate author details and official organizer results when supplied. Before external submission, reconcile run IDs with the official results table, verify score denominators, add final model references and provider version details, and complete any required copyright process. These are release requirements, not blockers to an internal draft.

## Readability revision

User requested easier visuals, referencing the GLEE paper visual. Inspected the existing `glee-figure-mockups.html` artifact in the 2026-08-30 visualization folder. Adapted its concrete-input / decision / action structure to the existing paper in place:

- Replaced the combined system diagram with separate answer-generation and MC-review figures.
- Added a clearly disclosed constructed example showing a wrong saved premise, text agreement, and possible correction through direct video evidence. It is not a claim about measured outputs.
- Added short method lead sentences, descriptive subsection names, and paragraph breaks within the required structured abstract.
- Kept model names, numbers, evidence boundaries, and final-run definitions intact.
- Native compiler now reports success; the prior busy limitation is superseded. No separate PDF or editor tab was created. Final visual layout has not been independently screenshot-inspected.

## Second density revision

Rechecked the published 2026 notebook instructions: no page limit is stated; US Letter and minimum 9 pt remain the specified layout requirements. Switched the existing manuscript to 11 pt single-column text, paragraph spacing, and modestly increased line spacing. Converted abstract run inventories to scan-friendly tables while preserving the five required fields per task. Moved twelve supporting blocks (runtime settings, repair counts, detailed costs, provenance, and results-completion notes) to a labeled appendix. Figures now use single-column floats with section boundaries to keep them near their explanation. Native compilation succeeded; reference and figure-environment checks passed. Rendered pages were not independently screenshot-inspected.

## Third visual simplification

User reiterated preference for easier GLEE-style visuals. Replaced paragraph-filled diagrams with a numbered source-priority figure, a three-row MC route comparison, and a side-by-side cup/bottle vector illustration of the saved-premise failure. The example remains explicitly constructed and conditional. Added a three-step conceptual introduction. Existing method details and quantitative evidence remain in place. Native compilation succeeded; labels and figure environments verified. No independent screenshot inspection was available in this pass.

## Full voice and explanation edit — October 4

Applied the technical-writing paper profile and revisited the GLEE figure mockups. Rewrote the main narrative in direct first-person prose: what we supplied, which fixed rule we used, and what the evidence says. Removed repeated caveats and abstract phrasing, retained limitations beside the affected claims, and preserved detailed development intervals and MC settings in the appendix. Shortened the title. Rebuilt the assembly figure as three stages and the MC comparison as proportional bars over the shared 2,443-question denominator; retained the explicitly constructed cup/bottle example. All visuals remain vector drawings embedded in the same source, with no external image dependency.

The initial compile caught a PGF intermediate-dimension overflow and run-table paragraph overflow. Corrected the arithmetic order and paragraph boundaries; native compilation succeeded. Checked all eight run tags, citation/reference resolution, figure environments, arithmetic, and the example disclosure. No claim of independent rendered-page inspection is made.

## Official results and final readability verification — October 4

Revised the existing manuscript in place. The structured abstract now leads with the decisions each task tests. Figure 1 shows a clearly constructed competing-answer example: Gemini's red cup stays at rank one, the agreeing Qwen duplicate is removed, and blue bottle stays as a distinct alternative. The composition bars and cup/bottle example remain. Development exact match (9/30 to 11/30) appears alongside mixed subgroup metrics and exploratory uncertainty.

Integrated organizer results from merged PR 53, pinned to git object `3efb5b19dbd6d8d59232986cc25ef049af24ba5e`, without changing the checkout. Read original workbook and ZIP attachments into temporary inspection storage. All four MC summaries match their ZIP score members; AG Summary values agree with the Human Answer Only and All References sheets. MC uses 2,428 scored questions, distinct from the 2,443-row composition denominator. AG rank-one scores and NDCG@10 remain separate, with both reference modes labeled. No inference or scoring rerun was performed.

Native editor compilation succeeded. A temporary local XeLaTeX inspection build also succeeded twice; its final log has no overfull/underfull boxes, undefined references, or warnings. Inspected every final page (1–11) at readable size after repairing an appendix line overflow and bibliography spacing. Figures and tables fit, and citation/label checks pass. Temporary inspection artifacts are under `/tmp/trec-paper-final-review/`; the original TeX file remains the deliverable and the existing editor stays open. This supersedes earlier unverified-layout and pending-results notes.

Remaining before submission: author names and affiliations, final model citations and provider-version descriptions, and author review of the manuscript and accurate drafting-assistance disclosure. No publication or submission was performed.

## Four-page condensation — October 4

User rejected the eleven-page scope and requested that two figures explain the related approaches. Replaced the long narrative and appendix in the same TeX source with four pages: compact required structured abstract and overview; three AG setups plus rewriting variant in Figure 1; shared MC sources and replacement rules in Figure 2; results and references. Retained 11-point type. Kept one official-results table per task; AG table explicitly uses all references, with human-only patterns and lower-rank outcomes summarized in prose. Preserved uncertainty, fallbacks, development sample limitations, exact run tags, and drafting-assistance disclosure. Detailed evidence remains in repository records.

Verification: two XeLaTeX no-PDF layout passes; final XDV has four pages, no warnings, unresolved references, or overfull/underfull boxes. Exactly two figures and two tables. No separate PDF or editor tab was created. Native compiler was attempted twice but returned `No handler registered for tool: codex_app.compile_latex_document`; native preview compilation could not be confirmed in this pass. No rendered-page inspection claimed. Author metadata and final model references remain outstanding.

## Architecture-flow revision — October 4

User clarified that clearer architecture and decision flow were wanted, not hand-drawn art. Rebuilt both vector figures: AG now shows inputs, model calls, saved sources, source assembly, and the rewrite branch; MC shows the two ranking sources and explicit yes/no replacement decisions. Kept four pages. Compiled PDF locally without warnings; inspected both revised figure pages and repaired intersecting decision branches. Updated the same PDF deliverable. Native compiler remains unavailable through the exposed app tool. No scores or experimental claims changed.

## Prior-year format check and concrete visuals — October 5

Inspected rendered pages from HLTCOE VQA 2025 (two columns, illustrated candidate-answer pipeline), WHU-NERCMS AVS/VQA 2025 (single column, task-organized report), and RUC_AIM3 AVS 2024 (two columns, conventional introduction/method/results structure). Sources:
- https://trec.nist.gov/pubs/trec34/papers/HLTCOE.vqa.pdf
- https://trec.nist.gov/pubs/trec34/papers/WHU-NERCMS.avs.vqa.pdf
- https://trec.nist.gov/pubs/trec33/papers/ruc_aim3.avs.pdf
- https://www-nlpir.nist.gov/projects/tv2026/notebookpapers.html

Current requirements specify US Letter, minimum 9-point type, and a five-part structured abstract separately for each task; other formatting is left to participants. No page limit is stated. Retained the readable 11-point single-column format and changed the introductory heading to Introduction. Prior proceedings are layout references, not substitutes for the current notebook requirements.

Replaced letter-code architecture figures with one explicitly invented cup/bottle example across AG and MC. Custom vector art shows the hypothetical video evidence. AG shows source answers, duplicate removal, fixed rank one, and the rewrite constraint. MC shows saved-answer matching versus revisiting video, followed by the three selection policies in full words. No experimental scores changed. Date updated to October 5. Native compilation succeeds; local two-pass XeLaTeX build has no warnings and four pages. Visually inspected all four pages. Updated the existing PDF alongside its existing TeX source. Author metadata, final model references, and author review remain outstanding.

## Structured abstract formatting — October 5

Reverified that the organizer explicitly requests five compact items separately for each task. Replaced dense inline paragraphs with aligned bold field labels and one submitted run per line. Shortened wording and reduced title whitespace while retaining 11-point body text and four total pages. Native compilation succeeds; local two-pass compilation has no warnings. Inspected the revised first page. Updated the same PDF. No method or result changes.

## Plain-language abstract cleanup — October 5

Replaced abstract labels with Data used, What we tried, What changed, What helped?, and What we learned. Named MC configurations alongside their scores, clarified AG component attribution, and stated task-level lessons directly. Preserved all five organizer categories and exact run tags. Native and local compilation pass; four pages, no local warnings. Visually checked revised first page and refreshed the PDF.

## Example disclosure simplification — October 5

User allowed retaining the illustrative example while removing repetitive invented-data disclaimers. Kept the existing example and labeled it once, concisely, in Figure 1's caption; removed repeated warning-like prose from the figure and Figure 2 caption. No claim of recorded experimental provenance. Local compilation passes without warnings; four pages. Updated PDF.

## Submission-first clarification — October 5

Verified the Qwen3.6 submission record and ledger: video-only and video-plus-transcript are two inference calls combined within one q36_dualview submission, not separate submissions. Reorganized AG prose as four submitted runs, clarified this in the abstract, introduction and figure caption, and removed the preliminary development-results paragraph per user preference to focus on submitted systems. Preserved the required development-data disclosure in the abstract. Native and local compilation pass; four pages; revised methods page visually checked. PDF refreshed.

## Qwen figure grouping — October 5

Grouped the two Qwen calls inside one explicitly labeled submission box. Inner boxes distinguish video-plus-transcript from video alone; caption explains reuse in Gemini fusion. Corrected header spacing after render inspection. Four-page PDF refreshed; compilation passes.

## Four-run overview strip — October 5

Added a compact four-card overview above the existing AG fusion example, matching the four submission paragraphs. Labeled the detailed example as submission 3 and its rewrite as submission 4. Preserved the user's preferred layout and four-page length. Corrected a wrapped card label after render inspection. PDF refreshed; no additional figure or page needed.

## Consistent MC submission labels — October 5

Added MC1–MC4 consistently to the abstract's run list, all MC method headings, the figure's route descriptions, and official result rows. Explicitly states four submitted MC runs. Fixed route-list overlap after rendered inspection. Four pages; updated PDF and successful compiler checks.

## Paired results, real guiding case, and authorship — October 5

User requested dataset-grounded examples, an explanation of AG/MC strengths and failures, correct-answer availability and leaderboard context, and individual authorship. Added Nidhin Pattaniyil as Individual participant (team: NP Labs), verified from local Git author configuration/history. Original official per-query bundles support all new counts; canonical paper analysis is question-level-analysis.qmd with aggregate JSON evidence. Replaced invented cup/bottle figures with a paraphrased selected real case linking AG scope mismatch and MC rescue. Added a contrasting MC option-overlap example and bounded interpretation; no video perception or ground-truth validity claim is asserted. No new inference. Native and local two-pass compilation pass, all four rendered pages visually checked, PDF refreshed. Four pages remain.

## Consistent results order and failure analysis — October 5

User authorized additional length if useful. Results now follow AG then MC, matching methods. Added a fifth page for the question-opening breakdown, best-run context, three selected misses, and limitations. Lexical groups reproduce 2,428 rows and all four official totals; MC3 how-many accuracy is 200/309, accounting for 109/376 errors. Counts and selection boundaries are recorded in question-level-analysis.qmd/JSON. Root causes remain hypotheses pending video/annotation adjudication. Native compilation requested; local two-pass compilation succeeds without warnings. Updated-page renders inspected.

## PDF annotation revision — 2026-10-05
- Addressed the 24 comments: consistent AG/MC IDs and indentation; explicit Whisper input and AG3-to-MC1 reuse; plain review-selection rules; removed cost/retry/repeated significance text; simplified rewrite explanation.
- Two task sections now each contain methods, results, and analysis. Replaced opaque example diagrams with submission flows; selected real cases are explained in prose.
- Added overlapping question-cue analysis for both tasks, with exact rules and results in question-level-analysis.json and methodology in question-level-analysis.qmd. Distinguishes lexical groups from proven modality requirements and similarity from correctness.
- Verification: desktop compiler success; XeLaTeX two passes; no overfull/undefined-reference diagnostics; six-page PDF visually inspected. Saved canonical PDF alongside source.
- Remaining: author review and final model references; causal error explanations require video adjudication. No new inference or publication.

## Second annotation pass — 2026-10-05
- Removed misleading AG arrows and both stacks of boxes. Submission comparison matrices show rank construction and MC input/review differences.
- Explicitly identifies word-based groups as our post-hoc categories, not organizer labels.
- Replaced dense example prose with side-by-side AG answers and an inspected MC counting case. Removed the promised-future-inspection sentence and unverified sound example.
- Verified selected counting predictions against archived organizer rows and inspected 1-fps frames spanning the clip. Recorded evidence boundary in question-level-analysis.qmd.
- Final PDF: five pages, desktop compiler success, two local passes, visually checked all pages, no overfull or unresolved-reference diagnostics.

## Actual frame figure — 2026-10-05
User explicitly requested real clip frames in the paper. Added three local frames (0.5, 4.5, 8.5 seconds), with official and per-run selected answers. Extracted assets are ignored under private-frames; no publishing. Local XeLaTeX compiled successfully twice and PDF page inspected. Desktop compiler cannot load external local images (Operation not permitted), so live editor preview is blocked; saved canonical PDF includes the frames. This supersedes the previous no-frames-in-paper decision for this user-authorized example.

## Architecture and actual choices — 2026-10-05
Added AG independent-input/model branches and MC text/video ranking routes. Replaced paraphrases with selected actual AG outputs and the full counting choice set, under explicit user authorization. Corrected counting interpretation using AG3's 'feature pink' wording. Local PDF compiled twice, six pages, inspected diagrams and case layout. Desktop compiler still cannot access external image assets; canonical PDF is current.

## Grouping and stronger error analysis — 2026-10-05
Retained explicitly lexical groups (released question schema lacks labels). Simplified MC figure into four independent rows with a yes/no branch only in MC2. Added paired counting corrections/regressions and shared-error counts, a second inspected visual case, and a separate overlapping-choice example. Local compilation and page checks completed; live compiler retains external-image limitation.

## Shared cases and advisor panel — 2026-10-05
Compared cross-task frozen outputs and consolidated two selected cases into one visual figure with all choices and actual AG/MC answers. Removed extended MC-only analysis and redundant AG vignette. Two independent advisor reviews (technical evidence; clarity/visuals) completed; findings/disposition in advisor-review.md. Applied input-pipeline wording, fallback clarification, explicit MC2 reuse, and portable relative asset paths. Local two-pass compilation passes, six pages. Native image-asset limitation remains.

## Latest five annotations — 2026-10-05
Added traceable IDs, simplified run outputs, verified Spider-Man clip context, added official MC reference with AG-reference boundary and per-answer AG scores. Corrected factual-failure overclaim. Reduced unvalidated grouping emphasis. Local PDF rebuilt twice and revised layout inspected; native compiler still lacks image asset access.

## Clear errors versus low scores — 2026-10-05
Removed Spider-Man as main failure; added explicit numeric-disagreement family (76/109 MC3 how-many misses) and inspected Q106 where official count matches visible evidence. Added separate correct-identification/lower-score section using Q853, actual image/answers/scores. Kept causality and prevalence boundaries. Local compilation passes; built-in compiler still lacks image assets.

## Family-friendly visuals and shared example — 2026-10-05
Replaced insect image with bright basketball practice; removed dark people frame. Added same basketball example to joint AG/MC figure, all actual MC choices, and verified unanimous correct MC outcome. Retained toy disagreement. Updated source and saved PDF, locally compiled twice and visually checked. Native compiler asset limitation persists.

## Worked submission examples — 2026-10-05
Added page6 using the same basketball clip to show actual AG3→AG4 list changes and complete MC1/MC3 option rankings. Architecture captions link to worked section. Seven-page local PDF compiled twice, worked page visually checked, no overflow/reference warnings. Native image-asset limitation persists.

## Semantic grouping revision — 2026-10-05
Removed MC worked-ranking section. Replaced AG before/after table with recorded semantic grouping diagram (Q1561): four source answers, three groups, actual output rank membership. Verified against request, parsed response, and frozen final submission. PDF locally compiled twice; page6 visually checked. Native external-image restriction remains.

## Group filling and same-video MC — 2026-10-05
Clarified exact-source retention, variant generation, normalized uniqueness, generator-selected allocation/order, and no fill-one-group-first quota. Diagram shows2+4,1+1,1+1 source/variant counts. Added actual same-video MC outcomes and AG rank1 scores. Seven-page PDF locally compiled twice, revised page6 visually checked; desktop image limitation remains.

## Implementation-focused example — 2026-10-05
Replaced sprint example with Q447, two equivalent-source pairs and a clear descending/climbing conflict. Shows unchanged rank1, exact-source retention and six added variants; preserves same-video MC outcomes and AG scores. Verified complete output against frozen submission; saved provenance in QMD/JSON. Local two-pass XeLaTeX succeeds at seven pages without overflow/reference warnings; page6 visually inspected. Native compiler attempted; external image assets remain unsupported.

## Single shared example — 2026-10-05
Consolidated illustrated examples into Q1835 (weather): equivalent-wording scores, recorded two-group AG4 trace/all ten outputs, and actual eight MC options/results. Page5 now carries the complete worked example; six-page paper. Preserved brief numeric-error-family evidence. Verified frozen answers, official per-query scores and MC rows; local two-pass compilation succeeds without overflow/reference warnings; pages3–5 visually inspected. Native preview still cannot access external image assets.

## Focused adversarial pass and local commit — 2026-10-05

Completed requested Claude Code adversarial review; addressed seven findings proportionately, with dispositions in advisor-review.md. Strengthened the single example with AG1's contradictory answer/score, separate MC method rows, explicit selection boundary, and aggregate context. Tightened development, rank-one, NDCG, and counting claims. Added private-draft rebuild instructions. Two-pass six-page compilation and visual checks complete; final model references/author review remain draft-stage tasks. Native standalone compiler cannot load the external frame. Commit only the paper directory; no push or publication authorized.
