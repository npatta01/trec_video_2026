# VQA Grouped-Rewrite Canary Design

## Objective

Test whether one text-only Gemini 3.7 Flash call can group genuine frozen answer
candidates per question and produce ten grounded wording variants while keeping
the current fusion rank-1 answer byte-exact at rank 1.

## Frozen panel and inputs

Use testing query IDs `1, 310, 1563, 2404, 2421`. Q1 and Q310 exercise
semantic agreement; Q1563 and Q2404 contain genuine conflicting candidate
meanings; Q2421 contains a severe count disagreement. For each query, include
only the released question and available rank-1 answers from the final Gemini
overlay, Qwen transcript, Qwen native, and Qwen3.5 control. Mark the source
selected at rank 1 in the final fusion provenance. Do not include baseline
ranks 2–10, labels, annotations, references, metrics, media, transcripts, or
prior grading information.

## Two one-call prompt strategies

Run two independent, identically bounded strategies over the same five rows.
`semantic_rewrite_a` asks for semantic grouping and ten grounded formulations.
`exact_group_hedge_b` adds the requirement that every group retain at least one
exact frozen source answer before paraphrases fill remaining ranks. Each
response must return each query separately, assign every source candidate to
exactly one semantic group, preserve conflicting groups, and produce exactly
ten normalized-distinct answers. Rank 1 must be an exact copy of the frozen
current rank-1 answer. Every answer must name one valid source group and add no
fact.

Use a fixed seed, JSON response mode, at most 8,192 output tokens, and a hard
observed/estimated cost cap of `$0.10` per strategy. Freeze each strategy's
input, request, raw response, parsed result, token usage, cost, and artifact
hashes in a distinct new private root.

## Canary decision

Advance the design only if all five queries have valid terminal output, every
source is grouped exactly once, every conflict remains represented, rank 1 is
byte-exact, each query has ten unique answers, no query crosses into another,
and manual label-free review finds no introduced fact. A pass establishes
pipeline safety only; it does not establish better hidden-label quality. Any
failure preserves the current five ready submissions unchanged.
