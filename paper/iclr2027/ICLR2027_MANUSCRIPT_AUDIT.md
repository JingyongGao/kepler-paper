# ICLR 2027 anonymous manuscript and package audit

Audit date: 2026-08-01 (Asia/Shanghai)

## Official requirements and template

- Official author guidelines:
  `https://iclr.cc/Conferences/2027/AuthorGuidelines`
- Official style archive:
  `https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip`
- Downloaded style archive SHA-256:
  `0d940dfa9398ae99a18f24a85a8a683f367204b6af6d17d2899e60a67102529e`
- The submission uses the official, byte-unmodified
  `iclr2027_conference.sty` and `iclr2027_conference.bst`.
- `\iclrfinalcopy` remains commented for double-blind review.

The audit applies the ICLR 2027 initial-submission limit of nine main-text
pages, permits references outside that limit, places the appendix after the
references, and includes the required AI use statement.

## PDF audit

- Output: `paper/output/pdf/kepler_paper_iclr2027_anonymous_draft.pdf`
- SHA-256: `e21ce75cfb36d28674ab9ed31a17ac51cdc42f5f9b658fa2425b1d2c2a684eee`
- Bytes: 380,294
- Pages: 13
- Main text: pages 1--9
- References begin: page 10
- Appendix begins after the references: page 11
- Required AI use statement: page 10
- Paper size: US Letter
- Anonymous author rendering: pass
- PDF text identity scan: pass
- Fonts: 31/31 records embedded
- Determinism: two consecutive Tectonic builds were byte-identical
- Layout warnings requiring intervention: 0

All 13 pages were rendered to PNG and visually inspected. The anonymous ICLR
header, title, text, figures, captions, compact main tables, post-confirmatory
paired tables, references, and appendix were legible and inside the official
page rectangle. No clipping, overlap, missing glyphs, broken images, or table
overflow was observed. Complete legacy audit fields were retained in the
machine-readable supplement instead of being typeset as illegible microtables.

## Anonymous code-package audit

- Output: `paper/output/iclr2027/kepler_iclr2027_anonymous_code.zip`
- SHA-256: `96cb2d5465ef5a70244912951585c7bb0088b6f7899675f761a7e8fb4f68c752`
- Bytes: 1,977,996
- Entries: 133
- ZIP integrity test: pass
- Source identity-token scan: pass
- Git history included: no
- Checkpoint binaries included: no
- TMLR submission correspondence included: no
- Extracted checkpoint-free test suite: 64/64 passed
- Five explicitly listed legacy checkpoint-dependent tests are unavailable in
  the checkpoint-free ZIP; all five pass in the full repository.

The archive contains code, tests, frozen manifests, result JSON, tabular source
data, and audit reports. The scale-result JSON records all 176 individual
checkpoint hash comparisons even though checkpoint binaries are omitted.

## Scientific-boundary audit

- Discovery v3.5 remains exploratory.
- The original v3.6 exact confirmatory result is unchanged:
  `p=9.5259869e-06`.
- The 1M/5M experiment is labeled post-confirmatory robustness in the abstract,
  introduction, methods, results, discussion, limitations, appendix, and code
  supplement.
- The scale extension is excluded from the original v3.6 p-value.
- No combined confirmatory p-value was computed.
- No scaling-law fit was computed or claimed.
- All eight new paired blocks are reported.
- The v1.3 Hessian audit appears only in an explicitly exploratory appendix
  subsection and is not presented as causal evidence.
- No recurrent bridge, p sweep, sampled-NLL training objective, or new
  architecture was run.
- The optional 3M/4M/5M early-removal continuation was not started because it
  was not separately authorized.

## Repository and frozen-state audit

- Tests: 69/69 passed.
- Scale trajectories: 16/16 complete.
- Registered scale evaluations: 192/192 present.
- Checkpoint SHA-256 sidecars: 176/176 valid.
- Zero-step 5M switch checks: 16/16 passed.
- Prior manifest/result hash locks: 9/9 unchanged.
- TMLR tag target:
  `kepler-paper-tmlr-desk-rejected^{}` =
  `57ef195b654066b1c5ce7d18fcb9843978694b8c`.
- Local and remote-tracking TMLR branch targets remain
  `57ef195b654066b1c5ce7d18fcb9843978694b8c`.

The frozen TMLR submission, v1.3 audit, v3.6 manifest and result, checkpoints,
pre-existing branches, and pre-existing tags were not modified by the ICLR
paper build.
