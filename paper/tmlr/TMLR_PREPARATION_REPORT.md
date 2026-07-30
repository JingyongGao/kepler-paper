# TMLR Submission Preparation Report

## Status

The anonymous TMLR PDF and lightweight supplemental ZIP are technically ready
for author review. They have not been submitted to OpenReview.

The local venue branch is `kepler-paper-tmlr-v1`, derived from the immutable
`kepler-paper-v1.2.1-submission` tag. The frozen v1.2.1 PDF remains
byte-identical with SHA-256
`3e70a70094f7e1b0a483b9c5e1633208fde314bc5f0a4e0f70f24a12906761dc`.

On 2026-07-30, the package was rebuilt and revalidated after the author's
OpenReview profile became active. The live TMLR Submission form is accessible.
The official template upstream remains at the recorded commit, all 12 PDF
pages passed a new rendered visual inspection, and all 62 repository tests
passed. Details are recorded in `OPENREVIEW_PREFLIGHT_RECORD.md`.

## Official template

- Source: official TMLR style repository.
- Upstream commit:
  `7bf90efe3a0debbba703c05c43f3ff7e4d4a2992`.
- Downloaded archive SHA-256:
  `48f9f972cc812d2cfc8032a78578c2f5d8939b92fbd4f61095e8cbeb3394f92d`.
- `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty`, and the upstream license are
  included without modification.
- The anonymous style mode is active; neither `accepted` nor `preprint` mode
  is used.

## Venue-only presentation changes

1. Migrated the frozen manuscript content to the official TMLR layout.
2. Placed references before the appendix, following the official template.
3. Added the required first-page LLM assistance disclosure:

   > LLM-based tools, including ChatGPT/Codex, Claude, and Gemini, were used
   > as assistive tools for iterative experimental-design discussion, code
   > generation and review, manuscript editing, and document preparation.
   > The authors selected and approved all hypotheses, frozen protocols,
   > analyses, interpretations, and claims; verified all reported results
   > through the frozen artifact and reproduction audits; and take full
   > responsibility for the manuscript.

4. Converted Unicode missing-value em dashes to the equivalent LaTeX `---`
   only in the TMLR table copies, avoiding glyph loss under the official
   Latin Modern font.
5. Preserved all scientific values, figure data, table data, statistical
   definitions, claims, and frozen source files.

## Anonymous PDF

- File: `paper/output/pdf/kepler_paper_tmlr_v1_submission.pdf`.
- SHA-256:
  `8f7b7d73fd2a02f039e66e05cc68d8ded4efdade49f39c2d09b463d44954bfb8`.
- Pages: 12, US Letter.
- References begin on page 9; the appendix begins on page 10.
- Main content is at most 9 pages before references.
- Deterministic build: two consecutive builds with `SOURCE_DATE_EPOCH=0`
  were byte-identical.
- Embedded font records: 44/44.
- Undefined references, missing glyphs, overfull boxes, or warnings requiring
  intervention: 0.
- Two official-template float-pagination `Underfull \vbox` notices remain;
  all 12 rendered pages were inspected and show no clipping, abnormal
  whitespace, overlap, or broken float placement.
- Anonymous source/PDF forbidden identity matches: 0.

## Anonymous supplemental ZIP

- File: `paper/output/tmlr/kepler_tmlr_supplement_v1.zip`.
- SHA-256:
  `ec21b36e4d9b2185b070d6fe2b5039b14300688d4ccb5b4b4ea004cb93a55118`.
- Size: 1,271,244 bytes, below the TMLR 100 MB limit.
- Files: 89.
- Checkpoint files: 0.
- Text files scanned for identity leakage: 89.
- Forbidden identity matches: 0.

The original frozen result files contain local absolute evaluation paths. The
supplement therefore contains anonymous review copies in which 649 absolute
path strings were replaced by relative paths. Sixteen non-scientific Git
provenance fields were removed from the review genealogy. Every changed JSON
field, original source SHA-256, and packaged SHA-256 is recorded in
`SOURCE_ARTIFACT_HASHES.json`. Scientific values changed: 0.

The unpacked lightweight verifier reproduced:

- distribution invariants;
- 54 transition cells;
- all 1,679,616 exact confirmatory assignments;
- upper-tail count 16;
- one-sided `p = 9.525986892242037e-06`;
- all publication figures and tables;
- zero optimizer steps.

## Integrity

- Frozen scientific artifacts remain byte-identical.
- Headline numerical claims reproduced: 39/39.
- Repository tests passed: 62/62.
- No training, optimizer step, new experiment, new hypothesis test, onset
  recoding, or statistical-definition change was performed.

## Remaining external actions

1. Confirm that the single profile currently selected in the OpenReview form
   is the complete and correctly ordered author list.
2. Confirm author quota, affiliations, publication records, conflicts,
   funding, competing interests, and human-subjects/IRB response.
3. Select candidate Action Editors.
4. Confirm there is no overlapping archival submission.
5. Conduct the planned non-author review.
6. Upload the anonymous PDF and supplemental ZIP only after the author
   completes the final OpenReview form review.
