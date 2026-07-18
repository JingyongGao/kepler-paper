# Project Kepler Paper v1.2.1 — Final PDF Preflight Report

## Decision

**Pass.** The anonymous 12-page PDF is portable, all fonts are embedded, the
main-text tables are readable at 100% zoom, and the frozen science remains
unchanged. No training, optimizer step, new scientific analysis, new
hypothesis test, statistical redefinition, onset recoding, or numerical-claim
change occurred.

## Frozen baseline

- Preflight source paper commit:
  `d1377ee1a67c37da09aef89ca6ab0c68c9a4ee61`.
- Frozen scientific tag:
  `kepler-v3.6-confirm1`.
- Frozen scientific commit:
  `f9fc22d85cd52863087afbb3f924466a63b30ba2`.
- Complete preflight hash map:
  `paper/final_preflight_baseline.json`.
- Tracked frozen scientific artifacts: 18.
- Frozen v3.5 result SHA-256:
  `74de64c64ae0ebf97be98611048b6a512fbbbcccbc60e37e1e72fafd8c146cf1`.
- Frozen v3.6 result SHA-256:
  `5fd26ca916bcf73fd583d95b03fbb724f09536f88b22c848e8049c08e0c9b7a0`.
- Before-task and after-task byte comparisons: pass, 18/18 unchanged.
- Checkpoint sidecars: 330/330 v3.5 and 264/264 v3.6 pass.

## PDF hashes

| Version | File | SHA-256 |
| --- | --- | --- |
| v1.2 baseline | `paper/output/pdf/kepler_paper_v1_2_submission.pdf` | `37e91cef9e9fc79b453083eaaeeda455dc9aef714e67857a4ecb9cfe74a4aa1c` |
| v1.2.1 pre-terminology patch | `paper/output/pdf/kepler_paper_v1_2_1_submission.pdf` | `1141e2ef441ca99fb23f455d16b717b8bb94f68b5ef97c315002ac43e4cbbb45` |
| v1.2.1 submission | `paper/output/pdf/kepler_paper_v1_2_1_submission.pdf` | `3e70a70094f7e1b0a483b9c5e1633208fde314bc5f0a4e0f70f24a12906761dc` |

The new PDF is anonymous, 12 pages, US letter, PDF 1.5.

## Presentation-only changes

1. Replaced figure core-font references with embedded Bitstream Vera TrueType
   subsets supplied by ReportLab. Figure dimensions, drawing coordinates,
   data, axes, values, intervals, colors, and panel content are unchanged.
2. Replaced the large main-text distribution audit with a representative
   R0/R1/R2 invariant summary.
3. Added compact narrative-facing main tables for exploratory outcomes,
   confirmatory outcomes, the primary exact test, paired differences, and
   cue-removal persistence.
4. Moved every complete registered table, including protocol integrity and
   claim boundaries, to Supplementary Section 14.7. No complete table was
   removed from the PDF or submission package.
5. Used normal-size unscaled main-table bodies and tightened only float/caption
   spacing. The bibliography uses a conventional compact two-column layout.
6. Copyedited the Abstract to say “exploratory study” and “confirmatory study,”
   reduced repeated internal version/freeze labels, and added a plain-language
   scaffold-removal sentence without changing claims or numbers.
7. Used compact BibTeX output while retaining the verified DOI/arXiv metadata.
8. Replaced Figure 3's panel title “Eight preregistered paired blocks” with
   “Eight confirmatory paired blocks” to match the manuscript's bounded
   registration language. Figure data, geometry, values, and source-data
   hashes are unchanged.

## Portability and layout checks

- `pdffonts` font records audited: 45.
- Non-embedded font records: 0.
- Final-PDF fonts with `emb=yes`: 31/31.
- Figure-PDF fonts with `emb=yes`: 14/14 across seven vector PDFs.
- Raster fallback: not used in the manuscript; all seven figures remain vector
  PDF assets, with SVG and 1800×1080 PNG companions.
- Figure source-data hashes changed: 0 of 8.
- Table CSV hashes changed: 0 of 8.
- Undefined citations/references: 0.
- Overfull/underfull warnings requiring intervention: 0.
- Clipped figures or tables: 0.
- Glyph substitutions observed: 0.
- Internal PDF links: 48 checked, 48 valid named destinations, 0 invalid.
- Reference verification targets: 13 checked. Ten returned direct HTTP 200;
  three publisher endpoints rejected automated curl requests but resolved to
  the verified article pages through independent page lookup.
- Anonymous metadata/source forbidden matches: 0.
- Visual inspection: all 12 latest rendered pages inspected; main tables were
  additionally inspected at 100% zoom.

The complete raw font listing is in `paper/font_embedding_audit.txt`; the
table-specific review is in `paper/table_readability_audit.md`.

## Scientific consistency checks

- Headline numerical claims rebuilt from frozen JSON: 39/39.
- Distribution invariants recomputed: pass.
- Transition cells recomputed: 54.
- Exact confirmatory assignments recomputed: 1,679,616.
- Observed Page-style statistic: 110.
- Upper-tail count, including the observed assignment: 16.
- Frozen one-sided p-value:
  `9.525986892242037e-06`.
- New scientific tests: 0.
- Optimizer steps: 0.

## Final status

The v1.2.1 PDF and source packages are ready for submission. Any later
venue-style substitution should rerun the font, link, anonymity, numerical,
and frozen-artifact checks before upload.
