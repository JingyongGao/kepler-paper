# ICLR 2027 Anonymity Audit

Audit date: 2026-08-01

## Scope

- Final PDF: `paper/output/pdf/iclr2027_final_anonymous_submission.pdf`
- Final supplement: `paper/output/iclr2027/iclr2027_final_anonymous_supplement.zip`
- Clean extraction root: `iclr2027_anonymous_supplement/`

## PDF audit

- PASS — ICLR 2027 review mode is active and the author block is anonymous.
- PASS — PDF metadata contains no Author, Title, Subject, custom metadata, or metadata stream.
- PASS — extracted text and LaTeX/BibTeX sources contain no author identity token, TMLR submission number or forum ID, private repository name, private remote, or local absolute path.
- PASS — the PDF contains no JavaScript, forms, attachments, or embedded files.
- PASS — all 32 font records are embedded.
- PASS — page size is US Letter.

The exact identity/path scan covered the known author-name variants, username,
private remote owner and repository name, TMLR paper number and forum ID, and
the `/Users/` and `/Volumes/` absolute-path prefixes. No match was found in the
final PDF text, manuscript source, or bibliography.

## Supplemental ZIP audit

- PASS — a new clean directory was used for extraction.
- PASS — 151 intended files were present; Git history and checkpoint binaries were absent.
- PASS — recursive byte-level scanning found no known identity token, TMLR submission identifier, private repository name/owner, or `/Users/` or `/Volumes/` path.
- PASS — the three included JSON checksum sidecars validate after extraction.
- PASS — the checkpoint-free test command selected 64 tests; all 64 passed and the five checkpoint-dependent tests were explicitly deselected.

Frozen repository JSON remains unchanged. For the anonymous archive only,
machine-local path prefixes are deterministically converted to archive-relative
paths and the historical private repository-name string is replaced with
`anonymous-repository`. Numerical fields and recorded checkpoint hashes are
unchanged; sidecars for path-normalized JSON copies are regenerated inside the
archive so they validate there.

## Conclusion

The final PDF and supplemental ZIP pass the double-blind anonymity audit. They
contain no TMLR submission identifiers and no private repository names or
paths.
