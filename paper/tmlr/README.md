# TMLR submission workspace

This directory is derived from the frozen
`kepler-paper-v1.2.1-submission` tag. It is restricted to venue formatting,
double-blind packaging, policy disclosure, and reproducibility materials.

- `template/` contains the unmodified official TMLR style files.
- `template_provenance.json` records the upstream commit and SHA-256 values.
- `submission/` is the generated anonymous TMLR LaTeX source tree.
- `supplement_package.json` records the anonymous supplemental ZIP.

Build the source tree and supplemental ZIP:

```sh
uv run python paper/scripts/build_tmlr_package.py
```

Compile the anonymous PDF deterministically:

```sh
uv run python paper/scripts/compile_tmlr_pdf.py
```

No command in the TMLR packaging path performs training or optimizer steps.
