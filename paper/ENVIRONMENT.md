# Clean Environment Specification

The scientific runtime is frozen by [`../uv.lock`](../uv.lock), SHA-256
`c228e2dedf4aa6754be20fd226b0592f2f46e14c6c3b2c641cdd9ed19b08aab8`. The project metadata SHA-256 is
`61ee1cf16c00e6831d31cba68b1833965d7c802ae653fb9cf6db75a944b250cf`.

- Python: `3.13.13 (main, May 10 2026, 19:20:37) [Clang 22.1.3 ]`
- Platform: `macOS-26.5.2-arm64-arm-64bit-Mach-O`
- Processor: `Apple M4`
- PyTorch: `2.13.0`
- NumPy: `2.5.1`
- SciPy: `1.18.0`
- MPS available: `True`
- MPS fallback: `0`
- MPS fast math: `0`

Paper rendering adds only `reportlab==4.4.9` and Poppler `pdftoppm`
`26.05.0`. The overlay is isolated from the scientific lock. Figures use
invariant PDF mode; PNG files are rendered from those PDFs at 180 dpi.
`environment_spec.json` is the machine-readable specification.
