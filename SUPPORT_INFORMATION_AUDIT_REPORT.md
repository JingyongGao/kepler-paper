# Project Kepler v3.4 — Exact Support-Induced Information Audit

## Protocol

This is an exact no-training enumeration of all 66,049 ordered parent pairs
under the frozen p=257 split.  Natural logarithms are used throughout.  For
each support, parent pairs are uniform on that support,
`Z = 45*X1 + 36*X2 mod 257`, and `Y` is produced by the registered eta=0.1
shifted-noise channel.  No empirical model result is used to construct or alter
the support.

The complete distributions, 257 per-parent KL values, and support-size
histograms are in `artifacts/support_information_audit.json`, SHA-256
`449c8fbde56bc148d014bec36feed52166d36d2a597c4c75f14f83cc911c35b4`.
The frozen split SHA-256 is
`273fea214893ffd818737a1c9faa75b8d47a5bc3cafb184796868580d757cab2`.

## Exact information quantities

| support | pairs | H(Z) | H(Z\|X1) | H(Z\|X2) | I(Z;X1) | I(Z;X2) |
|---|---:|---:|---:|---:|---:|---:|
| G1 full | 66,049 | 5.549076 | 5.549076 | 5.549076 | 0 | 0 |
| G2 train | 52,839 | 5.548585 | 5.326438 | 5.326401 | 0.222146 | 0.222184 |
| G2 held-out | 13,210 | 5.541191 | 3.947788 | 3.947386 | 1.593403 | 1.593805 |

| support | H(Y) | H(Y\|X1) | H(Y\|X2) | I(Y;X1) | I(Y;X2) | I(Y;X1,X2) | C_single |
|---|---:|---:|---:|---:|---:|---:|---:|
| G1 full | 5.549076 | 5.549076 | 5.549076 | 0 | 0 | 4.669475 | 0 |
| G2 train | 5.548678 | 5.396730 | 5.396708 | 0.151948 | 0.151970 | 4.669078 | 0.065092 |
| G2 held-out | 5.542703 | 4.336749 | 4.336456 | 1.205954 | 1.206246 | 4.663102 | 0.517295 |

The registered noise entropy `H(Y|X1,X2)` is 0.879601 nats for every support.

## Full-support analytic check

For uniform independent X1 and X2, fixing either singleton parent leaves the
other parent uniform.  Since both coefficients are nonzero in the field, its
contribution is a bijection, so Z remains uniform conditional on X1 or on X2.
Passing that uniform state through a shift-invariant noise channel leaves Y
uniform as well.  Therefore all four singleton mutual informations are
analytically zero.

The numerical enumeration gives exactly zero KL-based singleton mutual
information.  All six full-support entropy deviations from `log(257)` are at
most `8.89e-16`, below the preregistered numerical tolerance `1e-12`.

## Support sizes and per-parent KL

| support | row sizes | column sizes | target-state sizes |
|---|---:|---:|---:|
| G1 full | 257–257 | 257–257 | 257–257 |
| G2 train | 188–222 | 189–223 | 184–223 |
| G2 held-out | 35–69 | 34–68 | 34–73 |

For G1, every per-parent
`KL(P(Y|Xi=x) || P(Y))` is zero.  On G2 training support, the X1 KL range is
0.100206–0.212622 nats and the X2 range is 0.095410–0.211170; their
support-weighted means are the singleton mutual informations in the table.
On held-out support the corresponding ranges are 0.948245–1.555109 and
0.984768–1.536489 nats.  Exact arrays and the analogous Z KL values are retained
in the JSON artifact.

## Interpretation

The frozen G2 training distribution has nonzero singleton information:
`C_single = 0.065092`.  It must therefore not be described as a pure-synergy
training distribution.  G1 full support is the pure-synergy distribution in
the registered singleton-information sense.  The audit establishes a support
asymmetry that could provide a symmetry-breaking cue; by itself it does not
show that the optimizer used that cue or that the cue caused the observed G2
transition.
