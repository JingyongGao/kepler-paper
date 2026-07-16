# Project Kepler v3.5 — Full-Support Cue Distribution Audit

## Frozen construction

The cue family uses p=257, `z=45*x1+36*x2 mod 257`, and the preregistered
coefficient `c=1`, which is nonzero and differs from 45.  A single field
permutation is constructed by sorting each value by
`SHA256(seed_bytes || uint16_big_endian(value))`, with seed
`e6cccb69e7dba372d1a5db459b15625cc06c614ea8ba59db637d2d843ee282d9`.
Every `B_m` is the corresponding prefix of that permutation.

The complete permutation, subsets, solved conditions, probabilities, entropy
checks, and root brackets are in `artifacts/cue_distribution_audit.json`,
SHA-256 `e6ce056320307dddd21691e475e7c604f7d02f880f89821f8c89545803fddc0a`.

## Exact sampling and identities

The distribution is sampled as the registered mixture.  The base component
draws a uniform linear pair index.  The cue component draws uniform `x1` and a
uniform value from `B_m`, then solves the unique `x2`.  The lambda=0 branch uses
the original G1 linear-index draw directly, including identical RNG
consumption.

Analytically:

- X1 is uniform because both components draw it uniformly;
- X2 is uniform because `c != 45` makes the cue-component affine map from X1
  to X2 bijective for every fixed subset value;
- Z is uniform because `c != 0` makes `Z=c*X1+b` bijective in X1;
- the shifted noise channel preserves uniform Z, so Y is uniform;
- conditioning on either singleton parent produces the same multiset of `m`
  boosted and `p-m` baseline target probabilities, hence
  `I(Y;X1)=I(Y;X2)`.

Exhaustive enumeration of all 66,049 pairs confirms every identity.  The
maximum marginal-uniformity error is below `2e-17`; singleton-symmetry error is
below `9e-16`; and joint-information error is below `1.6e-14` nats.  For every
condition, `H(Y|X1,X2)=0.8796007178394043` nats and
`I(Y;X1,X2)=log(257)-H(nu_eta)=4.6694753670558` nats within enumeration
tolerance.

## Solved cue doses

Lambda is not a rounded hardcoded input.  Each nonzero value is selected by 80
deterministic bisection iterations over `[0,nextafter(1,0)]`, using exhaustive
enumeration of the registered cue index at each step.

| condition | m | target C_single | solved lambda | achieved C_single | I(Y;X1)=I(Y;X2) |
|---|---:|---:|---:|---:|---:|
| C0 | 64 | 0 | 0 | 7.61e-16 | 1.78e-15 |
| C1 | 64 | 0.016 | 0.18295450567904434 | 0.0159999999999990 | 0.0373558029364443 |
| C2 | 64 | 0.065092 | 0.3805503208347219 | 0.0650920000000006 | 0.151972745296201 |
| C3 | 64 | 0.13 | 0.5461664584309741 | 0.1299999999999998 | 0.303515898858628 |
| geometry-concentrated | 1 | 0.065092 | 0.0788263575575168 | 0.0650919999999999 | 0.151972745296199 |
| geometry-diffuse | 128 | 0.065092 | 0.5948892782779377 | 0.0650920000000001 | 0.151972745296199 |

The small nonzero displayed C0 value is floating-point entropy cancellation;
its pair probabilities are bitwise equal to the uniform G1 probability vector.

## Pair-probability geometry

| condition | minimum P(pair) | maximum P(pair) | H(pair) nats | expected minimum pair exposure at 5M |
|---|---:|---:|---:|---:|
| C0 | 1.51402746445821e-5 | 1.51402746445821e-5 | 11.0981521697904 | 75.7014 |
| C1 | 1.23702931811376e-5 | 2.34934999952818e-5 | 11.0523521768217 | 61.8515 |
| C2 | 9.37863827106055e-6 | 3.25152093334828e-5 | 10.9121607396996 | 46.8932 |
| C3 | 6.87116446227840e-6 | 4.00768100380915e-5 | 10.7257306419657 | 34.3558 |
| geometry-concentrated | 1.39468219419292e-5 | 3.20664166523707e-4 | 10.9218896177208 | 69.7341 |
| geometry-diffuse | 6.13348758833688e-6 | 2.42174272247041e-5 | 10.9076915248166 | 30.6674 |

All conditions retain strict full support.  Matching the scalar cue index does
not match probability concentration: m=1 places a much larger maximum mass on
a narrow geometry, while m=128 uses a larger mixture weight but a more diffuse
boost.  This is the intended geometry control.

`C_single` is a normalized sum of two singleton mutual informations.  It is not
an additive percentage of unique information and is not interpreted that way.
