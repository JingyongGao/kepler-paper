# Project Kepler — Mechanism Isolation Audit

## Scope and frozen provenance

This report covers only the registered no-training gradient audit and Runs
M1–M4.  All rows are exploratory, diagnostic-only, and excluded from the
confirmatory scaling fit.  No production or micro-pilot trajectory was launched,
no existing trajectory was extended, and no production optimizer or prior
manifest was changed.

- RC1 target commit: `a5d98f98bdcd4ccf891d662c7630fc795aff4563`
- completed learnability-audit target commit/tag: `32edf3f8839bd3f828d2793411630c64f41a127c` / `kepler-v3.2.2-audit1`
- mechanism implementation commit: `b96e13a5544897b7d546c6b1123bd34dda804022`
- production manifest SHA-256: `4331a9198ee43a374e908d3b2b0e95d34173fe94220e3cc3505f9deee7611f10`
- pilot manifest SHA-256: `3df30d340a80395b5185b5f38f6780c6a288d715e4d3eeb04cf527c9f8f097b0`
- learnability-audit manifest SHA-256: `95a21c5a89af958ae986b3d3284c9e610bd37c16a0d3e68dc2d01bd83c9f5af4`
- mechanism-isolation manifest SHA-256: `d854fa799a361287111c2f3ad996558992f72741a6560883517caf0979070b5a`
- initialization-gradient audit SHA-256: `fd8fb468f7ba6ec9ca9b125d9c36e0392d9d2c2529fbc118ff8b259af7b0a282`

The mechanism manifest records the three prior hashes and the exact coefficient
reuse: M1 uses `a1=45`; M2–M4 use `(a1,a2)=(45,36)`.

## Pure synergy result

For independent uniform parents and nonzero coefficients, any omitted parent
contributes a scaled uniform field element.  Its sum with the remaining terms
and independent noise is therefore uniform and independent of every fixed value
of the retained subset.  Consequently,

`I(Y; X_S)=0` for every proper parent subset `S`, while
`I(Y; X_1,...,X_k)=log(257)-H(nu_0.1)=4.669475367055815 nats`.

For the registered two-parent rule the analytic values are exactly 0, 0, and
4.669475367055815 nats for `X1`, `X2`, and `(X1,X2)`.  Two independent
262,144-sample stratified checks gave bias-corrected singleton MI estimates of
`-1.67e-5` and `-1.29e-4 nats`.  The small negative values are finite-sample
bias-correction noise, not negative information.  In this generator `k` is thus
both fan-in and the order of a pure synergistic interaction.

## Initialization gradient-signal audit

Each cell below is the mean over eight independently initialized 2M models.  For
each initialization, 32 independent Episode batches (512 scored positions per
batch) were accumulated online under exact soft-label loss.  The reported SNR is
`||mean(g)||^2 / mean(||g_i-mean(g)||^2)`.  Full per-initialization results,
mean-gradient norms, variances, mean pairwise cosines, batch-gradient norms, all
six block totals, and every block's Q/K/V/MLP/norm groups are in
`artifacts/initialization_gradient_signal_audit.json`; no set of full batch
gradient vectors was retained.

| task | embedding SNR | output SNR | Q SNR | K SNR | V SNR | MLP SNR | norm SNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| k1,d1, lag 1 | 0.6492 | 0.6254 | 0.4187 | 0.4278 | 0.1040 | 0.5874 | 0.2472 |
| k1,d4, lag 4 | 0.0401 | 0.0509 | 0.0359 | 0.0355 | 0.0554 | 0.0460 | 0.0748 |
| k2,d2, lags 1,2 | 0.0401 | 0.0477 | 0.0351 | 0.0351 | 0.0600 | 0.0465 | 0.0856 |
| k2,d4, lags 1,4 | 0.0393 | 0.0487 | 0.0347 | 0.0342 | 0.0549 | 0.0447 | 0.0836 |

The mean pairwise cosines show the same split.  For example, embedding/output
cosines were `0.377/0.368` for k1,d1, but only `0.0075/0.0177` for k1,d4,
`0.0074/0.0147` for k2,d2, and `0.0067/0.0156` for k2,d4.  The two-parent
tasks are indistinguishable at this resolution across spans 2 and 4.  Crucially,
k1,d4 also begins with low SNR but M1 later learns completely.  Low
initialization SNR is therefore a learning-cost indicator here, not a sufficient
failure test or impossibility result.

## Behavioral results

All `R` values use the exact conditional distribution.  M1/M2 confidence
intervals bootstrap independent Episodes; M3/M4 enumerate all 66,049 parent
pairs and bootstrap the 257 first-parent rows.  Checkpoint JSON files contain
training/fresh NLL, entropy, center probability, per-block gradient norms, and
per-block update/parameter ratios at every registered checkpoint.

| run | final exposure | exact KL (nats) | R (95% CI) | center accuracy | P(center) | entropy (nats) | fresh NLL |
|---|---:|---:|---|---:|---:|---:|---:|
| M1: 2M recurrent, k1 lag4 | 10M tokens | 0.02136 | 0.004575 [0.004557, 0.004593] | 1.0000 | 0.8964 | 0.8855 | 0.9047 |
| M2: 10M recurrent, k2 lags1,2 | 40M tokens | 4.66951 | 1.000008 [0.999987, 1.000028] | 0.004700 | 0.003891 | 5.5491 | 5.5491 |
| M3: parent-exposed Transformer | 10M pairs | 2.15879 | 0.4623 [0.4564, 0.4685] | 0.1688 | 0.08243 | 3.4284 | 3.0335 |
| M4: concatenated-parent MLP | 10M pairs | 4.67110 | 1.00035 [1.00032, 1.00037] | 0.004315 | 0.003891 | 5.5475 | 5.5510 |

M1 crossed from `R=0.9978` at 0.5M to `0.4877` at 2M, `0.00832` at 5M,
and `0.00457` at 10M.  Its layer-3 and layer-6 linear and fixed-capacity MLP
probes all reached 1.0 test accuracy; the layer-6 fixed-Fourier readout reached
variance-normalized R2 `0.5530`.  Lag-4 retrieval is therefore learnable under
the frozen architecture and optimizer.

M2 remained at the exact uniform baseline at every checkpoint: `R=1.00061`,
`1.00034`, `1.00015`, and `1.000008` at 5M, 10M, 20M, and 40M.  At 40M its
layer-6 linear/MLP probe accuracies were `0.00488/0.00195`, and Fourier R2 was
`-0.2110`; no predictive-state representation formed.  This is not a
noise-calibration-only failure, because the modular center was never decoded.

M3 remained at `R≈1` through 5M, then reached `R=0.4623` at 10M.  At 10M its
layer-6 linear/MLP probe accuracies were `0.2285/0.3047`, and its Fourier R2 was
`0.0927`.  A partial predictive-state representation formed and the output used
it, but neither representation nor conditional distribution was oracle-like at
the registered endpoint.

M4 remained at the uniform baseline at all five checkpoints.  This is not a
noise-calibration-only failure: center accuracy and center probability remained
at chance, so the modular center itself was not learned.

### Parent-pair frequency and held-out diagnostic

M3/M4 evaluation records accuracy and `R` separately for unseen, low-, middle-,
and high-frequency parent pairs.  Because the main streams are genuinely iid
uniform over all 66,049 pairs, empirical zero-frequency pairs exist only early:
at 0.5M, M3 had 34 and M4 had 43.  Their accuracies were respectively `1/34`
and `0/43`, providing no early structured-rule evidence.  By 1M every pair had
been observed, so later “structured versus lookup” status is explicitly marked
inconclusive rather than relabeling seen pairs as held out.  The full-support
metrics are exact, but the present design cannot prove that M3's late partial
gain is algebraic rather than distributed lookup.  No training samples were
withheld post hoc, because doing so would violate the registered iid-uniform
parent-pair stream.

## Mechanism interpretation

1. **M1 succeeds.**  The original k2,d4 failure cannot be attributed primarily
   to an inability to retrieve position `t-4`.
2. **M2 fails at the uniform baseline.**  Together with successful M1, this
   selects the registered “M1 succeeds and M2 fails” branch: the multi-parent
   pure-synergy interaction is the primary obstacle isolated here, not lag-4
   retrieval.  Shortening the two-parent rule to lags 1 and 2 and increasing the
   model from 2M to 10M did not restore learning through 40M tokens.
3. **M3 is a positive but incomplete mechanism control.**  Direct parent
   exposure produces a substantial exact-KL reduction and measurable internal
   state, unlike M2.  The registered matrix therefore points to recurrence,
   distractors, or support discovery as a major added obstacle.  The endpoint
   does not show complete modular composition.
4. **M4 fails while M3 improves.**  The branch “M4 succeeds but M3 fails” does
   not apply.  Removing attention is not sufficient under this raw embedding
   and fixed-depth GELU MLP interface.
5. The k1,d4 gradient audit and M1 outcome show that low initial population
   gradient SNR can be overcome.  By contrast, similarity of k2,d2 and k2,d4
   SNR supports the view that shortening span does not restore a first-order
   gradient signal at initialization.

These findings are specific to the registered field, parameterizations,
optimizer, schedules, and exposure limits.  They are not a universal
impossibility theorem for Transformers or MLPs.

## Stop-condition audit

After completing the gradient audit and M1–M4, execution stopped.  No p=17 run,
Fourier initialization, curriculum, global optimizer change, remaining
micro-pilot, or production trajectory was launched.  A field-size sweep or
Fourier-initialization audit, if scientifically desired after this result, must
be separately preregistered and is not performed here.
