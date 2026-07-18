# Internal Rebuttal Playbook

This is an internal preparation document, not a prewritten response for
upload. Actual responses must address the reviewers' specific wording and
should not promise unregistered experiments.

## Why is only one approximately 2M-parameter model family studied?

The paper asks a controlled causal question about whether early
training-distribution geometry changes access to a task-sufficient
representation while the target conditional, information budget,
architecture, optimizer, and objective remain fixed. Holding the model family
fixed is part of that isolation strategy. The independent confirmatory study
adds held-out coefficient rules, cue permutations, and initialization seeds,
but it does not estimate a scale law. Accordingly, the manuscript restricts
the claim to the tested parent-exposed task and approximately 2M-parameter
Transformer family, explicitly lists scale dependence as open, and excludes
these results from the original scaling fit. The appropriate response is to
clarify this boundary, not to extrapolate across scale. Relevant locations:
Limitations; Claim-boundary Table 14; confirmatory manifest fields describing
the fixed 2,045,568-core-parameter architecture.

Acceptable revision: make the fixed-scale estimand more prominent.
Do not promise: a new scale sweep during review.

## Why use a parent-exposed interface?

The interface is an intentional mechanism-isolation control. It separates
identifying which variables in a recurrent context matter from composing two
already exposed parents into the modular predictive center. The experiment
therefore tests representation formation after variable discovery has been
controlled, rather than claiming that the model solved the earlier recurrent
production task. This makes the causal intervention interpretable: R0, R1,
and R2 differ in pair-distribution geometry while the input variables, target
conditional, model, and optimization protocol remain matched. The paper
states this separation in the Introduction and again treats recurrent context
discovery as future work in Limitations. Relevant evidence includes the
parent-exposed task schematic, the fixed full-pair enumeration, and the
confirmatory manifest.

Acceptable revision: sharpen the “variable discovery versus composition”
sentence.
Do not promise: a recurrent bridge experiment during review.

## Why use exact soft-label training?

Exact soft labels isolate the optimization of the predictive conditional from
variance introduced by sampling a single noisy target label. The target
distribution is not changed: it is the same registered
`q(y | x1, x2)` used for exact evaluation, with the same irreducible entropy
and Bayes-optimal center. This choice narrows the result to representation
formation under an analytically controlled objective. The paper does not
claim that the observed effect automatically transfers to sampled-token
training; sampled-label learning is explicitly listed as open. Relevant
locations are Supplementary Methods A.1, the Objective limitation, and the
confirmatory manifest objective field.

Acceptable revision: state more directly that soft labels remove label-sampling
variance but not optimization difficulty.
Do not promise: a sampled-NLL replication during review.

## Why is the effect not attributable to `C_single` alone?

The randomized causal object is the registered cue-bearing joint
distribution, not the scalar summary `C_single`. The index records the sum of
the two singleton mutual informations relative to joint predictive
information; it is not a PID unique-information percentage and does not
identify the remaining geometry of the pair distribution. Two pieces of
evidence bound a scalar-only interpretation. First, exploratory matched-index
geometry controls with the same registered `C_single` but different support
geometry showed different dynamics. Second, the confirmatory mid-dose
response differed across held-out rules while the high dose was robust. The
paper therefore claims that cue-bearing distributions changed onset, while
explicitly declining to identify singleton mutual information alone as the
causal mediator.

Acceptable revision: emphasize the distinction between intervention and
summary statistic.
Do not promise: a universal PID or mediation analysis.
