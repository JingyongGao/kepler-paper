# Reviewer Response Matrix

All responses are manuscript or submission-package changes. No row required a
new scientific experiment, optimizer step, statistical test, transition
redefinition, or numerical-result change.

| Item | Reviewer concern | Manuscript/package location | Disposition | New experiment required |
| --- | --- | --- | --- | --- |
| Claude W1 | Title, “higher-order,” confirmatory-freeze terminology, and wording that could imply frozen trained parameters | Title; Abstract; Introduction; Prospectively frozen confirmatory replication | Addressed | No |
| Claude W2 | The cue construction and operational meaning of `C_single` were too far from the main argument | Full-support cue-family construction; Figure 1; Supplementary cue-family derivation | Addressed | No |
| Claude W3 | Pure synergy was described too strongly as a theorem about vanishing gradients | Discussion / Possible mechanism: symmetry breaking and optimization accessibility | Addressed | No |
| Claude W4 | Inferential unit, within-block exchangeability, tail counting, and stratified tests needed clarification | Prospectively frozen confirmatory replication; Supplementary Exact test; Table 4 | Addressed | No |
| Claude W5 | Related work did not adequately cover delayed generalization, curricula, self-paced and continuation methods, probes, path dependence, and PID caveats | Related Work; `references.bib`; `reference_verification.json` | Addressed | No |
| Claude W6 | Figure 1 did not expose the task identities, distribution geometry, and cue-removal schedule together | Figure 1 and source JSON | Addressed using frozen distribution JSON | No |
| Claude W7 | Warmup timing, joint-parent dependence, remaining axes of generality, and manifest provenance needed explicit treatment | Limitations; Supplementary Model and optimization; Reproducibility statement | Addressed / bounded limitation | No |
| Claude Q1 | Was the confirmatory study publicly preregistered? | Abstract; Introduction; confirmatory section | Bounded: described as “prospectively frozen,” with no public-registration claim | No |
| Claude Q2 | What exactly changes under the cue intervention? | Full-support cue-family construction; Figure 1 | Addressed: joint parent distribution changes while marginals, support, target conditional, Bayes entropy, and total predictable information are preserved | No |
| Claude Q3 | Are the 24 trajectories independent replicates? | Prospectively frozen confirmatory replication; Table 3 | Addressed: eight paired blocks are the inference units; 24 trajectories are condition observations | No |
| Claude Q4 | Do probes or gradient statistics establish causal use or mediation? | Related Work / Mechanistic interpretability; Discussion mechanism subsection | Bounded limitation: extractability and correlational dynamics only | No |
| Claude Q5 | How broadly should the result generalize? | Practical implication; Limitations; claim-boundary check | Deferred: scale, field size, noise, input interface, sampled labels, and optimizer-phase timing remain open | No |

## Outcome

- Addressed in manuscript or package: W1–W7, Q1–Q4.
- Explicitly bounded or deferred rather than overclaimed: W3, W7, Q1, Q4, Q5.
- New experiments required for this revision: **0**.
