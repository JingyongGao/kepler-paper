# Canonical Artifact Index

This index separates the v3.5 exploratory discovery from the v3.6 independent confirmatory replication. The v3.5 result is never combined into the primary v3.6 p-value.

| Artifact | Version | Status | Frozen source | SHA-256 |
| --- | --- | --- | --- | --- |
| v3.5-cue-distribution-audit | Project Kepler v3.5 | exploratory_discovery | `artifacts/cue_distribution_audit.json` | `e6ce056320307dddd21691e475e7c604f7d02f880f89821f8c89545803fddc0a` |
| v3.5-cue-dose-results | Project Kepler v3.5 | exploratory_discovery | `artifacts/cue_dose_results.json` | `74de64c64ae0ebf97be98611048b6a512fbbbcccbc60e37e1e72fafd8c146cf1` |
| v3.6-confirmatory-distribution-audit | Project Kepler v3.6 | confirmatory_replication | `artifacts/confirmatory_distribution_audit.json` | `f67d261bc6e97fcb9fb7f25e12d7dfdae29fe0567acc930677507d5bf454c3c9` |
| v3.6-confirmatory-results | Project Kepler v3.6 | confirmatory_replication | `artifacts/cue_dose_confirmatory_results.json` | `5fd26ca916bcf73fd583d95b03fbb724f09536f88b22c848e8049c08e0c9b7a0` |

## Metric definitions

- **exact_conditional_kl:** Mean over the registered full-support parent-pair evaluation set of D_KL(q_h || p_theta).
- **normalized_R:** Exact conditional KL divided by log(p) - H(nu_eta); a uniform predictor has R=1 and the analytic oracle has R=0.
- **row_bootstrap_interval:** Registered 95% interval from 2,000 bootstrap resamples of first-parent rows.
- **center_accuracy:** Indicator that argmax_y p_theta(y|h) equals the true center z.
- **mean_center_probability:** Mean model probability assigned to the true center z.
- **predictive_entropy:** Mean entropy, in nats, of p_theta(.|h).
- **representation_onset:** First registered probe checkpoint with full-support R<0.9 and layer-6 linear-probe accuracy>0.5, with both retained at the next registered probe checkpoint.
- **representation_establishment:** First registered probe checkpoint with full-support R<0.1, center accuracy>0.9, and layer-6 linear-probe accuracy>0.9.
- **page_style_test:** One-sided exact blocked Page-style within-block label-permutation test using doubled within-block midranks; larger statistics align earlier onset with increasing registered cue dose.
- **persistence:** Retention of the frozen onset threshold at 6M and 10M among cells that were threshold-positive at the 5M pre-switch evaluation.

## Destinations and provenance

### v3.5-cue-distribution-audit

- Figures: `paper/figures/fig01_task_cue_schematic`, `paper/figures/fig07_v35_geometry_comparison`
- Tables: `paper/tables/table01_distribution_invariants`
- Provenance: `{"manifest_seed_summary":{"data_seeds":[1206845142,1454497354,4190121015,4207843790,4265698507],"evaluation_seeds":[1950082767,2634833603,2794286904,3591743300,4219147053],"model_initialization_seeds":[0,1,2,3,4],"probe_seeds":[252955824,427251465,2087411460,2243163775,3194591644]},"permutation_seed_sha256":"e6cccb69e7dba372d1a5db459b15625cc06c614ea8ba59db637d2d843ee282d9","trained":false}`

### v3.5-cue-dose-results

- Figures: `paper/figures/fig02_v35_dose_trajectories`, `paper/figures/fig07_v35_geometry_comparison`
- Tables: `paper/tables/table02_v35_exploratory_outcomes`, `paper/tables/table06_cue_removal_persistence`, `paper/tables/table07_protocol_integrity`
- Provenance: `{"cell_count":30,"checkpoint_hash_count":300,"checkpoint_reference_count":330,"checkpoint_sha256":["0079a338cba78a9693dff6c6a44bf5b74fef7819613239ad14be7db7d9d7deed","014ad8f7a28e3569ddf4d1224422671d69b1ae43d28b59d71baffdf05c93d169","01c4f271948f42054d761286d11fa2bcdf44508414bf151249b379e9167d8a1c","01dfe15215aa31ec8b18ca3f705248272ca4ee72274036b7febb242ee1bb00b0","02968aea6382ba76ebbcf6884c07696a0847218f163c6c8bb0221333c0ffe592","03a33f509c82a5edda77584819e21b2b21851f3b4ecf33313fce659c7a669451","0424e7b68a22165ef1edd61a29ae372a8f1edd316439f404bbf31ca592c49981","059d8a6a21e90d3f981326b81ba7...`

### v3.6-confirmatory-distribution-audit

- Figures: `paper/figures/fig01_task_cue_schematic`
- Tables: `paper/tables/table01_distribution_invariants`
- Provenance: `{"generation_seeds_sha256":{"block_rng_streams":"183338fcb7b65f902ba342dc62fa235813a33f430cb595083e68a1505feb23b6","cue_permutations":["1f31b4ee7abe61520cbfd1d12ee54908e4cdcc5d6153fdfbcbae64d65b57f642","25805e63757592cc4a5f60fe9e4b40fb8b100113b6ee6b99b0d41f8e6fcdc262"],"model_initializations":"5cff01a7cab164ce82f20939a9b8a99bfc35692925fff2e824dfcb2943b98856","rules":"b27ac3dd3035423188193159d1b1e2ad954adef65f3843022a63b8ff7fefff2e"},"manifest_seed_summary":{"data_seeds":[230412755,266665298,352952622,700838186,857749246,896133475,1260919668,1470659661],"evaluation_seeds":[57506884,647469400...`

### v3.6-confirmatory-results

- Figures: `paper/figures/fig03_v36_paired_onset`, `paper/figures/fig04_v36_exact_null`, `paper/figures/fig05_v36_persistence`, `paper/figures/fig06_v36_rule_heterogeneity`
- Tables: `paper/tables/table03_v36_confirmatory_outcomes`, `paper/tables/table04_exact_tests`, `paper/tables/table05_paired_R_differences`, `paper/tables/table06_cue_removal_persistence`, `paper/tables/table07_protocol_integrity`, `paper/tables/table08_claim_boundaries`
- Provenance: `{"cell_count":24,"checkpoint_hash_count":240,"checkpoint_reference_count":264,"checkpoint_sha256":["000e47b0446eaa2957019238a6833a232b8f0bff9add318faa955e98466ef82e","00707092ddb82c650aa387f0b514222c53f5fe2d25a8261c9ea84ecaec8cb7f4","01d23b57b3b99a19947a25406184157870a6660daf4e0d3516c602d55d30d349","03444f75761e46d3df27d6e8b7faa88bd6f576872866cbbbd31bf1b2722212c7","05d0cc4ace308456b0ecd5cd0eb5cab382dac8171f88e20d0b3b6a419799ec2b","0798f979435d035379f82dd79747b051c7bb50a6598b20f0894ad183f8c89d20","088967ae59bdf2a7ef48532ab02abdca6e594c2318fb321287aa5aee0526847e","09d8563d76ead4d8fde45ecc2c59...`
