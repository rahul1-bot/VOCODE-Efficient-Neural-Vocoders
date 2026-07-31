# Study 3 — Cross-Model Efficiency–Quality Synthesis

## Research Question

Which frozen Study 1 baselines occupy the strongest fidelity–RTF and fidelity–complexity frontiers under the common project protocol, and what separate same-lane deployment effects are observed for the admitted Study 2 interventions?

## Evidence Design

Study 3 synthesizes the twelve project-trained baseline configurations from Study 1 and separately summarizes the 91 included model–variant groups from Study 2. Baseline quality uses the 525-utterance LJSpeech adaptive evaluation partition and evaluation seeds 42, 43, and 44. Scores on this partition previously informed some Study 1 budget gates and an inference correction, so they are descriptive benchmark measurements rather than untouched final-test estimates. Efficiency comparisons use isolated, same-lane timing; intervention speedups are never compared across hardware lanes. Study 1 baseline real-time factors and Study 2 intervention timings also use different timing instruments, one complete evaluation pass versus three synchronized calls per utterance, and are never cross-study comparable. Evaluation seeds repeat inference from one trained state and are not independent training replicas.

The primary baseline quality axis is true-length wide-band PESQ. Recorded padded-batch PESQ remains visible because padding sensitivity is itself a robustness result, especially for VocosFormer. Objective quality is limited to the named signal-based and learned-proxy metrics; no listening study, multi-speaker evaluation, or out-of-distribution evaluation was performed.

## Baseline Synthesis

| Model | True-length PESQ | B200 RTF | Parameters | Study 2 groups |
|---|---:|---:|---:|---:|
| RNDVoC | 3.8953 | 0.0559 | 3,572,361 | 6 |
| RFWave | 3.7846 | 0.0148 | 18,139,296 | 13 |
| Vocos | 3.6137 | 0.0047 | 13,531,650 | 11 |
| VocosFormer | 3.5125 | 0.0122 | 13,778,690 | 8 |
| FreeV | 3.3863 | 0.0077 | 18,220,557 | 7 |
| BigVGAN-base | 3.3201 | 0.1573 | 14,025,154 | 8 |
| HiFi-GAN V1 | 3.3051 | 0.1197 | 13,936,130 | 9 |
| HiFi-GAN V3 | 2.5501 | 0.0580 | 1,464,322 | 8 |
| APNet2 | 2.5150 | 0.0095 | 31,425,539 | 6 |
| HiFi-GAN V2 | 2.3997 | 0.1140 | 928,514 | 7 |
| LPCNet | 1.0519 | 11.3633 | 1,232,992 | 0 |
| MelGAN | 1.0430 | 0.0742 | 4,266,050 | 8 |

The strict B200 speed–quality frontier is Vocos → RFWave → RNDVoC. It excludes LPCNet because that PyTorch timing row contains one seed and only 27 timed utterances after five warm-ups; including the labelled sensitivity point does not change frontier membership. The parameter–quality frontier is HiFi-GAN V2 → HiFi-GAN V3 → RNDVoC. RNDVoC provides the strongest measured baseline quality at 3.57 million parameters. Among the seven parallel models with true-length PESQ of at least 3.30, Vocos has the lowest measured B200 and CPU real-time factors, while RFWave occupies the intermediate B200 speed–quality frontier position. VocosFormer reaches near-Vocos true-length quality but is slower and strongly padding-sensitive. No architecture dominates fidelity, RTF, and footprint simultaneously.

## Intervention Synthesis

The included Study 2 set contains 19 beneficial, 32 harmful, 18 inconclusive, and 22 same-lane denominator groups under a post hoc descriptive taxonomy selected after execution. The classification is neither pre-registered nor a population-level significance claim.

Compilation and reduced precision are architecture-dependent rather than universal accelerators. Executable half precision is principally a storage reduction; RTF gains depend on the execution path. Dynamic or weight-only quantization helps only when the transformed operations cover a meaningful bottleneck. Dense masked pruning preserves tensor shapes and therefore does not constitute physical compression or sparse-kernel acceleration; recovered quality must also be interpreted against the absence of matched dense-continuation controls.

## Contents

| Path | Content |
|---|---|
| `results.csv` | Twelve baseline synthesis rows with recorded and true-length quality, complexity, three hardware timing lanes, and Pareto membership. |
| `interventions.csv` | The 91 included Study 2 model–variant groups and their same-lane effects. |
| `intervention_exclusions.csv` | Complete, duplicate, superseded, or incomplete executions excluded from numerical synthesis. |
| `frontiers.csv` | Ordered memberships of the baseline speed-, parameter-, and size-quality frontiers. |
| `models/` | One compact baseline and intervention summary for each model. |
| `figures/` | Cross-model trade-off, deployment, and intervention figures. |
