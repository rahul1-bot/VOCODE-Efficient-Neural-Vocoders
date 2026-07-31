# Experiments

The measured evidence is organized as three studies under `artifacts/`, which together answer the three research questions: Study 1 establishes the trained baselines (RQ1), Study 2 measures the deployment interventions (RQ2), and Study 3 synthesizes frontiers and selection across models (RQ3). The three are dependent analyses of one experiment: transformed variants inherit the trained checkpoints, and selection consumes their measurements.

| Study | Contents | Registers |
|---|---|---|
| `study_1/` | Twelve project-trained baselines with training histories, seed-level test evaluations, and baseline figures | Per-model `experiments.csv` |
| `study_2/` | Same-lane deployment interventions over the trained checkpoints | `results.csv` (91 groups), `exclusions.csv` (29 records) |
| `study_3/` | Cross-model synthesis: the baseline surface, frontier membership, and intervention conclusions | `results.csv`, `interventions.csv`, `intervention_exclusions.csv`, `frontiers.csv` |

## The Executed Surface

The intervention surface contains 22 pre-transformation controls and 69 transformed groups, which is 91 groups measured over 273 executions. Each group aggregates three executions (inference seeds 42, 43, and 44) of one frozen artifact on one hardware lane; except for static ONNX INT8, the three executions repeat inference from one fixed artifact, so their spread reflects execution variability rather than statistical uncertainty. The 29 exclusion records document 88 rejected attempts, each with seeds, run identifiers, a machine-readable reason code, and a full-sentence reason.

## The Decision Rule

Every transformed group is labeled on its paired same-lane deltas: PESQ change, real-time-factor speedup, and serialized-size ratio. A group is harmful when its PESQ change falls below negative 0.10, or when speedup falls below 0.90 while the size ratio exceeds 0.90. A group is beneficial when the PESQ change is at least negative 0.05, speedup is at least 0.90, and either speedup is at least 1.10 or the size ratio is at most 0.55. Every other group is inconclusive. Dense-masked pruning and recovery rows are labeled on quality alone, and MelGAN rows are fixed as inconclusive because their PESQ is floor-censored. Each register row carries its label in `classification` and the governing sentence in `classification_basis`.

The rule labeled the 69 transformed groups as 19 beneficial, 18 inconclusive, and 32 harmful. Quality contrasts pair the execution-averaged utterance values before a Student t interval, with a 5,000-replicate percentile bootstrap as sensitivity analysis; these are fixed-set descriptive intervals, not population inference.

## Sensitivity

A grid of 243 threshold profiles varies every cutoff of the rule over three levels. The beneficial, harmful, and inconclusive totals ranged over 12 to 24, 29 to 34, and 12 to 27; median and minimum agreement with the default taxonomy were 91.3 and 82.6 percent. Stability stratifies by rule structure: 27 of the 28 rows governed by the dense-mask and MelGAN overrides never changed label, whereas 25 of the 41 threshold-governed rows never changed, so aggregate stability partly reflects deterministic rule structure rather than threshold insensitivity.

## Outcome Composition

| Variant | Groups | Beneficial / Inconclusive / Harmful |
|---|---:|---|
| Compilation, default mode | 11 | 4 / 5 / 2 |
| Compilation, reduce-overhead | 4 | 1 / 0 / 3 |
| FP16 weight cast | 10 | 8 / 2 / 0 |
| INT8 dynamic | 5 | 2 / 1 / 2 |
| INT8 weight-only | 4 | 2 / 2 / 0 |
| ONNX FP32 export | 4 | 1 / 2 / 1 |
| ONNX INT8 static | 4 | 0 / 1 / 3 |
| Pruning at 30 / 50 / 70 percent | 4 / 11 / 4 | 0 / 0 / 4, then 0 / 1 / 10, then 0 / 0 / 4 |
| Pruning, 50 percent with recovery | 5 | 0 / 4 / 1 |
| RFWave at 8 / 4 / 2 ODE steps | 1 each | 1 / 0 / 0, then 0 / 0 / 1, then 0 / 0 / 1 |

The per-variant medians and ranges are reported in Table 3 of the report and reproduced in the repository's root `README.md`.
