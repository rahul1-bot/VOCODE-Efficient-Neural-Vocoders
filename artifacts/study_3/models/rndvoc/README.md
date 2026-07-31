# RNDVoC

This directory presents the Study 3 synthesis for RNDVoC: one project-trained baseline row and the included deployment interventions evaluated for this architecture.

## Baseline

| Measure | Value |
|---|---:|
| True-length wide-band PESQ | 3.8953 |
| Recorded padded-lane PESQ | 3.8568 |
| STOI | 0.9871 |
| Parameters | 3,572,361 |
| Serialized size | 14.94 MB |
| B200 real-time factor | 0.0559 |
| M3 Max real-time factor | 0.0748 |
| 8-core CPU real-time factor | 0.0838 |
| Baseline Pareto memberships | B200 speed–quality, parameter–quality, size–quality |

## Intervention Coverage

This model contributes 6 included Study 2 groups: 2 denominator, 1 beneficial, 1 harmful, and 2 inconclusive. 3 additional execution attempts are represented only in the exclusion table and do not contribute to numerical conclusions.

## Interpretation Boundary

The intervention labels describe same-lane objective quality, RTF, and serialized-size effects under post hoc descriptive thresholds selected after execution. They do not establish preregistration, perceptual equivalence, population-level significance across training replicas, or performance outside the LJSpeech evaluation domain.

## Contents

`baseline.csv` contains the model's Study 1 synthesis row. `interventions.csv` contains its included Study 2 groups. `figures/` contains the model-highlighted cross-study profile.
