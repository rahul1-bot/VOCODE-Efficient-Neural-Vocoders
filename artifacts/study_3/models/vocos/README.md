# Vocos

This directory presents the Study 3 synthesis for Vocos: one project-trained baseline row and the included deployment interventions evaluated for this architecture.

## Baseline

| Measure | Value |
|---|---:|
| True-length wide-band PESQ | 3.6137 |
| Recorded padded-lane PESQ | 3.5653 |
| STOI | 0.9793 |
| Parameters | 13,531,650 |
| Serialized size | 51.62 MB |
| B200 real-time factor | 0.0047 |
| M3 Max real-time factor | 0.0044 |
| 8-core CPU real-time factor | 0.0098 |
| Baseline Pareto memberships | B200 speed–quality |

## Intervention Coverage

This model contributes 11 included Study 2 groups: 2 denominator, 3 beneficial, 5 harmful, and 1 inconclusive. 13 additional execution attempts are represented only in the exclusion table and do not contribute to numerical conclusions.

## Interpretation Boundary

The intervention labels describe same-lane objective quality, RTF, and serialized-size effects under post hoc descriptive thresholds selected after execution. They do not establish preregistration, perceptual equivalence, population-level significance across training replicas, or performance outside the LJSpeech evaluation domain.

## Contents

`baseline.csv` contains the model's Study 1 synthesis row. `interventions.csv` contains its included Study 2 groups. `figures/` contains the model-highlighted cross-study profile.
