# FreeV

This directory presents the Study 3 synthesis for FreeV: one project-trained baseline row and the included deployment interventions evaluated for this architecture.

## Baseline

| Measure | Value |
|---|---:|
| True-length wide-band PESQ | 3.3863 |
| Recorded padded-lane PESQ | 3.3394 |
| STOI | 0.9695 |
| Parameters | 18,220,557 |
| Serialized size | 69.66 MB |
| B200 real-time factor | 0.0077 |
| M3 Max real-time factor | 0.0075 |
| 8-core CPU real-time factor | 0.0157 |
| Baseline Pareto memberships | none |

## Intervention Coverage

This model contributes 7 included Study 2 groups: 2 denominator, 0 beneficial, 3 harmful, and 2 inconclusive. 9 additional execution attempts are represented only in the exclusion table and do not contribute to numerical conclusions.

## Interpretation Boundary

The intervention labels describe same-lane objective quality, RTF, and serialized-size effects under post hoc descriptive thresholds selected after execution. They do not establish preregistration, perceptual equivalence, population-level significance across training replicas, or performance outside the LJSpeech evaluation domain.

## Contents

`baseline.csv` contains the model's Study 1 synthesis row. `interventions.csv` contains its included Study 2 groups. `figures/` contains the model-highlighted cross-study profile.
