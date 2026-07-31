# RFWave

This directory presents the Study 3 synthesis for RFWave: one project-trained baseline row and the included deployment interventions evaluated for this architecture.

## Baseline

| Measure | Value |
|---|---:|
| True-length wide-band PESQ | 3.7846 |
| Recorded padded-lane PESQ | 3.6692 |
| STOI | 0.9809 |
| Parameters | 18,139,296 |
| Serialized size | 69.21 MB |
| B200 real-time factor | 0.0148 |
| M3 Max real-time factor | 0.0798 |
| 8-core CPU real-time factor | 0.7491 |
| Baseline Pareto memberships | B200 speed–quality |

## Intervention Coverage

This model contributes 13 included Study 2 groups: 2 denominator, 5 beneficial, 6 harmful, and 0 inconclusive. 12 additional execution attempts are represented only in the exclusion table and do not contribute to numerical conclusions.

## Interpretation Boundary

The intervention labels describe same-lane objective quality, RTF, and serialized-size effects under post hoc descriptive thresholds selected after execution. They do not establish preregistration, perceptual equivalence, population-level significance across training replicas, or performance outside the LJSpeech evaluation domain.

## Contents

`baseline.csv` contains the model's Study 1 synthesis row. `interventions.csv` contains its included Study 2 groups. `figures/` contains the model-highlighted cross-study profile.
