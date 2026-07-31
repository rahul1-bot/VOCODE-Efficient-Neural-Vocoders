# HiFi-GAN V3

This directory presents the Study 3 synthesis for HiFi-GAN V3: one project-trained baseline row and the included deployment interventions evaluated for this architecture.

## Baseline

| Measure | Value |
|---|---:|
| True-length wide-band PESQ | 2.5501 |
| Recorded padded-lane PESQ | 2.5765 |
| STOI | 0.9488 |
| Parameters | 1,464,322 |
| Serialized size | 5.59 MB |
| B200 real-time factor | 0.0580 |
| M3 Max real-time factor | 0.0252 |
| 8-core CPU real-time factor | 0.0366 |
| Baseline Pareto memberships | parameter–quality, size–quality |

## Intervention Coverage

This model contributes 8 included Study 2 groups: 2 denominator, 2 beneficial, 2 harmful, and 2 inconclusive. 3 additional execution attempts are represented only in the exclusion table and do not contribute to numerical conclusions.

## Interpretation Boundary

The intervention labels describe same-lane objective quality, RTF, and serialized-size effects under post hoc descriptive thresholds selected after execution. They do not establish preregistration, perceptual equivalence, population-level significance across training replicas, or performance outside the LJSpeech evaluation domain.

## Contents

`baseline.csv` contains the model's Study 1 synthesis row. `interventions.csv` contains its included Study 2 groups. `figures/` contains the model-highlighted cross-study profile.
