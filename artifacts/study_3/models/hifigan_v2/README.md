# HiFi-GAN V2

This directory presents the Study 3 synthesis for HiFi-GAN V2: one project-trained baseline row and the included deployment interventions evaluated for this architecture.

## Baseline

| Measure | Value |
|---|---:|
| True-length wide-band PESQ | 2.3997 |
| Recorded padded-lane PESQ | 2.4561 |
| STOI | 0.9421 |
| Parameters | 928,514 |
| Serialized size | 3.54 MB |
| B200 real-time factor | 0.1140 |
| M3 Max real-time factor | 0.0704 |
| 8-core CPU real-time factor | 0.0507 |
| Baseline Pareto memberships | parameter–quality, size–quality |

## Intervention Coverage

This model contributes 7 included Study 2 groups: 2 denominator, 2 beneficial, 3 harmful, and 0 inconclusive. 6 additional execution attempts are represented only in the exclusion table and do not contribute to numerical conclusions.

## Interpretation Boundary

The intervention labels describe same-lane objective quality, RTF, and serialized-size effects under post hoc descriptive thresholds selected after execution. They do not establish preregistration, perceptual equivalence, population-level significance across training replicas, or performance outside the LJSpeech evaluation domain.

## Contents

`baseline.csv` contains the model's Study 1 synthesis row. `interventions.csv` contains its included Study 2 groups. `figures/` contains the model-highlighted cross-study profile.
