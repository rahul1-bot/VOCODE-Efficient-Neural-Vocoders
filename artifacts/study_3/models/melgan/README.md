# MelGAN

This directory presents the Study 3 synthesis for MelGAN: one project-trained baseline row and the included deployment interventions evaluated for this architecture.

## Baseline

| Measure | Value |
|---|---:|
| True-length wide-band PESQ | 1.0430 |
| Recorded padded-lane PESQ | 1.0597 |
| STOI | 0.4181 |
| Parameters | 4,266,050 |
| Serialized size | 16.27 MB |
| B200 real-time factor | 0.0742 |
| M3 Max real-time factor | 0.0455 |
| 8-core CPU real-time factor | 0.0456 |
| Baseline Pareto memberships | none |

## Intervention Coverage

This model contributes 8 included Study 2 groups: 2 denominator, 0 beneficial, 0 harmful, and 6 inconclusive. 3 additional execution attempts are represented only in the exclusion table and do not contribute to numerical conclusions.

## Interpretation Boundary

The intervention labels describe same-lane objective quality, RTF, and serialized-size effects under post hoc descriptive thresholds selected after execution. They do not establish preregistration, perceptual equivalence, population-level significance across training replicas, or performance outside the LJSpeech evaluation domain.

## Contents

`baseline.csv` contains the model's Study 1 synthesis row. `interventions.csv` contains its included Study 2 groups. `figures/` contains the model-highlighted cross-study profile.
