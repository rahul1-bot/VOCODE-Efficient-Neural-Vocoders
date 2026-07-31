# LPCNet

This directory presents the Study 3 synthesis for LPCNet: one project-trained baseline row and the included deployment interventions evaluated for this architecture.

## Baseline

| Measure | Value |
|---|---:|
| True-length wide-band PESQ | 1.0519 |
| Recorded padded-lane PESQ | 1.0770 |
| STOI | 0.5267 |
| Parameters | 1,232,992 |
| Serialized size | 4.70 MB |
| B200 real-time factor | 11.3633 |
| M3 Max real-time factor | unsupported |
| 8-core CPU real-time factor | 9.2015 |
| Baseline Pareto memberships | none |

## Intervention Coverage

This model contributes 0 included Study 2 groups: 0 denominator, 0 beneficial, 0 harmful, and 0 inconclusive. 15 additional execution attempts are represented only in the exclusion table and do not contribute to numerical conclusions.

## Interpretation Boundary

The intervention labels describe same-lane objective quality, RTF, and serialized-size effects under post hoc descriptive thresholds selected after execution. They do not establish preregistration, perceptual equivalence, population-level significance across training replicas, or performance outside the LJSpeech evaluation domain.

## Contents

`baseline.csv` contains the model's Study 1 synthesis row. `interventions.csv` contains its included Study 2 groups. `figures/` contains the model-highlighted cross-study profile.
