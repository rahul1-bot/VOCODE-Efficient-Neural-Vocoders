# LPCNet

This directory contains the Study 2 intervention results for LPCNet under the common LJSpeech evaluation design.

## Results

No execution group completed the full 525-utterance measurement contract for this architecture.

## Excluded Executions

| Variant | Category | Attempts | Reason |
|---|---|---|---|
| baseline_cpu | incomplete_lpcnet_execution | 3 | The executions terminated before the complete 525-utterance measurement contract and are not used as results. |
| int8_dynamic | incomplete_lpcnet_execution | 3 | The executions terminated before the complete 525-utterance measurement contract and are not used as results. |
| pruned_50 | incomplete_lpcnet_execution | 6 | The executions terminated before the complete 525-utterance measurement contract and are not used as results. |
| pruned_50_recovered | incomplete_lpcnet_execution | 3 | The executions terminated before the complete 525-utterance measurement contract and are not used as results. |

## Interpretation

All quality statements refer only to the named objective and learned-proxy metrics on single-speaker LJSpeech. Evaluation seeds repeat the same retained model state. Dense masked pruning is not interpreted as physical size or sparse-runtime acceleration.

## Contents

`experiments.csv` and `statistics.csv` contain headers only because no execution group was admitted for this architecture. The grouped reasons are reported above and in the study-level exclusion table. `figures/` contains the architecture status figure.
