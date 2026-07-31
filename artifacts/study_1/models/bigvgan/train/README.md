# BigVGAN Training

This directory contains the accepted Study 1 training evidence for BigVGAN-base. The model was trained from random initialization on LJSpeech using training seed 1234; validation was performed during the same execution on the deterministic validation partition.

## Experiment

| Field | Value |
|---|---|
| Run identifier | `bigvgan_full_bf16bs32_20260704_0735` |
| Architecture | BigVGAN-base (`base_24khz_100band`) |
| Dataset | LJSpeech 1.1, resampled to 24 kHz |
| Partition | Identifier-ordered holdout: 12,475 training, 100 validation, 525 test utterances |
| Initialization | Random; no author-released generator weights |
| Training seed | 1234 |
| Hardware | NVIDIA L40S |
| Precision | BF16 mixed precision |
| Batch size | 32 |
| Training budget | 320 epochs; 124,800 optimizer steps |
| Wall-clock time | 58,348.9 seconds |
| Status | Completed |

Dataset membership is not randomized: records are sorted by utterance identifier, the final 525 form the adaptive evaluation partition, and the preceding 100 form the validation partition. Source audio is resampled to 24 kHz before 8,192-sample training segments are formed. The execution uses separate AdamW optimizers for the generator and discriminators, with per-step exponential learning-rate decay. Training metrics were logged every five steps, validation was performed every 1,000 steps, and durable checkpoints were written every 5,000 steps.

## Selected Checkpoint

The three accepted adaptive-benchmark executions evaluate the durable checkpoint at epoch 307 and step 120,000.

The saved checkpoint with the lowest validation loss occurs at epoch 294 and step 115,000, while the global validation minimum at step 118,000 falls between checkpoint boundaries. The validation-best and evaluated checkpoint states are not equivalent. The curated bundle therefore retains the evaluated step-120,000 state rather than relabeling the validation-best state as evidence for measurements it did not produce. The evaluated and validation-best states remain distinguished in this record.


## Files

| File | Purpose |
|---|---|
| `config.yaml` | Resolved training, data, architecture, optimization, and checkpoint configuration. |
| `execution.log` | Complete log from the accepted training execution. |
| `metrics.csv` | Step-indexed training and validation metric history. |
