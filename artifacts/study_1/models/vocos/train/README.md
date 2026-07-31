# Vocos Training

This directory contains the accepted Study 1 training evidence for Vocos. The model was trained from random initialization on the deterministic LJSpeech training partition using seed 1234.

## Experiment

| Field | Value |
|---|---|
| Run identifier | `vocos_full_bf16bs32_20260704_0625` |
| Architecture | Vocos (`charactr_mel_24khz`) |
| Dataset | LJSpeech 1.1, resampled to 24 kHz |
| Partition | Identifier-ordered holdout: 12,475 training, 100 validation, 525 test utterances |
| Initialization | Random; no author-released generator weights |
| Training seed | 1234 |
| Hardware | NVIDIA L40S |
| Precision | BF16 mixed precision |
| Batch size | 32 |
| Training budget | 320 epochs; 124,800 optimizer steps |
| Wall-clock time | 17,430.6 seconds |
| Status | Completed |

Source audio is resampled before 16,384-sample training crops are formed, and training-only random peak gain is sampled between -6 and -1 dB. Separate AdamW optimizers use per-step cosine schedules. Training metrics were logged every five steps, validation was performed every 1,000 steps, and durable checkpoints were written every 5,000 steps.

## Selected Checkpoint

All accepted adaptive-benchmark executions evaluate `last.ckpt` at epoch 307 and step 120,000. The separately written validation-best checkpoint represents the same epoch and global step, and all 406 model tensors are identical. The lower validation value at step 124,000 occurs after the final scheduled durable boundary and has no checkpoint.


## Files

| File | Purpose |
|---|---|
| `config.yaml` | Resolved training, data, architecture, optimization, and checkpoint configuration. |
| `execution.log` | Complete log from the accepted training execution. |
| `metrics.csv` | Source-ordered training and validation metric history. |
