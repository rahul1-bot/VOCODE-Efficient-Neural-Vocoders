# RNDVoC Training

This directory contains the accepted Study 1 training evidence for RNDVoC. The model was trained from random initialization on the deterministic LJSpeech training partition using seed 1234.

## Experiment

| Field | Value |
|---|---|
| Run identifier | `rndvoc_full_bf16bs32_20260707_0341` |
| Architecture | RNDVoC (`andong_22k_ljspeech`) |
| Dataset | LJSpeech 1.1 |
| Partition | Identifier-ordered holdout: 12,475 training, 100 validation, 525 test utterances |
| Initialization | Random; no author-released model weights |
| Training seed | 1234 |
| Hardware | NVIDIA L40S |
| Precision | BF16 mixed precision |
| Batch size | 32 |
| Training budget | 320 epochs; 124,800 alternating optimizer steps |
| Generator updates | 62,400 |
| Wall-clock time | 15,491.9 seconds |
| Status | Completed |

The reference alternating-update discipline is preserved: discriminator and generator updates occur on opposite batch parities. The gate therefore contains 62,400 generator updates, equivalent to 12.5% of the reference generator-update scale after batch-size normalization. Training metrics were logged every five steps, validation was performed every 1,000 steps, and durable checkpoints were written every 5,000 steps.

## Selected Checkpoint

All accepted adaptive-benchmark executions evaluate `last.ckpt` at epoch 307 and step 120,000. The saved validation-best boundary is epoch 153 and step 60,000; 409 of 417 model tensors differ from the evaluated state. The global validation minimum at step 48,000 falls between checkpoint boundaries and has no saved checkpoint.


## Files

| File | Purpose |
|---|---|
| `config.yaml` | Resolved training, data, architecture, optimization, and checkpoint configuration. |
| `execution.log` | Complete log from the accepted training execution. |
| `metrics.csv` | Source-ordered training and validation metric history. |
