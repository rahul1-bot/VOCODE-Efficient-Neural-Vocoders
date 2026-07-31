# LPCNet Training

This directory contains the accepted Study 1 training evidence for LPCNet. The model was trained from random initialization on the deterministic LJSpeech training partition using seed 1234.

## Experiment

| Field | Value |
|---|---|
| Run identifier | `lpcnet_full_fp32bs128_20260707_0326` |
| Architecture | LPCNet (`xiph_reference_ljspeech`) |
| Dataset | LJSpeech 1.1, resampled to 16 kHz |
| Partition | Identifier-ordered holdout: 12,475 training, 100 validation, 525 test utterances |
| Initialization | Random; no author-released model weights |
| Training seed | 1234 |
| Hardware | NVIDIA L40S |
| Precision | FP32 |
| Batch size | 128 |
| Training budget | 319 epochs; 31,262 executed optimizer steps |
| Wall-clock time | 8,382.5 seconds |
| Status | Completed |

The 2,000-to-20,000-step GRU-A sparsification schedule completed within the registered budget. The execution finished at step 31,262, while the last durable checkpoint was written at epoch 306 and step 30,000.

## Selected Checkpoint

All accepted adaptive-benchmark executions evaluate `last.ckpt` at epoch 306 and step 30,000. The separately written validation-best checkpoint has the same epoch and global step and contains the same 26 model tensors, so it represents the same learned state.


## Files

| File | Purpose |
|---|---|
| `config.yaml` | Resolved training, data, architecture, optimization, and checkpoint configuration. |
| `execution.log` | Complete log from the accepted training execution. |
| `metrics.csv` | Source-ordered training and validation metric history. |
