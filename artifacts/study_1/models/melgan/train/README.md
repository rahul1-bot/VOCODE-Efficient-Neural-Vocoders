# MelGAN Training

This directory contains the accepted Study 1 training evidence for MelGAN. The model was trained from random initialization on the deterministic LJSpeech training partition using seed 1234.

## Experiment

| Field | Value |
|---|---|
| Run identifier | `melgan_full_bf16bs32_20260704_0530` |
| Architecture | MelGAN (`seungwon`) |
| Dataset | LJSpeech 1.1 |
| Partition | Identifier-ordered holdout: 12,475 training, 100 validation, 525 test utterances |
| Initialization | Random; no author-released generator weights |
| Training seed | 1234 |
| Hardware | NVIDIA L40S |
| Precision | BF16 mixed precision |
| Batch size | 32 |
| Training budget | 320 epochs; 124,800 optimizer steps |
| Wall-clock time | 10,702.1 seconds |
| Status | Completed |

MelGAN uses one Adam optimizer for the generator and one for the three-scale discriminator. The objective combines least-squares adversarial losses with feature matching weighted by 10. Training metrics were logged every five steps, validation was performed every 1,000 steps, and durable checkpoints were written every 5,000 steps.

## Selected Checkpoint

All accepted adaptive-benchmark executions evaluate `last.ckpt` at epoch 307 and step 120,000. The saved validation-best boundary is epoch 192 and step 75,000; 189 of 191 model tensors differ from the evaluated state. The global validation minimum at step 109,000 falls between checkpoint boundaries and has no saved checkpoint.


## Files

| File | Purpose |
|---|---|
| `config.yaml` | Resolved training, data, architecture, optimization, and checkpoint configuration. |
| `execution.log` | Complete log from the accepted training execution. |
| `metrics.csv` | Source-ordered training and validation metric history. |
