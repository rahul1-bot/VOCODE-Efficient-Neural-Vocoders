# APNet2 Training

This directory contains the accepted Study 1 training evidence for APNet2. The model was trained from random initialization on LJSpeech using training seed 1234; validation was performed during the same execution on the deterministic validation partition.

## Experiment

| Field | Value |
|---|---|
| Run identifier | `apnet2_full_bf16bs32_20260704_1005` |
| Architecture | APNet2 (`redmist328_ljspeech`) |
| Dataset | LJSpeech 1.1 |
| Partition | Identifier-ordered holdout: 12,475 training, 100 validation, 525 test utterances |
| Initialization | Random; no published generator weights |
| Training seed | 1234 |
| Hardware | NVIDIA L40S |
| Precision | BF16 mixed precision |
| Batch size | 32 |
| Training budget | 320 epochs; 124,800 optimizer steps |
| Status | Completed |

Dataset membership is not randomized: records are sorted by utterance identifier, the final 525 form the adaptive evaluation partition, and the preceding 100 form the validation partition. The execution used two AdamW optimizers, one for the generator and one for the discriminators. Training metrics were logged every five steps, validation was performed every 1,000 steps, and checkpoint selection was evaluated every 5,000 steps.

## Best Checkpoint

The minimum validation loss among saved checkpoint boundaries was 443.45948486328126 at epoch 307 and step 120,000. That selected state supplied the admitted adaptive-benchmark executions.


## Files

| File | Purpose |
|---|---|
| `config.yaml` | Resolved training, data, architecture, optimization, and checkpoint-selection configuration. |
| `execution.log` | Complete log from the accepted training execution. |
| `metrics.csv` | Step-indexed training and validation metric history. |
