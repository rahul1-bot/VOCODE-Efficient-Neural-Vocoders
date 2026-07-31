# FreeV Training

This directory contains the accepted Study 1 training evidence for FreeV. The model was trained from random initialization on LJSpeech using training seed 1234; validation was performed during the same execution on the deterministic validation partition.

## Experiment

| Field | Value |
|---|---|
| Run identifier | `freev_full_bf16bs32_20260704_1215` |
| Architecture | FreeV (`official_ljspeech`) |
| Dataset | LJSpeech 1.1 at 22.05 kHz |
| Partition | Identifier-ordered holdout: 12,475 training, 100 validation, 525 test utterances |
| Initialization | Random; no author-released generator weights |
| Training seed | 1234 |
| Hardware | NVIDIA L40S |
| Precision | BF16 mixed precision |
| Batch size | 32 |
| Training budget | 320 epochs; 124,800 optimizer steps |
| Wall-clock time | 9,078.9 seconds |
| Status | Completed |

Dataset membership is not randomized: records are sorted by utterance identifier, the final 525 form the adaptive evaluation partition, and the preceding 100 form the validation partition. Training uses shuffled 8,192-sample waveform segments at the native LJSpeech sample rate. The pseudo-inverse mel projection is an architectural amplitude prior computed inside the generator; it is not a pretrained parameter source. Training metrics were logged every five steps, validation was performed every 1,000 steps, and durable checkpoints were written every 5,000 steps.

## Best Checkpoint

The minimum validation loss among saved checkpoint boundaries was 442.2043695068359 at epoch 307 and step 120,000. That selected state supplied the admitted adaptive-benchmark executions.

The training history continued to a lower validation value of 441.706643371582 at step 124,000, but no checkpoint was scheduled at that non-5,000-step boundary. The accepted adaptive-benchmark executions used the separately written `last.ckpt` at epoch 307 and step 120,000. Its 252 model tensors are exactly equal to the retained best checkpoint, so `best.ckpt` reproduces the evaluated model state.


## Files

| File | Purpose |
|---|---|
| `config.yaml` | Resolved training, data, architecture, optimization, and checkpoint-selection configuration. |
| `execution.log` | Complete log from the accepted training execution. |
| `metrics.csv` | Step-indexed training and validation metric history. |
