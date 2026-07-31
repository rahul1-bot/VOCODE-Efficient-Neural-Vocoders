# HiFi-GAN V2 Training

This directory contains the accepted Study 1 training evidence for HiFi-GAN V2. The model was trained from random initialization on LJSpeech using seed 1234; validation was performed during the same gate-one execution on the deterministic validation partition.

## Experiment

| Field | Value |
|---|---|
| Run identifier | hifigan_v2_full_bf16bs32_20260704_0520 |
| Architecture | HiFi-GAN V2 (v2) |
| Dataset | LJSpeech 1.1 at 22.05 kHz |
| Partition | Identifier-ordered holdout: 12,475 training, 100 validation, 525 test utterances |
| Initialization | Random; no author-released generator weights |
| Training seed | 1234 |
| Hardware | NVIDIA L40S |
| Precision | BF16 mixed precision |
| Batch size | 32 |
| Final global step | 124,800 |
| Executed optimizer updates | 124,800 |
| Trainer elapsed time | 18,334.5 seconds |
| Status | Completed and stopped by the registered gate-one rule |

Dataset membership is not randomized: records are sorted by utterance identifier, the final 525 form the adaptive evaluation partition, and the preceding 100 form the validation partition. Training uses shuffled 8,192-sample waveform segments after peak normalization to 0.95; validation uses complete, unshuffled utterances. Separate AdamW optimizers update the generator and discriminators, and their exponential schedulers advance once per epoch.

The metric table is a source-ordered scientific projection containing 24,960 train records, 125 validation records, and 320 learning-rate records. The table contains the scientific training and validation series; runtime-only telemetry is outside this measurement table. Every numeric value in the tracked scientific projection is finite.

## Selected Checkpoint

All three accepted adaptive-benchmark executions evaluate last.ckpt at epoch 307 and step 120,000.

The saved validation-best checkpoint occurs at epoch 269 and step 105,000. It differs from the evaluated state in 403 of 404 model tensors and therefore cannot reproduce the reported test rows. The global validation minimum of 0.3833152514696121 occurs at step 121,000, where no checkpoint was scheduled. The admitted evaluations use the step-120,000 state while this record distinguishes the validation-best and unsaved global-minimum boundaries.


## Files

| File | Purpose |
|---|---|
| config.yaml | Resolved data, architecture, optimization, gate, and checkpoint configuration. |
| execution.log | Complete source-equal gate-one training log. |
| metrics.csv | Source-ordered train, validation, and learning-rate metric history. |
