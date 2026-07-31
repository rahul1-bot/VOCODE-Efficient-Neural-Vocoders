# HiFi-GAN V3 Training

This directory contains the accepted Study 1 training evidence for HiFi-GAN V3. The model was trained from random initialization on LJSpeech using seed 1234; validation was performed during the same two-gate run identity on the deterministic validation partition.

## Experiment

| Field | Value |
|---|---|
| Run identifier | hifigan_v3_full_bf16bs32_20260704_0520 |
| Architecture | HiFi-GAN V3 (v3) |
| Dataset | LJSpeech 1.1 at 22.05 kHz |
| Partition | Identifier-ordered holdout: 12,475 training, 100 validation, 525 test utterances |
| Initialization | Random; no author-released generator weights |
| Training seed | 1234 |
| Hardware | NVIDIA L40S |
| Precision | BF16 mixed precision |
| Batch size | 32 |
| Final global step | 249,990 |
| Executed optimizer updates | 254,790 across both training calls |
| Trainer elapsed time | 37,624.9 seconds across both gates |
| Status | Completed at the registered gate-two ceiling |

Dataset membership is not randomized: records are sorted by utterance identifier, the final 525 form the adaptive evaluation partition, and the preceding 100 form the validation partition. Training uses shuffled 8,192-sample waveform segments after peak normalization to 0.95; validation uses complete, unshuffled utterances. Separate AdamW optimizers update the generator and discriminators, and their exponential schedulers advance once per epoch.

Gate 1 reached global step 124,800. Because checkpoints were written every 5,000 steps, gate 2 resumed from the durable step-120,000 state and replayed 4,800 optimizer updates before continuing to step 249,990. The metric table preserves this history in source order: it contains 50,958 train records, 255 validation records, and 654 learning-rate records. The repeated step range comprises 960 train records, 5 validation records, and 13 learning-rate records and is intentional evidence of the resume boundary.

## Selected Checkpoint

All three accepted adaptive-benchmark executions evaluate last.ckpt at epoch 628 and step 245,000.

The saved validation-best checkpoint occurs at epoch 589 and step 230,000. It differs from the evaluated state in 238 of 239 model tensors and therefore cannot reproduce the reported test rows. The global validation minimum of 0.3556865441799164 occurs at step 226,000, where no checkpoint was scheduled. The admitted evaluations use the step-245,000 state while this record distinguishes the validation-best and unsaved global-minimum boundaries.


## Files

| File | Purpose |
|---|---|
| config.yaml | Resolved data, architecture, optimization, gate, and checkpoint configuration. |
| execution.log | Complete source-equal log spanning both training gates. |
| metrics.csv | Source-ordered train, validation, and learning-rate metric history. |
