# Training

Every configuration was fitted through one recorded-seed training trajectory under its native objective and a finite registered budget. Training execution is implemented by `ReproductionTrainingRunner` in `code/vocode/trainers/reproduction.py` on top of the `syntheticmind` trainer.

## Execution

All models are PyTorch modules executed through the SyntheticMind harness, which standardizes fitting, validation, metric reduction, and checkpoint restore. The adversarial vocoders use the framework's manual-optimization path for their alternating generator and discriminator updates. The recorded training seed is 1234; the accepted runs predate the repair that seeds model construction and workers deterministically, so 1234 is a recorded rather than reconstructible seed, retained states are immutable, and bitwise regeneration is not claimed.

Eleven configurations trained on a 48 GB NVIDIA L40S and RFWave on an 80 GB NVIDIA A100, scheduled as isolated single-device jobs on Modal. Recorded training sums to 75.31 GPU hours as a lower bound. Architecture-native rather than equalized budgets make the retained states controlled project realizations, not attainable quality ceilings.

## Durable Checkpoints and Resumability

Durable checkpoints were written every 5,000 updates by the framework's checkpoint callback. A durable checkpoint carries model, optimizer, scheduler, callback, datamodule, epoch, step, and random-number-generator state, so an interrupted run continues from its last durable state rather than restarting; the HiFi-GAN V1 and V3 budgets include 4,800 replayed updates after one such durable resume.

## The Registered Selection Gate

Checkpoint selection followed a registered gate rather than a retrospective search: validation loss informed continuation, a gain below 0.02 PESQ across gates indicated a plateau, and every run kept a registered ceiling. Where a commensurate published or released-checkpoint PESQ anchor existed, the registered quality target was 93 percent of that anchor; RFWave used a registered PESQ floor of 3.55; LPCNet and VocosFormer had no commensurate anchor. For each configuration, evaluation used the terminal durable checkpoint admitted by this gate, the Retained Project Checkpoint; the validation-best, global-minimum, and evaluated states remain identified in the resolved records.

## Records

Each model's training trajectory is preserved under `artifacts/study_1/models/<architecture>/train/`: the resolved `config.yaml`, the appended `execution.log`, the per-step `metrics.csv` with every logged loss term, and a rendered convergence figure under `figures/`.
