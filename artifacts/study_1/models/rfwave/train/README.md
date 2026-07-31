# RFWave Training Evidence

The accepted run is `rfwave_full_bf16bs64_20260707_2110`: 641 epochs and 124,995 optimizer steps on an NVIDIA A100 80 GB using BF16 mixed precision, batch size 64, training seed 1234, and random initialization. The complete execution took 27,061.1 seconds.

All accepted evaluations load `last.ckpt`, the durable epoch-615, step-120,000 state. The validation-best saved checkpoint is `checkpoint-epoch487-step95000.ckpt`; it was not used by the accepted adaptive-benchmark executions.
