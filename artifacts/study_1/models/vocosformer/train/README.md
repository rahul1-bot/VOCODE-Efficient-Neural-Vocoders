# VocosFormer Training Evidence

The accepted run is `vocosformer_full_bf16bs32_20260707_0940`: 320 epochs and 124,800 optimizer steps on an NVIDIA L40S using BF16 mixed precision, batch size 32, training seed 1234, and random initialization. The complete execution took 22,714.6 seconds.

All accepted evaluations load `last.ckpt`, the durable epoch-307, step-120,000 state. The run also retained the lower-loss saved checkpoint `checkpoint-epoch294-step115000.ckpt`; it was not used by the accepted adaptive-benchmark executions.
