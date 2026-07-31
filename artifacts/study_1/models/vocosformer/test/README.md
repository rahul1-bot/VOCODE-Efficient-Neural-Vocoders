# VocosFormer Test Evidence

`seed_42`, `seed_43`, and `seed_44` contain the complete B200 FP32 evaluations of the same retained checkpoint over the fixed 525-utterance adaptive evaluation partition. The recorded batch-16 PESQ lane is padding-sensitive for this global-attention configuration; each capsule therefore also preserves the true-length batch-1 quality measurement from the same selected state.
