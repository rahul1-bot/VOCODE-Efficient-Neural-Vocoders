# Study 1 — Budget-Controlled Training Reproduction

## Research Question

Can twelve representative neural-vocoder configurations produce useful waveform-synthesis behaviour when their weights are trained from random initialization on the same LJSpeech partition under explicit project compute budgets?

“Trained from scratch” describes weight initialization and optimization. It does not claim that the project invented the published architectures. Eleven rows are project-trained implementations or reimplementations of published model families; VocosFormer is the project label for a literature-derived, parameter-matched attention-augmented Vocos adaptation.

## Evidence Contract

Every executed model has one accepted seed-1234 training capsule and three evaluation capsules using inference seeds 42, 43, and 44. Each model directory follows the same compact contract: one model README, one experiment index, one training capsule, three seed-specific evaluation capsules stored under the stable `test/` schema path, and model-specific figures. No separate narrative folders are required.

Dataset membership is fixed by identifier order: 12,475 training utterances, 100 validation utterances, and 525 adaptive-evaluation utterances. This partition does not use a split seed. Its PESQ scores informed some continue/stop gates and an RFWave inference correction, so it is not an untouched final test set; the `test/` directory name is retained only for package-schema stability. Evaluation seeds repeat inference from the same retained checkpoint and are not independent training replicas. LPCNet has three complete quality executions but only seed 42 carries one PyTorch B200 RTF observation over 27 timed utterances after five warm-ups. That timing point is excluded from strict speed-frontier inference and is not comparable to optimized LPCNet C execution.

## Study 1 Measurements

Study 1 uses one explicit measurement set selected for its research question:

1. Perceptual and intelligibility evidence: PESQ and STOI, with true-length PESQ/STOI used where padded batches alter the result.
2. Spectral and signal diagnostics: model-protocol mel error, multi-resolution STFT error, MCD, F0 RMSE, periodicity RMSE, V/UV F1, UTMOS, and LAS-RMSE.
3. Efficiency and complexity evidence: real-time factor, trainable parameter count, serialized checkpoint size, and MACs per second of audio where the graph is traceable.

Parameter count and serialized size are complexity descriptors, not audio-quality metrics. Mel error is model-protocol dependent across 16 kHz, 22.05 kHz, and 24 kHz configurations and must not be ranked as if it were one cross-model instrument. MCD and LAS-RMSE are not used as headline RFWave comparators because suspected group delay invalidates a naive frame-aligned reading.

## Twelve-Model Result Set

| Model | Parameters | PESQ | STOI | Mel error | B200 RTF |
|---|---:|---:|---:|---:|---:|
| APNet2 | 31,425,539 | 2.541039 | 0.930045 | 0.543382 | 0.009451 |
| BigVGAN | 14,025,154 | 3.256982 | 0.973688 | 0.638439 | 0.157267 |
| FreeV | 18,220,557 | 3.339431 | 0.969317 | 0.174622 | 0.007677 |
| HiFi-GAN V1 | 13,936,130 | 3.317037 | 0.974800 | 0.270400 | 0.119740 |
| HiFi-GAN V2 | 928,514 | 2.456062 | 0.945346 | 0.393529 | 0.114028 |
| HiFi-GAN V3 | 1,464,322 | 2.576487 | 0.951972 | 0.360738 | 0.057986 |
| LPCNet | 1,232,992 | 1.077017 | 0.527543 | 2.189054 | 11.363275 (n=1) |
| MelGAN | 4,266,050 | 1.059750 | 0.418274 | 1.791550 | 0.074215 |
| RNDVoC | 3,572,361 | 3.856795 | 0.987090 | 0.138808 | 0.055900 |
| Vocos | 13,531,650 | 3.565251 | 0.979314 | 3.046101 | 0.004725 |
| VocosFormer | 13,778,690 | 2.479785 recorded / 3.512457 true-length | 0.958694 recorded / 0.975776 true-length | 2.910696 | 0.012170 |
| RFWave | 18,139,296 | 3.669211 recorded / 3.784551 true-length | 0.980810 recorded / 0.980918 true-length | 3.837596 | 0.014810 |

RNDVoC provides the strongest measured quality and Vocos the fastest B200 inference. HiFi-GAN V2 is the smallest model. RFWave reaches the second-highest recorded PESQ and joins the speed-quality Pareto frontier, while VocosFormer demonstrates near-parity true-length quality but slower inference and severe padding sensitivity relative to its matched Vocos control. No architecture dominates quality, speed, and footprint simultaneously.

## Scope Boundary

The executed cohort is exactly twelve models: APNet2, BigVGAN, FreeV, HiFi-GAN V1, HiFi-GAN V2, HiFi-GAN V3, LPCNet, MelGAN, RNDVoC, Vocos, VocosFormer, and RFWave. `models/hiftnet/` remains only as a documented exclusion because no admissible project-trained checkpoint was produced; it is not a thirteenth result. Half-width controls and post-training interventions belong outside Study 1.


Study 1 demonstrates the controlled project workflow at registered budgets. It does not imply author-scale convergence, equal training compute across architectures, or strict comparability to paper values measured under different datasets, sample rates, checkpoints, or evaluators. The figures visualize only the frozen measurements and training histories reported in this package.
