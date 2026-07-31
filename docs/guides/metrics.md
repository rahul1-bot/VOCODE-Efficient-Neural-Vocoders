# Metrics

Every measured quantity is implemented as an atomic module in `code/vocode/metrics/`, and all configurations are measured by these implementations, so cross-model numbers share one evaluator by construction.

## Quality Proxies

| Proxy | Operating point | Implementation |
|---|---|---|
| PESQ | Wideband at 16 kHz | `pesq.py`, the primary intrusive measure |
| STOI | Classic non-extended at 10 kHz | `stoi.py` |
| UTMOS | SpeechMOS UTMOS22 strong 1.2.0 at 16 kHz, reference-free | `utmos.py`, through the pinned torch.hub release |

Reference and synthesized waveforms are truncated to their common unpadded length before resampling; the report calls this true-length evaluation, and it guarantees that no quality proxy ever scores padding. PESQ was withdrawn by ITU-T in favor of the P.863 family yet remains the dominant quality proxy in the vocoder literature; the report treats all three proxies as proxies, not listener judgments.

For each artifact, execution values are averaged within each utterance and then over the fixed 525-item evaluation set, with inference seeds 42, 43, and 44 re-executing one frozen state. The spread across executions reflects execution variability, except for static ONNX INT8, whose per-execution recalibration also carries calibration-set variation.

## Diagnostics

Mel L1, multi-resolution STFT error, mel-cepstral distortion, log-amplitude spectrum RMSE, and the pitch-domain measures (F0 RMSE, periodicity RMSE, voicing F1 over one shared pitch extraction) are diagnostics only: architecture-native or alignment-sensitive values are not cross-model endpoints.

## Timing and Footprint

Two timing regimes exist and are never cross-compared. Pre-transformation baseline evaluation measures one complete pass at batch one on the NVIDIA B200 after five warm-up batches; the CPU column of the report's baseline table is a retained comparative record from three executions on an 8-core CPU host. Deployment evaluation measures three synchronized calls per utterance for the 520 post-warm-up utterances on B200 or CPU, each timed call following one untimed prediction of the same input, so those quantities describe warm repeated synthesis only. Reported p50 and p95 latencies are per-execution percentiles of the per-utterance three-repetition means, averaged over executions.

Footprint endpoints are the continuous process-RSS high-water mark, the one-time ONNX session-construction time, deployable parameter counts, serialized bytes, and multiply-accumulate operations from one traced second of audio. LPCNet's timing record covers 27 utterances in one execution, omits the authors' optimized sparse C path, and is excluded from strict speed frontiers.
