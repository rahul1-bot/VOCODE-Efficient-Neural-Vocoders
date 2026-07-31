# vocode.metrics: The Measurement Stack

`metrics/` implements every measured quantity of the study as an atomic, independently testable module, together with the registry that selects and orders the measurements. All configurations are measured by these implementations, so cross-model numbers share one evaluator by construction. Reference and synthesized waveforms are truncated to their common unpadded length before resampling, so no quality proxy ever scores padding.

| Module | Measurement |
|---|---|
| `pesq.py` | `Pesq` computes wideband PESQ at 16 kHz, the primary intrusive quality proxy. |
| `stoi.py` | `Stoi` computes classic non-extended STOI at 10 kHz. |
| `utmos.py` | `UtmosPredictor` computes reference-free UTMOS through the pinned torch.hub release `tarepan/SpeechMOS:v1.2.0`, loaded lazily on first use. |
| `mel.py` | `MelError` computes the mel-spectrogram L1 error. |
| `stft.py` | `MultiResolutionStftError` computes the multi-resolution STFT error. |
| `mcd.py` | `MelCepstralDistortion` computes mel-cepstral distortion. |
| `las.py` | `LogAmplitudeSpectrumRmse` computes the log-amplitude spectrum RMSE. |
| `pitch.py` | `PitchExtractor` computes the shared pitch features consumed by the pitch-domain diagnostics. |
| `f0.py` | `F0Rmse` computes the fundamental-frequency RMSE over the shared pitch features. |
| `periodicity.py` | `PeriodicityRmse` computes the periodicity RMSE over the shared pitch features. |
| `voicing.py` | `VoicingF1` computes the voicing-decision F1 score over the shared pitch features. |
| `rtf.py` | `RealTimeFactorMonitor` measures the warm real-time factor and per-utterance latency from synchronized timed calls. |
| `macs.py` | `MacsProfiler` counts multiply-accumulate operations from one traced second of audio through a `SynthesisProbe`. |
| `parameters.py` | `ParameterCount` counts deployable parameters. |
| `size.py` | `ModelSize` measures the serialized artifact size. |
| `registry.py` | `MetricRegistry` and `MetricSelection` define the closed metric vocabulary and the ordered, validated selection for a run. |
| `sequence.py` | `MetricSequence` executes the selected metrics over synthesized and reference waveforms and assembles the measured values. |

The three pitch-domain diagnostics consume one shared extraction from `PitchExtractor`, so they measure the same pitch analysis rather than three divergent ones. PESQ, STOI, and UTMOS are the quality proxies reported in the study; the spectral and prosodic errors serve as diagnostics only.

## Related Components

The assembled values flow into `../loggers/`, which persists them into run records; the curated registers under `../../../artifacts/` and the results chapter of the report are derived from those records. The mirrored tests live in `../../tests/vocode/metrics/`.
