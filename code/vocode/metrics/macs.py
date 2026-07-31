# This module:
# 1. Profiles synthesis compute as giga-MACs per second of audio by tracing
#    one seeded probe synthesis through the fvcore JIT flop counter
# 2. Reports a structured status record (profiled, profiled_partial,
#    unsupported, failed) with the unsupported-operator inventory, so a
#    non-numeric outcome is recorded evidence rather than a silent absence
#
# Harness contract (syntheticmind):
# - The profiled object is a harness Module that additionally satisfies the
#   Vocoder protocol (network, mel protocol, synthesize); modules without
#   that traceable mel-conditioned contract report unsupported instead of
#   being forced through an unfaithful trace
#
# Design decisions:
# - The probe mel comes from seeded noise of a fixed duration through the
#   model's own conditioning protocol, so every architecture is profiled on
#   an input grid it actually accepts
# - The synthesis call is wrapped in a plain torch.nn.Module probe because
#   the tracer requires a module boundary, not a bound method
# - A custom handle counts aten::_convolution_mode (used by same-padding
#   convolutions) with the standard convolution arithmetic, which fvcore
#   does not cover by default
# - Remaining unsupported operators downgrade the status to
#   profiled_partial and are listed by name, keeping partial counts honest
#
# Author: Rahul Sawhney

from collections import Counter
from typing import ClassVar, Literal, cast

import torch
from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import conv_flop_count, get_shape
from pydantic import BaseModel, ConfigDict, PositiveFloat

from syntheticmind.core.module import Module

from vocode.models.vocoder import Vocoder
from vocode.transforms.mel import MelConfig, MelSpectrogram

__all__: list[str] = [
    "MacsProfileResult",
    "MacsProfileStatus",
    "MacsProfiler",
    "MacsProfilerConfig"
]

type MacsProfileStatus = Literal[
    "profiled",
    "profiled_partial",
    "unsupported",
    "failed"
]


class MacsProfilerConfig(BaseModel):
    # Frozen probe settings: the audio duration the count normalizes over
    # and the seed of the probe waveform.
    #
    # Fields:
    #     audio_seconds: Duration of audio the probe synthesizes, serving
    #         both as the length of the generated conditioning and as the
    #         denominator the raw count is normalized by. Because the
    #         reported quantity is a rate, this value changes the total
    #         count but not the normalized count. Default: ``1.0``.
    #     probe_seed: Seed of the generator drawing the probe waveform, so
    #         repeated profiles of one module reproduce the same count.
    #         Default: ``0``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    audio_seconds: PositiveFloat = 1.0
    probe_seed: int = 0


class MacsProfileResult(BaseModel):
    # Frozen profiling outcome: status, normalized and total counts, probe
    # geometry, the unsupported-operator inventory, and the failure reason
    # when one exists.
    #
    # The record is designed so that a non-numeric outcome is still
    # evidence. The four numeric fields are populated together or not at
    # all: an unsupported or failed profile leaves every one of them
    # ``None`` rather than defaulting to zero, because a zero compute cost
    # is a claim about the model whereas ``None`` is an honest absence of
    # measurement. The metric sequence serializes the whole record into the
    # run log before deciding whether a column can be reported.
    #
    # Fields:
    #     status: The structured outcome. ``"profiled"`` means every traced
    #         operator was counted; ``"profiled_partial"`` means the count
    #         is an undercount and names what it omitted;
    #         ``"unsupported"`` means the module exposes no traceable
    #         mel-conditioned synthesis surface; ``"failed"`` means probe
    #         construction or tracing raised.
    #     method: The measurement technique, recorded so a published count
    #         states how it was obtained rather than leaving the reader to
    #         infer it. Default: ``"fvcore_jit_trace"``.
    #     giga_macs_per_audio_second: The total count divided by the probe
    #         duration and by one billion, which is the comparable rate the
    #         study reports. ``None`` for a non-numeric outcome.
    #     total_macs: Raw multiply-accumulate count over the traced probe
    #         synthesis, before normalization. ``None`` for a non-numeric
    #         outcome.
    #     audio_seconds: The probe duration the count was normalized by,
    #         travelling with the result so the rate stays interpretable
    #         without consulting the configuration. ``None`` for a
    #         non-numeric outcome.
    #     mel_frames: Conditioning frames in the probe mel, recording the
    #         input geometry the trace actually ran on. ``None`` for a
    #         non-numeric outcome.
    #     unsupported_operators: Alphabetically sorted names of traced
    #         operators the counter has no handler for and therefore
    #         charged nothing. Non-empty exactly when the status is
    #         ``"profiled_partial"``.
    #     reason: The machine-readable cause under ``"unsupported"``, or
    #         the exception type and message under ``"failed"``. ``None``
    #         whenever a count was produced.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    status: MacsProfileStatus
    method: Literal["fvcore_jit_trace"] = "fvcore_jit_trace"
    giga_macs_per_audio_second: float | None
    total_macs: float | None
    audio_seconds: float | None
    mel_frames: int | None
    unsupported_operators: tuple[str, ...]
    reason: str | None


class SynthesisProbe(torch.nn.Module):
    # Minimal module boundary around the vocoder's synthesize call, giving
    # the JIT tracer the module interface it requires.
    def __init__(self, vocoder: Vocoder) -> None:
        # Binds the vocoder whose synthesis the probe exposes.
        super().__init__()
        self._vocoder: Vocoder = vocoder

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Forwards the conditioning mel through the vocoder's synthesis.
        return self._vocoder.synthesize(mel)


class MacsProfiler:
    # Trace-based synthesis compute profiler with structured outcomes.
    #
    # The mechanism is a single traced probe synthesis rather than an
    # analytic walk over layer geometry, so an architecture is charged for
    # the arithmetic it actually executes, including control flow the
    # declared module structure does not reveal. The probe conditioning is
    # built from seeded noise of the configured duration through the
    # model's own mel protocol, which means every architecture is profiled
    # on an input grid it genuinely accepts instead of a grid imposed by
    # the profiler. Because the probe is seeded and the trace is
    # deterministic, repeated profiles of one module agree exactly.
    #
    # The counted unit is the multiply-accumulate, not the floating-point
    # operation: a convolution is charged output_channels times
    # input_channels times kernel width per output frame, counting each
    # multiply-add once. The result is normalized to giga-MACs per second
    # of synthesized audio, which is a rate and therefore invariant to the
    # probe duration.
    #
    # Coverage is disclosed rather than assumed. The trace counter has a
    # handler per operator and silently charges nothing for operators it
    # does not recognize, so a bare total could be a silent undercount.
    # This profiler therefore adds a handler for the same-padding
    # convolution operator, which the counter omits by default and which
    # would otherwise erase a genuine convolution from the total; it then
    # collects whatever operators remain uncovered, lists them by name, and
    # downgrades the status to partial. The counter's own warning channels
    # are silenced precisely because that inventory replaces them with a
    # structured, recorded disclosure. An uncovered operator is not
    # automatically material, because an elementwise activation
    # legitimately costs no multiply-accumulates, but naming it lets a
    # reader judge the omission instead of trusting the total unexamined.
    #
    # No outcome escapes as an exception. A module without the traceable
    # contract and a synthesis that raises both return records, so the
    # caller always receives a decision it can log.
    def __init__(self, configuration: MacsProfilerConfig) -> None:
        # Binds the probe settings. The profiler holds no per-module state,
        # so one instance profiles any number of modules.
        self._configuration: MacsProfilerConfig = configuration

    def profile(self, module: Module) -> MacsProfileResult:
        # Profiles one module: contract check first (unsupported without a
        # traceable mel-conditioned synthesis surface), then a traced probe
        # synthesis under the flop counter with the custom convolution
        # handle; any tracing exception is captured as a failed record and
        # remaining unsupported operators downgrade the status to partial.
        #
        # Args:
        #     module: The harness module to profile. It must additionally
        #         satisfy the vocoder surface by exposing a network module,
        #         a mel protocol record, and a callable synthesis; all
        #         three are resolved by attribute lookup rather than by
        #         declared inheritance, so a structurally conforming module
        #         qualifies.
        #
        # Returns:
        #     A MacsProfileResult in one of the four statuses. Every path
        #     returns a record; no path raises, because a profiling outcome
        #     is evidence the run must be able to record even when no
        #     number was obtained.
        # Contract: a module without all three members is reported
        # unsupported rather than forced through an unfaithful trace.
        network: object = getattr(module, "network", None)
        mel_protocol: object = getattr(module, "mel_protocol", None)
        synthesize: object = getattr(module, "synthesize", None)
        if (
            not isinstance(network, torch.nn.Module)
            or not isinstance(mel_protocol, MelConfig)
            or not callable(synthesize)
        ):
            return MacsProfileResult(
                status="unsupported",
                giga_macs_per_audio_second=None,
                total_macs=None,
                audio_seconds=None,
                mel_frames=None,
                unsupported_operators=(),
                reason="module_has_no_traceable_mel_conditioned_synthesis_contract"
            )
        vocoder: Vocoder = cast(Vocoder, module)
        # Trace: the probe runs in evaluation mode without gradients, and
        # the counter's warning channels are silenced because the returned
        # inventory discloses uncovered operators structurally instead.
        try:
            mel: torch.Tensor = self._build_probe_mel(vocoder)
            probe: SynthesisProbe = SynthesisProbe(vocoder)
            probe.eval()
            with torch.no_grad():
                analysis: FlopCountAnalysis = FlopCountAnalysis(probe, (mel,))
                analysis.set_op_handle("aten::_convolution_mode", self._convolution_mode_macs)
                analysis.unsupported_ops_warnings(False)
                analysis.uncalled_modules_warnings(False)
                total_macs: float = float(analysis.total())
                unsupported_operators: tuple[str, ...] = tuple(
                    sorted(analysis.unsupported_ops().keys())
                )
        except Exception as caught:
            return MacsProfileResult(
                status="failed",
                giga_macs_per_audio_second=None,
                total_macs=None,
                audio_seconds=None,
                mel_frames=None,
                unsupported_operators=(),
                reason=f"{type(caught).__name__}: {caught}"
            )
        # Normalization: an uncovered operator makes the total an
        # undercount, which the status records rather than hides.
        audio_seconds: float = self._configuration.audio_seconds
        profile_status: MacsProfileStatus = (
            "profiled_partial" if unsupported_operators else "profiled"
        )
        return MacsProfileResult(
            status=profile_status,
            giga_macs_per_audio_second=total_macs / audio_seconds / 1.0e9,
            total_macs=total_macs,
            audio_seconds=audio_seconds,
            mel_frames=int(mel.shape[-1]),
            unsupported_operators=unsupported_operators,
            reason=None
        )

    @property
    def configuration(self) -> MacsProfilerConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration

    def _convolution_mode_macs(
        self,
        inputs: list[torch._C.Value],
        outputs: list[torch._C.Value]
    ) -> Counter[str]:
        # Counts one aten::_convolution_mode node with the standard
        # convolution arithmetic; nodes without shape information are a
        # hard error because a shapeless count would be fabricated.
        #
        # Same-padding convolutions trace to this operator rather than to
        # the ordinary convolution node, and the counter ships no handler
        # for it. Without this handle such a layer would contribute nothing
        # and merely appear in the uncovered inventory, understating a
        # genuinely expensive layer. The arithmetic delegated to here is
        # the same one the ordinary convolution handler applies, so the two
        # padding spellings of one layer produce identical counts.
        #
        # Args:
        #     inputs: Traced node inputs, whose first two entries are the
        #         activation and the weight.
        #     outputs: Traced node outputs, whose first entry carries the
        #         result geometry the count is taken over.
        #
        # Raises:
        #     ValueError: If the activation, weight, or result carries no
        #         recorded shape, since any count derived without geometry
        #         would be invented rather than measured.
        input_shape: list[int] | None = get_shape(inputs[0])
        weight_shape: list[int] | None = get_shape(inputs[1])
        output_shape: list[int] | None = get_shape(outputs[0])
        if input_shape is None or weight_shape is None or output_shape is None:
            raise ValueError("aten::_convolution_mode node carries no tensor shape information.")
        return cast(
            Counter[str],
            conv_flop_count(input_shape, weight_shape, output_shape, transposed=False)
        )

    def _build_probe_mel(self, vocoder: Vocoder) -> torch.Tensor:
        # Builds the probe conditioning: seeded noise of the configured
        # duration through the vocoder's own mel protocol, batched and
        # placed on the network's device.
        #
        # The noise is drawn from a private generator rather than the
        # global one, so profiling cannot perturb a run's random stream,
        # and it is attenuated to a speech-like amplitude before analysis.
        # Routing it through the model's declared protocol rather than a
        # fixed grid is what lets one profiler serve architectures that
        # condition on different mel geometries. A parameter-free network
        # leaves no device to read, so the probe stays on the processor.
        mel_transform: MelSpectrogram = MelSpectrogram(vocoder.mel_protocol)
        sample_count: int = int(round(vocoder.mel_protocol.sample_rate * self._configuration.audio_seconds))
        generator: torch.Generator = torch.Generator().manual_seed(self._configuration.probe_seed)
        waveform: torch.Tensor = torch.randn(1, sample_count, generator=generator) * 0.1
        mel: torch.Tensor = mel_transform(waveform)
        if mel.ndim == 2:
            mel: torch.Tensor = mel.unsqueeze(0)
        parameter: torch.nn.Parameter | None = next(iter(vocoder.network.parameters()), None)
        device: torch.device = parameter.device if parameter is not None else torch.device("cpu")
        return mel.to(device)
