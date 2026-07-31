# This module:
# 1. Verifies the traced multiply-accumulate count over a minimal
#    convolutional synthesis probe whose arithmetic is known exactly from
#    its geometry
# 2. Verifies the normalization onto giga-MACs per audio second and the
#    recorded probe geometry
# 3. Verifies the custom same-padding convolution handle, which fvcore does not
#    cover by default
# 4. Verifies the four structured outcomes: profiled, profiled_partial with the
#    unsupported-operator inventory, unsupported for a module without the
#    traceable synthesis contract, and failed for a synthesis that raises
#
# Design decisions:
# - The probe vocoders condition on a deliberately small mel protocol (one
#   kilohertz, sixty-four-point FFT, eight bands) over a quarter second, so the
#   traced graph stays minimal while the counted arithmetic remains exact
# - Counts are asserted against the convolution identity
#   output_channels * input_channels * kernel * frames rather than against a
#   recorded number, so the assertion states why the count is what it is
# - The fakes satisfy the Vocoder protocol structurally rather than by explicit
#   inheritance, because the profiler resolves the contract through attribute
#   lookup and a Protocol base would displace torch.nn.Module initialization
# - No real project recipe is profiled here; the heavy architectures are
#   construction-cost boundaries and their arithmetic is not what this module
#   verifies
#
# Author: Rahul Sawhney

import unittest

import torch
from pydantic import ValidationError
from torch import nn

from syntheticmind.core.module import Module

from vocode.metrics.macs import MacsProfiler, MacsProfilerConfig, MacsProfileResult, MacsProfileStatus
from vocode.transforms.mel import MelConfig


class ProbeMelProtocol:
    # Builds the deliberately small mel protocol the probe vocoders condition
    # on, keeping the traced graph and the probe waveform minimal.
    #
    # A one-kilohertz rate over a quarter second is 250 samples, which at a
    # sixteen-sample hop under centered framing yields sixteen conditioning
    # frames. Every count assertion in this module is written against that
    # frame number times the convolution geometry, so the protocol is
    # small enough to keep the trace instant while still producing an
    # arithmetic total that is exact and derivable by hand.
    def create(self) -> MelConfig:
        # Produces the one-kilohertz eight-band protocol the probes condition on.
        return MelConfig(
            sample_rate=1000,
            n_fft=64,
            hop_length=16,
            win_length=64,
            n_mels=8,
            fmin=0.0,
            fmax=None,
            mel_scale="htk",
            center=True,
            pad_mode="reflect",
            power=1.0,
            normalize_mel_basis=False,
            log_clamp_min=1e-5,
            log_base="natural"
        )


class ConvolutionVocoder(Module):
    # Traceable probe vocoder whose synthesis is one injected convolution, so
    # its multiply-accumulate count follows directly from the layer geometry.
    #
    # This is the fully supported probe: every operator in its traced graph
    # has a counting handler, so it must earn the unqualified status with
    # an empty uncovered inventory. Injecting the convolution rather than
    # fixing it lets one class serve the width, padding, and
    # proportionality comparisons.
    #
    # The four probe modules in this file satisfy the vocoder surface
    # structurally rather than by inheriting a protocol, because the
    # profiler resolves that surface through attribute lookup and a
    # Protocol base would displace the harness module initialization these
    # classes need.
    def __init__(self, mel_protocol: MelConfig, network: nn.Module) -> None:
        # Binds the conditioning protocol and the traced synthesis network.
        super().__init__()
        self._mel_protocol: MelConfig = mel_protocol
        self._network: nn.Module = network

    def synthesize(self, mel: torch.Tensor) -> torch.Tensor:
        # Applies the single convolution, so the traced arithmetic is exactly
        # the layer geometry.
        return self._network(mel)

    @property
    def network(self) -> nn.Module:
        # Returns the network the profiler traces.
        return self._network

    @property
    def mel_protocol(self) -> MelConfig:
        # Returns the protocol the probe waveform is built on.
        return self._mel_protocol

    @property
    def metric_mel_protocol(self) -> MelConfig:
        # Returns the same protocol under the name the metric layer reads.
        return self._mel_protocol


class SaturatingVocoder(Module):
    # Probe vocoder whose synthesis ends in a hyperbolic tangent, an operator
    # fvcore neither counts nor ignores, so the profile downgrades to partial.
    #
    # The activation is the sharpest available demonstration that a partial
    # count is still an honest count: it genuinely costs no
    # multiply-accumulates, so the convolution total must be unchanged from
    # the fully supported probe while the status and the inventory both
    # record that something went uncounted.
    def __init__(self, mel_protocol: MelConfig, network: nn.Module) -> None:
        # Binds the conditioning protocol and the traced synthesis network.
        super().__init__()
        self._mel_protocol: MelConfig = mel_protocol
        self._network: nn.Module = network

    def synthesize(self, mel: torch.Tensor) -> torch.Tensor:
        # Ends in the uncounted activation, leaving one operator outside the
        # profiler's coverage.
        return torch.tanh(self._network(mel))

    @property
    def network(self) -> nn.Module:
        # Returns the network the profiler traces.
        return self._network

    @property
    def mel_protocol(self) -> MelConfig:
        # Returns the protocol the probe waveform is built on.
        return self._mel_protocol

    @property
    def metric_mel_protocol(self) -> MelConfig:
        # Returns the same protocol under the name the metric layer reads.
        return self._mel_protocol


class RejectingVocoder(Module):
    # Probe vocoder whose synthesis raises, exercising the captured-failure path.
    def __init__(self, mel_protocol: MelConfig, network: nn.Module) -> None:
        # Binds the conditioning protocol and the traced synthesis network.
        super().__init__()
        self._mel_protocol: MelConfig = mel_protocol
        self._network: nn.Module = network

    def synthesize(self, mel: torch.Tensor) -> torch.Tensor:
        # Refuses the probe, naming the shape it received so the recorded
        # failure reason is traceable.
        raise ValueError(f"probe of {tuple(mel.shape)} rejected")

    @property
    def network(self) -> nn.Module:
        # Returns the network the profiler traces.
        return self._network

    @property
    def mel_protocol(self) -> MelConfig:
        # Returns the protocol the probe waveform is built on.
        return self._mel_protocol

    @property
    def metric_mel_protocol(self) -> MelConfig:
        # Returns the same protocol under the name the metric layer reads.
        return self._mel_protocol


class ContractlessModule(Module):
    # Harness module exposing none of the synthesis contract, so the profiler
    # must report unsupported rather than force an unfaithful trace.
    def __init__(self) -> None:
        # Initializes the harness module and declares nothing further.
        super().__init__()


class ConditioningOnlyModule(Module):
    # Module exposing a network and a mel protocol but no callable synthesis,
    # so the contract check fails on the synthesis member alone.
    #
    # Where ContractlessModule proves the check rejects a module missing
    # everything, this one proves the check is a conjunction: two of the
    # three members are present and correctly typed, and the profile must
    # still report unsupported. The synthesis member is present but
    # uncallable rather than absent, which exercises the callability test
    # specifically instead of the attribute-presence test again.
    def __init__(self, mel_protocol: MelConfig, network: nn.Module) -> None:
        # Binds the conditioning protocol and the network, then shadows the
        # synthesis member with an integer so the contract check meets a
        # present but uncallable attribute.
        super().__init__()
        self._mel_protocol: MelConfig = mel_protocol
        self._network: nn.Module = network
        self.synthesize: int = 0

    @property
    def network(self) -> nn.Module:
        # Returns the network the profiler would have traced.
        return self._network

    @property
    def mel_protocol(self) -> MelConfig:
        # Returns the protocol the probe waveform would have been built on.
        return self._mel_protocol


class MacsProfilerConfigurationTest(unittest.TestCase):
    # Verifies the frozen probe settings and their validation boundaries.
    def test_default_probe_normalizes_over_one_audio_second(self) -> None:
        # The default denominator is one second of synthesized audio.
        self.assertEqual(MacsProfilerConfig().audio_seconds, 1.0)

    def test_default_probe_seed_is_fixed(self) -> None:
        # A fixed seed makes the probe waveform reproducible across runs.
        self.assertEqual(MacsProfilerConfig().probe_seed, 0)

    def test_configuration_is_frozen(self) -> None:
        # The probe protocol cannot drift while a profile is being taken.
        configuration: MacsProfilerConfig = MacsProfilerConfig()
        with self.assertRaises(ValidationError):
            configuration.audio_seconds: float = 2.0

    def test_zero_audio_duration_is_rejected(self) -> None:
        # A zero denominator would make the normalized count undefined.
        with self.assertRaises(ValidationError):
            MacsProfilerConfig(audio_seconds=0.0)

    def test_negative_audio_duration_is_rejected(self) -> None:
        # Negative audio is not a measurable quantity.
        with self.assertRaises(ValidationError):
            MacsProfilerConfig(audio_seconds=-1.0)

    def test_extra_fields_are_rejected(self) -> None:
        # A misspelled setting must fail loudly rather than be ignored.
        with self.assertRaises(ValidationError):
            MacsProfilerConfig(audio_second=1.0)

    def test_profiler_exposes_the_bound_configuration(self) -> None:
        # The profiler reports the exact settings it was constructed with.
        configuration: MacsProfilerConfig = MacsProfilerConfig(audio_seconds=0.25)
        self.assertIs(MacsProfiler(configuration).configuration, configuration)


class MacsProfileCountingTest(unittest.TestCase):
    # Verifies the counted arithmetic, its normalization, and the recorded
    # probe geometry over a single-convolution synthesis.
    #
    # Counts are asserted against the convolution identity of output
    # channels times input channels times kernel width times frames, rather
    # than against a number recorded from an earlier run. Stating the
    # identity makes the assertion explain why the count is what it is, and
    # keeps it valid if the probe geometry is ever retuned.
    def setUp(self) -> None:
        # Prepares a quarter-second probe and an eight-to-four convolution
        # whose multiply-accumulate count is known from its geometry.
        self._mel_protocol: MelConfig = ProbeMelProtocol().create()
        self._profiler: MacsProfiler = MacsProfiler(
            MacsProfilerConfig(audio_seconds=0.25, probe_seed=0)
        )
        self._vocoder: ConvolutionVocoder = ConvolutionVocoder(
            self._mel_protocol,
            nn.Conv1d(8, 4, kernel_size=3, padding=1)
        )

    def test_convolution_synthesis_counts_the_standard_multiply_accumulates(self) -> None:
        # A convolution costs out_channels * in_channels * kernel per output frame.
        result: MacsProfileResult = self._profiler.profile(self._vocoder)
        self.assertIsNotNone(result.mel_frames)
        self.assertEqual(result.total_macs, float(4 * 8 * 3 * result.mel_frames))

    def test_probe_geometry_records_the_conditioning_frame_count(self) -> None:
        # A quarter second at one kilohertz over a sixteen-sample hop is sixteen centered frames.
        result: MacsProfileResult = self._profiler.profile(self._vocoder)
        self.assertEqual(result.mel_frames, 250 // 16 + 1)

    def test_probe_records_the_audio_duration_it_normalized_over(self) -> None:
        # The denominator travels with the result so the count is interpretable.
        result: MacsProfileResult = self._profiler.profile(self._vocoder)
        self.assertEqual(result.audio_seconds, 0.25)

    def test_normalized_count_divides_the_total_by_the_probe_duration(self) -> None:
        # Giga-MACs per audio second is the total over the duration over one billion.
        result: MacsProfileResult = self._profiler.profile(self._vocoder)
        self.assertIsNotNone(result.total_macs)
        self.assertIsNotNone(result.audio_seconds)
        self.assertAlmostEqual(
            result.giga_macs_per_audio_second,
            result.total_macs / result.audio_seconds / 1.0e9,
            places=12
        )

    def test_wider_synthesis_costs_proportionally_more(self) -> None:
        # Doubling the output channels doubles the counted arithmetic.
        wide_vocoder: ConvolutionVocoder = ConvolutionVocoder(
            self._mel_protocol,
            nn.Conv1d(8, 8, kernel_size=3, padding=1)
        )
        narrow_result: MacsProfileResult = self._profiler.profile(self._vocoder)
        wide_result: MacsProfileResult = self._profiler.profile(wide_vocoder)
        self.assertEqual(wide_result.total_macs, 2.0 * narrow_result.total_macs)

    def test_longer_probe_raises_the_total_but_not_the_normalized_count(self) -> None:
        # The normalized count is a rate and must be invariant to probe length.
        long_profiler: MacsProfiler = MacsProfiler(
            MacsProfilerConfig(audio_seconds=0.5, probe_seed=0)
        )
        short_result: MacsProfileResult = self._profiler.profile(self._vocoder)
        long_result: MacsProfileResult = long_profiler.profile(self._vocoder)
        self.assertGreater(long_result.total_macs, short_result.total_macs)
        self.assertAlmostEqual(
            long_result.giga_macs_per_audio_second,
            short_result.giga_macs_per_audio_second,
            places=9
        )

    def test_repeated_profiles_of_one_module_agree(self) -> None:
        # The seeded probe makes profiling deterministic across invocations.
        first_result: MacsProfileResult = self._profiler.profile(self._vocoder)
        second_result: MacsProfileResult = self._profiler.profile(self._vocoder)
        self.assertEqual(first_result.total_macs, second_result.total_macs)
        self.assertEqual(first_result.mel_frames, second_result.mel_frames)

    def test_same_padding_convolution_is_counted_through_the_custom_handle(self) -> None:
        # Same-padding traces as aten::_convolution_mode, which fvcore ignores without the handle.
        same_padding_vocoder: ConvolutionVocoder = ConvolutionVocoder(
            self._mel_protocol,
            nn.Conv1d(8, 4, kernel_size=3, padding="same")
        )
        result: MacsProfileResult = self._profiler.profile(same_padding_vocoder)
        self.assertEqual(result.status, "profiled")
        self.assertEqual(result.unsupported_operators, ())
        self.assertEqual(result.total_macs, float(4 * 8 * 3 * result.mel_frames))

    def test_same_padding_and_explicit_padding_agree(self) -> None:
        # The custom handle reproduces the standard convolution arithmetic exactly.
        same_padding_vocoder: ConvolutionVocoder = ConvolutionVocoder(
            self._mel_protocol,
            nn.Conv1d(8, 4, kernel_size=3, padding="same")
        )
        self.assertEqual(
            self._profiler.profile(same_padding_vocoder).total_macs,
            self._profiler.profile(self._vocoder).total_macs
        )


class MacsProfileStatusTest(unittest.TestCase):
    # Verifies the four structured outcomes and the evidence each one carries.
    #
    # One network is shared across all four probes so the outcome is
    # attributable to the synthesis surface alone rather than to any
    # difference in the traced weights.
    def setUp(self) -> None:
        # Prepares one shared network the four outcome probes are built around.
        self._mel_protocol: MelConfig = ProbeMelProtocol().create()
        self._network: nn.Conv1d = nn.Conv1d(8, 4, kernel_size=3, padding=1)
        self._profiler: MacsProfiler = MacsProfiler(
            MacsProfilerConfig(audio_seconds=0.25, probe_seed=0)
        )

    def test_fully_supported_trace_reports_profiled(self) -> None:
        # A graph of counted operators alone earns the unqualified status.
        result: MacsProfileResult = self._profiler.profile(
            ConvolutionVocoder(self._mel_protocol, self._network)
        )
        self.assertEqual(result.status, "profiled")
        self.assertEqual(result.unsupported_operators, ())
        self.assertIsNone(result.reason)

    def test_uncounted_operator_downgrades_the_status_to_partial(self) -> None:
        # A count over an incompletely covered graph must not claim completeness.
        result: MacsProfileResult = self._profiler.profile(
            SaturatingVocoder(self._mel_protocol, self._network)
        )
        self.assertEqual(result.status, "profiled_partial")

    def test_uncounted_operators_are_disclosed_by_name(self) -> None:
        # The inventory names what the partial count omitted.
        result: MacsProfileResult = self._profiler.profile(
            SaturatingVocoder(self._mel_protocol, self._network)
        )
        self.assertIn("aten::tanh", result.unsupported_operators)

    def test_partial_profile_still_reports_the_counted_arithmetic(self) -> None:
        # The convolution is still counted even though the activation is not.
        result: MacsProfileResult = self._profiler.profile(
            SaturatingVocoder(self._mel_protocol, self._network)
        )
        self.assertEqual(result.total_macs, float(4 * 8 * 3 * result.mel_frames))

    def test_module_without_the_synthesis_contract_reports_unsupported(self) -> None:
        # A module with no traceable mel-conditioned surface is never forced through a trace.
        result: MacsProfileResult = self._profiler.profile(ContractlessModule())
        self.assertEqual(result.status, "unsupported")
        self.assertEqual(
            result.reason,
            "module_has_no_traceable_mel_conditioned_synthesis_contract"
        )

    def test_module_without_a_callable_synthesis_reports_unsupported(self) -> None:
        # Every member of the contract is required, not the network and protocol alone.
        result: MacsProfileResult = self._profiler.profile(
            ConditioningOnlyModule(self._mel_protocol, self._network)
        )
        self.assertEqual(result.status, "unsupported")

    def test_failing_synthesis_is_recorded_as_a_failed_profile(self) -> None:
        # A tracing exception becomes recorded evidence rather than a crash.
        result: MacsProfileResult = self._profiler.profile(
            RejectingVocoder(self._mel_protocol, self._network)
        )
        self.assertEqual(result.status, "failed")

    def test_failure_reason_names_the_exception_type(self) -> None:
        # The reason must identify what went wrong during tracing.
        result: MacsProfileResult = self._profiler.profile(
            RejectingVocoder(self._mel_protocol, self._network)
        )
        self.assertIn("ValueError", result.reason)

    def test_non_numeric_outcomes_carry_no_counts(self) -> None:
        # A status without a number must not leave a stale number behind.
        for result in (
            self._profiler.profile(ContractlessModule()),
            self._profiler.profile(RejectingVocoder(self._mel_protocol, self._network))
        ):
            with self.subTest(status=result.status):
                self.assertIsNone(result.total_macs)
                self.assertIsNone(result.giga_macs_per_audio_second)
                self.assertIsNone(result.audio_seconds)
                self.assertIsNone(result.mel_frames)

    def test_every_outcome_records_the_trace_method(self) -> None:
        # The method field makes the measurement technique auditable.
        result: MacsProfileResult = self._profiler.profile(ContractlessModule())
        self.assertEqual(result.method, "fvcore_jit_trace")


class MacsProfileResultRecordTest(unittest.TestCase):
    # Verifies the frozen profiling record itself.
    def setUp(self) -> None:
        # Builds one numberless record, the shape every non-numeric outcome takes.
        self._result: MacsProfileResult = MacsProfileResult(
            status="unsupported",
            giga_macs_per_audio_second=None,
            total_macs=None,
            audio_seconds=None,
            mel_frames=None,
            unsupported_operators=(),
            reason="module_has_no_traceable_mel_conditioned_synthesis_contract"
        )

    def test_record_is_frozen(self) -> None:
        # A recorded profile cannot be edited after the measurement.
        with self.assertRaises(ValidationError):
            self._result.status: MacsProfileStatus = "profiled"

    def test_record_rejects_extra_fields(self) -> None:
        # An unrecognized field must fail loudly rather than be dropped.
        with self.assertRaises(ValidationError):
            MacsProfileResult(
                status="unsupported",
                giga_macs_per_audio_second=None,
                total_macs=None,
                audio_seconds=None,
                mel_frames=None,
                unsupported_operators=(),
                reason=None,
                gflops=1.0
            )

    def test_record_rejects_an_unregistered_status(self) -> None:
        # The status vocabulary is closed to the four declared outcomes.
        with self.assertRaises(ValidationError):
            MacsProfileResult(
                status="partial",
                giga_macs_per_audio_second=None,
                total_macs=None,
                audio_seconds=None,
                mel_frames=None,
                unsupported_operators=(),
                reason=None
            )

    def test_record_serializes_for_the_run_log(self) -> None:
        # The sequence callback logs the record as JSON, so it must serialize.
        self.assertIn("unsupported", self._result.model_dump_json())


if __name__ == "__main__":
    unittest.main()
