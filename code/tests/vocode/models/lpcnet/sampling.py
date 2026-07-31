# This module:
# 1. Verifies the LPCNet sampler configuration record: the reference
#    sharpening and probability-floor settings, immutability, and the
#    rejection of unknown fields
# 2. Verifies the autoregressive synthesis contract: the frame-to-sample
#    output length, the int16 clamping of generated samples, batch
#    preservation, and reproducibility under a fixed seed
# 3. Verifies the call-time network resolution: the sampler asks its
#    provider on every synthesis call, so a replaced network is the network
#    that actually synthesizes
#
# Design decisions:
# - The autoregressive loop runs at sample rate, so every synthesis here
#   uses a miniature network at frame size two over a handful of frames; the
#   reference geometry (frame size 160 at 16 kHz) is never synthesized
# - Sampled indices are drawn from the global generator, so reproducibility
#   is asserted under a fixed seed rather than by assuming determinism
# - The degenerate probability floor is exercised to cover the uniform
#   fallback branch, and bounded by validity rather than by pinned values
#
# Author: Rahul Sawhney

import unittest

import torch
from pydantic import ValidationError

from vocode.models.lpcnet.network import LpcnetNetwork
from vocode.models.lpcnet.sampling import LpcnetSampler, LpcnetSamplerConfig


class MiniatureNetworkRecipe:
    # Builds the miniature LPCNet networks the autoregressive loop runs over.
    # The reduction is far more aggressive than elsewhere in the suite, and
    # deliberately so: synthesis cost is linear in the sample count and every
    # sample requires a full pass through the core, so a reference-sized
    # network over a reference-sized frame would make these assertions
    # prohibitively slow. Nothing asserted here depends on width.
    def __init__(self, frame_size: int) -> None:
        # Binds the frame size, which is the loop length per conditioning
        # frame.
        #
        # Args:
        #     frame_size: Samples generated per conditioning frame. This is
        #         also the property the network-replacement test observes,
        #         since it is directly visible in the output length.
        self._frame_size: int = frame_size

    def build(self) -> LpcnetNetwork:
        # Returns the miniature network at the bound frame size. Training
        # noise is set to zero, which keeps the network's noise injection out
        # of the way; the sampler's own randomness is the only stochastic
        # element these assertions should observe.
        return LpcnetNetwork(
            feature_dimension=6,
            condition_dimension=8,
            embedding_dimension=4,
            first_gru_dimension=8,
            second_gru_dimension=4,
            lpc_order=2,
            frame_size=self._frame_size,
            training_noise_standard_deviation=0.0
        )


class FrameFeatureBuilder:
    # Builds the frame-rate synthesis inputs the sampler consumes. Each input
    # is produced by its own method rather than as one bundle, so a test can
    # vary one while holding the rest fixed.
    def __init__(self, batch_size: int, frame_count: int) -> None:
        # Binds the batch and frame geometry every emitted input carries.
        self._batch_size: int = batch_size
        self._frame_count: int = frame_count

    def build_conditioning(self) -> torch.Tensor:
        # Returns one six-dimensional feature vector per frame, matching the
        # miniature network's declared feature width.
        return torch.randn(self._batch_size, self._frame_count, 6)

    def build_pitch_index(self) -> torch.Tensor:
        # Returns one pitch-table index per frame, drawn inside the
        # two-hundred-fifty-six-entry table so the embedding lookup is always
        # in range.
        return torch.randint(0, 256, (self._batch_size, self._frame_count))

    def build_pitch_correlation(self) -> torch.Tensor:
        # Returns per-frame correlations in the unit interval. Drawing across
        # the whole interval means a single synthesis exercises both
        # sharpening regimes, since values below one third leave the
        # distribution unsharpened and values above it do not.
        return torch.rand(self._batch_size, self._frame_count)

    def build_lpc_coefficients(self) -> torch.Tensor:
        # Returns small second-order coefficients. The scaling matters: the
        # synthesis loop feeds each generated sample back through the
        # prediction filter, so large coefficients would make that recursion
        # diverge and drive every sample to the clamp, hiding whatever the
        # test meant to measure.
        return torch.randn(self._batch_size, self._frame_count, 2) * 0.1


class CountingNetworkProvider:
    # Resolves one fixed network and records how often the sampler asked for
    # it. The count is the observable that distinguishes call-time resolution
    # from a reference captured at construction.
    def __init__(self, network: LpcnetNetwork) -> None:
        # Binds the network this provider always returns and opens the request
        # counter.
        self._network: LpcnetNetwork = network
        self._call_count: int = 0

    @property
    def call_count(self) -> int:
        # Returns how many times the sampler has resolved a network through this provider.
        return self._call_count

    def __call__(self) -> LpcnetNetwork:
        # Records the request and returns the bound network.
        self._call_count: int = self._call_count + 1
        return self._network


class SwappingNetworkProvider:
    # Resolves a different network on each call, standing in for an
    # intervention that replaces one. This is the fixture that makes the
    # indirection's purpose testable: it simulates what dynamic quantization
    # does to the module's network attribute, and a sampler holding a captured
    # reference would keep synthesizing through the original.
    def __init__(self, networks: list[LpcnetNetwork]) -> None:
        # Copies the network sequence so the fixture cannot mutate the
        # caller's list.
        self._networks: list[LpcnetNetwork] = list(networks)
        self._call_index: int = 0

    def __call__(self) -> LpcnetNetwork:
        # Advances through the sequence and holds at the last network once it
        # is exhausted, so an extra synthesis call cannot raise; the sequence
        # defines the first several resolutions rather than a fixed budget.
        resolved_index: int = min(self._call_index, len(self._networks) - 1)
        self._call_index: int = self._call_index + 1
        return self._networks[resolved_index]


class LpcnetSamplerConfigurationTest(unittest.TestCase):
    # Verifies the reference sampling settings and the validation behavior of the record.
    def setUp(self) -> None:
        # Reads the default sampler record, which is the reference inference setting.
        self._configuration: LpcnetSamplerConfig = LpcnetSamplerConfig()

    def test_defaults_follow_the_reference_sampling_scheme(self) -> None:
        # Pitch-adaptive sharpening and the constant probability floor follow the reference inference path.
        self.assertEqual(self._configuration.correlation_sharpening_scale, 1.5)
        self.assertEqual(self._configuration.correlation_sharpening_offset, -0.5)
        self.assertEqual(self._configuration.probability_floor, 0.002)

    def test_configuration_is_immutable(self) -> None:
        # Frozen settings cannot drift between synthesis calls.
        with self.assertRaises(ValidationError):
            self._configuration.probability_floor = 0.5

    def test_unknown_field_is_rejected(self) -> None:
        # Extra fields are forbidden so silent typos cannot enter an experiment record.
        with self.assertRaises(ValidationError):
            LpcnetSamplerConfig(unknown_setting=1)

    def test_negative_probability_floor_is_rejected(self) -> None:
        # The floor is subtracted from probabilities, so a negative value
        # would add mass rather than remove it and would leave the
        # distribution unnormalized in the opposite direction. The field's
        # non-negative type is what forecloses that, and zero remains legal
        # because it disables the mechanism cleanly.
        with self.assertRaises(ValidationError):
            LpcnetSamplerConfig(probability_floor=-0.1)


class LpcnetAutoregressiveSynthesisTest(unittest.TestCase):
    # Verifies the synthesis shape contract, the int16 clamping, and seeded
    # reproducibility. Sample values cannot be asserted, since the network is
    # untrained and the excitation is drawn at random; what is established is
    # that the loop produces the declared number of samples, that every one
    # stays representable, and that the whole sequence repeats exactly under a
    # repeated seed.
    def setUp(self) -> None:
        # Builds the frame-size-two sampler over three frames, a six-step
        # autoregressive loop. Six steps is enough to exercise the state
        # carry-over across a frame boundary, which is the only structural
        # transition inside the nested loop.
        torch.manual_seed(1234)
        self._network: LpcnetNetwork = MiniatureNetworkRecipe(frame_size=2).build()
        self._network.eval()
        self._provider: CountingNetworkProvider = CountingNetworkProvider(self._network)
        self._sampler: LpcnetSampler = LpcnetSampler(self._provider, LpcnetSamplerConfig())
        self._features: FrameFeatureBuilder = FrameFeatureBuilder(batch_size=1, frame_count=3)
        self._conditioning: torch.Tensor = self._features.build_conditioning()
        self._pitch_index: torch.Tensor = self._features.build_pitch_index()
        self._pitch_correlation: torch.Tensor = self._features.build_pitch_correlation()
        self._lpc_coefficients: torch.Tensor = self._features.build_lpc_coefficients()

    def test_synthesis_length_is_frames_times_frame_size(self) -> None:
        # Every conditioning frame is unrolled into frame_size autoregressive steps.
        synthesized: torch.Tensor = self._sampler.synthesize(
            self._conditioning,
            self._pitch_index,
            self._pitch_correlation,
            self._lpc_coefficients
        )
        self.assertEqual(tuple(synthesized.shape), (1, 6))

    def test_samples_stay_inside_the_int16_range(self) -> None:
        # Generated samples are clamped to the int16 domain the reference synthesizes in.
        synthesized: torch.Tensor = self._sampler.synthesize(
            self._conditioning,
            self._pitch_index,
            self._pitch_correlation,
            self._lpc_coefficients
        )
        self.assertLessEqual(float(synthesized.abs().max()), 32767.0)
        self.assertTrue(bool(torch.isfinite(synthesized).all()))

    def test_batch_dimension_is_preserved(self) -> None:
        # Batched features synthesize one waveform row per batch element.
        batched: FrameFeatureBuilder = FrameFeatureBuilder(batch_size=2, frame_count=3)
        synthesized: torch.Tensor = self._sampler.synthesize(
            batched.build_conditioning(),
            batched.build_pitch_index(),
            batched.build_pitch_correlation(),
            batched.build_lpc_coefficients()
        )
        self.assertEqual(tuple(synthesized.shape), (2, 6))

    def test_synthesis_is_reproducible_under_a_fixed_seed(self) -> None:
        # Excitation indices are sampled, so the same seed must reproduce the same waveform.
        torch.manual_seed(11)
        first_waveform: torch.Tensor = self._sampler.synthesize(
            self._conditioning,
            self._pitch_index,
            self._pitch_correlation,
            self._lpc_coefficients
        )
        torch.manual_seed(11)
        second_waveform: torch.Tensor = self._sampler.synthesize(
            self._conditioning,
            self._pitch_index,
            self._pitch_correlation,
            self._lpc_coefficients
        )
        self.assertTrue(bool(torch.equal(first_waveform, second_waveform)))

    def test_degenerate_probability_floor_falls_back_to_uniform_sampling(self) -> None:
        # A floor that erases the whole distribution still yields valid
        # excitation indices. A floor of one is unreachable by any probability,
        # so it drives every level to zero and forces the fallback branch on
        # every step, which is the only way to reach that branch
        # deterministically. The assertion is validity rather than a value:
        # what matters is that synthesis completes with finite samples instead
        # of dividing by a zero total.
        flooded_sampler: LpcnetSampler = LpcnetSampler(
            self._provider,
            LpcnetSamplerConfig(probability_floor=1.0)
        )
        synthesized: torch.Tensor = flooded_sampler.synthesize(
            self._conditioning,
            self._pitch_index,
            self._pitch_correlation,
            self._lpc_coefficients
        )
        self.assertEqual(tuple(synthesized.shape), (1, 6))
        self.assertTrue(bool(torch.isfinite(synthesized).all()))

    def test_synthesis_runs_without_building_a_gradient_graph(self) -> None:
        # Inference is wrapped in no-grad, so the synthesized waveform carries
        # no graph. This is a memory contract rather than a stylistic one: the
        # loop is sequential and each step consumes the previous step's
        # output, so a retained graph would grow linearly in the sample count
        # and exhaust memory on any realistic utterance.
        synthesized: torch.Tensor = self._sampler.synthesize(
            self._conditioning,
            self._pitch_index,
            self._pitch_correlation,
            self._lpc_coefficients
        )
        self.assertFalse(synthesized.requires_grad)


class LpcnetNetworkResolutionTest(unittest.TestCase):
    # Verifies that the sampler resolves its network through the provider at
    # every synthesis call. This is a measurement-integrity property rather
    # than a functional one: the study measures interventions that replace the
    # network, such as dynamic integer quantization, and a sampler bound to a
    # stale reference would report the unmodified network's behavior under the
    # intervention's name. Both halves are asserted, that the provider is
    # consulted each time and that its answer is what actually runs.
    def setUp(self) -> None:
        # Builds one conditioning set shared by both provider fixtures, so the
        # two tests differ only in the provider they install.
        torch.manual_seed(1234)
        self._features: FrameFeatureBuilder = FrameFeatureBuilder(batch_size=1, frame_count=3)
        self._conditioning: torch.Tensor = self._features.build_conditioning()
        self._pitch_index: torch.Tensor = self._features.build_pitch_index()
        self._pitch_correlation: torch.Tensor = self._features.build_pitch_correlation()
        self._lpc_coefficients: torch.Tensor = self._features.build_lpc_coefficients()

    def test_provider_is_asked_once_per_synthesis_call(self) -> None:
        # A retained network reference would invalidate intervention
        # measurements. The count is asserted as zero before any synthesis,
        # which proves the sampler's constructor does not resolve the network,
        # and then as exactly two after two calls, which proves resolution
        # happens once per call rather than once overall or once per frame.
        provider: CountingNetworkProvider = CountingNetworkProvider(MiniatureNetworkRecipe(frame_size=2).build())
        sampler: LpcnetSampler = LpcnetSampler(provider, LpcnetSamplerConfig())
        self.assertEqual(provider.call_count, 0)
        sampler.synthesize(
            self._conditioning,
            self._pitch_index,
            self._pitch_correlation,
            self._lpc_coefficients
        )
        sampler.synthesize(
            self._conditioning,
            self._pitch_index,
            self._pitch_correlation,
            self._lpc_coefficients
        )
        self.assertEqual(provider.call_count, 2)

    def test_replaced_network_drives_the_next_synthesis(self) -> None:
        # Swapping the resolved network changes the synthesis geometry
        # immediately. Frame size is chosen as the distinguishing property
        # precisely because it is visible in the output length, so the
        # assertion can tell which network ran from the result alone rather
        # than by inspecting the sampler's internals; the doubled frame size
        # doubles the sample count on the second call.
        provider: SwappingNetworkProvider = SwappingNetworkProvider(
            [MiniatureNetworkRecipe(frame_size=2).build(), MiniatureNetworkRecipe(frame_size=4).build()]
        )
        sampler: LpcnetSampler = LpcnetSampler(provider, LpcnetSamplerConfig())
        first_waveform: torch.Tensor = sampler.synthesize(
            self._conditioning,
            self._pitch_index,
            self._pitch_correlation,
            self._lpc_coefficients
        )
        second_waveform: torch.Tensor = sampler.synthesize(
            self._conditioning,
            self._pitch_index,
            self._pitch_correlation,
            self._lpc_coefficients
        )
        self.assertEqual(tuple(first_waveform.shape), (1, 6))
        self.assertEqual(
            tuple(second_waveform.shape),
            (1, 12),
            msg="The second synthesis must run on the replaced network, not the constructed one"
        )
