# This module:
# 1. Verifies that every named MelConfig factory reproduces the reference
#    extraction protocol it documents: sample rate, band count, frequency
#    ceiling, mel scale, STFT centering, basis normalization, and log floor
# 2. Verifies that delegating factories inherit their target protocol exactly
#    and that conditioning and reconstruction recipes of one family differ in
#    the frequency ceiling alone
# 3. Verifies MelSpectrogram path selection, accepted and rejected waveform
#    ranks, frame counts on minimal synthetic waveforms, and the clamp-then-log
#    compression contract including its response to amplitude gain
#
# Design decisions:
# - Protocol fields are asserted directly, because a mel protocol is a
#   measurement contract whose published constants are the object under test
#   rather than an implementation detail
# - Executable assertions bound behavior (frame counts, ranks, log floors,
#   gain response) instead of pinning spectrogram values, which are not stable
#   across torchaudio releases
# - The amplitude-gain identity is asserted only on bins sitting clearly above
#   the clamp floor, because clamped bins are deliberately insensitive to gain
# - Compact 256-point protocols built directly from the public record cover the
#   log10 branch and the manual-path basis-normalization switch, neither of
#   which any published recipe exercises
# - Waveforms stay at or below 4096 samples so every extraction produces at
#   most seventeen frames and the whole file runs in about a second
#
# Author: Rahul Sawhney

import math
import unittest
from collections.abc import Callable

import torch
from pydantic import ValidationError

from vocode.transforms.mel import MelConfig, MelSpectrogram


class ToneWaveformBuilder:
    # Builds deterministic tonal, broadband, and silent waveforms for extraction.
    def __init__(self, sample_rate: int, sample_count: int) -> None:
        # Binds the analysis grid and the fixed seed backing the broadband draw.
        self._sample_rate: int = sample_rate
        self._sample_count: int = sample_count
        self._noise_seed: int = 20260730

    def tone(self, frequency: float, amplitude: float) -> torch.Tensor:
        # Produces a single-channel sine at the requested frequency and peak.
        times: torch.Tensor = torch.arange(self._sample_count, dtype=torch.float32) / self._sample_rate
        return amplitude * torch.sin(2.0 * torch.pi * frequency * times)

    def broadband(self, amplitude: float) -> torch.Tensor:
        # Draws a reproducible gaussian sequence covering the whole band.
        torch.manual_seed(self._noise_seed)
        return amplitude * torch.randn(self._sample_count)

    def silence(self) -> torch.Tensor:
        # Produces an all-zero waveform that must reach the protocol log floor.
        return torch.zeros(self._sample_count)


class PublishedRecipeRegistry:
    # Names every published protocol factory the study registers on the record.
    def __init__(self) -> None:
        # Binds the closed list of factory names checked by the sweep assertions.
        self._recipe_names: tuple[str, ...] = (
            "hifigan_v1",
            "hifigan_conditioning",
            "hifigan_v2",
            "hifigan_v3",
            "hifigan_reconstruction",
            "hifigan_half_width_v1",
            "melgan_seungwon",
            "vocos_charactr_mel_24khz",
            "bigvgan_nvidia_base_24khz_100band",
            "bigvgan_nvidia_base_reconstruction_24khz_100band",
            "apnet2_redmist328",
            "apnet2_redmist328_reconstruction",
            "freev_official",
            "freev_official_reconstruction",
            "hiftnet_yl4579",
            "rndvoc_andong",
            "rndvoc_andong_reconstruction",
            "lpcnet_metric_16khz"
        )

    def names(self) -> tuple[str, ...]:
        # Returns the registered factory names.
        return self._recipe_names

    def protocols(self) -> dict[str, MelConfig]:
        # Builds every registered protocol keyed by its factory name.
        built_protocols: dict[str, MelConfig] = {}
        recipe_name: str
        for recipe_name in self._recipe_names:
            factory: Callable[[], MelConfig] = getattr(MelConfig, recipe_name)
            built_protocols[recipe_name] = factory()
        return built_protocols


class CompactProtocolBuilder:
    # Builds small 256-point protocols covering branches no published recipe reaches.
    def __init__(self) -> None:
        # Binds the shared compact analysis grid used by every variant below.
        self._sample_rate: int = 16000
        self._n_fft: int = 256
        self._hop_length: int = 64
        self._n_mels: int = 8

    def normalized_manual(self) -> MelConfig:
        # Compact manual-padding protocol with a Slaney area-normalized basis.
        return MelConfig(
            sample_rate=self._sample_rate,
            n_fft=self._n_fft,
            hop_length=self._hop_length,
            win_length=self._n_fft,
            n_mels=self._n_mels,
            fmin=0.0,
            fmax=8000.0,
            mel_scale="slaney",
            center=False,
            pad_mode="reflect",
            power=1.0,
            normalize_mel_basis=True,
            log_clamp_min=1e-5,
            log_base="natural",
            manual_reflect_padding=True
        )

    def unnormalized_manual(self) -> MelConfig:
        # Compact manual-padding protocol whose triangular filters keep unit peaks.
        return self.normalized_manual().model_copy(update={"normalize_mel_basis": False})

    def htk_manual(self) -> MelConfig:
        # Compact manual-padding protocol on the HTK mel scale.
        return self.normalized_manual().model_copy(update={"mel_scale": "htk"})

    def base_ten_centered(self) -> MelConfig:
        # Compact centered protocol compressing with the base-ten logarithm.
        return self.normalized_manual().model_copy(
            update={"log_base": "log10", "center": True, "manual_reflect_padding": False}
        )


class MelProtocolRecipeTest(unittest.TestCase):
    # Verifies that each named factory carries its documented reference protocol.
    def setUp(self) -> None:
        # Collects the published conditioning protocols compared across families.
        self._registry: PublishedRecipeRegistry = PublishedRecipeRegistry()
        self._hifigan: MelConfig = MelConfig.hifigan_conditioning()
        self._melgan: MelConfig = MelConfig.melgan_seungwon()
        self._vocos: MelConfig = MelConfig.vocos_charactr_mel_24khz()
        self._bigvgan: MelConfig = MelConfig.bigvgan_nvidia_base_24khz_100band()

    def test_hifigan_conditioning_carries_the_published_extraction_protocol(self) -> None:
        # The jik876 LJSpeech recipe: 22.05 kHz, 80 bands to 8 kHz, uncentered.
        self.assertEqual(self._hifigan.sample_rate, 22050)
        self.assertEqual(self._hifigan.n_fft, 1024)
        self.assertEqual(self._hifigan.hop_length, 256)
        self.assertEqual(self._hifigan.win_length, 1024)
        self.assertEqual(self._hifigan.n_mels, 80)
        self.assertEqual(self._hifigan.fmin, 0.0)
        self.assertEqual(self._hifigan.fmax, 8000.0)
        self.assertEqual(self._hifigan.mel_scale, "slaney")
        self.assertEqual(self._hifigan.log_clamp_min, 1e-5)
        self.assertEqual(self._hifigan.log_base, "natural")

    def test_hifigan_family_uses_uncentered_manual_padding_with_normalized_basis(self) -> None:
        # The published HiFi-GAN extraction pads manually and never centers frames.
        self.assertFalse(self._hifigan.center, msg="HiFi-GAN conditioning must not center STFT frames")
        self.assertTrue(
            self._hifigan.manual_reflect_padding,
            msg="HiFi-GAN conditioning must use the manual reflect-padded STFT path"
        )
        self.assertTrue(
            self._hifigan.normalize_mel_basis,
            msg="HiFi-GAN conditioning must use the Slaney-normalized mel basis"
        )

    def test_melgan_uses_centered_stft_without_manual_padding(self) -> None:
        # The Seungwon Park MelGAN recipe centers frames on the torchaudio path.
        self.assertTrue(self._melgan.center, msg="MelGAN conditioning must center STFT frames")
        self.assertFalse(
            self._melgan.manual_reflect_padding,
            msg="MelGAN conditioning must not use the manual reflect-padded path"
        )
        self.assertTrue(self._melgan.normalize_mel_basis)
        self.assertEqual(self._melgan.sample_rate, 22050)
        self.assertEqual(self._melgan.n_mels, 80)
        self.assertEqual(self._melgan.fmax, 8000.0)

    def test_hifigan_and_melgan_differ_only_in_the_stft_grid(self) -> None:
        # Two families share every band setting yet disagree on frame centering.
        self.assertEqual(self._hifigan.sample_rate, self._melgan.sample_rate)
        self.assertEqual(self._hifigan.n_mels, self._melgan.n_mels)
        self.assertEqual(self._hifigan.fmax, self._melgan.fmax)
        self.assertEqual(self._hifigan.mel_scale, self._melgan.mel_scale)
        self.assertNotEqual(
            self._hifigan.center,
            self._melgan.center,
            msg="The HiFi-GAN and MelGAN recipes must disagree on STFT centering"
        )

    def test_delegating_hifigan_factories_inherit_the_conditioning_protocol(self) -> None:
        # V1, V2, V3, the half-width variant, and HiFTNet all reuse one protocol.
        delegating_factories: tuple[Callable[[], MelConfig], ...] = (
            MelConfig.hifigan_v1,
            MelConfig.hifigan_v2,
            MelConfig.hifigan_v3,
            MelConfig.hifigan_half_width_v1,
            MelConfig.hiftnet_yl4579
        )
        factory: Callable[[], MelConfig]
        for factory in delegating_factories:
            delegated_protocol: MelConfig = factory()
            self.assertEqual(
                delegated_protocol,
                self._hifigan,
                msg=f"{factory.__name__} must reproduce the HiFi-GAN conditioning protocol exactly"
            )

    def test_hifigan_reconstruction_differs_from_conditioning_only_in_frequency_ceiling(self) -> None:
        # References condition on a band-limited mel and reconstruct on the full band.
        reconstruction: MelConfig = MelConfig.hifigan_reconstruction()
        self.assertEqual(self._hifigan.fmax, 8000.0)
        self.assertIsNone(reconstruction.fmax, msg="The reconstruction protocol must reach Nyquist")
        differing_fields: set[str] = {
            field_name
            for field_name in self._hifigan.model_dump()
            if getattr(self._hifigan, field_name) != getattr(reconstruction, field_name)
        }
        self.assertEqual(
            differing_fields,
            {"fmax"},
            msg=f"Conditioning and reconstruction must differ in fmax alone, differed in {differing_fields}"
        )

    def test_every_reconstruction_recipe_extends_the_basis_to_nyquist(self) -> None:
        # Reconstruction losses see the full band, so no ceiling may be set.
        reconstruction_factories: tuple[Callable[[], MelConfig], ...] = (
            MelConfig.hifigan_reconstruction,
            MelConfig.bigvgan_nvidia_base_reconstruction_24khz_100band,
            MelConfig.apnet2_redmist328_reconstruction,
            MelConfig.freev_official_reconstruction,
            MelConfig.rndvoc_andong_reconstruction
        )
        factory: Callable[[], MelConfig]
        for factory in reconstruction_factories:
            protocol: MelConfig = factory()
            self.assertIsNone(
                protocol.fmax,
                msg=f"{factory.__name__} must leave fmax unset so the basis reaches Nyquist"
            )

    def test_vocos_recipe_uses_htk_scale_without_basis_normalization(self) -> None:
        # The charactr Vocos protocol is the only unnormalized HTK recipe.
        self.assertEqual(self._vocos.sample_rate, 24000)
        self.assertEqual(self._vocos.n_mels, 100)
        self.assertEqual(self._vocos.mel_scale, "htk")
        self.assertFalse(
            self._vocos.normalize_mel_basis,
            msg="The Vocos protocol must keep the unnormalized mel basis"
        )
        self.assertIsNone(self._vocos.fmax)
        self.assertEqual(self._vocos.log_clamp_min, 1e-7)

    def test_bigvgan_recipe_band_limits_at_twelve_kilohertz_on_the_manual_path(self) -> None:
        # BigVGAN-base conditions on 100 bands to 12 kHz through the manual path.
        self.assertEqual(self._bigvgan.sample_rate, 24000)
        self.assertEqual(self._bigvgan.n_mels, 100)
        self.assertEqual(self._bigvgan.fmax, 12000.0)
        self.assertEqual(self._bigvgan.mel_scale, "slaney")
        self.assertTrue(self._bigvgan.manual_reflect_padding)
        self.assertFalse(self._bigvgan.center)

    def test_centered_conditioning_recipes_share_the_hifigan_band_settings(self) -> None:
        # APNet2, FreeV, and RNDVoC condition on 80 bands to 8 kHz with centering.
        centered_factories: tuple[Callable[[], MelConfig], ...] = (
            MelConfig.apnet2_redmist328,
            MelConfig.freev_official,
            MelConfig.rndvoc_andong
        )
        factory: Callable[[], MelConfig]
        for factory in centered_factories:
            protocol: MelConfig = factory()
            self.assertEqual(protocol.sample_rate, 22050, msg=f"{factory.__name__} rate mismatch")
            self.assertEqual(protocol.n_mels, 80, msg=f"{factory.__name__} band-count mismatch")
            self.assertEqual(protocol.fmax, 8000.0, msg=f"{factory.__name__} ceiling mismatch")
            self.assertTrue(protocol.center, msg=f"{factory.__name__} must center STFT frames")

    def test_lpcnet_metric_recipe_operates_at_sixteen_kilohertz_on_the_full_band(self) -> None:
        # The LPCNet mel-error metric measures at the LPCNet corpus rate.
        protocol: MelConfig = MelConfig.lpcnet_metric_16khz()
        self.assertEqual(protocol.sample_rate, 16000)
        self.assertEqual(protocol.n_mels, 80)
        self.assertIsNone(protocol.fmax)
        self.assertTrue(protocol.center)

    def test_manual_padding_is_declared_exactly_for_uncentered_recipes(self) -> None:
        # Manual reflect padding replaces centering; the two are never both set.
        recipe_name: str
        protocol: MelConfig
        for recipe_name, protocol in self._registry.protocols().items():
            self.assertEqual(
                protocol.manual_reflect_padding,
                not protocol.center,
                msg=f"{recipe_name} pairs center={protocol.center} with manual padding"
            )

    def test_no_recipe_places_its_ceiling_above_nyquist(self) -> None:
        # A mel basis above half the sample rate would measure empty filters.
        recipe_name: str
        protocol: MelConfig
        for recipe_name, protocol in self._registry.protocols().items():
            if protocol.fmax is None:
                continue
            self.assertLessEqual(
                protocol.fmax,
                protocol.sample_rate / 2.0,
                msg=f"{recipe_name} sets fmax {protocol.fmax} above Nyquist"
            )

    def test_every_recipe_declares_a_consistent_stft_grid(self) -> None:
        # A usable analysis grid overlaps frames and fits the window in the transform.
        recipe_name: str
        protocol: MelConfig
        for recipe_name, protocol in self._registry.protocols().items():
            self.assertLessEqual(
                protocol.win_length,
                protocol.n_fft,
                msg=f"{recipe_name} declares a window longer than its FFT"
            )
            self.assertLess(
                protocol.hop_length,
                protocol.win_length,
                msg=f"{recipe_name} declares a hop that leaves gaps between frames"
            )
            self.assertEqual(protocol.power, 1.0, msg=f"{recipe_name} must keep magnitude spectra")


class MelProtocolValidationTest(unittest.TestCase):
    # Verifies the frozen, strict, extra-forbidding record semantics.
    def setUp(self) -> None:
        # Binds one published protocol and its field mapping as the mutation target.
        self._protocol: MelConfig = MelConfig.hifigan_v1()
        self._fields: dict[str, object] = self._protocol.model_dump()

    def test_protocol_rejects_field_mutation(self) -> None:
        # A bound protocol cannot drift after the recipe has been selected.
        with self.assertRaises(ValidationError):
            self._protocol.sample_rate: int = 16000

    def test_dumped_fields_round_trip_through_strict_validation(self) -> None:
        # The comparison baseline for the rejection checks must itself validate.
        self.assertEqual(MelConfig.model_validate(self._fields), self._protocol)

    def test_protocol_rejects_unknown_field(self) -> None:
        # A forbidden extra turns a mistyped protocol field into a construction failure.
        with self.assertRaises(ValidationError):
            MelConfig.model_validate({**self._fields, "n_mel_bands": 80})

    def test_protocol_rejects_float_where_an_integer_field_is_declared(self) -> None:
        # Strict validation refuses a fractional sample rate or band count.
        with self.assertRaises(ValidationError):
            MelConfig.model_validate({**self._fields, "sample_rate": 22050.0})
        with self.assertRaises(ValidationError):
            MelConfig.model_validate({**self._fields, "n_mels": 80.0})

    def test_protocol_rejects_non_boolean_centering_flag(self) -> None:
        # Strict validation refuses an integer or string where centering is declared.
        with self.assertRaises(ValidationError):
            MelConfig.model_validate({**self._fields, "center": 1})
        with self.assertRaises(ValidationError):
            MelConfig.model_validate({**self._fields, "manual_reflect_padding": "yes"})

    def test_protocol_rejects_unknown_mel_scale(self) -> None:
        # The mel-scale literal admits only the HTK and Slaney formulas.
        with self.assertRaises(ValidationError):
            MelConfig.model_validate({**self._fields, "mel_scale": "bark"})

    def test_protocol_rejects_non_positive_band_count(self) -> None:
        # A protocol without bands cannot express a measurement.
        with self.assertRaises(ValidationError):
            MelConfig.model_validate({**self._fields, "n_mels": 0})

    def test_protocol_rejects_negative_frequency_floor(self) -> None:
        # A negative lower bound has no meaning on the frequency axis.
        with self.assertRaises(ValidationError):
            MelConfig.model_validate({**self._fields, "fmin": -100.0})


class MelSpectrogramPathSelectionTest(unittest.TestCase):
    # Verifies which extraction path a protocol selects and what it registers.
    def setUp(self) -> None:
        # Builds one executable transform per extraction path.
        self._manual: MelSpectrogram = MelSpectrogram(MelConfig.hifigan_conditioning())
        self._centered: MelSpectrogram = MelSpectrogram(MelConfig.melgan_seungwon())

    def test_manual_protocol_registers_the_window_and_filter_bank(self) -> None:
        # The manual path owns its analysis window and mel basis as buffers.
        buffer_names: set[str] = {name for name, _ in self._manual.named_buffers(recurse=False)}
        self.assertEqual(buffer_names, {"_manual_window", "_manual_filter_bank"})
        window: torch.Tensor = self._manual.get_buffer("_manual_window")
        filter_bank: torch.Tensor = self._manual.get_buffer("_manual_filter_bank")
        self.assertEqual(window.shape, (1024,))
        self.assertEqual(filter_bank.shape, (80, 513))

    def test_manual_protocol_keeps_derived_buffers_out_of_the_state_dictionary(self) -> None:
        # Window and basis are derivable from the protocol, so checkpoints omit them.
        self.assertEqual(
            list(self._manual.state_dict().keys()),
            [],
            msg="Manual-path buffers must be non-persistent and stay out of checkpoints"
        )

    def test_centered_protocol_builds_the_torchaudio_transform(self) -> None:
        # Centered protocols delegate extraction to the torchaudio transform.
        self.assertEqual(
            {name for name, _ in self._centered.named_buffers(recurse=False)},
            set(),
            msg="The centered path must not register the manual-path buffers"
        )
        self.assertTrue(
            any(key.startswith("_spectrogram") for key in self._centered.state_dict()),
            msg="The centered path must own a torchaudio transform submodule"
        )

    def test_configuration_property_returns_the_injected_protocol(self) -> None:
        # The transform exposes exactly the protocol it was constructed with.
        protocol: MelConfig = MelConfig.vocos_charactr_mel_24khz()
        transform: MelSpectrogram = MelSpectrogram(protocol)
        self.assertIs(transform.configuration, protocol)

    def test_every_registered_recipe_constructs_an_executable_transform(self) -> None:
        # No published protocol may fail to build its extraction machinery, and
        # each one lands on the path its padding declaration selects.
        recipe_name: str
        protocol: MelConfig
        for recipe_name, protocol in PublishedRecipeRegistry().protocols().items():
            transform: MelSpectrogram = MelSpectrogram(protocol)
            expected_buffers: set[str] = (
                {"_manual_window", "_manual_filter_bank"}
                if protocol.manual_reflect_padding
                else set()
            )
            self.assertEqual(
                {name for name, _ in transform.named_buffers(recurse=False)},
                expected_buffers,
                msg=f"{recipe_name} built the wrong extraction path"
            )
            self.assertIs(transform.configuration, protocol)


class MelSpectrogramShapeTest(unittest.TestCase):
    # Verifies frame counts and accepted waveform ranks on both extraction paths.
    def setUp(self) -> None:
        # Prepares both paths and a 4096-sample tonal waveform at 22.05 kHz.
        self._manual: MelSpectrogram = MelSpectrogram(MelConfig.hifigan_conditioning())
        self._centered: MelSpectrogram = MelSpectrogram(MelConfig.melgan_seungwon())
        self._builder: ToneWaveformBuilder = ToneWaveformBuilder(22050, 4096)
        self._waveform: torch.Tensor = self._builder.tone(440.0, 0.5)
        self._hop_length: int = 256

    def test_uncentered_manual_path_produces_one_frame_per_hop(self) -> None:
        # Manual padding by (n_fft - hop) / 2 yields exactly time / hop frames.
        spectrogram: torch.Tensor = self._manual(self._waveform.unsqueeze(0))
        self.assertEqual(spectrogram.shape, (1, 80, self._waveform.shape[-1] // self._hop_length))

    def test_centered_path_produces_one_frame_more_than_the_hop_count(self) -> None:
        # Centering adds the boundary frame the uncentered grid omits.
        spectrogram: torch.Tensor = self._centered(self._waveform.unsqueeze(0))
        self.assertEqual(spectrogram.shape, (1, 80, self._waveform.shape[-1] // self._hop_length + 1))

    def test_manual_path_promotes_unbatched_waveforms_to_a_batch(self) -> None:
        # The manual path normalizes every accepted rank to [batch, mels, frames].
        spectrogram: torch.Tensor = self._manual(self._waveform)
        self.assertEqual(spectrogram.shape, (1, 80, 16))

    def test_manual_path_accepts_single_channel_batched_waveforms(self) -> None:
        # A [batch, 1, time] buffer loses its channel axis before the STFT.
        batched: torch.Tensor = self._waveform.unsqueeze(0).unsqueeze(0).repeat(2, 1, 1)
        spectrogram: torch.Tensor = self._manual(batched)
        self.assertEqual(spectrogram.shape, (2, 80, 16))

    def test_centered_path_preserves_the_unbatched_rank(self) -> None:
        # The torchaudio transform maps [time] onto [mels, frames] unchanged.
        spectrogram: torch.Tensor = self._centered(self._waveform)
        self.assertEqual(spectrogram.shape, (80, 17))

    def test_manual_path_rejects_multichannel_waveforms(self) -> None:
        # Multi-channel material has no defined single-track mel protocol.
        with self.assertRaises(ValueError):
            self._manual(torch.zeros(2, 3, 1024))

    def test_manual_path_pads_waveforms_shorter_than_the_reflect_padding(self) -> None:
        # Reflect padding needs more source samples than it inserts, so short
        # utterances are zero-extended before padding instead of raising.
        spectrogram: torch.Tensor = self._manual(torch.zeros(1, 100))
        self.assertEqual(spectrogram.shape, (1, 80, 1))

    def test_batched_extraction_matches_single_utterance_extraction(self) -> None:
        # Frames of one utterance never depend on its neighbours in the batch.
        other: torch.Tensor = self._builder.tone(880.0, 0.25)
        batched: torch.Tensor = self._manual(torch.stack([self._waveform, other]))
        single: torch.Tensor = self._manual(other.unsqueeze(0))
        self.assertTrue(
            torch.allclose(batched[1:2], single, atol=1e-6),
            msg="Batched extraction must equal single-utterance extraction"
        )


class MelCompressionTest(unittest.TestCase):
    # Verifies the clamp-then-log compression contract shared by every protocol.
    def setUp(self) -> None:
        # Prepares both paths, the compact branch protocols, and the test material.
        self._manual: MelSpectrogram = MelSpectrogram(MelConfig.hifigan_conditioning())
        self._centered: MelSpectrogram = MelSpectrogram(MelConfig.melgan_seungwon())
        self._compact_builder: CompactProtocolBuilder = CompactProtocolBuilder()
        self._builder: ToneWaveformBuilder = ToneWaveformBuilder(22050, 4096)
        self._log_floor: float = math.log(1e-5)

    def test_silence_maps_to_the_protocol_log_floor_on_the_centered_path(self) -> None:
        # A silent frame clamps at the floor and compresses to its logarithm.
        spectrogram: torch.Tensor = self._centered(self._builder.silence().unsqueeze(0))
        self.assertAlmostEqual(
            float(spectrogram.min()),
            self._log_floor,
            places=4,
            msg="Silence must compress to log(log_clamp_min) exactly"
        )
        self.assertAlmostEqual(float(spectrogram.max()), self._log_floor, places=4)

    def test_silence_maps_to_the_protocol_log_floor_on_the_manual_path(self) -> None:
        # The manual path's magnitude epsilon still lands under the clamp floor.
        spectrogram: torch.Tensor = self._manual(self._builder.silence().unsqueeze(0))
        self.assertAlmostEqual(float(spectrogram.min()), self._log_floor, places=4)
        self.assertAlmostEqual(float(spectrogram.max()), self._log_floor, places=4)

    def test_no_output_falls_below_the_protocol_log_floor(self) -> None:
        # Clamping before compression bounds every bin from below on both paths.
        waveform: torch.Tensor = self._builder.broadband(0.05).unsqueeze(0)
        transform: MelSpectrogram
        for transform in (self._manual, self._centered):
            spectrogram: torch.Tensor = transform(waveform)
            self.assertTrue(
                torch.isfinite(spectrogram).all(),
                msg="Log-mel extraction must stay finite on broadband material"
            )
            self.assertGreaterEqual(
                float(spectrogram.min()),
                self._log_floor - 1e-4,
                msg=f"An output bin fell below the protocol floor {self._log_floor}"
            )

    def test_base_ten_protocol_compresses_with_the_decimal_logarithm(self) -> None:
        # The log10 branch floors silence at the decimal logarithm of the clamp.
        transform: MelSpectrogram = MelSpectrogram(self._compact_builder.base_ten_centered())
        spectrogram: torch.Tensor = transform(torch.zeros(1, 1024))
        self.assertAlmostEqual(float(spectrogram.max()), math.log10(1e-5), places=4)
        self.assertAlmostEqual(float(spectrogram.min()), math.log10(1e-5), places=4)

    def test_amplitude_gain_shifts_the_centered_log_mel_by_the_log_of_the_gain(self) -> None:
        # Magnitude spectra are linear in gain, so log compression turns it into a shift.
        # This is the strongest available statement that the extraction is a faithful
        # magnitude measurement: it holds for the whole pipeline at once, without pinning any
        # spectrogram value that a torchaudio release could legitimately change.
        waveform: torch.Tensor = self._builder.tone(440.0, 0.25).unsqueeze(0)
        quiet: torch.Tensor = self._centered(waveform)
        loud: torch.Tensor = self._centered(2.0 * waveform)
        # Bins sitting at or near the clamp floor are deliberately insensitive to gain, so the
        # identity is asserted only where the quiet spectrogram already stands two nats clear
        # of the floor and neither side of the comparison can have been clamped.
        unclamped: torch.Tensor = quiet > self._log_floor + 2.0
        self.assertTrue(bool(unclamped.any()), msg="The tone must excite bins above the clamp floor")
        deviation: float = float(((loud - quiet)[unclamped] - math.log(2.0)).abs().max())
        self.assertLess(
            deviation,
            1e-4,
            msg=f"Doubling amplitude deviated from a log(2) shift by {deviation}"
        )

    def test_amplitude_gain_shifts_the_manual_log_mel_by_the_log_of_the_gain(self) -> None:
        # The manual path shares the linearity, up to its magnitude epsilon. That epsilon is
        # added inside the square root and is therefore not proportional to the signal, so it
        # perturbs the two amplitudes unequally; the tolerance below is an order of magnitude
        # looser than the centered path's for exactly that reason, and not because the manual
        # path is held to a weaker contract.
        waveform: torch.Tensor = self._builder.tone(440.0, 0.25).unsqueeze(0)
        quiet: torch.Tensor = self._manual(waveform)
        loud: torch.Tensor = self._manual(2.0 * waveform)
        unclamped: torch.Tensor = quiet > self._log_floor + 2.0
        self.assertTrue(bool(unclamped.any()), msg="The tone must excite bins above the clamp floor")
        deviation: float = float(((loud - quiet)[unclamped] - math.log(2.0)).abs().max())
        self.assertLess(
            deviation,
            1e-3,
            msg=f"Doubling amplitude deviated from a log(2) shift by {deviation}"
        )

    def test_extraction_is_deterministic_across_repeated_calls(self) -> None:
        # A measurement protocol must return the same value for the same input.
        waveform: torch.Tensor = self._builder.tone(440.0, 0.4).unsqueeze(0)
        self.assertTrue(torch.equal(self._manual(waveform), self._manual(waveform)))


class MelBasisNormalizationTest(unittest.TestCase):
    # Verifies that the basis flags reach the filter bank the manual path builds.
    def setUp(self) -> None:
        # Builds compact transforms differing only in basis normalization or scale.
        self._builder: CompactProtocolBuilder = CompactProtocolBuilder()
        self._normalized: MelSpectrogram = MelSpectrogram(self._builder.normalized_manual())
        self._unnormalized: MelSpectrogram = MelSpectrogram(self._builder.unnormalized_manual())

    def test_unnormalized_filters_keep_unit_triangular_peaks(self) -> None:
        # Without area normalization each triangle peaks at unit gain.
        filter_bank: torch.Tensor = self._unnormalized.get_buffer("_manual_filter_bank")
        self.assertTrue(bool((filter_bank >= 0.0).all()), msg="Mel weights must be non-negative")
        self.assertLessEqual(float(filter_bank.max()), 1.0 + 1e-6)
        self.assertGreater(
            float(filter_bank.max()),
            0.9,
            msg="Unnormalized triangular filters must approach unit peak gain"
        )

    def test_slaney_normalization_shrinks_every_filter_peak(self) -> None:
        # Area normalization divides each filter by its bandwidth in hertz.
        normalized_bank: torch.Tensor = self._normalized.get_buffer("_manual_filter_bank")
        unnormalized_bank: torch.Tensor = self._unnormalized.get_buffer("_manual_filter_bank")
        self.assertEqual(normalized_bank.shape, unnormalized_bank.shape)
        self.assertTrue(
            bool((normalized_bank.max(dim=1).values < unnormalized_bank.max(dim=1).values).all()),
            msg="Every Slaney-normalized filter must peak below its unnormalized counterpart"
        )

    def test_mel_scale_choice_changes_the_filter_bank(self) -> None:
        # HTK and Slaney place band edges by different formulas.
        htk_transform: MelSpectrogram = MelSpectrogram(self._builder.htk_manual())
        htk_bank: torch.Tensor = htk_transform.get_buffer("_manual_filter_bank")
        slaney_bank: torch.Tensor = self._normalized.get_buffer("_manual_filter_bank")
        self.assertEqual(htk_bank.shape, slaney_bank.shape)
        self.assertFalse(
            torch.allclose(htk_bank, slaney_bank),
            msg="The HTK and Slaney mel scales must produce different filter banks"
        )


if __name__ == "__main__":
    unittest.main()
