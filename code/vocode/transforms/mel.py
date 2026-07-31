# This module:
# 1. Defines MelConfig, the frozen mel-spectrogram protocol record, with one
#    factory per reference recipe covering conditioning, reconstruction-loss,
#    and metric protocols
# 2. Implements MelSpectrogram, which executes a protocol either through the
#    torchaudio transform or through a manual reflect-padded STFT path that
#    reproduces the published HiFi-GAN-family extraction exactly
#
# Design decisions:
# - Mel protocols are per-architecture because reference recipes disagree on
#   sample rate, band count, frequency ceiling, mel scale, STFT centering,
#   and basis normalization; conflating them would corrupt both training
#   conditioning and the mel-error metric
# - Conditioning and reconstruction factories differ deliberately within a
#   family: references condition on a band-limited mel while their
#   reconstruction losses use the full band (fmax None)
# - The manual reflect-padding path exists because the published HiFi-GAN
#   extraction pads the waveform by (n_fft - hop) / 2 and runs an uncentered
#   STFT, a sequence the centered torchaudio transform cannot reproduce
# - The small epsilon inside the magnitude square root matches the published
#   extraction and keeps gradients finite at silent frames
# - Log compression clamps at the protocol floor before applying the
#   protocol's log base, so silence maps to a finite floor value
#
# Author: Rahul Sawhney

from typing import ClassVar, Literal, override

import torch
import torchaudio
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, PositiveFloat, PositiveInt
from torch import nn
from torch.nn import functional as F

__all__: list[str] = ["MelConfig", "MelSpectrogram"]


class MelConfig(BaseModel):
    # Frozen mel-protocol record. Instances come from the named factories;
    # each factory is one reference recipe's extraction protocol. Because a
    # mel protocol is the measurement contract of both the training
    # conditioning and the mel-error metric, every field below is a
    # published constant of a reference recipe rather than a tunable.
    #
    # The factories realize the native conditioning interface each
    # Project-Trained Configuration retains: the registered protocols share
    # a 1,024-point Hann transform with a 256-sample hop, and differ in
    # band count, mel scale, basis normalization, frequency ceiling, and
    # sample rate. A consequence is that mel error is protocol dependent
    # across the 16 kHz, 22.05 kHz, and 24 kHz configurations and is a
    # per-configuration diagnostic rather than one cross-model instrument.
    #
    # Fields:
    #     sample_rate: Waveform sample rate the protocol assumes, in hertz.
    #         It is an assumption, not a conversion: the transform does not
    #         resample, so material must already arrive at this rate.
    #     n_fft: FFT size of the STFT, which fixes the bin count at
    #         ``n_fft // 2 + 1``.
    #     hop_length: Hop between STFT frames, in samples. It sets the frame
    #         rate and therefore the time resolution of the conditioning.
    #     win_length: Analysis window length, in samples; never longer than
    #         the FFT size across the registered recipes.
    #     n_mels: Number of mel bands, which is the channel count of the
    #         extracted spectrogram.
    #     fmin: Lower frequency bound of the mel basis, in hertz.
    #     fmax: Upper frequency bound of the mel basis in hertz; ``None``
    #         extends the basis to the Nyquist frequency. Conditioning
    #         recipes band-limit here while their paired reconstruction
    #         recipes leave it unset, which is the only field in which the
    #         two members of a family differ.
    #     mel_scale: Mel-scale formula, ``"htk"`` or ``"slaney"``. The two
    #         formulas place band edges differently and therefore produce
    #         different filter banks from otherwise identical fields.
    #     center: Whether STFT frames are centered on their timestamps.
    #         Centering pads the signal internally and yields one boundary
    #         frame more than the hop count; uncentered extraction yields
    #         exactly ``time // hop_length`` frames and pairs with manual
    #         padding across every registered recipe.
    #     pad_mode: Padding mode applied by the STFT at the signal edges.
    #     power: Spectrogram magnitude exponent; one keeps amplitude rather
    #         than energy, and every registered recipe uses one.
    #     normalize_mel_basis: Whether the mel basis rows are Slaney
    #         area-normalized. Normalization divides each triangular filter
    #         by its bandwidth, so filter peaks fall below unit gain and
    #         wide high-frequency bands stop dominating narrow low ones;
    #         unnormalized filters keep unit triangular peaks.
    #     log_clamp_min: Magnitude floor applied before log compression. It
    #         bounds the output from below, so a silent frame compresses to
    #         the finite value ``log(log_clamp_min)`` instead of diverging.
    #     log_base: Log-compression base, ``"natural"`` or ``"log10"``.
    #     manual_reflect_padding: Whether extraction uses the manual
    #         reflect-padded uncentered STFT path instead of the torchaudio
    #         transform. The manual path exists because the published
    #         HiFi-GAN-family extraction pads by ``(n_fft - hop) / 2`` and
    #         then runs an uncentered STFT, which the centered torchaudio
    #         transform cannot reproduce. Default: ``False``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    sample_rate: PositiveInt
    n_fft: PositiveInt
    hop_length: PositiveInt
    win_length: PositiveInt
    n_mels: PositiveInt
    fmin: NonNegativeFloat
    fmax: PositiveFloat | None
    mel_scale: Literal["htk", "slaney"]
    center: bool
    pad_mode: Literal["reflect", "constant", "replicate"]
    power: PositiveFloat
    normalize_mel_basis: bool
    log_clamp_min: PositiveFloat
    log_base: Literal["natural", "log10"]
    manual_reflect_padding: bool = False

    @classmethod
    def hifigan_v1(cls) -> MelConfig:
        # Builds the HiFi-GAN V1 mel protocol used by the jik876 LJSpeech recipe.
        return cls.hifigan_conditioning()

    @classmethod
    def hifigan_conditioning(cls) -> MelConfig:
        # Builds the HiFi-GAN mel protocol used for generator conditioning.
        return cls(
            sample_rate=22050,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=80,
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

    @classmethod
    def hifigan_v2(cls) -> MelConfig:
        # Builds the HiFi-GAN V2 mel protocol used by the jik876 LJSpeech recipe.
        return cls.hifigan_v1()

    @classmethod
    def hifigan_v3(cls) -> MelConfig:
        # Builds the HiFi-GAN V3 mel protocol used by the jik876 LJSpeech recipe.
        return cls.hifigan_v1()

    @classmethod
    def hifigan_reconstruction(cls) -> MelConfig:
        # Builds the HiFi-GAN mel protocol used by the reconstruction loss.
        return cls(
            sample_rate=22050,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=80,
            fmin=0.0,
            fmax=None,
            mel_scale="slaney",
            center=False,
            pad_mode="reflect",
            power=1.0,
            normalize_mel_basis=True,
            log_clamp_min=1e-5,
            log_base="natural",
            manual_reflect_padding=True
        )

    @classmethod
    def hifigan_half_width_v1(cls) -> MelConfig:
        # Builds the mel protocol for the project-defined half-width HiFi-GAN V1 variant.
        return cls.hifigan_v1()

    @classmethod
    def melgan_seungwon(cls) -> MelConfig:
        # Builds the MelGAN mel protocol aligned with the Seungwon Park reference recipe.
        return cls(
            sample_rate=22050,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=80,
            fmin=0.0,
            fmax=8000.0,
            mel_scale="slaney",
            center=True,
            pad_mode="reflect",
            power=1.0,
            normalize_mel_basis=True,
            log_clamp_min=1e-5,
            log_base="natural"
        )

    @classmethod
    def vocos_charactr_mel_24khz(cls) -> MelConfig:
        # Builds the Vocos 24 kHz 100-band mel protocol used by the charactr model.
        return cls(
            sample_rate=24000,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=100,
            fmin=0.0,
            fmax=None,
            mel_scale="htk",
            center=True,
            pad_mode="reflect",
            power=1.0,
            normalize_mel_basis=False,
            log_clamp_min=1e-7,
            log_base="natural"
        )

    @classmethod
    def bigvgan_nvidia_base_24khz_100band(cls) -> MelConfig:
        # Builds the BigVGAN-base 24 kHz 100-band conditioning mel protocol.
        return cls(
            sample_rate=24000,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=100,
            fmin=0.0,
            fmax=12000.0,
            mel_scale="slaney",
            center=False,
            pad_mode="reflect",
            power=1.0,
            normalize_mel_basis=True,
            log_clamp_min=1e-5,
            log_base="natural",
            manual_reflect_padding=True
        )

    @classmethod
    def bigvgan_nvidia_base_reconstruction_24khz_100band(cls) -> MelConfig:
        # Builds the BigVGAN-base full-band mel protocol used for reconstruction losses.
        return cls(
            sample_rate=24000,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=100,
            fmin=0.0,
            fmax=None,
            mel_scale="slaney",
            center=False,
            pad_mode="reflect",
            power=1.0,
            normalize_mel_basis=True,
            log_clamp_min=1e-5,
            log_base="natural",
            manual_reflect_padding=True
        )

    @classmethod
    def apnet2_redmist328(cls) -> MelConfig:
        # Builds the APNet2 conditioning mel protocol with the center-true STFT grid.
        return cls(
            sample_rate=22050,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=80,
            fmin=0.0,
            fmax=8000.0,
            mel_scale="slaney",
            center=True,
            pad_mode="reflect",
            power=1.0,
            normalize_mel_basis=True,
            log_clamp_min=1e-5,
            log_base="natural"
        )

    @classmethod
    def apnet2_redmist328_reconstruction(cls) -> MelConfig:
        # Builds the APNet2 full-band mel protocol used for reconstruction losses.
        return cls(
            sample_rate=22050,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=80,
            fmin=0.0,
            fmax=None,
            mel_scale="slaney",
            center=True,
            pad_mode="reflect",
            power=1.0,
            normalize_mel_basis=True,
            log_clamp_min=1e-5,
            log_base="natural"
        )

    @classmethod
    def freev_official(cls) -> MelConfig:
        # Builds the FreeV conditioning mel protocol with the center-true STFT grid.
        return cls(
            sample_rate=22050,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=80,
            fmin=0.0,
            fmax=8000.0,
            mel_scale="slaney",
            center=True,
            pad_mode="reflect",
            power=1.0,
            normalize_mel_basis=True,
            log_clamp_min=1e-5,
            log_base="natural"
        )

    @classmethod
    def freev_official_reconstruction(cls) -> MelConfig:
        # Builds the FreeV full-band mel protocol used for reconstruction losses.
        return cls(
            sample_rate=22050,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=80,
            fmin=0.0,
            fmax=None,
            mel_scale="slaney",
            center=True,
            pad_mode="reflect",
            power=1.0,
            normalize_mel_basis=True,
            log_clamp_min=1e-5,
            log_base="natural"
        )

    @classmethod
    def hiftnet_yl4579(cls) -> MelConfig:
        # Builds the HiFTNet mel protocol aligned with the yl4579 reference recipe.
        return cls.hifigan_v1()

    @classmethod
    def rndvoc_andong(cls) -> MelConfig:
        # Builds the RNDVoC conditioning mel protocol with the center-true STFT grid.
        return cls(
            sample_rate=22050,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=80,
            fmin=0.0,
            fmax=8000.0,
            mel_scale="slaney",
            center=True,
            pad_mode="reflect",
            power=1.0,
            normalize_mel_basis=True,
            log_clamp_min=1e-5,
            log_base="natural"
        )

    @classmethod
    def rndvoc_andong_reconstruction(cls) -> MelConfig:
        # Builds the RNDVoC full-band mel protocol used for reconstruction losses.
        return cls(
            sample_rate=22050,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=80,
            fmin=0.0,
            fmax=None,
            mel_scale="slaney",
            center=True,
            pad_mode="reflect",
            power=1.0,
            normalize_mel_basis=True,
            log_clamp_min=1e-5,
            log_base="natural"
        )

    @classmethod
    def lpcnet_metric_16khz(cls) -> MelConfig:
        # Builds the 16 kHz mel protocol used only by the LPCNet mel-error test metric.
        return cls(
            sample_rate=16000,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=80,
            fmin=0.0,
            fmax=None,
            mel_scale="slaney",
            center=True,
            pad_mode="reflect",
            power=1.0,
            normalize_mel_basis=True,
            log_clamp_min=1e-5,
            log_base="natural"
        )


class MelSpectrogram(nn.Module):
    # Executable form of one mel protocol. The extraction path is chosen at
    # construction: the manual reflect-padded STFT for protocols that demand
    # it, the torchaudio transform otherwise. The protocol is bound for the
    # lifetime of the instance, so one transform measures under exactly one
    # contract and a recipe change requires a new instance.
    #
    # Integration: models compose this transform to build their
    # conditioning input and losses compose it to compare a synthesized
    # waveform against its reference. Both sides construct it from a named
    # MelConfig factory rather than assembling fields by hand, which is
    # what keeps a family's conditioning and reconstruction protocols
    # distinguishable.
    def __init__(self, configuration: MelConfig) -> None:
        # Prepares the selected path: the manual path registers the window
        # and mel filter bank as non-persistent buffers (they follow device
        # movement but stay out of checkpoints), while the torchaudio path
        # constructs the equivalent transform from the protocol fields.
        #
        # Args:
        #     configuration: Mel protocol this transform executes. Its
        #         manual_reflect_padding field selects the extraction path
        #         and therefore what the instance registers: the manual
        #         path owns two buffers and no submodule, the torchaudio
        #         path owns a submodule and no buffer.
        super().__init__()
        self._configuration: MelConfig = configuration
        self._spectrogram: torchaudio.transforms.MelSpectrogram | None = None
        if configuration.manual_reflect_padding:
            manual_window: torch.Tensor = torch.hann_window(configuration.win_length)
            manual_filter_bank: torch.Tensor = torchaudio.functional.melscale_fbanks(
                n_freqs=configuration.n_fft // 2 + 1,
                f_min=configuration.fmin,
                f_max=configuration.fmax if configuration.fmax is not None else configuration.sample_rate / 2,
                n_mels=configuration.n_mels,
                sample_rate=configuration.sample_rate,
                norm="slaney" if configuration.normalize_mel_basis else None,
                mel_scale=configuration.mel_scale
            ).transpose(0, 1)
            self.register_buffer("_manual_window", manual_window, persistent=False)
            self.register_buffer("_manual_filter_bank", manual_filter_bank, persistent=False)
        else:
            self._spectrogram: torchaudio.transforms.MelSpectrogram | None = torchaudio.transforms.MelSpectrogram(
                sample_rate=configuration.sample_rate,
                n_fft=configuration.n_fft,
                win_length=configuration.win_length,
                hop_length=configuration.hop_length,
                n_mels=configuration.n_mels,
                f_min=configuration.fmin,
                f_max=configuration.fmax,
                power=configuration.power,
                center=configuration.center,
                pad_mode=configuration.pad_mode,
                norm="slaney" if configuration.normalize_mel_basis else None,
                mel_scale=configuration.mel_scale
            )

    @property
    def configuration(self) -> MelConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration

    @override
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Extracts the log-mel spectrogram: magnitude extraction through the
        # protocol's path, clamping at the protocol floor, then compression
        # with the protocol's log base. Clamping strictly before compression
        # is what bounds the output from below, so no bin can fall under
        # ``log(log_clamp_min)`` however quiet the input is.
        #
        # Args:
        #     waveform: Audio at the protocol's sample rate. The manual
        #         path accepts ``[time]``, ``[batch, time]``, or
        #         ``[batch, 1, time]``; the torchaudio path applies the
        #         transform's own rank handling.
        #
        # Returns:
        #     The log-mel spectrogram. The manual path always returns
        #     ``[batch, n_mels, frames]`` because it promotes every
        #     accepted rank to a batch first, whereas the torchaudio path
        #     preserves the leading rank and maps ``[time]`` onto
        #     ``[n_mels, frames]``. Frame counts follow the protocol's
        #     centering: ``time // hop_length`` uncentered, one more when
        #     centered.
        #
        # Raises:
        #     ValueError: If the manual path receives a waveform rank it
        #         does not accept, or if the protocol declares a log base
        #         outside the supported pair.
        #     RuntimeError: If a torchaudio-path protocol reaches
        #         extraction without its transform having been built.
        if self._configuration.manual_reflect_padding:
            mel_magnitude: torch.Tensor = self._manual_mel_spectrogram(waveform)
        else:
            if self._spectrogram is None:
                raise RuntimeError("torchaudio spectrogram is not initialized for this mel protocol")
            mel_magnitude: torch.Tensor = self._spectrogram(waveform)
        clamped: torch.Tensor = torch.clamp(mel_magnitude, min=self._configuration.log_clamp_min)
        match self._configuration.log_base:
            case "natural":
                return torch.log(clamped)
            case "log10":
                return torch.log10(clamped)
            case _:
                raise ValueError(f"Unsupported log_base: {self._configuration.log_base}")

    def _manual_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        # Applies explicit reflect padding before STFT to match the published extraction protocol.
        # The published extraction is a fixed sequence: normalize the rank, reflect-pad the
        # waveform by half the difference between the FFT size and the hop, run an uncentered
        # STFT under the registered Hann window, take magnitudes, and project them onto the mel
        # basis. Padding outside the transform rather than inside it is precisely what the
        # centered torchaudio path cannot express.
        prepared_waveform: torch.Tensor = self._prepare_waveform(waveform)
        padding_amount: int = (self._configuration.n_fft - self._configuration.hop_length) // 2
        if padding_amount > 0:
            prepared_waveform: torch.Tensor = self._pad_for_reflect_stft(prepared_waveform, padding_amount)
        # The window and basis are moved rather than cast in place, so one transform serves
        # inputs on any device and in any floating dtype without mutating its buffers.
        window: torch.Tensor = self._manual_window.to(
            device=prepared_waveform.device,
            dtype=prepared_waveform.dtype
        )
        complex_spectrum: torch.Tensor = torch.stft(
            prepared_waveform,
            n_fft=self._configuration.n_fft,
            hop_length=self._configuration.hop_length,
            win_length=self._configuration.win_length,
            window=window,
            center=self._configuration.center,
            pad_mode=self._configuration.pad_mode,
            normalized=False,
            onesided=True,
            return_complex=True
        )
        # The epsilon inside the square root matches the published extraction: it leaves audible
        # magnitudes unchanged while keeping the derivative of the root finite at silent bins,
        # where an exact zero would otherwise produce an infinite gradient.
        magnitude: torch.Tensor = torch.sqrt(torch.abs(complex_spectrum).pow(2) + 1e-9)
        filter_bank: torch.Tensor = self._manual_filter_bank.to(
            device=magnitude.device,
            dtype=magnitude.dtype
        )
        return torch.matmul(filter_bank, magnitude)

    def _prepare_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        # Converts accepted waveform shapes to the [batch, time] tensor expected by torch.stft.
        # A single channel is squeezed out rather than averaged, because a genuinely
        # multi-channel signal has no defined single-track mel protocol and is rejected instead.
        #
        # Args:
        #     waveform: Audio shaped ``[time]``, ``[batch, time]``, or
        #         ``[batch, 1, time]``.
        #
        # Raises:
        #     ValueError: If the rank or the channel axis is anything else,
        #         which names the received shape so the caller can see what
        #         reached the transform.
        if waveform.ndim == 1:
            return waveform.unsqueeze(0)
        if waveform.ndim == 2:
            return waveform
        if waveform.ndim == 3 and waveform.shape[1] == 1:
            return waveform.squeeze(1)
        raise ValueError(
            f"Expected waveform shape [time], [batch, time], or [batch, 1, time], "
            f"got {tuple(waveform.shape)}"
        )

    def _pad_for_reflect_stft(self, waveform: torch.Tensor, padding_amount: int) -> torch.Tensor:
        # Reflect padding requires the source length to be larger than the requested padding.
        # An utterance shorter than the padding is zero-extended to the minimum admissible
        # length first and only then reflected, so a short clip yields a short spectrogram
        # instead of raising and removing the utterance from the measurement altogether.
        if waveform.shape[-1] <= padding_amount:
            minimum_source_samples: int = padding_amount + 1
            missing_samples: int = minimum_source_samples - waveform.shape[-1]
            waveform: torch.Tensor = F.pad(waveform, (0, missing_samples))
        return F.pad(waveform.unsqueeze(1), (padding_amount, padding_amount), mode="reflect").squeeze(1)
