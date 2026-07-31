# This module:
# 1. Implements FreevNetwork, the FreeV generator: the pseudo-inverse mel
#    prior producing the analytic amplitude estimate, a refinement stream
#    over that estimate, and the phase stream, combined through an inverse
#    STFT into the waveform
# 2. Defines FreevGeneratorOutput, the frozen record carrying every spectral
#    quantity the FreeV loss composition supervises alongside the waveform
# 3. Defines the primitives the refinement and phase stacks share: the
#    global response normalization applied to expanded features and the
#    residual block itself
#
# Design decisions:
# - The pseudo-inverse projection of the mel basis is precomputed and
#   fixed; the amplitude stream learns only the residual correction,
#   which is the architecture's parameter-efficiency idea
# - Phase is predicted as real and imaginary components whose arctangent
#   yields wrapped phase directly
# - The amplitude stream carries no learned entry or exit projection at all:
#   the analytic estimate is already on the STFT bin grid, so the refinement
#   blocks operate at bin width and the stream has no head to speak of
# - The projection is a non-persistent buffer, so it is recomputed at
#   construction rather than restored, and the strict author load neither
#   expects nor supplies it
# - The polar recombination and the inverse STFT are forced to float32
#   because torch complex operations reject reduced precision and
#   exponentiating an unbounded log-amplitude overflows in half precision
#
# Author: Rahul Sawhney

from contextlib import nullcontext
from typing import ClassVar, override

import torch
import torchaudio
from pydantic import BaseModel, ConfigDict
from torch import nn

__all__: list[str] = ["FreevGeneratorOutput", "FreevNetwork"]


class FreevGeneratorOutput(BaseModel):
    # Frozen bundle of everything one generator pass produces. The FreeV
    # objective supervises the spectral members directly rather than the
    # waveform alone, so all of them travel to the loss composition together;
    # freezing the record guarantees that no loss term can overwrite a
    # component another term has already consumed.
    #
    # Fields:
    #     log_amplitude: Natural logarithm of the predicted amplitude
    #         spectrum, shaped ``[batch, n_fft // 2 + 1, frames]``; it is the
    #         refined analytic prior, not a projection learned from scratch.
    #     phase: Wrapped phase spectrum on the same grid, obtained as the
    #         arctangent of the imaginary over the real projection and
    #         therefore confined to ``[-pi, pi]``.
    #     real_spectrum: Real part of the recombined complex spectrum,
    #         ``exp(log_amplitude) * cos(phase)``.
    #     imaginary_spectrum: Imaginary part of the recombined complex
    #         spectrum, ``exp(log_amplitude) * sin(phase)``.
    #     waveform: Synthesized audio shaped ``[batch, 1, samples]``, the
    #         centered inverse STFT of the recombined spectrum.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    log_amplitude: torch.Tensor
    phase: torch.Tensor
    real_spectrum: torch.Tensor
    imaginary_spectrum: torch.Tensor
    waveform: torch.Tensor


class _GlobalResponseNormalization(nn.Module):
    # Global response normalization of the expanded features inside each
    # residual block. For every feature the L2 norm across the time axis is
    # divided by the mean of those norms over all features, producing a
    # relative-response factor that amplifies features responding above the
    # layer average and attenuates the rest. The learned per-feature gamma and
    # beta modulate that rescaled signal and the untouched input is added back.
    def __init__(self, dimension: int) -> None:
        # Allocates the per-feature modulation pair at zero, so the module is
        # an exact identity at initialization and the calibration it applies
        # is learned rather than imposed from the first step.
        #
        # Args:
        #     dimension: Feature width of the expanded activation this module
        #         normalizes, matching the block's intermediate dimension.
        super().__init__()
        self.gamma: nn.Parameter = nn.Parameter(torch.zeros(1, 1, dimension))
        self.beta: nn.Parameter = nn.Parameter(torch.zeros(1, 1, dimension))

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Computes the relative response of a channel-last
        # ``[batch, frames, features]`` activation and applies the learned
        # modulation on top of the preserved input.
        response_norm: torch.Tensor = torch.norm(x, p=2, dim=1, keepdim=True)
        normalized_response: torch.Tensor = response_norm / (response_norm.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * normalized_response) + self.beta + x


class _ConvnextBlock(nn.Module):
    # One residual block of either stream. A depthwise convolution mixes seven
    # neighboring frames within each channel, the channel-last stage then
    # normalizes, expands to the intermediate width, activates, calibrates
    # through the global response normalization, and projects back, and the
    # block input is added to the result. The phase stack instantiates it at
    # the phase working width while the amplitude stack instantiates it at the
    # STFT bin width, which is what lets the refinement operate directly on the
    # analytic estimate without any projection around it.
    def __init__(self, dimension: int, intermediate_dimension: int) -> None:
        # Builds the depthwise-then-pointwise chain. The depthwise convolution
        # pads by three on each side so its seven-tap kernel leaves the frame
        # count unchanged, and its group count equals its channel count, which
        # is what makes it depthwise.
        #
        # Args:
        #     dimension: Channel width the block consumes and returns.
        #     intermediate_dimension: Expanded width at which the activation
        #         and the response normalization operate.
        super().__init__()
        self.dwconv: nn.Conv1d = nn.Conv1d(
            dimension,
            dimension,
            kernel_size=7,
            padding=3,
            groups=dimension
        )
        self.norm: nn.LayerNorm = nn.LayerNorm(dimension, eps=1e-6)
        self.pwconv1: nn.Linear = nn.Linear(dimension, intermediate_dimension)
        self.act: nn.GELU = nn.GELU()
        self.grn: _GlobalResponseNormalization = _GlobalResponseNormalization(intermediate_dimension)
        self.pwconv2: nn.Linear = nn.Linear(intermediate_dimension, dimension)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Runs the block over a channel-first ``[batch, channels, frames]``
        # activation and returns the residual sum. The two transposes bracket
        # the pointwise stage because nn.LayerNorm and nn.Linear act on the
        # trailing axis while the convolution and the block interface are
        # channel-first.
        residual: torch.Tensor = x
        x: torch.Tensor = self.dwconv(x)
        x: torch.Tensor = x.transpose(1, 2)
        x: torch.Tensor = self.norm(x)
        x: torch.Tensor = self.pwconv1(x)
        x: torch.Tensor = self.act(x)
        x: torch.Tensor = self.grn(x)
        x: torch.Tensor = self.pwconv2(x)
        x: torch.Tensor = x.transpose(1, 2)
        return residual + x


class FreevNetwork(nn.Module):
    # The FreeV generator, and the architecture's whole claim in one class.
    # The phase branch conditions on the mel spectrogram and the amplitude branch refines the
    # pseudo-inverse mel projection, matching the released FreeV checkpoint structure exactly,
    # including two reference LayerNorm modules that the reference forward pass never invokes.
    #
    # The amplitude branch is what distinguishes this network. A mel
    # spectrogram is a known linear projection of the amplitude spectrum, so
    # the least-squares inverse of that projection already recovers a usable
    # amplitude estimate analytically, with no parameters at all. The network
    # therefore precomputes the pseudo-inverse of the mel basis once, applies
    # it at every forward pass, and spends its amplitude capacity purely on
    # refining the result. Because the estimate already lives on the STFT bin
    # grid, the refinement stack needs no entry projection to enter that grid
    # and no head to leave it, and the whole branch collapses to a small number
    # of residual blocks; the published recipe uses one.
    #
    # The phase branch has no such analytic shortcut, since a mel spectrogram
    # discards phase entirely, and is consequently the larger of the two: it
    # projects the mel to its working width, refines through its own residual
    # stack, and ends in two sibling projections whose arctangent is the
    # wrapped phase. Predicting a real and imaginary pair rather than a scalar
    # keeps phase wrapped by construction, and no unwrapping, branch selection,
    # or iterative phase-recovery procedure appears anywhere in the pass;
    # phase is produced in one shot alongside amplitude.
    #
    # The branches meet only at the end, where the polar recombination
    # ``exp(log_amplitude) * (cos(phase) + i * sin(phase))`` forms the complex
    # spectrum that one centered inverse STFT converts into the waveform.
    #
    # In the report's generation-mechanism taxonomy this places FreeV among
    # the adversarially trained spectral models with inverse-STFT heads,
    # conditioned by the official recipe on 80 mel bands at 22.05 kHz.
    #
    # Integration: the module wrapper (vocode.models.freev.freev.Freev) calls
    # predict_components during training and validation, because the loss
    # composition supervises the spectral members and not only the waveform,
    # and calls forward when it needs synthesis alone. The published release is
    # loaded strictly onto this class by
    # vocode.models.freev.weights.FreevWeights, so every attribute name
    # declared here is part of the release contract, including the two
    # normalization modules that exist solely to satisfy it.
    def __init__(
        self,
        num_mels: int,
        n_fft: int,
        hop_size: int,
        win_size: int,
        sampling_rate: int,
        fmin: float,
        fmax: float | None,
        psp_channel: int,
        psp_input_conv_kernel_size: int,
        psp_output_r_conv_kernel_size: int,
        psp_output_i_conv_kernel_size: int,
        convnext_layer_count: int = 8,
        amplitude_refinement_layer_count: int = 1,
        convnext_intermediate_dimension: int = 1536
    ) -> None:
        # Builds the analytic prior, both stacks, and the phase projections,
        # then applies the reference initialization to every convolution and
        # linear layer.
        #
        # The mel basis is reconstructed here from the same band edges and
        # scale the conditioning protocol uses, because the prior is only a
        # valid inverse of the mel the network is actually fed; a mismatch
        # between the two would make the amplitude estimate systematically
        # wrong before any refinement runs.
        #
        # Args:
        #     num_mels: Band count of the conditioning mel and the column count
        #         of the projection.
        #     n_fft: Transform size; the one-sided bin count is both the
        #         projection's row count and the width the refinement stack
        #         operates at.
        #     hop_size: Hop of the inverse STFT, the samples each conditioning
        #         frame expands into.
        #     win_size: Window length of the inverse STFT.
        #     sampling_rate: Rate the mel basis is constructed against; it also
        #         supplies the Nyquist ceiling when no upper edge is given.
        #     fmin: Lower band edge of the mel basis.
        #     fmax: Upper band edge, or ``None`` to use the Nyquist frequency.
        #     psp_channel: Working width of the phase stream.
        #     psp_input_conv_kernel_size: Kernel of the phase entry
        #         projection.
        #     psp_output_r_conv_kernel_size: Kernel of the real phase head.
        #     psp_output_i_conv_kernel_size: Kernel of the imaginary phase
        #         head.
        #     convnext_layer_count: Residual block depth of the phase stream.
        #         Default: ``8``.
        #     amplitude_refinement_layer_count: Residual block depth of the
        #         amplitude refinement; the published recipe refines with a
        #         single block. Default: ``1``.
        #     convnext_intermediate_dimension: Expanded width inside every
        #         residual block of both streams. Default: ``1536``.
        super().__init__()
        self._n_fft: int = n_fft
        self._hop_size: int = hop_size
        self._win_size: int = win_size
        stft_bin_count: int = n_fft // 2 + 1
        # The filter bank is produced as bins by bands and transposed to bands
        # by bins, which is the orientation that maps a spectrum onto a mel;
        # its pseudo-inverse therefore maps a mel back onto a spectrum.
        mel_filter_bank: torch.Tensor = torchaudio.functional.melscale_fbanks(
            n_freqs=stft_bin_count,
            f_min=fmin,
            f_max=fmax if fmax is not None else float(sampling_rate / 2),
            n_mels=num_mels,
            sample_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney"
        ).transpose(0, 1)
        # Non-persistent: the projection is a deterministic function of the
        # constructor arguments, so it is rebuilt on every construction and
        # deliberately kept out of the state dictionary, which is what lets the
        # strict author load succeed without the release carrying it.
        self.register_buffer("_inverse_mel_filter_bank", mel_filter_bank.pinverse(), persistent=False)
        self.PSP_input_conv: nn.Conv1d = nn.Conv1d(
            num_mels,
            psp_channel,
            psp_input_conv_kernel_size,
            1,
            padding=int((psp_input_conv_kernel_size - 1) / 2)
        )
        self.PSP_output_R_conv: nn.Conv1d = nn.Conv1d(
            psp_channel,
            stft_bin_count,
            psp_output_r_conv_kernel_size,
            1,
            padding=int((psp_output_r_conv_kernel_size - 1) / 2)
        )
        self.PSP_output_I_conv: nn.Conv1d = nn.Conv1d(
            psp_channel,
            stft_bin_count,
            psp_output_i_conv_kernel_size,
            1,
            padding=int((psp_output_i_conv_kernel_size - 1) / 2)
        )
        # norm and final_layer_norm bracket the phase stack. The suffixed pair
        # is the amplitude stream's counterpart in the released checkpoint, but
        # the reference forward pass never applies them, since the analytic
        # estimate is not normalized; they are materialized at the phase width
        # purely so the strict load finds every key it carries.
        self.norm: nn.LayerNorm = nn.LayerNorm(psp_channel, eps=1e-6)
        self.norm2: nn.LayerNorm = nn.LayerNorm(psp_channel, eps=1e-6)
        self.convnext: nn.ModuleList = nn.ModuleList(
            [
                _ConvnextBlock(psp_channel, convnext_intermediate_dimension)
                for _ in range(convnext_layer_count)
            ]
        )
        # The refinement stack runs at bin width, not at a projected working
        # width, because its input is already the amplitude estimate itself.
        self.convnext2: nn.ModuleList = nn.ModuleList(
            [
                _ConvnextBlock(stft_bin_count, convnext_intermediate_dimension)
                for _ in range(amplitude_refinement_layer_count)
            ]
        )
        self.final_layer_norm: nn.LayerNorm = nn.LayerNorm(psp_channel, eps=1e-6)
        self.final_layer_norm2: nn.LayerNorm = nn.LayerNorm(psp_channel, eps=1e-6)
        self.apply(self._initialize_weights)

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Synthesis entry point: runs the full component prediction and returns
        # only the waveform, discarding the spectral members that the training
        # path consumes through predict_components.
        return self.predict_components(mel).waveform

    def predict_components(self, mel: torch.Tensor) -> FreevGeneratorOutput:
        # Runs both branches over the conditioning mel and returns every
        # spectral quantity together with the synthesized waveform.
        #
        # The amplitude branch exponentiates the log-mel back to the linear
        # domain, lifts it to the bin grid through the fixed projection, takes
        # the magnitude because a least-squares inverse can return negative
        # entries that no amplitude spectrum has, floors the result so the
        # logarithm is finite, and hands the resulting estimate to the
        # refinement blocks. The phase branch projects, refines, normalizes,
        # and emits its paired projections. The two are then recombined in
        # polar form and inverted by a centered inverse STFT under a Hann
        # window built on the mel's device.
        #
        # Args:
        #     mel: Conditioning mel shaped ``[batch, bands, frames]``; the
        #         batch axis must be explicit, because the branches cannot
        #         distinguish an unbatched mel from a batch of single-band
        #         frames.
        #
        # Raises:
        #     ValueError: If the mel does not carry exactly three dimensions.
        #
        # Returns:
        #     A FreevGeneratorOutput whose spectral members share the shape
        #     ``[batch, n_fft // 2 + 1, frames]`` and whose waveform is shaped
        #     ``[batch, 1, (frames - 1) * hop_size]`` under the centered
        #     inverse-STFT arithmetic.
        if mel.ndim != 3:
            raise ValueError(f"Expected mel shape [batch, channels, frames], got {tuple(mel.shape)}")
        # The pseudo-inverse projection is the amplitude estimate itself, so its matmul and
        # exponential run outside autocast to keep the prior at full precision.
        with self._full_precision_context(mel.device.type):
            inverse_amplitude: torch.Tensor = (
                self._inverse_mel_filter_bank.to(mel.device) @ torch.exp(mel.float())
            ).abs().clamp_min(1e-5)
        log_amplitude: torch.Tensor = inverse_amplitude.log()
        # Amplitude refinement: a residual correction on the analytic estimate,
        # applied in place on the bin grid with no projection on either side.
        for convnext_block in self.convnext2:
            log_amplitude: torch.Tensor = convnext_block(log_amplitude)
        # Phase branch: entry projection, normalization, stack, exit
        # normalization, then the paired projections whose ratio carries phase.
        phase_features: torch.Tensor = self.PSP_input_conv(mel)
        phase_features: torch.Tensor = self.norm(phase_features.transpose(1, 2)).transpose(1, 2)
        for convnext_block in self.convnext:
            phase_features: torch.Tensor = convnext_block(phase_features)
        phase_features: torch.Tensor = self.final_layer_norm(phase_features.transpose(1, 2)).transpose(1, 2)
        real_phase_projection: torch.Tensor = self.PSP_output_R_conv(phase_features)
        imaginary_phase_projection: torch.Tensor = self.PSP_output_I_conv(phase_features)
        # The complex reconstruction and inverse STFT run in float32 because reduced-precision
        # complex operations are unsupported and exp(log_amplitude) can overflow in half precision.
        log_amplitude: torch.Tensor = log_amplitude.float()
        phase: torch.Tensor = torch.atan2(imaginary_phase_projection.float(), real_phase_projection.float())
        real_spectrum: torch.Tensor = torch.exp(log_amplitude) * torch.cos(phase)
        imaginary_spectrum: torch.Tensor = torch.exp(log_amplitude) * torch.sin(phase)
        complex_spectrum: torch.Tensor = torch.complex(real_spectrum, imaginary_spectrum)
        audio: torch.Tensor = torch.istft(
            complex_spectrum,
            self._n_fft,
            hop_length=self._hop_size,
            win_length=self._win_size,
            window=torch.hann_window(self._win_size, device=mel.device),
            center=True
        )
        return FreevGeneratorOutput(
            log_amplitude=log_amplitude,
            phase=phase,
            real_spectrum=real_spectrum,
            imaginary_spectrum=imaginary_spectrum,
            waveform=audio.unsqueeze(1)
        )

    def _initialize_weights(self, module: nn.Module) -> None:
        # Reference initialization, dispatched to every submodule through
        # nn.Module.apply at the end of construction. Convolutions and linear
        # layers receive truncated-normal weights at the reference deviation
        # with zero bias; normalization layers are deliberately untouched and
        # keep the torch defaults the reference also relies on. The projection
        # is a buffer and is therefore never visited.
        #
        # Args:
        #     module: Submodule visited by the apply traversal; anything other
        #         than a Conv1d or Linear is left as constructed.
        if isinstance(module, nn.Conv1d | nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.constant_(module.bias, 0)

    def _full_precision_context(self, device_type: str) -> torch.autocast | nullcontext[None]:
        # Returns an autocast-disabled context on devices where mixed precision is active.
        # The device type is taken as an argument rather than read from the
        # module, because the network is an nn.Module with no harness device
        # mirror and the caller already holds the conditioning tensor.
        if device_type in ("cuda", "cpu"):
            return torch.autocast(device_type=device_type, enabled=False)
        return nullcontext()
