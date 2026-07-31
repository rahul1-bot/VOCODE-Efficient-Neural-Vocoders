# This module:
# 1. Implements Apnet2Network, the APNet2 generator: two structurally
#    identical ConvNeXt-style stacks run in parallel over the conditioning
#    mel, one emitting the log-amplitude spectrum and one emitting the real
#    and imaginary phase projections, recombined through a single inverse
#    STFT into the waveform
# 2. Defines Apnet2GeneratorOutput, the frozen record carrying every spectral
#    quantity the APNet2 loss composition supervises alongside the waveform
# 3. Defines the primitives both stacks share: the length-invariant padding
#    rule, the global response normalization applied to expanded features,
#    and the residual block itself
#
# Design decisions:
# - Amplitude and phase are predicted by separate streams because their
#   loss supervision differs (log-amplitude regression versus
#   anti-wrapping phase losses), per the reference
# - The phase head predicts real and imaginary components whose
#   arctangent yields wrapped phase directly, avoiding unwrapping
# - Each stream owns its entry and exit layer normalizations and its own
#   ConvNeXt stack, so the conditioning mel is the only tensor they share
#   and neither stream can perturb the other's features
# - The polar recombination and the inverse STFT are forced to float32
#   because torch complex operations reject reduced precision and
#   exponentiating an unbounded log-amplitude overflows in half precision
# - The analysis window is rebuilt on the mel's device at every call rather
#   than registered as a buffer, so the network holds no window state that
#   the strict author load would have to account for
#
# Author: Rahul Sawhney

from typing import ClassVar, override

import torch
from pydantic import BaseModel, ConfigDict
from torch import nn

__all__: list[str] = ["Apnet2GeneratorOutput", "Apnet2Network"]


class Apnet2GeneratorOutput(BaseModel):
    # Frozen bundle of everything one generator pass produces. The APNet2
    # objective supervises the spectral members directly rather than the
    # waveform alone, so all of them travel to the loss composition together;
    # freezing the record guarantees that no loss term can overwrite a
    # component another term has already consumed.
    #
    # Fields:
    #     log_amplitude: Natural logarithm of the predicted amplitude
    #         spectrum, shaped ``[batch, n_fft // 2 + 1, frames]`` and cast to
    #         float32 before the recombination.
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


class _ConvolutionPadding:
    # Padding rule shared by every stream convolution. Both streams operate at
    # frame rate and the synthesis length is derived from the conditioning
    # frame count, so padding is computed from the kernel geometry rather than
    # chosen per call site, and no convolution may alter the time axis.
    def compute(self, kernel_size: int, dilation: int = 1) -> int:
        # Returns the symmetric padding that leaves the time axis unchanged
        # for a stride-one convolution of this kernel and dilation, which is
        # half the dilated kernel extent.
        #
        # Args:
        #     kernel_size: Kernel extent along the time axis.
        #     dilation: Spacing between kernel taps. Default: ``1``.
        return int((kernel_size * dilation - dilation) / 2)


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
    # block input is added to the result. Both streams instantiate this block
    # at their own width, which is what makes the two stacks structurally
    # identical and leaves the heads as their only architectural difference.
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


class Apnet2Network(nn.Module):
    # The APNet2 generator. Two independent streams read the same conditioning
    # mel and exchange no intermediate tensor. The amplitude stream projects
    # the mel to asp_channel features, refines them through its own stack of
    # residual blocks, and projects onto one log-amplitude value per one-sided
    # STFT bin. The phase stream performs the identical sequence at
    # psp_channel width but terminates in two sibling projections, one real
    # and one imaginary, whose arctangent is the phase.
    #
    # The split exists because the two quantities answer to incompatible
    # objectives: amplitude is supervised by direct regression on the log
    # spectrum, while phase is supervised by anti-wrapping losses that are only
    # meaningful on a wrapped quantity. Emitting phase as a real and imaginary
    # pair rather than as a scalar is what keeps it wrapped by construction,
    # because the arctangent of any pair already lies in the principal
    # interval; no unwrapping, branch selection, or iterative phase recovery
    # occurs anywhere in the pass.
    #
    # The predictions meet only at the end, where the polar recombination
    # ``exp(log_amplitude) * (cos(phase) + i * sin(phase))`` forms the complex
    # spectrum that one centered inverse STFT converts into the waveform. The
    # whole generator is therefore a single frame-level pass with no
    # autoregression, and the synthesis length follows directly from the
    # conditioning frame count.
    #
    # In the report's generation-mechanism taxonomy this places APNet2 among
    # the adversarially trained spectral models with inverse-STFT heads,
    # conditioned by the redmist328 recipe on 80 mel bands at 22.05 kHz.
    #
    # Integration: the module wrapper
    # (vocode.models.apnet2.apnet2.Apnet2) calls predict_components during
    # training and validation, because the loss composition supervises the
    # spectral members and not only the waveform, and calls forward when it
    # needs synthesis alone. The published redmist328 release is loaded
    # strictly onto this class by vocode.models.apnet2.weights.Apnet2Weights,
    # so every attribute name declared here is part of the release contract
    # and renaming any of them breaks that load.
    def __init__(
        self,
        num_mels: int,
        n_fft: int,
        hop_size: int,
        win_size: int,
        asp_channel: int,
        psp_channel: int,
        asp_input_conv_kernel_size: int,
        asp_output_conv_kernel_size: int,
        psp_input_conv_kernel_size: int,
        psp_output_r_conv_kernel_size: int,
        psp_output_i_conv_kernel_size: int,
        convnext_layer_count: int = 8,
        convnext_intermediate_dimension: int = 1536
    ) -> None:
        # Builds both streams at their configured widths and applies the
        # reference initialization to every convolution and linear layer. The
        # STFT grid is retained because the inverse transform at the end of
        # predict_components must invert the same analysis the loss targets
        # were extracted under.
        #
        # Args:
        #     num_mels: Band count of the conditioning mel, the input width of
        #         both entry convolutions.
        #     n_fft: Transform size; the one-sided bin count ``n_fft // 2 + 1``
        #         is the output width of all three head projections.
        #     hop_size: Hop of the inverse STFT, the samples each conditioning
        #         frame expands into.
        #     win_size: Window length of the inverse STFT.
        #     asp_channel: Working width of the amplitude stream.
        #     psp_channel: Working width of the phase stream.
        #     asp_input_conv_kernel_size: Kernel of the amplitude entry
        #         projection.
        #     asp_output_conv_kernel_size: Kernel of the log-amplitude head.
        #     psp_input_conv_kernel_size: Kernel of the phase entry
        #         projection.
        #     psp_output_r_conv_kernel_size: Kernel of the real phase head.
        #     psp_output_i_conv_kernel_size: Kernel of the imaginary phase
        #         head.
        #     convnext_layer_count: Residual block depth of each stream; both
        #         stacks receive the same depth. Default: ``8``.
        #     convnext_intermediate_dimension: Expanded width inside every
        #         residual block of both streams. Default: ``1536``.
        super().__init__()
        padding: _ConvolutionPadding = _ConvolutionPadding()
        self._n_fft: int = n_fft
        self._hop_size: int = hop_size
        self._win_size: int = win_size
        stft_bin_count: int = n_fft // 2 + 1
        self.ASP_input_conv: nn.Conv1d = nn.Conv1d(
            num_mels,
            asp_channel,
            asp_input_conv_kernel_size,
            1,
            padding=padding.compute(asp_input_conv_kernel_size, 1)
        )
        self.PSP_input_conv: nn.Conv1d = nn.Conv1d(
            num_mels,
            psp_channel,
            psp_input_conv_kernel_size,
            1,
            padding=padding.compute(psp_input_conv_kernel_size, 1)
        )
        self.ASP_output_conv: nn.Conv1d = nn.Conv1d(
            asp_channel,
            stft_bin_count,
            asp_output_conv_kernel_size,
            1,
            padding=padding.compute(asp_output_conv_kernel_size, 1)
        )
        self.PSP_output_R_conv: nn.Conv1d = nn.Conv1d(
            psp_channel,
            stft_bin_count,
            psp_output_r_conv_kernel_size,
            1,
            padding=padding.compute(psp_output_r_conv_kernel_size, 1)
        )
        self.PSP_output_I_conv: nn.Conv1d = nn.Conv1d(
            psp_channel,
            stft_bin_count,
            psp_output_i_conv_kernel_size,
            1,
            padding=padding.compute(psp_output_i_conv_kernel_size, 1)
        )
        # The unsuffixed normalizations and stack belong to the phase stream
        # and the suffixed ones to the amplitude stream; the naming is the
        # author's and is preserved because the strict load matches on it.
        self.norm: nn.LayerNorm = nn.LayerNorm(psp_channel, eps=1e-6)
        self.norm2: nn.LayerNorm = nn.LayerNorm(asp_channel, eps=1e-6)
        self.convnext: nn.ModuleList = nn.ModuleList(
            [
                _ConvnextBlock(psp_channel, convnext_intermediate_dimension)
                for _ in range(convnext_layer_count)
            ]
        )
        self.convnext2: nn.ModuleList = nn.ModuleList(
            [
                _ConvnextBlock(asp_channel, convnext_intermediate_dimension)
                for _ in range(convnext_layer_count)
            ]
        )
        self.final_layer_norm: nn.LayerNorm = nn.LayerNorm(psp_channel, eps=1e-6)
        self.final_layer_norm2: nn.LayerNorm = nn.LayerNorm(asp_channel, eps=1e-6)
        self.apply(self._initialize_weights)

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Synthesis entry point: runs the full component prediction and returns
        # only the waveform, discarding the spectral members that the training
        # path consumes through predict_components.
        return self.predict_components(mel).waveform

    def predict_components(self, mel: torch.Tensor) -> Apnet2GeneratorOutput:
        # Runs both streams over the conditioning mel and returns every
        # spectral quantity together with the synthesized waveform.
        #
        # The amplitude stream projects the mel, normalizes channel-last,
        # passes through its residual stack, normalizes again, and projects to
        # the log-amplitude spectrum. The phase stream repeats that sequence
        # and ends in the paired real and imaginary projections. The two
        # results are then recombined in polar form and inverted by a centered
        # inverse STFT under a Hann window built on the mel's device.
        #
        # Args:
        #     mel: Conditioning mel shaped ``[batch, bands, frames]``; the
        #         batch axis must be explicit, because the streams cannot
        #         distinguish an unbatched mel from a batch of single-band
        #         frames.
        #
        # Raises:
        #     ValueError: If the mel does not carry exactly three dimensions.
        #
        # Returns:
        #     An Apnet2GeneratorOutput whose spectral members share the shape
        #     ``[batch, n_fft // 2 + 1, frames]`` and whose waveform is shaped
        #     ``[batch, 1, (frames - 1) * hop_size]`` under the centered
        #     inverse-STFT arithmetic.
        if mel.ndim != 3:
            raise ValueError(f"Expected mel shape [batch, channels, frames], got {tuple(mel.shape)}")
        # Amplitude stream: entry projection, stack, exit normalization, head.
        log_amplitude: torch.Tensor = self.ASP_input_conv(mel)
        log_amplitude: torch.Tensor = self.norm2(log_amplitude.transpose(1, 2)).transpose(1, 2)
        for convnext_block in self.convnext2:
            log_amplitude: torch.Tensor = convnext_block(log_amplitude)
        log_amplitude: torch.Tensor = self.final_layer_norm2(log_amplitude.transpose(1, 2)).transpose(1, 2)
        log_amplitude: torch.Tensor = self.ASP_output_conv(log_amplitude)

        # Phase stream: the same sequence, ending in the paired projections
        # whose ratio carries the phase.
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
        return Apnet2GeneratorOutput(
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
        # keep the torch defaults the reference also relies on.
        #
        # Args:
        #     module: Submodule visited by the apply traversal; anything other
        #         than a Conv1d or Linear is left as constructed.
        if isinstance(module, nn.Conv1d | nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.constant_(module.bias, 0)
