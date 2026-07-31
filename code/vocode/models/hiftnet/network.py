# This module:
# 1. Implements the HiFTNet generator network: the neural source-filter
#    chain in which the F0 contour drives a harmonic source signal that
#    the inverse-STFT generator filters into the waveform
# 2. Implements the JDC pitch network that supplies that F0 contour, together
#    with the state-dictionary adapter that maps the published pitch release
#    onto this implementation's member names
# 3. Defines HiftnetNetworkConfig and HiftnetGeneratorOutput, the frozen
#    topology record and the frozen bundle of predicted spectra returned to
#    the training step
#
# Design decisions:
# - The harmonic source injects explicit periodicity so the filter network
#   spends capacity on spectral shaping rather than pitch generation, the
#   architecture's defining idea
# - The ISTFT head performs the final sample-rate reconstruction per the
#   reference recipe: the network predicts a magnitude and a phase field at
#   frame rate and one inverse transform produces every output sample, so
#   the last upsampling factor costs no transposed convolution
# - Sine phase is accumulated at a reduced rate and interpolated back up
#   rather than accumulated per sample, which is the reference recipe's
#   guard against phase drift and aliasing in the harmonic source
# - The source module wraps sine generation in a no-grad block: the sine
#   bank is a deterministic function of F0 with no parameters of its own,
#   so excluding it from the graph removes a long cumulative-sum path
#   without discarding any learnable signal
# - The inverse-STFT window is a non-persistent buffer, so it is absent
#   from the state dictionary and strict loading of an author release
#   neither expects nor rejects it
#
# Author: Rahul Sawhney

import math
from pathlib import Path
from typing import ClassVar, override

import torch
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

__all__: list[str] = ["HiftnetGeneratorOutput", "HiftnetNetwork", "HiftnetNetworkConfig"]


class HiftnetNetworkConfig(BaseModel):
    # Frozen HiFTNet network topology and source-module settings. Every field
    # carries the reference default, so the network is constructible without
    # arguments for shape and gradient testing; a study run supplies the
    # values through HiftnetConfig instead.
    #
    # Fields:
    #     input_mel_channels: Mel band count of the conditioning input. The
    #         pitch network's shape contract depends on this value: its three
    #         residual stages halve the band axis and its final pool divides
    #         it by four, and the recurrent classifier is declared with a
    #         five-hundred-twelve-wide input, so only the reference band count
    #         of eighty produces a consistent chain. Default: ``80``.
    #     sampling_rate: Waveform rate in hertz used by the sine bank to
    #         convert harmonic frequencies into per-sample phase increments.
    #         Default: ``22050``.
    #     upsample_rates: Per-stage temporal upsampling factors of the
    #         transposed-convolution stack. Default: ``(8, 8)``.
    #     upsample_kernel_sizes: Kernel widths paired positionally with
    #         ``upsample_rates``. Default: ``(16, 16)``.
    #     upsample_initial_channel: Channels entering the first upsampling
    #         stage; each stage halves this count. Default: ``512``.
    #     resblock_kernel_sizes: Kernel widths of the multi-receptive-field
    #         residual blocks whose outputs are averaged after each upsampling
    #         stage. Default: ``(3, 7, 11)``.
    #     resblock_dilation_sizes: Dilation triples paired positionally with
    #         ``resblock_kernel_sizes``. Default: ``((1, 3, 5), (1, 3, 5),
    #         (1, 3, 5))``.
    #     gen_istft_n_fft: Transform size of the inverse-STFT head. The head
    #         emits ``gen_istft_n_fft + 2`` channels, split evenly into
    #         ``gen_istft_n_fft // 2 + 1`` magnitude bins and the same number
    #         of phase bins. Default: ``16``.
    #     gen_istft_hop_size: Hop of the inverse-STFT head and the final
    #         upsampling factor of the chain. Default: ``4``.
    #     f0_checkpoint_path: Location of the published JDC pitch release.
    #         ``None`` or a nonexistent path leaves the pitch network randomly
    #         initialized without raising. Default: ``None``.
    #     sine_amplitude: Peak amplitude of each harmonic in the source
    #         signal, and, divided by three, the noise amplitude used in
    #         unvoiced regions. Default: ``0.1``.
    #     noise_standard_deviation: Noise amplitude used in voiced regions,
    #         far below the unvoiced level so periodic structure dominates
    #         wherever pitch is present. Default: ``0.003``.
    #     voiced_threshold: Frequency in hertz above which a frame counts as
    #         voiced. Default: ``10.0``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True, arbitrary_types_allowed=True)
    input_mel_channels: PositiveInt = 80
    sampling_rate: PositiveInt = 22050
    upsample_rates: tuple[PositiveInt, ...] = (8, 8)
    upsample_kernel_sizes: tuple[PositiveInt, ...] = (16, 16)
    upsample_initial_channel: PositiveInt = 512
    resblock_kernel_sizes: tuple[PositiveInt, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[PositiveInt, PositiveInt, PositiveInt], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5)
    )
    gen_istft_n_fft: PositiveInt = 16
    gen_istft_hop_size: PositiveInt = 4
    f0_checkpoint_path: Path | None = None
    sine_amplitude: PositiveFloat = 0.1
    noise_standard_deviation: PositiveFloat = 0.003
    voiced_threshold: float = 10.0


class HiftnetGeneratorOutput(BaseModel):
    # Frozen bundle of one generator prediction. The training step needs the
    # waveform for the critics and the reconstruction term, so returning the
    # spectra alongside it lets a caller inspect the head's output without a
    # second forward pass. All three tensors carry the autograd graph; the
    # training step detaches at the point of use rather than here.
    #
    # Fields:
    #     magnitude: Non-negative magnitude field of shape
    #         ``[batch, gen_istft_n_fft // 2 + 1, frames]``, produced by
    #         exponentiating the head's first channel group.
    #     phase: Phase field of the same shape, produced by taking the sine
    #         of the head's second channel group and therefore bounded to
    #         ``[-1, 1]``.
    #     waveform: Reconstructed waveform of shape ``[batch, 1, samples]``.
    #         Its sample count follows from the frame count and hop and need
    #         not equal the reference length.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    magnitude: torch.Tensor
    phase: torch.Tensor
    waveform: torch.Tensor


class _ConvolutionPadding:
    # Stateless helper computing the symmetric padding that keeps a dilated
    # convolution length-preserving. It exists as a named object rather than a
    # free function so the padding rule is stated once and every convolution
    # in the family is visibly built from the same rule.
    def compute(self, kernel_size: int, dilation: int = 1) -> int:
        # Computes the padding that keeps the time axis length-invariant. The
        # expression equals half the dilated kernel's extent beyond a single
        # sample, which is exact for the odd kernel sizes this family uses;
        # an even kernel would lose one sample, so callers must pass odd
        # widths where length invariance matters.
        return int((kernel_size * dilation - dilation) / 2)


class _JdcResidualBlock(nn.Module):
    # One residual stage of the JDC pitch network. Each stage normalizes and
    # activates its input, halves the frequency axis by max pooling, and adds
    # a two-convolution residual branch. When the channel count changes, the
    # skip path is projected through a pointwise convolution so the addition
    # stays well defined; when it does not, the skip is the identity.
    def __init__(self, in_channels: int, out_channels: int, leaky_relu_slope: float = 0.01) -> None:
        # Builds the pre-activation, the residual branch, and the projection.
        # The projection is created only for a channel-changing stage, which
        # is why it is typed as optional and why the published release
        # contains projection weights for exactly the widening stages.
        super().__init__()
        self._downsample: bool = in_channels != out_channels
        self._pre_convolution: nn.Sequential = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2))
        )
        self._convolution: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        )
        self._projection: nn.Conv2d | None = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False) if self._downsample else None
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Applies the pre-activation first, so both the residual branch and
        # the skip see the pooled tensor and the addition is dimensionally
        # consistent.
        x: torch.Tensor = self._pre_convolution(x)
        if self._projection is None:
            return self._convolution(x) + x
        return self._convolution(x) + self._projection(x)


class _JdcNet(nn.Module):
    # The JDC pitch network that supplies HiFTNet's F0 contour. A
    # convolutional trunk reduces the mel band axis through three residual
    # stages, a bidirectional recurrent classifier reads the pooled result as
    # a sequence, and a linear head emits one value per frame.
    #
    # The shape chain is rigid and ties this network to the eighty-band mel
    # protocol: the three residual stages halve the band axis to ten, the
    # final pool divides it by four to two, and the trunk's two hundred
    # fifty-six channels times those two bands is the five-hundred-twelve-wide
    # input the recurrent layer is declared with. A different band count
    # therefore fails at the recurrent layer rather than degrading quietly.
    def __init__(self, num_class: int = 1, leaky_relu_slope: float = 0.01) -> None:
        # Builds the trunk, the three widening residual stages, the pooling
        # block, the bidirectional recurrent classifier, and the linear head,
        # then applies the reference initialization to every submodule.
        super().__init__()
        self._num_class: int = num_class
        self._conv_block: nn.Sequential = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False)
        )
        self._res_block_1: _JdcResidualBlock = _JdcResidualBlock(in_channels=64, out_channels=128)
        self._res_block_2: _JdcResidualBlock = _JdcResidualBlock(in_channels=128, out_channels=192)
        self._res_block_3: _JdcResidualBlock = _JdcResidualBlock(in_channels=192, out_channels=256)
        self._pool_batch_norm: nn.BatchNorm2d = nn.BatchNorm2d(num_features=256)
        self._pool_activation: nn.LeakyReLU = nn.LeakyReLU(leaky_relu_slope, inplace=True)
        self._pool: nn.MaxPool2d = nn.MaxPool2d(kernel_size=(1, 4))
        self._bilstm_classifier: nn.LSTM = nn.LSTM(
            input_size=512,
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )
        self._classifier: nn.Linear = nn.Linear(in_features=512, out_features=self._num_class)
        self.apply(self._initialize_weights)

    @override
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Predicts a per-frame pitch value from a conditioning mel.
        #
        # Args:
        #     x: Mel of shape ``[batch, 1, channels, frames]``. The frame
        #         count is captured before any reshaping, because the
        #         recurrent stage must be reassembled against it.
        #
        # Returns:
        #     A triple of the pitch contour of shape ``[batch, frames]``, the
        #     trunk feature map taken before the final pool, and the pooled
        #     trunk output. The pitch contour is passed through an absolute
        #     value, so it is non-negative by construction and the voiced
        #     threshold in the source module compares against a magnitude.
        #     The generator consumes only the first element; the other two
        #     exist for the reference network's auxiliary heads.
        sequence_length: int = x.shape[-1]
        # The band and frame axes are exchanged so that the pooling kernels,
        # which act only on the last axis, reduce bands and leave the frame
        # count intact. The cast keeps the pitch trunk in float32 regardless
        # of the caller's dtype.
        x: torch.Tensor = x.float().transpose(-1, -2)
        conv_out: torch.Tensor = self._conv_block(x)
        res_1: torch.Tensor = self._res_block_1(conv_out)
        res_2: torch.Tensor = self._res_block_2(res_1)
        res_3: torch.Tensor = self._res_block_3(res_2)
        pool_out: torch.Tensor = self._pool_batch_norm(res_3)
        pool_out: torch.Tensor = self._pool_activation(pool_out)
        # The auxiliary feature map is taken before the final pool, so it
        # retains the full band resolution the pooled path discards.
        gan_feature: torch.Tensor = pool_out.transpose(-1, -2)
        pool_out: torch.Tensor = self._pool(pool_out)
        # Channels and remaining bands are flattened into one feature vector
        # per frame, which is where the trunk width and the surviving band
        # count must multiply to the declared recurrent input width.
        classifier_out: torch.Tensor = pool_out.permute(0, 2, 1, 3).contiguous().view((-1, sequence_length, 512))
        classifier_out, _ = self._bilstm_classifier(classifier_out)
        classifier_out: torch.Tensor = classifier_out.contiguous().view((-1, 512))
        classifier_out: torch.Tensor = self._classifier(classifier_out)
        classifier_out: torch.Tensor = classifier_out.view((-1, sequence_length, self._num_class))
        return torch.abs(classifier_out.squeeze(-1)), gan_feature, pool_out

    def _initialize_weights(self, module: nn.Module) -> None:
        # Applies the reference initialization scheme per submodule type:
        # Kaiming-uniform with zeroed bias for linear layers, Xavier-normal
        # for convolutions, and orthogonal initialization for recurrent
        # weight matrices with normal initialization for their bias vectors.
        # Orthogonal recurrent weights keep the repeated multiplications
        # along the sequence norm-preserving at the start of training, which
        # matters here because the classifier runs over the full frame axis.
        # The method is applied through nn.Module.apply, so it is invoked for
        # every descendant and must ignore types it does not recognize.
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.LSTM | nn.LSTMCell):
            for parameter in module.parameters():
                if parameter.data is None:
                    continue
                if len(parameter.shape) >= 2:
                    nn.init.orthogonal_(parameter.data)
                else:
                    nn.init.normal_(parameter.data)


class _HiftnetResidualBlock(nn.Module):
    # Multi-receptive-field residual block of the filter network. Each of the
    # three sub-blocks applies a dilated convolution followed by an
    # undilated refinement convolution, with a residual connection around the
    # pair, so one block mixes three different receptive-field extents.
    #
    # The activation is the Snake nonlinearity, whose periodic term gives the
    # filter network an inductive bias toward the oscillatory structure that
    # dominates speech. Its sharpness is controlled by a learnable per-channel
    # coefficient, so each channel can settle on its own periodicity rather
    # than sharing one global setting; separate coefficients are held for the
    # dilated and the refinement position because the two operate on
    # differently scaled activations.
    def __init__(self, channels: int, kernel_size: int, dilation: tuple[int, int, int]) -> None:
        # Builds one dilated convolution, one refinement convolution, and two
        # activation coefficients per entry of the dilation triple. Every
        # convolution is weight-normalized, which is what makes the published
        # release's paired magnitude and direction tensors loadable.
        super().__init__()
        padding: _ConvolutionPadding = _ConvolutionPadding()
        self._dilated_convolutions: nn.ModuleList = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=value, padding=padding.compute(kernel_size, value)))
            for value in dilation
        ])
        self._refinement_convolutions: nn.ModuleList = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=padding.compute(kernel_size, 1)))
            for _ in dilation
        ])
        self._alpha_1: nn.ParameterList = nn.ParameterList([
            nn.Parameter(torch.ones(1, channels, 1)) for _ in dilation
        ])
        self._alpha_2: nn.ParameterList = nn.ParameterList([
            nn.Parameter(torch.ones(1, channels, 1)) for _ in dilation
        ])

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Runs the three sub-blocks in sequence, each adding its result back
        # into the running tensor, so the block output accumulates all three
        # receptive-field scales rather than selecting one. The strict zip
        # asserts that the convolution lists and coefficient lists were built
        # from the same dilation triple.
        for first, second, alpha_first, alpha_second in zip(
            self._dilated_convolutions,
            self._refinement_convolutions,
            self._alpha_1,
            self._alpha_2,
            strict=True
        ):
            candidate: torch.Tensor = x + (1.0 / alpha_first) * torch.sin(alpha_first * x).pow(2)
            candidate: torch.Tensor = first(candidate)
            candidate: torch.Tensor = candidate + (1.0 / alpha_second) * torch.sin(alpha_second * candidate).pow(2)
            candidate: torch.Tensor = second(candidate)
            x: torch.Tensor = candidate + x
        return x


class _SineGenerator(nn.Module):
    # Parameter-free bank that turns a per-sample F0 contour into a set of
    # harmonically related sine waves plus additive noise. The fundamental and
    # its harmonics are generated together, each frame is classified as voiced
    # or unvoiced against the frequency threshold, and the two regimes receive
    # different noise levels: voiced frames get the small standard deviation
    # so periodic structure dominates, unvoiced frames get a much larger noise
    # amplitude and have their sine content masked away entirely.
    def __init__(
        self,
        sampling_rate: int,
        upsample_scale: int,
        harmonic_count: int,
        sine_amplitude: float,
        noise_standard_deviation: float,
        voiced_threshold: float
    ) -> None:
        # Records the generation settings. This module holds no parameters
        # and registers no buffers, so it contributes nothing to the state
        # dictionary and its behavior is fully determined by these values.
        super().__init__()
        self._sampling_rate: int = sampling_rate
        self._upsample_scale: int = upsample_scale
        self._harmonic_count: int = harmonic_count
        self._sine_amplitude: float = sine_amplitude
        self._noise_standard_deviation: float = noise_standard_deviation
        self._voiced_threshold: float = voiced_threshold

    @override
    def forward(self, f0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Generates the harmonic source for one F0 contour.
        #
        # Args:
        #     f0: Per-sample fundamental frequency in hertz, of shape
        #         ``[batch, samples, 1]``.
        #
        # Returns:
        #     A triple of the masked and noised harmonic bank of shape
        #     ``[batch, samples, harmonic_count + 1]``, the voiced mask, and
        #     the noise that was added, the last two returned so a caller can
        #     reuse the same voicing decision without recomputing it.
        # The index range runs from the fundamental through the requested
        # harmonic count inclusive, giving one more column than the harmonic
        # count; the source module's projection is declared with exactly that
        # width.
        harmonic_indices: torch.Tensor = torch.arange(
            1,
            self._harmonic_count + 2,
            device=f0.device,
            dtype=f0.dtype
        ).view(1, 1, -1)
        harmonic_frequencies: torch.Tensor = f0 * harmonic_indices
        # Frequencies become normalized per-sample phase increments, wrapped
        # into the unit interval so that accumulating them cannot lose
        # precision to an ever-growing magnitude.
        radian_values: torch.Tensor = (harmonic_frequencies / self._sampling_rate) % 1
        # Phase is accumulated at the reduced frame rate rather than per
        # sample: the increments are downsampled, summed, and the resulting
        # phase is interpolated back to sample rate with a compensating gain.
        # Accumulating at the lower rate keeps the cumulative sum short and is
        # the reference recipe's guard against drift and aliasing in long
        # utterances.
        radian_values: torch.Tensor = torch.nn.functional.interpolate(
            radian_values.transpose(1, 2),
            scale_factor=1 / self._upsample_scale,
            mode="linear"
        ).transpose(1, 2)
        phase: torch.Tensor = torch.cumsum(radian_values, dim=1) * 2 * math.pi
        phase: torch.Tensor = torch.nn.functional.interpolate(
            phase.transpose(1, 2) * self._upsample_scale,
            scale_factor=self._upsample_scale,
            mode="linear"
        ).transpose(1, 2)
        sine_waves: torch.Tensor = torch.sin(phase) * self._sine_amplitude
        # Voicing is decided per sample by comparing against the threshold;
        # the pitch network's output is already non-negative, so this is a
        # magnitude comparison.
        voiced_mask: torch.Tensor = (f0 > self._voiced_threshold).type(torch.float32)
        noise_amplitude: torch.Tensor = (
            voiced_mask * self._noise_standard_deviation
            + (1 - voiced_mask) * self._sine_amplitude / 3
        )
        noise: torch.Tensor = noise_amplitude * torch.randn_like(sine_waves)
        # Unvoiced samples keep only noise, because the sine content is
        # multiplied away by the mask; voiced samples keep the harmonics with
        # a small noise floor added.
        return sine_waves * voiced_mask + noise, voiced_mask, noise


class _SourceModule(nn.Module):
    # Wraps the sine bank with the single learnable step of the source path: a
    # linear projection collapsing the harmonic columns into one excitation
    # channel, followed by a hyperbolic tangent that bounds the result. This
    # is the only place the network can learn how to weight the fundamental
    # against its harmonics; everything upstream is a fixed function of F0.
    def __init__(
        self,
        sampling_rate: int,
        upsample_scale: int,
        harmonic_count: int,
        sine_amplitude: float,
        noise_standard_deviation: float,
        voiced_threshold: float
    ) -> None:
        # Builds the sine bank and the projection that mixes its harmonic
        # columns. The projection's input width is the harmonic count plus
        # one, matching the fundamental-inclusive bank the generator emits.
        super().__init__()
        self._sine_generator: _SineGenerator = _SineGenerator(
            sampling_rate=sampling_rate,
            upsample_scale=upsample_scale,
            harmonic_count=harmonic_count,
            sine_amplitude=sine_amplitude,
            noise_standard_deviation=noise_standard_deviation,
            voiced_threshold=voiced_threshold
        )
        self._linear: nn.Linear = nn.Linear(harmonic_count + 1, 1)
        self._tanh: nn.Tanh = nn.Tanh()
        self._sine_amplitude: float = sine_amplitude

    @override
    def forward(self, f0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Produces the excitation signal the filter network is conditioned on.
        # The sine bank runs under no-grad: it holds no parameters, so nothing
        # learnable is lost, and excluding it removes the long cumulative-sum
        # path from the graph. The consequence is that no gradient flows back
        # into the pitch network through this route, so the F0 extractor is
        # trained only by whatever other paths reach it.
        #
        # Returns:
        #     A triple of the projected excitation, an independently drawn
        #     noise signal, and the voiced mask. The generator consumes only
        #     the first element; the other two are part of the reference
        #     interface.
        with torch.no_grad():
            sine_waves, voiced_mask, _ = self._sine_generator(f0)
        sine_merge: torch.Tensor = self._tanh(self._linear(sine_waves))
        noise: torch.Tensor = torch.randn_like(voiced_mask) * self._sine_amplitude / 3
        return sine_merge, noise, voiced_mask


class _TorchIstft(nn.Module):
    # Short-time Fourier analysis and synthesis pair used at both ends of the
    # source-filter chain: it analyzes the harmonic excitation into the
    # spectral features the filter network conditions on, and it reconstructs
    # the output waveform from the head's predicted magnitude and phase.
    def __init__(self, filter_length: int, hop_length: int, win_length: int) -> None:
        # Records the transform geometry and registers the Hann window as a
        # non-persistent buffer, so it moves with the module across devices
        # but never appears in the state dictionary. That exclusion is what
        # lets an author release load strictly without carrying a window
        # tensor.
        super().__init__()
        self._filter_length: int = filter_length
        self._hop_length: int = hop_length
        self._win_length: int = win_length
        self.register_buffer("_window", torch.hann_window(win_length), persistent=False)

    def transform(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Analyzes the waveform into magnitude and phase spectra.
        stft: torch.Tensor = torch.stft(
            waveform,
            self._filter_length,
            self._hop_length,
            self._win_length,
            window=self._window.to(waveform.device),
            return_complex=True
        )
        return stft.abs(), torch.angle(stft)

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        # Reconstructs the waveform from magnitude and phase spectra by
        # recombining them into a complex spectrum and inverting the
        # transform. A channel axis is added so the result matches the
        # [batch, 1, samples] layout the discriminators and the training step
        # expect.
        waveform: torch.Tensor = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self._filter_length,
            self._hop_length,
            self._win_length,
            window=self._window.to(magnitude.device)
        )
        return waveform.unsqueeze(1)


class HiftnetNetwork(nn.Module):
    # The complete HiFTNet generator. A conditioning mel enters two parallel
    # paths: the pitch network extracts an F0 contour that drives the harmonic
    # source, whose spectral analysis is injected into every upsampling stage,
    # while the mel itself is projected and progressively upsampled by the
    # filter stack. The final stage emits a magnitude and a phase field that
    # one inverse transform converts into the output waveform.
    #
    # The topology is bound by three consistency requirements. The product of
    # the upsampling rates times the inverse-STFT hop is the network's total
    # upsampling and must equal the conditioning mel's hop length. The head's
    # channel count is the transform size plus two, which is exactly twice the
    # bin count, so it splits evenly into a magnitude field and a phase field.
    # The source-injection convolutions accept that same channel count,
    # because the harmonic analysis is a magnitude and a phase spectrum
    # concatenated along channels.
    #
    # Integration: the pitch network is a member of this module, so its
    # parameters appear in parameters() and reach any optimizer built over
    # them. Callers that intend to hold pitch fixed must exclude or freeze it
    # explicitly; neither this class nor Hiftnet does so.
    def __init__(self, configuration: HiftnetNetworkConfig) -> None:
        # Builds the whole chain and, when a checkpoint path is configured and
        # present, initializes the pitch network from the published release.
        # The total upsampling computed first is shared by the source module,
        # which needs it to set the phase-accumulation rate, and by the F0
        # upsampler, which needs it to expand the frame-rate contour to sample
        # rate.
        super().__init__()
        self._configuration: HiftnetNetworkConfig = configuration
        self._leaky_relu_slope: float = 0.1
        upsample_scale: int = math.prod(configuration.upsample_rates) * configuration.gen_istft_hop_size
        self._f0_model: _JdcNet = _JdcNet(num_class=1)
        self._load_f0_checkpoint(configuration.f0_checkpoint_path)
        self._pre_convolution: nn.Module = weight_norm(
            nn.Conv1d(configuration.input_mel_channels, configuration.upsample_initial_channel, 7, 1, padding=3)
        )
        self._source_module: _SourceModule = _SourceModule(
            sampling_rate=configuration.sampling_rate,
            upsample_scale=upsample_scale,
            harmonic_count=8,
            sine_amplitude=configuration.sine_amplitude,
            noise_standard_deviation=configuration.noise_standard_deviation,
            voiced_threshold=configuration.voiced_threshold
        )
        self._f0_upsampler: nn.Upsample = nn.Upsample(scale_factor=upsample_scale)
        self._upsample_layers: nn.ModuleList = nn.ModuleList()
        self._noise_convolutions: nn.ModuleList = nn.ModuleList()
        self._noise_residuals: nn.ModuleList = nn.ModuleList()
        # Upsampling stack. Each stage halves the channel count while
        # multiplying the time axis by its rate, and each is paired with a
        # source-injection convolution that must reduce the harmonic features
        # to the time resolution prevailing at that stage. Intermediate stages
        # therefore stride by the product of all remaining rates, while the
        # last stage, already at output resolution, needs only a pointwise
        # projection.
        for index, (rate, kernel_size) in enumerate(zip(configuration.upsample_rates, configuration.upsample_kernel_sizes, strict=True)):
            in_channels: int = configuration.upsample_initial_channel // (2 ** index)
            out_channels: int = configuration.upsample_initial_channel // (2 ** (index + 1))
            self._upsample_layers.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        rate,
                        padding=(kernel_size - rate) // 2
                    )
                )
            )
            if index + 1 < len(configuration.upsample_rates):
                stride_f0: int = math.prod(configuration.upsample_rates[index + 1:])
                self._noise_convolutions.append(
                    nn.Conv1d(
                        configuration.gen_istft_n_fft + 2,
                        out_channels,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=(stride_f0 + 1) // 2
                    )
                )
                self._noise_residuals.append(_HiftnetResidualBlock(out_channels, 7, (1, 3, 5)))
            else:
                self._noise_convolutions.append(nn.Conv1d(configuration.gen_istft_n_fft + 2, out_channels, kernel_size=1))
                self._noise_residuals.append(_HiftnetResidualBlock(out_channels, 11, (1, 3, 5)))
        # Residual blocks are stored in one flat list ordered stage-major,
        # so the forward pass walks it with a running index rather than
        # indexing a nested structure.
        self._residual_blocks: nn.ModuleList = nn.ModuleList()
        for index in range(len(self._upsample_layers)):
            channels: int = configuration.upsample_initial_channel // (2 ** (index + 1))
            for kernel_size, dilation in zip(configuration.resblock_kernel_sizes, configuration.resblock_dilation_sizes, strict=True):
                self._residual_blocks.append(_HiftnetResidualBlock(channels, kernel_size, dilation))
        post_channels: int = configuration.upsample_initial_channel // (2 ** len(self._upsample_layers))
        self._post_convolution: nn.Module = weight_norm(nn.Conv1d(post_channels, configuration.gen_istft_n_fft + 2, 7, 1, padding=3))
        self._reflection_pad: nn.ReflectionPad1d = nn.ReflectionPad1d((1, 0))
        self._stft: _TorchIstft = _TorchIstft(
            filter_length=configuration.gen_istft_n_fft,
            hop_length=configuration.gen_istft_hop_size,
            win_length=configuration.gen_istft_n_fft
        )

    @override
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # Synthesizes a waveform from a conditioning mel, discarding the
        # intermediate spectra. Callers that need those spectra use
        # predict_components instead, so no forward pass is ever repeated
        # merely to recover them.
        return self.predict_components(mel).waveform

    def predict_components(self, mel: torch.Tensor) -> HiftnetGeneratorOutput:
        # Runs the full chain and returns the predicted magnitude and phase
        # fields alongside the reconstructed waveform.
        #
        # Args:
        #     mel: Conditioning mel of shape ``[batch, channels, frames]``.
        #
        # Returns:
        #     The frozen bundle of magnitude, phase, and waveform, all three
        #     carrying the autograd graph.
        magnitude, phase = self._predict_spectrum(mel)
        waveform: torch.Tensor = self._stft.inverse(magnitude, phase)
        return HiftnetGeneratorOutput(magnitude=magnitude, phase=phase, waveform=waveform)

    @property
    def configuration(self) -> HiftnetNetworkConfig:
        # Returns the immutable configuration attached to this component.
        return self._configuration

    def _predict_spectrum(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Runs both paths of the source-filter chain and returns the head's
        # magnitude and phase fields.
        #
        # Args:
        #     mel: Conditioning mel of shape ``[batch, channels, frames]``.
        #
        # Raises:
        #     ValueError: If the mel is not rank three.
        #     RuntimeError: If the residual block list is empty, which would
        #         mean the configuration declared no residual kernels.
        #
        # Returns:
        #     The magnitude field, strictly positive because it is produced by
        #     exponentiation, and the phase field, bounded to ``[-1, 1]``
        #     because it is produced by a sine.
        if mel.ndim != 3:
            raise ValueError(f"Expected mel shape [batch, channels, frames], got {tuple(mel.shape)}")
        # Source path: the pitch network yields a frame-rate contour, which is
        # expanded to sample rate, converted into the harmonic excitation, and
        # analyzed back into a spectrum whose magnitude and phase are
        # concatenated along channels for injection into the filter stack.
        f0, _, _ = self._f0_model(mel.unsqueeze(1))
        if f0.ndim == 1:
            f0: torch.Tensor = f0.unsqueeze(0)
        f0: torch.Tensor = self._f0_upsampler(f0[:, None]).transpose(1, 2)
        harmonic_source, _, _ = self._source_module(f0)
        harmonic_source: torch.Tensor = harmonic_source.transpose(1, 2).squeeze(1)
        harmonic_magnitude, harmonic_phase = self._stft.transform(harmonic_source)
        harmonic_features: torch.Tensor = torch.cat([harmonic_magnitude, harmonic_phase], dim=1)
        # Filter path: the mel is projected to the initial width and then
        # upsampled stage by stage, receiving the source injection at every
        # stage.
        x: torch.Tensor = self._pre_convolution(mel)
        residual_index: int = 0
        for upsample_index, upsample_layer in enumerate(self._upsample_layers):
            x: torch.Tensor = torch.nn.functional.leaky_relu(x, self._leaky_relu_slope)
            source_features: torch.Tensor = self._noise_convolutions[upsample_index](harmonic_features)
            source_features: torch.Tensor = self._noise_residuals[upsample_index](source_features)
            x: torch.Tensor = upsample_layer(x)
            # The last stage is padded by one frame on the left, which
            # reconciles the transposed convolution's output length with the
            # frame count the inverse transform requires.
            if upsample_index == len(self._upsample_layers) - 1:
                x: torch.Tensor = self._reflection_pad(x)
            x: torch.Tensor = x + source_features
            # The stage's residual blocks are averaged rather than summed, so
            # the activation scale stays independent of how many kernel sizes
            # the configuration declares.
            residual_sum: torch.Tensor | None = None
            for _ in range(len(self._configuration.resblock_kernel_sizes)):
                residual_output: torch.Tensor = self._residual_blocks[residual_index](x)
                residual_sum: torch.Tensor | None = residual_output if residual_sum is None else residual_sum + residual_output
                residual_index += 1
            if residual_sum is None:
                raise RuntimeError("HiFTNet residual block list is empty")
            x: torch.Tensor = residual_sum / len(self._configuration.resblock_kernel_sizes)
        x: torch.Tensor = torch.nn.functional.leaky_relu(x, self._leaky_relu_slope)
        x: torch.Tensor = self._post_convolution(x)
        # The head's channels split evenly at the bin count: the first group
        # is exponentiated into a strictly positive magnitude, and the second
        # is passed through a sine that bounds the phase without the
        # discontinuity a wrapping operation would introduce.
        magnitude: torch.Tensor = torch.exp(x[:, : self._configuration.gen_istft_n_fft // 2 + 1, :])
        phase: torch.Tensor = torch.sin(x[:, self._configuration.gen_istft_n_fft // 2 + 1:, :])
        return magnitude, phase

    def _load_f0_checkpoint(self, checkpoint_path: Path | None) -> None:
        # Initializes the pitch network from a published JDC release.
        #
        # An unset path, or one that does not exist, returns silently and
        # leaves the pitch network randomly initialized. This is deliberate:
        # the architecture must stay constructible without an external
        # download for construction, shape, and gradient testing. The
        # consequence is that a missing release is invisible at run time, so a
        # reproduction run must confirm the file's presence separately.
        #
        # When a file is present the contract is exact. Loading passes
        # strict=False and then asserts that neither missing nor unexpected
        # keys remain, which enforces full agreement while producing a
        # domain-specific error naming the offending keys instead of the
        # framework's generic report. The relaxed flag is what allows the
        # adapter to drop the release's unused voicing-detector entries
        # without those becoming missing-key failures.
        #
        # Args:
        #     checkpoint_path: Location of the release, or ``None``.
        #
        # Raises:
        #     ValueError: If the payload is not a mapping carrying a
        #         ``"model"`` entry.
        #     TypeError: If that entry is not a state dictionary.
        #     RuntimeError: If any key remains unmatched after adaptation in
        #         either direction, which means the release and this
        #         implementation describe different architectures.
        if checkpoint_path is None or not checkpoint_path.exists():
            return
        checkpoint: object = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ValueError(f"HiFTNet F0 checkpoint must contain key `model`: {checkpoint_path}")
        state_dict: object = checkpoint["model"]
        if not isinstance(state_dict, dict):
            raise TypeError(f"HiFTNet F0 checkpoint `model` entry must be a state dict: {checkpoint_path}")
        adapted_state_dict: dict[str, torch.Tensor] = self._adapt_f0_state_dict(state_dict)
        load_result: torch.nn.modules.module._IncompatibleKeys = self._f0_model.load_state_dict(adapted_state_dict, strict=False)
        missing_keys: list[str] = list(load_result.missing_keys)
        unexpected_keys: list[str] = list(load_result.unexpected_keys)
        if missing_keys:
            raise RuntimeError(f"HiFTNet F0 checkpoint load missing keys after adaptation: {missing_keys[:10]}")
        if unexpected_keys:
            raise RuntimeError(f"HiFTNet F0 checkpoint load has unexpected keys after adaptation: {unexpected_keys[:10]}")

    def _adapt_f0_state_dict(self, upstream_state_dict: dict[object, object]) -> dict[str, torch.Tensor]:
        # Rewrites the published pitch release's key names onto this
        # implementation's private member names. The release predates the
        # naming conventions used here, so every renamed member needs an
        # explicit rule; the rules are ordered from most specific to least so
        # that a broad pattern cannot consume a key a narrower rule owns.
        #
        # Two classes of entry are dropped rather than renamed. Non-tensor
        # values are skipped because a state dictionary may carry bookkeeping
        # scalars this network has no slot for. Voicing-detector entries are
        # skipped because the reference network's detector head is not
        # reproduced here; only the pitch-regression path is. Because the
        # caller enforces that no unexpected keys survive, dropping them here
        # is the mechanism that makes the release loadable at all.
        #
        # Note:
        #     This adapter handles the standalone pitch release, whose keys
        #     are unprefixed. The full generator release handled by
        #     vocode.models.hiftnet.weights nests the same tensors under a
        #     pitch-model prefix and therefore needs its own rules.
        adapted: dict[str, torch.Tensor] = {}
        for raw_key, raw_value in upstream_state_dict.items():
            if not isinstance(raw_value, torch.Tensor):
                continue
            key: str = str(raw_key)
            if key.startswith("detector_conv.") or key.startswith("bilstm_detector.") or key.startswith("detector."):
                continue
            key: str = key.replace("conv_block.", "_conv_block.", 1)
            key: str = key.replace("res_block1.", "_res_block_1.", 1)
            key: str = key.replace("res_block2.", "_res_block_2.", 1)
            key: str = key.replace("res_block3.", "_res_block_3.", 1)
            key: str = key.replace(".pre_conv.", "._pre_convolution.")
            key: str = key.replace(".conv1by1.", "._projection.")
            key: str = key.replace(".conv.", "._convolution.")
            key: str = key.replace("pool_block.0.", "_pool_batch_norm.", 1)
            key: str = key.replace("bilstm_classifier.", "_bilstm_classifier.", 1)
            if key.startswith("classifier."):
                key: str = key.replace("classifier.", "_classifier.", 1)
            adapted[key] = raw_value
        return adapted
