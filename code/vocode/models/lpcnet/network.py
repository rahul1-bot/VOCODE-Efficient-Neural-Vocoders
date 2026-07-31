# This module:
# 1. Implements the LPCNet network: the frame-rate conditioning encoder
#    over analysis features and the sample-rate core (embeddings, two GRUs,
#    and the dual-softmax output) predicting the mu-law excitation
#    distribution
#
# Rate structure:
# - The conditioning encoder runs once per analysis frame and the
#   autoregressive core once per sample. The encoder's output is repeated
#   across the samples its frame governs, which is the only point where the
#   two rates meet
# - The two convolutions of the encoder are length-preserving, so one
#   conditioning vector is emitted per input frame and the frame axis is
#   never resampled inside the encoder
#
# Design decisions:
# - The sample-rate path consumes the previous sample, the linear
#   prediction, and the previous excitation as embedded inputs, exactly
#   the reference's autoregressive conditioning
# - Training injects the configured noise into teacher-forced inputs so
#   the model learns robustness to its own sampling errors
# - The signal embedding interpolates between adjacent table rows rather
#   than indexing one, because its inputs are continuous mu-law values: the
#   companding produces fractional indices, and the training noise moves
#   them further off the integer grid
# - The output distribution is a binary tree over 255 internal nodes rather
#   than a flat 256-way softmax, so the per-sample output stage stays small
#   enough to evaluate inside a sequential loop
# - The first GRU is the sparsification target; its dimensions follow the
#   reference deployment geometry, and the second GRU is deliberately
#   narrow because it runs unpruned
#
# Author: Rahul Sawhney

import math
from typing import ClassVar, override

import torch
from pydantic import BaseModel, ConfigDict
from torch import nn

from vocode.transforms.lpc import LpcnetMuLaw

__all__: list[str] = ["LpcnetNetwork", "LpcnetNetworkOutput"]


class LpcnetNetworkOutput(BaseModel):
    # Frozen result of one teacher-forced training pass. Both members are
    # required by the objective: the target excitation is the residual between
    # the target sample and the prediction, so the loss cannot be computed
    # from the node outputs alone.
    #
    # Fields:
    #     prediction: Per-sample linear prediction of shape
    #         ``[batch, samples, 1]`` on the int16 scale, computed from the
    #         frame-rate coefficients and the past samples.
    #     node_outputs: Binary-tree node activations of shape
    #         ``[batch, samples, 256]``, each in ``(0, 1)``. Entries one
    #         through two hundred fifty-five address the tree's internal
    #         nodes; entry zero is unused.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    prediction: torch.Tensor
    node_outputs: torch.Tensor


class LpcnetFractionalEmbedding(nn.Module):
    # Embedding table addressed by continuous rather than integer indices. A
    # lookup interpolates linearly between the two adjacent rows bracketing
    # the requested position, which is required here because every input to
    # this table is a mu-law value: companding produces fractional indices,
    # and the training noise displaces them further. Rounding to the nearest
    # row instead would quantize the model's own inputs and discard the
    # sub-step precision the companding provides.
    def __init__(self, dictionary_size: int, embedding_dimension: int) -> None:
        # Builds the table at the reference initialization. The weight is a
        # plain parameter rather than an embedding submodule because the
        # lookup is a manual gather of two rows, not a standard index
        # operation.
        super().__init__()
        self._dictionary_size: int = dictionary_size
        initial_weight: torch.Tensor = self._pcm_initialization(dictionary_size, embedding_dimension)
        self.weight: nn.Parameter = nn.Parameter(initial_weight)

    @override
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        # Looks up continuous indices by interpolating between adjacent rows.
        #
        # Two clamps are applied for different reasons. The first bounds the
        # index into the table's domain, which matters because the training
        # noise can push a companded value past either end. The second bounds
        # the lower row to the second-to-last entry, so that reading the row
        # above it is always in range; at the very top of the table this makes
        # the interpolation weight zero on a valid pair rather than indexing
        # past the end.
        #
        # Args:
        #     indices: Continuous mu-law positions of any shape.
        #
        # Returns:
        #     The interpolated embeddings, with one additional trailing axis
        #     of the embedding width.
        clamped: torch.Tensor = indices.clamp(0.0, float(self._dictionary_size - 1))
        lower_index: torch.Tensor = clamped.floor().clamp(0.0, float(self._dictionary_size - 2)).to(torch.long)
        fraction: torch.Tensor = (clamped - lower_index.to(clamped.dtype)).unsqueeze(-1)
        lower_embedding: torch.Tensor = self.weight[lower_index]
        upper_embedding: torch.Tensor = self.weight[lower_index + 1]
        return (1.0 - fraction) * lower_embedding + fraction * upper_embedding

    def _pcm_initialization(self, dictionary_size: int, embedding_dimension: int) -> torch.Tensor:
        # Builds the reference initialization: uniform noise added to a linear
        # ramp that runs across the table. The ramp is the substantive part.
        # A mu-law index is an ordered quantity, so neighboring rows should
        # begin as neighbors in embedding space; seeding every dimension with
        # a shared monotone component gives the table that ordering from the
        # start, which a purely random initialization would have to discover.
        # The ramp is centered and scaled to unit variance so it neither
        # biases nor dominates the noise it is added to.
        #
        # The generator is seeded locally rather than drawing from the global
        # stream, so this initialization is identical across runs and cannot
        # shift because unrelated code consumed randomness first.
        generator: torch.Generator = torch.Generator().manual_seed(1234)
        uniform: torch.Tensor = (
            torch.rand(dictionary_size, embedding_dimension, generator=generator) * 2.0 - 1.0
        ) * 1.7321
        ramp: torch.Tensor = (
            math.sqrt(12.0)
            * (torch.arange(dictionary_size, dtype=torch.float32) - 0.5 * dictionary_size + 0.5)
            / dictionary_size
        )
        return 0.1 * (uniform + ramp.unsqueeze(-1))


class LpcnetDualFullyConnected(nn.Module):
    # Output stage producing one probability per binary-tree node. Two
    # independent bounded branches are combined under learnable per-output
    # weights before the final gate, which lets each node compose two
    # differently-shaped responses instead of one; the reference recipe uses
    # this in place of a wider single layer, because the output stage runs
    # once per sample and its cost is on the critical path.
    #
    # Both branches saturate before mixing, so the pre-gate sum is bounded by
    # the sum of the two factor magnitudes and the node probability cannot be
    # driven arbitrarily close to zero or one by activation growth alone.
    def __init__(self, input_dimension: int, output_dimension: int) -> None:
        # Builds the two branches and their mixing factors. The factors are
        # per-output rather than scalar, so each node weights the branches
        # independently, and they start at one so the stage begins as a plain
        # sum.
        super().__init__()
        self._first_branch: nn.Linear = nn.Linear(input_dimension, output_dimension)
        self._second_branch: nn.Linear = nn.Linear(input_dimension, output_dimension)
        self.first_factor: nn.Parameter = nn.Parameter(torch.ones(output_dimension))
        self.second_factor: nn.Parameter = nn.Parameter(torch.ones(output_dimension))

    @override
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Produces one node probability per output, each strictly inside the
        # open unit interval because the final gate is a sigmoid over a finite
        # sum. That strictness matters downstream: the tree expansion
        # multiplies these values and their complements, so a node at exactly
        # zero or one would make an entire branch of the distribution
        # unreachable.
        first_activation: torch.Tensor = torch.tanh(self._first_branch(features))
        second_activation: torch.Tensor = torch.tanh(self._second_branch(features))
        return torch.sigmoid(self.first_factor * first_activation + self.second_factor * second_activation)


class LpcnetTreeProbability(nn.Module):
    # Expands the binary-tree node activations into an explicit distribution
    # over all mu-law levels. Each level of the tree splits the remaining
    # index range in half, so a level's probability is the product of the
    # branch decisions taken to reach it; with eight levels the tree addresses
    # two hundred fifty-six leaves through two hundred fifty-five internal
    # nodes, which is why the node vector's first entry is never read.
    #
    # Only synthesis needs this expansion, because sampling requires the full
    # distribution. Training does not use it at all: the objective evaluates
    # the likelihood of one known target by walking its eight-node path
    # directly, which is far cheaper than materializing every leaf.
    def __init__(self, level_count: int = 8) -> None:
        # Fixes the tree depth and derives the leaf count from it.
        #
        # Args:
        #     level_count: Depth of the tree. Default: ``8``, giving the
        #         two-hundred-fifty-six-level mu-law alphabet.
        super().__init__()
        self._level_count: int = level_count
        self._pcm_levels: int = 2 ** level_count

    @override
    def forward(self, node_outputs: torch.Tensor) -> torch.Tensor:
        # Accumulates the leaf distribution level by level. Each iteration
        # reads the nodes belonging to one depth, pairs each node's activation
        # with its complement to form the two branch probabilities, and
        # repeats each pair across the leaves it governs before multiplying it
        # into the running product. Because every level contributes exactly
        # one factor to every leaf, the result is normalized by construction
        # and needs no explicit division.
        #
        # Args:
        #     node_outputs: Node activations whose last axis has one entry per
        #         leaf, of which entries one through two hundred fifty-five
        #         are the internal nodes.
        #
        # Returns:
        #     The leaf distribution, with the same leading axes as the input.
        probabilities: torch.Tensor = torch.ones(
            *node_outputs.shape[:-1],
            self._pcm_levels,
            device=node_outputs.device,
            dtype=node_outputs.dtype
        )
        for level in range(self._level_count):
            node_count: int = 2 ** level
            level_nodes: torch.Tensor = node_outputs[..., node_count: 2 * node_count]
            branch_pair: torch.Tensor = torch.stack([1.0 - level_nodes, level_nodes], dim=-1)
            repeat_count: int = self._pcm_levels // (2 * node_count)
            expanded: torch.Tensor = branch_pair.repeat_interleave(repeat_count, dim=-1).reshape(
                *node_outputs.shape[:-1],
                self._pcm_levels
            )
            probabilities: torch.Tensor = probabilities * expanded
        return probabilities


class LpcnetLinearPrediction(nn.Module):
    # Applies the frame-rate linear-prediction coefficients at sample rate.
    # This module holds no parameters: the coefficients come from the fixed
    # analysis front end, and the network's task is to predict what this
    # filter cannot explain rather than to learn the filter itself.
    #
    # Evaluating the whole sequence at once is possible only under teacher
    # forcing, where the true past samples are known in advance. The
    # synthesis path cannot use this module and computes the same quantity one
    # sample at a time against its own generated history.
    def __init__(self, lpc_order: int, frame_size: int) -> None:
        # Records the filter order and the frame size, the latter being the
        # factor by which frame-rate coefficients are expanded to sample rate.
        super().__init__()
        self._lpc_order: int = lpc_order
        self._frame_size: int = frame_size

    @override
    def forward(self, samples: torch.Tensor, lpc_coefficients: torch.Tensor) -> torch.Tensor:
        # Computes the prediction for every sample in one pass.
        #
        # The lagged inputs are assembled by left-padding the sequence with
        # the filter order and then slicing one window per tap, so tap zero
        # sees the immediately preceding sample and tap n sees the sample n
        # positions back. The padding is what makes the opening samples well
        # defined: their missing history reads as zeros rather than wrapping
        # around from the end of the sequence.
        #
        # Args:
        #     samples: Past samples of shape ``[batch, samples, 1]`` on the
        #         int16 scale. Under teacher forcing these are the true
        #         samples shifted by one.
        #     lpc_coefficients: Frame-rate coefficients of shape
        #         ``[batch, frames, lpc_order]``, expanded here to sample rate
        #         and truncated to the sample count, so a coefficient set
        #         covering more frames than there are samples is tolerated.
        #
        # Returns:
        #     The prediction of shape ``[batch, samples, 1]``. The sign is
        #     negated because the analysis produces coefficients of the
        #     inverse filter, whose convention makes the prediction the
        #     negated weighted sum of the history.
        sample_rate_coefficients: torch.Tensor = lpc_coefficients.repeat_interleave(self._frame_size, dim=1)
        padded: torch.Tensor = torch.nn.functional.pad(samples, (0, 0, self._lpc_order, 0))
        sample_count: int = samples.shape[1]
        lagged_columns: list[torch.Tensor] = [
            padded[:, self._lpc_order - tap_index: self._lpc_order - tap_index + sample_count, 0]
            for tap_index in range(self._lpc_order)
        ]
        lagged: torch.Tensor = torch.stack(lagged_columns, dim=-1)
        prediction: torch.Tensor = -(sample_rate_coefficients[:, :sample_count] * lagged).sum(dim=-1, keepdim=True)
        return prediction


class LpcnetNetwork(nn.Module):
    # The complete LPCNet network across both rates. The frame-rate encoder
    # turns analysis features and a pitch embedding into one conditioning
    # vector per frame; the sample-rate core embeds three autoregressive
    # inputs, concatenates the repeated conditioning vector, and runs two
    # recurrent layers into the binary-tree output stage.
    #
    # The widths are interlocked. The first recurrent layer's input is three
    # embeddings plus the conditioning width, because three signal values are
    # embedded per sample. The second layer's input is the first layer's
    # hidden width plus the conditioning width, so the conditioning is
    # presented to both layers rather than only at the bottom. The output
    # stage emits one activation per mu-law level, of which all but the first
    # are tree nodes.
    #
    # Integration: this class exposes the sample-rate pieces the autoregressive
    # sampler drives directly. The sampler steps both recurrent layers itself,
    # carrying their hidden states across samples, and calls the embedding,
    # output stage, and tree expansion in turn; forward is the teacher-forced
    # training path only and is never used for synthesis.
    def __init__(
        self,
        feature_dimension: int,
        condition_dimension: int,
        embedding_dimension: int,
        first_gru_dimension: int,
        second_gru_dimension: int,
        lpc_order: int,
        frame_size: int,
        training_noise_standard_deviation: float
    ) -> None:
        # Builds both rate paths. The encoder's convolutions use
        # length-preserving padding so the frame count survives them
        # unchanged, and the pitch embedding contributes a fixed
        # sixty-four-wide vector concatenated onto the analysis features,
        # which is why the first convolution's input width exceeds the
        # feature width by that amount.
        super().__init__()
        self._frame_size: int = frame_size
        self._training_noise_standard_deviation: float = training_noise_standard_deviation
        self._mulaw: LpcnetMuLaw = LpcnetMuLaw()
        self.pitch_embedding: nn.Embedding = nn.Embedding(256, 64)
        self.first_feature_convolution: nn.Conv1d = nn.Conv1d(
            feature_dimension + 64,
            condition_dimension,
            kernel_size=3,
            padding="same"
        )
        self.second_feature_convolution: nn.Conv1d = nn.Conv1d(
            condition_dimension,
            condition_dimension,
            kernel_size=3,
            padding="same"
        )
        self.first_feature_dense: nn.Linear = nn.Linear(condition_dimension, condition_dimension)
        self.second_feature_dense: nn.Linear = nn.Linear(condition_dimension, condition_dimension)
        self.signal_embedding: LpcnetFractionalEmbedding = LpcnetFractionalEmbedding(256, embedding_dimension)
        self.first_gru: nn.GRU = nn.GRU(
            embedding_dimension * 3 + condition_dimension,
            first_gru_dimension,
            batch_first=True
        )
        self.second_gru: nn.GRU = nn.GRU(
            first_gru_dimension + condition_dimension,
            second_gru_dimension,
            batch_first=True
        )
        self.dual_fully_connected: LpcnetDualFullyConnected = LpcnetDualFullyConnected(second_gru_dimension, 256)
        self._tree_probability: LpcnetTreeProbability = LpcnetTreeProbability()
        self._linear_prediction: LpcnetLinearPrediction = LpcnetLinearPrediction(lpc_order, frame_size)

    @property
    def mulaw(self) -> LpcnetMuLaw:
        # Returns the mu-law codec shared by training, loss, and synthesis paths.
        return self._mulaw

    @property
    def frame_size(self) -> int:
        # Returns the number of samples represented by one conditioning frame.
        return self._frame_size

    def encode_conditioning(self, conditioning: torch.Tensor, pitch_index: torch.Tensor) -> torch.Tensor:
        # Encodes frame features and the pitch embedding into the frame-rate
        # conditioning vector. The pitch period enters as a learned embedding
        # rather than a numeric feature, because period is an index into a
        # perceptually non-linear scale and a raw value would impose a linear
        # geometry the model would have to undo.
        #
        # The two convolutions give each conditioning vector a receptive field
        # spanning several frames, so a sample's conditioning reflects the
        # local trajectory rather than one frame in isolation. Both are
        # length-preserving, and the two dense layers that follow mix channels
        # per frame without touching the time axis.
        #
        # Args:
        #     conditioning: Analysis features of shape
        #         ``[batch, frames, feature_dimension]``.
        #     pitch_index: Integer pitch-period indices of shape
        #         ``[batch, frames]``.
        #
        # Returns:
        #     Conditioning vectors of shape
        #     ``[batch, frames, condition_dimension]``, bounded because every
        #     stage of the encoder ends in a hyperbolic tangent.
        pitch_vectors: torch.Tensor = self.pitch_embedding(pitch_index)
        stacked: torch.Tensor = torch.cat([conditioning, pitch_vectors], dim=-1).transpose(1, 2)
        convolved: torch.Tensor = torch.tanh(self.first_feature_convolution(stacked))
        convolved: torch.Tensor = torch.tanh(self.second_feature_convolution(convolved)).transpose(1, 2)
        densified: torch.Tensor = torch.tanh(self.first_feature_dense(convolved))
        return torch.tanh(self.second_feature_dense(densified))

    def compute_prediction(self, samples: torch.Tensor, lpc_coefficients: torch.Tensor) -> torch.Tensor:
        # Computes the per-sample linear prediction from the frame-rate LPC coefficients.
        return self._linear_prediction(samples, lpc_coefficients)

    def expand_tree_probabilities(self, node_outputs: torch.Tensor) -> torch.Tensor:
        # Expands the binary-tree node outputs into the 256-level excitation distribution.
        return self._tree_probability(node_outputs)

    @override
    def forward(
        self,
        target_samples: torch.Tensor,
        conditioning: torch.Tensor,
        pitch_index: torch.Tensor,
        lpc_coefficients: torch.Tensor
    ) -> LpcnetNetworkOutput:
        # Executes the teacher-forced training pass over int16-scaled sample
        # sequences, evaluating every sample in one shot.
        #
        # Args:
        #     target_samples: True samples of shape ``[batch, samples, 1]`` on
        #         the int16 scale.
        #     conditioning: Frame-rate analysis features.
        #     pitch_index: Frame-rate pitch-period indices.
        #     lpc_coefficients: Frame-rate prediction coefficients.
        #
        # Returns:
        #     The per-sample prediction and the tree node activations. The
        #     objective derives the target excitation from the first and
        #     scores it against the second.
        condition_vectors: torch.Tensor = self.encode_conditioning(conditioning, pitch_index)
        # Teacher forcing: each step is shown the previous true sample, so the
        # target sequence is shifted forward by one and its opening position
        # zero-filled. The zero fill replaces what the roll wrapped around
        # from the end, which would otherwise leak the last sample of the
        # utterance into its first step.
        input_samples: torch.Tensor = torch.roll(target_samples, shifts=1, dims=1)
        input_samples: torch.Tensor = torch.cat([torch.zeros_like(input_samples[:, :1]), input_samples[:, 1:]], dim=1)
        prediction: torch.Tensor = self.compute_prediction(input_samples, lpc_coefficients)
        # The previous excitation is reconstructed as the residual between the
        # previous input sample and the prediction made for it, which requires
        # shifting the prediction sequence by one under the same zero fill.
        previous_prediction: torch.Tensor = torch.roll(prediction, shifts=1, dims=1)
        previous_prediction: torch.Tensor = torch.cat(
            [torch.zeros_like(previous_prediction[:, :1]), previous_prediction[:, 1:]],
            dim=1
        )
        past_excitation: torch.Tensor = self._mulaw.linear_to_mulaw(input_samples - previous_prediction)
        # The three autoregressive inputs are companded onto the mu-law index
        # domain and concatenated, so the embedding table sees all three under
        # one shared representation.
        mulaw_inputs: torch.Tensor = torch.cat(
            [
                self._mulaw.linear_to_mulaw(input_samples),
                self._mulaw.linear_to_mulaw(prediction),
                past_excitation
            ],
            dim=-1
        )
        # Noise is injected before embedding and only while training. Teacher
        # forcing otherwise shows the model none of the errors it will
        # encounter when driven by its own output, so perturbing the inputs is
        # what teaches it to recover from them; the fractional embedding is
        # what makes a perturbed, non-integer index meaningful.
        if self.training and self._training_noise_standard_deviation > 0.0:
            mulaw_inputs: torch.Tensor = mulaw_inputs + torch.randn_like(mulaw_inputs) * self._training_noise_standard_deviation
        embedded: torch.Tensor = self.signal_embedding(mulaw_inputs)
        flat_embedded: torch.Tensor = embedded.reshape(embedded.shape[0], embedded.shape[1], -1)
        # The frame-rate conditioning is expanded to sample rate and truncated
        # to the sample count, which absorbs a trailing frame whose samples
        # were cut away upstream.
        repeated_condition: torch.Tensor = condition_vectors.repeat_interleave(self._frame_size, dim=1)
        repeated_condition: torch.Tensor = repeated_condition[:, : flat_embedded.shape[1]]
        first_gru_output: torch.Tensor
        first_gru_output, _ = self.first_gru(torch.cat([flat_embedded, repeated_condition], dim=-1))
        second_gru_output: torch.Tensor
        second_gru_output, _ = self.second_gru(torch.cat([first_gru_output, repeated_condition], dim=-1))
        node_outputs: torch.Tensor = self.dual_fully_connected(second_gru_output)
        return LpcnetNetworkOutput(prediction=prediction, node_outputs=node_outputs)
