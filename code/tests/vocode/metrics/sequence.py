# This module:
# 1. Verifies the static complexity panel computed once per pass: the parameter
#    and size columns, and the preference for a module's declared deployable
#    quantities over live-object measurements
# 2. Verifies metric composition by ordered selection: unselected metrics
#    produce no columns and selection order does not change the reported panel
# 3. Verifies per-utterance evaluation: padding never enters a measurement, the
#    utterance denominator counts evaluated pairs, and accumulation state resets
#    between passes
# 4. Verifies failure accounting, the loud reduction failure of a metric with no
#    valid values, the module and batch contract errors, and the per-utterance
#    table written for paired statistical analysis
#
# Design decisions:
# - Composition is exercised with the inexpensive members of the panel only.
#   The mel error runs over a minimal mel protocol; PESQ, STOI, MCD, and the
#   pitch family are excluded because they bind heavyweight analysis grids
#   or, for UTMOS, a downloaded predictor, and none of them is what the
#   sequencing logic under test decides
# - Metric values are asserted as exact zero for identical signals and as
#   strictly positive for differing signals, rather than pinned to a magnitude
# - The logger stand-in records payloads in memory; its save directory is an
#   existing temporary directory and its log directory is never resolved, so the
#   only file these tests create is the per-utterance table inside a temporary
#   directory that is removed on teardown
#
# Author: Rahul Sawhney

import csv
import tempfile
import unittest
from pathlib import Path
from typing import override

import torch
from torch import nn

from syntheticmind.core.module import Module
from syntheticmind.core.trainer import Trainer
from syntheticmind.loggers.logger import Logger
from syntheticmind.utilities.types import HyperparameterDict

from vocode.metrics.registry import MetricSelection
from vocode.metrics.sequence import MetricSequence
from vocode.metrics.size import ModelSize
from vocode.transforms.mel import MelConfig


class RecordingLogger(Logger):
    # Logger stand-in retaining every published metric payload in memory so the
    # panel's publication points are inspectable without touching the filesystem.
    def __init__(self, save_dir: Path) -> None:
        # Binds the logger identity and prepares the empty payload record.
        super().__init__(save_dir)
        self._payloads: list[dict[str, float]] = []

    @override
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        # Retains a copy of one published payload.
        self._payloads.append(dict(metrics))

    @override
    def log_hyperparams(self, params: HyperparameterDict) -> None:
        # Accepts hyperparameters without recording them; the panel publishes none.
        pass

    @override
    def finalize(self) -> None:
        # Holds no open resource to release.
        pass

    @property
    def payloads(self) -> list[dict[str, float]]:
        # Returns a copy of every payload published so far.
        return list(self._payloads)


class ProbeMelProtocol:
    # Builds the deliberately small metric mel protocol the evaluated modules
    # declare, keeping every extracted spectrogram minimal.
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


class EvaluatedModule(Module):
    # Minimal evaluated module satisfying the metric-model contract: a
    # synthesis network for complexity metrics and a declared metric protocol.
    def __init__(self, mel_protocol: MelConfig, network: nn.Module) -> None:
        # Binds the synthesis network and the declared metric mel protocol.
        super().__init__()
        self._mel_protocol: MelConfig = mel_protocol
        self._network: nn.Module = network

    @property
    def network(self) -> nn.Module:
        # Returns the network the complexity metrics measure.
        return self._network

    @property
    def metric_mel_protocol(self) -> MelConfig:
        # Returns the protocol the mel error extracts its spectrograms on.
        return self._mel_protocol


class DeployableModule(Module):
    # Evaluated module additionally declaring the deployable quantities a
    # transformed variant would actually ship.
    #
    # The declared values are deliberately unrelated to the live network's
    # own geometry, so a test cannot pass by coincidence: a panel reporting
    # the declaration and a panel reporting the live measurement produce
    # visibly different numbers. This stands in for a quantized or exported
    # variant, whose live object no longer reflects what would be shipped.
    # Dense masked pruning is not such a case, because it changes neither
    # tensor shape nor serialized size.
    def __init__(
        self,
        mel_protocol: MelConfig,
        network: nn.Module,
        declared_parameter_count: int,
        declared_artifact_bytes: int
    ) -> None:
        # Binds the network, the metric protocol, and the shipped quantities the
        # module declares in place of its live measurements.
        super().__init__()
        self._mel_protocol: MelConfig = mel_protocol
        self._network: nn.Module = network
        self._declared_parameter_count: int = declared_parameter_count
        self._declared_artifact_bytes: int = declared_artifact_bytes

    @property
    def network(self) -> nn.Module:
        # Returns the network the complexity metrics measure.
        return self._network

    @property
    def metric_mel_protocol(self) -> MelConfig:
        # Returns the protocol the mel error extracts its spectrograms on.
        return self._mel_protocol

    @property
    def deployable_parameter_count(self) -> int:
        # Returns the weight count a transformed variant would actually ship.
        return self._declared_parameter_count

    @property
    def deployable_artifact_bytes(self) -> int:
        # Returns the exported artifact size a transformed variant would ship.
        return self._declared_artifact_bytes


class ContractlessModule(Module):
    # Harness module declaring neither a network nor a metric mel protocol, so
    # the metric-model contract check must reject it.
    def __init__(self) -> None:
        # Initializes the harness module and declares nothing further.
        super().__init__()


class SyntheticUtteranceBatch:
    # Builds one collated test batch: reference waveforms padded beyond their
    # declared true length, the declared lengths, and the batch sample rate.
    #
    # The padding is a constant tail rather than zeros, and it differs from
    # whatever the candidate carries in the same region. That makes the
    # crop observable: a measurement that honours the declared length
    # scores exactly zero on an otherwise identical pair, while one that
    # reads into the padding cannot. Passing zero padding produces an
    # ordinary unpadded batch for the cases where the crop is not the
    # subject.
    def __init__(self, waveform: torch.Tensor, sample_rate: int, padding_samples: int) -> None:
        # Binds the true waveforms, the batch rate, and how far to pad them.
        self._waveform: torch.Tensor = waveform
        self._sample_rate: int = sample_rate
        self._padding_samples: int = padding_samples

    def create(self) -> dict[str, torch.Tensor | int]:
        # Appends a constant tail past the declared length, so any measurement
        # that reads the padding is distinguishable from one that crops first.
        padding: torch.Tensor = torch.full(
            (self._waveform.shape[0], self._padding_samples),
            0.5
        )
        padded: torch.Tensor = torch.cat([self._waveform, padding], dim=-1)
        lengths: torch.Tensor = torch.full(
            (self._waveform.shape[0],),
            self._waveform.shape[-1]
        )
        return {
            "waveform": padded,
            "waveform_length": lengths,
            "sample_rate": self._sample_rate
        }


class MetricSequenceStaticPanelTest(unittest.TestCase):
    # Verifies the complexity panel computed once per pass from the network.
    def setUp(self) -> None:
        # Prepares a CPU trainer, a four-by-three module, and a static selection.
        self._logger: RecordingLogger = RecordingLogger(Path(tempfile.gettempdir()))
        self._trainer: Trainer = Trainer(
            accelerator="cpu",
            strategy="single_device",
            enable_progress_bar=False,
            logger=self._logger
        )
        self._module: EvaluatedModule = EvaluatedModule(
            ProbeMelProtocol().create(),
            nn.Linear(4, 3)
        )
        self._sequence: MetricSequence = MetricSequence(
            MetricSelection(names=("parameters", "size"))
        )

    def test_static_selection_reports_the_complexity_columns(self) -> None:
        # Selecting parameters and size yields both definitions of each quantity.
        self._sequence.on_test_start(self._trainer, self._module)
        self._sequence.on_test_end(self._trainer, self._module)
        self.assertEqual(
            set(self._sequence.results),
            {
                "parameter_count",
                "residual_parameter_count",
                "model_size_megabytes",
                "deployable_size_megabytes"
            }
        )

    def test_parameter_count_matches_the_live_network_total(self) -> None:
        # A four-by-three linear layer holds twelve weights and three biases.
        self._sequence.on_test_start(self._trainer, self._module)
        self.assertEqual(self._sequence.results["parameter_count"], float(4 * 3 + 3))

    def test_residual_count_accompanies_the_deployable_count(self) -> None:
        # The unquantized remainder is always reported beside the deployable count.
        self._sequence.on_test_start(self._trainer, self._module)
        self.assertEqual(self._sequence.results["residual_parameter_count"], float(4 * 3 + 3))

    def test_model_size_matches_the_in_memory_parameter_bytes(self) -> None:
        # Fifteen float32 parameters occupy sixty bytes.
        self._sequence.on_test_start(self._trainer, self._module)
        self.assertEqual(
            self._sequence.results["model_size_megabytes"],
            (4 * 3 + 3) * 4 / (1024.0 * 1024.0)
        )

    def test_deployable_size_measures_the_serialized_state(self) -> None:
        # Without a declared artifact, the deployable size is the serialized state.
        self._sequence.on_test_start(self._trainer, self._module)
        self.assertEqual(
            self._sequence.results["deployable_size_megabytes"],
            ModelSize().serialized_megabytes(self._module.network)
        )

    def test_static_metrics_publish_before_the_first_batch(self) -> None:
        # Static metrics must exist even if a later batch fails the run.
        self._sequence.on_test_start(self._trainer, self._module)
        self.assertEqual(len(self._logger.payloads), 1)
        self.assertIn("parameter_count", self._logger.payloads[0])

    def test_static_only_pass_records_no_utterance_denominator(self) -> None:
        # A pass measuring no waveforms has no utterance count to report.
        self._sequence.on_test_start(self._trainer, self._module)
        self._sequence.on_test_end(self._trainer, self._module)
        self.assertNotIn("test_utterance_count", self._sequence.results)


class MetricSequenceDeployablePreferenceTest(unittest.TestCase):
    # Verifies that declared deployable quantities take precedence over
    # live-object measurements, so transformed variants report what ships.
    def setUp(self) -> None:
        # Prepares a CPU trainer, the probe protocol, and a static selection.
        self._logger: RecordingLogger = RecordingLogger(Path(tempfile.gettempdir()))
        self._trainer: Trainer = Trainer(
            accelerator="cpu",
            strategy="single_device",
            enable_progress_bar=False,
            logger=self._logger
        )
        self._mel_protocol: MelConfig = ProbeMelProtocol().create()
        self._sequence: MetricSequence = MetricSequence(
            MetricSelection(names=("parameters", "size"))
        )

    def test_declared_parameter_count_overrides_the_live_count(self) -> None:
        # A packed network would otherwise appear to have lost its weights.
        module: DeployableModule = DeployableModule(
            self._mel_protocol,
            nn.Linear(4, 3),
            7,
            2 * 1024 * 1024
        )
        self._sequence.on_test_start(self._trainer, module)
        self.assertEqual(self._sequence.results["parameter_count"], 7.0)

    def test_residual_count_still_reports_the_registered_parameters(self) -> None:
        # The unquantized remainder is reported independently of the declaration.
        module: DeployableModule = DeployableModule(
            self._mel_protocol,
            nn.Linear(4, 3),
            7,
            2 * 1024 * 1024
        )
        self._sequence.on_test_start(self._trainer, module)
        self.assertEqual(self._sequence.results["residual_parameter_count"], float(4 * 3 + 3))

    def test_declared_artifact_bytes_override_the_serialized_size(self) -> None:
        # An exported artifact is the authoritative deployable size when declared.
        module: DeployableModule = DeployableModule(
            self._mel_protocol,
            nn.Linear(4, 3),
            7,
            2 * 1024 * 1024
        )
        self._sequence.on_test_start(self._trainer, module)
        self.assertEqual(self._sequence.results["deployable_size_megabytes"], 2.0)

    def test_live_size_column_is_unaffected_by_the_declaration(self) -> None:
        # The in-memory column keeps continuity with the untransformed rows.
        module: DeployableModule = DeployableModule(
            self._mel_protocol,
            nn.Linear(4, 3),
            7,
            2 * 1024 * 1024
        )
        self._sequence.on_test_start(self._trainer, module)
        self.assertEqual(
            self._sequence.results["model_size_megabytes"],
            (4 * 3 + 3) * 4 / (1024.0 * 1024.0)
        )

    def test_non_positive_declarations_fall_back_to_live_measurement(self) -> None:
        # A zero declaration is an absent declaration, not a zero-sized model.
        module: DeployableModule = DeployableModule(self._mel_protocol, nn.Linear(4, 3), 0, 0)
        self._sequence.on_test_start(self._trainer, module)
        self.assertEqual(self._sequence.results["parameter_count"], float(4 * 3 + 3))
        self.assertEqual(
            self._sequence.results["deployable_size_megabytes"],
            ModelSize().serialized_megabytes(module.network)
        )


class MetricSequenceCompositionTest(unittest.TestCase):
    # Verifies that the panel is composed from the selected names alone and is
    # independent of the order in which those names were declared.
    #
    # Two properties of the ordered-name contract are separated here. A
    # name absent from the selection must produce no column at all, which
    # is what makes a selection a statement of scope rather than a hint.
    # And two selections differing only in order must produce equal
    # panels, because the reduced result is a mapping keyed by result
    # column; order remains observable elsewhere, in the column layout of
    # the per-utterance table, which the table tests cover separately.
    def setUp(self) -> None:
        # Seeds the waveform draws and prepares a CPU trainer and evaluated module.
        torch.manual_seed(7)
        self._logger: RecordingLogger = RecordingLogger(Path(tempfile.gettempdir()))
        self._trainer: Trainer = Trainer(
            accelerator="cpu",
            strategy="single_device",
            enable_progress_bar=False,
            logger=self._logger
        )
        self._module: EvaluatedModule = EvaluatedModule(
            ProbeMelProtocol().create(),
            nn.Linear(4, 3)
        )

    def test_results_are_empty_before_the_pass(self) -> None:
        # A sequence that has measured nothing reports nothing.
        sequence: MetricSequence = MetricSequence(MetricSelection(names=("parameters",)))
        self.assertEqual(sequence.results, {})

    def test_unselected_metrics_produce_no_columns(self) -> None:
        # Selecting parameters alone must not compute or report size.
        sequence: MetricSequence = MetricSequence(MetricSelection(names=("parameters",)))
        sequence.on_test_start(self._trainer, self._module)
        sequence.on_test_end(self._trainer, self._module)
        self.assertIn("parameter_count", sequence.results)
        self.assertNotIn("model_size_megabytes", sequence.results)

    def test_selection_order_does_not_change_the_reported_columns(self) -> None:
        # Selection is a set of names to compute, not an output ordering.
        forward_sequence: MetricSequence = MetricSequence(
            MetricSelection(names=("parameters", "size"))
        )
        reversed_sequence: MetricSequence = MetricSequence(
            MetricSelection(names=("size", "parameters"))
        )
        forward_sequence.on_test_start(self._trainer, self._module)
        forward_sequence.on_test_end(self._trainer, self._module)
        reversed_sequence.on_test_start(self._trainer, self._module)
        reversed_sequence.on_test_end(self._trainer, self._module)
        self.assertEqual(forward_sequence.results, reversed_sequence.results)

    def test_waveform_selection_reports_a_mean_and_a_failure_count(self) -> None:
        # Every per-utterance metric carries its own failure denominator.
        sequence: MetricSequence = MetricSequence(MetricSelection(names=("mel",)))
        sequence.on_test_start(self._trainer, self._module)
        waveform: torch.Tensor = torch.randn(1, 256) * 0.1
        sequence.on_test_batch_end(
            self._trainer,
            self._module,
            {"synthesized_waveform": waveform.clone()},
            SyntheticUtteranceBatch(waveform, 1000, 0).create(),
            0
        )
        sequence.on_test_end(self._trainer, self._module)
        self.assertIn("mel_error", sequence.results)
        self.assertEqual(sequence.results["mel_error_failure_count"], 0.0)

    def test_static_and_waveform_families_compose_into_one_panel(self) -> None:
        # A mixed selection reports both lifecycles in a single result mapping.
        sequence: MetricSequence = MetricSequence(
            MetricSelection(names=("mel", "parameters"))
        )
        sequence.on_test_start(self._trainer, self._module)
        waveform: torch.Tensor = torch.randn(1, 256) * 0.1
        sequence.on_test_batch_end(
            self._trainer,
            self._module,
            {"synthesized_waveform": waveform.clone()},
            SyntheticUtteranceBatch(waveform, 1000, 0).create(),
            0
        )
        sequence.on_test_end(self._trainer, self._module)
        self.assertIn("parameter_count", sequence.results)
        self.assertIn("mel_error", sequence.results)

    def test_results_are_returned_as_a_defensive_copy(self) -> None:
        # A caller mutating the returned panel must not corrupt the record.
        sequence: MetricSequence = MetricSequence(MetricSelection(names=("parameters",)))
        sequence.on_test_start(self._trainer, self._module)
        sequence.on_test_end(self._trainer, self._module)
        borrowed_results: dict[str, float] = sequence.results
        borrowed_results["parameter_count"] = -1.0
        self.assertEqual(sequence.results["parameter_count"], float(4 * 3 + 3))


class MetricSequenceWaveformEvaluationTest(unittest.TestCase):
    # Verifies per-utterance evaluation over reference and candidate pairs,
    # including length handling and state reset between passes.
    def setUp(self) -> None:
        # Seeds the waveform draws and prepares a mel-only per-utterance panel.
        torch.manual_seed(11)
        self._logger: RecordingLogger = RecordingLogger(Path(tempfile.gettempdir()))
        self._trainer: Trainer = Trainer(
            accelerator="cpu",
            strategy="single_device",
            enable_progress_bar=False,
            logger=self._logger
        )
        self._module: EvaluatedModule = EvaluatedModule(
            ProbeMelProtocol().create(),
            nn.Linear(4, 3)
        )
        self._sequence: MetricSequence = MetricSequence(MetricSelection(names=("mel",)))

    def test_identical_reference_and_candidate_score_zero_error(self) -> None:
        # A perfect reconstruction has no mel distance from its reference.
        waveform: torch.Tensor = torch.randn(1, 256) * 0.1
        self._sequence.on_test_start(self._trainer, self._module)
        self._sequence.on_test_batch_end(
            self._trainer,
            self._module,
            {"synthesized_waveform": waveform.clone()},
            SyntheticUtteranceBatch(waveform, 1000, 0).create(),
            0
        )
        self._sequence.on_test_end(self._trainer, self._module)
        self.assertEqual(self._sequence.results["mel_error"], 0.0)

    def test_differing_candidate_scores_a_positive_error(self) -> None:
        # A distorted candidate must be distinguishable from a perfect one.
        waveform: torch.Tensor = torch.randn(1, 256) * 0.1
        self._sequence.on_test_start(self._trainer, self._module)
        self._sequence.on_test_batch_end(
            self._trainer,
            self._module,
            {"synthesized_waveform": waveform * 0.25},
            SyntheticUtteranceBatch(waveform, 1000, 0).create(),
            0
        )
        self._sequence.on_test_end(self._trainer, self._module)
        self.assertGreater(self._sequence.results["mel_error"], 0.0)

    def test_padding_beyond_the_declared_length_never_enters_the_measurement(self) -> None:
        # Reference and candidate agree on the true region and disagree only in
        # the padded tail, so a nonzero error would prove padding was compared.
        waveform: torch.Tensor = torch.randn(1, 256) * 0.1
        candidate: torch.Tensor = torch.cat([waveform, torch.full((1, 64), 0.9)], dim=-1)
        self._sequence.on_test_start(self._trainer, self._module)
        self._sequence.on_test_batch_end(
            self._trainer,
            self._module,
            {"synthesized_waveform": candidate},
            SyntheticUtteranceBatch(waveform, 1000, 64).create(),
            0
        )
        self._sequence.on_test_end(self._trainer, self._module)
        self.assertEqual(self._sequence.results["mel_error"], 0.0)

    def test_missing_lengths_fall_back_to_the_padded_width(self) -> None:
        # The same disagreeing tail becomes measurable once no length is declared.
        waveform: torch.Tensor = torch.randn(1, 256) * 0.1
        padded_reference: torch.Tensor = torch.cat(
            [waveform, torch.full((1, 64), 0.5)],
            dim=-1
        )
        candidate: torch.Tensor = torch.cat([waveform, torch.full((1, 64), 0.9)], dim=-1)
        self._sequence.on_test_start(self._trainer, self._module)
        self._sequence.on_test_batch_end(
            self._trainer,
            self._module,
            {"synthesized_waveform": candidate},
            {"waveform": padded_reference, "sample_rate": 1000},
            0
        )
        self._sequence.on_test_end(self._trainer, self._module)
        self.assertGreater(self._sequence.results["mel_error"], 0.0)
        self.assertEqual(self._sequence.results["test_utterance_count"], 1.0)

    def test_utterance_denominator_counts_every_evaluated_pair(self) -> None:
        # The denominator accumulates across batches, not within one.
        waveform: torch.Tensor = torch.randn(2, 256) * 0.1
        self._sequence.on_test_start(self._trainer, self._module)
        for batch_index in range(2):
            self._sequence.on_test_batch_end(
                self._trainer,
                self._module,
                {"synthesized_waveform": waveform.clone()},
                SyntheticUtteranceBatch(waveform, 1000, 0).create(),
                batch_index
            )
        self._sequence.on_test_end(self._trainer, self._module)
        self.assertEqual(self._sequence.results["test_utterance_count"], 4.0)

    def test_accumulation_state_resets_between_passes(self) -> None:
        # A second pass measures its own utterances, never the previous pass's.
        waveform: torch.Tensor = torch.randn(2, 256) * 0.1
        self._sequence.on_test_start(self._trainer, self._module)
        self._sequence.on_test_batch_end(
            self._trainer,
            self._module,
            {"synthesized_waveform": waveform.clone()},
            SyntheticUtteranceBatch(waveform, 1000, 0).create(),
            0
        )
        self._sequence.on_test_end(self._trainer, self._module)
        single_waveform: torch.Tensor = waveform[:1]
        self._sequence.on_test_start(self._trainer, self._module)
        self._sequence.on_test_batch_end(
            self._trainer,
            self._module,
            {"synthesized_waveform": single_waveform.clone()},
            SyntheticUtteranceBatch(single_waveform, 1000, 0).create(),
            0
        )
        self._sequence.on_test_end(self._trainer, self._module)
        self.assertEqual(self._sequence.results["test_utterance_count"], 1.0)

    def test_final_panel_publishes_through_the_experiment_logger(self) -> None:
        # The reduced panel reaches the logger at the end of the pass.
        waveform: torch.Tensor = torch.randn(1, 256) * 0.1
        self._sequence.on_test_start(self._trainer, self._module)
        self._sequence.on_test_batch_end(
            self._trainer,
            self._module,
            {"synthesized_waveform": waveform.clone()},
            SyntheticUtteranceBatch(waveform, 1000, 0).create(),
            0
        )
        self._sequence.on_test_end(self._trainer, self._module)
        self.assertIn("mel_error", self._logger.payloads[-1])


class MetricSequenceFailureAccountingTest(unittest.TestCase):
    # Verifies that a failed per-utterance metric is counted rather than
    # silently dropped, and that a metric with no valid value fails the run.
    #
    # Failures are induced with a non-finite candidate, which is the
    # realistic failure mode for a diverged synthesis and reaches the same
    # accounting path as a raised exception. The two boundaries either side
    # of the same rule are covered together: one failed utterance among
    # several must be counted while the surviving mean stays intact, and a
    # metric that fails on every utterance must stop the run rather than
    # reduce to nothing.
    def setUp(self) -> None:
        # Seeds the waveform draws and prepares a mel-only per-utterance panel.
        torch.manual_seed(13)
        self._logger: RecordingLogger = RecordingLogger(Path(tempfile.gettempdir()))
        self._trainer: Trainer = Trainer(
            accelerator="cpu",
            strategy="single_device",
            enable_progress_bar=False,
            logger=self._logger
        )
        self._module: EvaluatedModule = EvaluatedModule(
            ProbeMelProtocol().create(),
            nn.Linear(4, 3)
        )
        self._sequence: MetricSequence = MetricSequence(MetricSelection(names=("mel",)))

    def test_non_finite_value_is_counted_as_a_failure(self) -> None:
        # A non-finite value is a failure rather than a member of the mean.
        waveform: torch.Tensor = torch.randn(2, 256) * 0.1
        candidate: torch.Tensor = waveform.clone()
        candidate[1] = torch.full((256,), float("nan"))
        self._sequence.on_test_start(self._trainer, self._module)
        self._sequence.on_test_batch_end(
            self._trainer,
            self._module,
            {"synthesized_waveform": candidate},
            SyntheticUtteranceBatch(waveform, 1000, 0).create(),
            0
        )
        self._sequence.on_test_end(self._trainer, self._module)
        self.assertEqual(self._sequence.results["mel_error_failure_count"], 1.0)

    def test_failure_leaves_the_surviving_mean_intact(self) -> None:
        # The successful utterance still contributes its value to the mean.
        waveform: torch.Tensor = torch.randn(2, 256) * 0.1
        candidate: torch.Tensor = waveform.clone()
        candidate[1] = torch.full((256,), float("nan"))
        self._sequence.on_test_start(self._trainer, self._module)
        self._sequence.on_test_batch_end(
            self._trainer,
            self._module,
            {"synthesized_waveform": candidate},
            SyntheticUtteranceBatch(waveform, 1000, 0).create(),
            0
        )
        self._sequence.on_test_end(self._trainer, self._module)
        self.assertEqual(self._sequence.results["mel_error"], 0.0)

    def test_failed_utterances_still_count_toward_the_denominator(self) -> None:
        # The utterance denominator counts evaluated pairs, not successful ones.
        waveform: torch.Tensor = torch.randn(2, 256) * 0.1
        candidate: torch.Tensor = waveform.clone()
        candidate[1] = torch.full((256,), float("nan"))
        self._sequence.on_test_start(self._trainer, self._module)
        self._sequence.on_test_batch_end(
            self._trainer,
            self._module,
            {"synthesized_waveform": candidate},
            SyntheticUtteranceBatch(waveform, 1000, 0).create(),
            0
        )
        self._sequence.on_test_end(self._trainer, self._module)
        self.assertEqual(self._sequence.results["test_utterance_count"], 2.0)

    def test_metric_without_any_valid_value_fails_the_reduction(self) -> None:
        # A broken metric can never produce a silently absent column.
        waveform: torch.Tensor = torch.randn(1, 256) * 0.1
        self._sequence.on_test_start(self._trainer, self._module)
        self._sequence.on_test_batch_end(
            self._trainer,
            self._module,
            {"synthesized_waveform": torch.full((1, 256), float("nan"))},
            SyntheticUtteranceBatch(waveform, 1000, 0).create(),
            0
        )
        with self.assertRaisesRegex(RuntimeError, "produced no valid values"):
            self._sequence.on_test_end(self._trainer, self._module)


class MetricSequenceContractValidationTest(unittest.TestCase):
    # Verifies the module and batch contracts the panel requires before any
    # measurement uses them.
    def setUp(self) -> None:
        # Seeds the waveform draws and prepares one reference utterance to
        # pair against the deliberately malformed batches.
        torch.manual_seed(17)
        self._logger: RecordingLogger = RecordingLogger(Path(tempfile.gettempdir()))
        self._trainer: Trainer = Trainer(
            accelerator="cpu",
            strategy="single_device",
            enable_progress_bar=False,
            logger=self._logger
        )
        self._module: EvaluatedModule = EvaluatedModule(
            ProbeMelProtocol().create(),
            nn.Linear(4, 3)
        )
        self._sequence: MetricSequence = MetricSequence(MetricSelection(names=("mel",)))
        self._waveform: torch.Tensor = torch.randn(1, 256) * 0.1

    def test_module_without_the_metric_model_contract_is_rejected(self) -> None:
        # A module without a network and mel protocol cannot be measured.
        with self.assertRaisesRegex(TypeError, "metric model contract"):
            self._sequence.on_test_start(self._trainer, ContractlessModule())

    def test_batch_without_a_waveform_tensor_is_rejected(self) -> None:
        # The reference signal is mandatory for every waveform metric.
        self._sequence.on_test_start(self._trainer, self._module)
        with self.assertRaisesRegex(TypeError, "waveform"):
            self._sequence.on_test_batch_end(
                self._trainer,
                self._module,
                {"synthesized_waveform": self._waveform},
                {"sample_rate": 1000},
                0
            )

    def test_step_output_without_a_synthesized_waveform_is_rejected(self) -> None:
        # The candidate signal must travel under the declared output key.
        self._sequence.on_test_start(self._trainer, self._module)
        with self.assertRaisesRegex(TypeError, "synthesized_waveform"):
            self._sequence.on_test_batch_end(
                self._trainer,
                self._module,
                {"waveform": self._waveform},
                SyntheticUtteranceBatch(self._waveform, 1000, 0).create(),
                0
            )

    def test_mismatched_reference_and_candidate_batch_sizes_are_rejected(self) -> None:
        # Pairing utterances across differing batch sizes would misalign them.
        self._sequence.on_test_start(self._trainer, self._module)
        with self.assertRaisesRegex(ValueError, "does not match"):
            self._sequence.on_test_batch_end(
                self._trainer,
                self._module,
                {"synthesized_waveform": torch.randn(2, 256) * 0.1},
                SyntheticUtteranceBatch(self._waveform, 1000, 0).create(),
                0
            )

    def test_length_count_must_match_the_batch_size(self) -> None:
        # One declared length per utterance, or the crop targets the wrong signal.
        self._sequence.on_test_start(self._trainer, self._module)
        with self.assertRaisesRegex(ValueError, "does not match batch size"):
            self._sequence.on_test_batch_end(
                self._trainer,
                self._module,
                {"synthesized_waveform": self._waveform.clone()},
                {
                    "waveform": self._waveform,
                    "waveform_length": torch.tensor([256, 256]),
                    "sample_rate": 1000
                },
                0
            )

    def test_unsupported_length_payload_is_rejected(self) -> None:
        # A length payload the collator never produces is a contract violation.
        self._sequence.on_test_start(self._trainer, self._module)
        with self.assertRaisesRegex(TypeError, "waveform_length"):
            self._sequence.on_test_batch_end(
                self._trainer,
                self._module,
                {"synthesized_waveform": self._waveform.clone()},
                {
                    "waveform": self._waveform,
                    "waveform_length": "256",
                    "sample_rate": 1000
                },
                0
            )

    def test_unsupported_sample_rate_payload_is_rejected(self) -> None:
        # Metrics bind their analysis grid to the rate, so it cannot be guessed.
        self._sequence.on_test_start(self._trainer, self._module)
        with self.assertRaisesRegex(TypeError, "sample_rate"):
            self._sequence.on_test_batch_end(
                self._trainer,
                self._module,
                {"synthesized_waveform": self._waveform.clone()},
                {
                    "waveform": self._waveform,
                    "waveform_length": torch.tensor([256]),
                    "sample_rate": "1000"
                },
                0
            )

    def test_empty_utterance_is_rejected(self) -> None:
        # A zero-length overlap leaves nothing to measure.
        self._sequence.on_test_start(self._trainer, self._module)
        with self.assertRaisesRegex(ValueError, "Empty waveform"):
            self._sequence.on_test_batch_end(
                self._trainer,
                self._module,
                {"synthesized_waveform": self._waveform.clone()},
                {
                    "waveform": self._waveform,
                    "waveform_length": torch.tensor([0]),
                    "sample_rate": 1000
                },
                0
            )

    def test_pass_without_waveform_metrics_ignores_batches_entirely(self) -> None:
        # A static-only selection never inspects the batch contract.
        static_sequence: MetricSequence = MetricSequence(
            MetricSelection(names=("parameters",))
        )
        static_sequence.on_test_start(self._trainer, self._module)
        static_sequence.on_test_batch_end(self._trainer, self._module, {}, {}, 0)
        static_sequence.on_test_end(self._trainer, self._module)
        self.assertIn("parameter_count", static_sequence.results)


class MetricSequencePerUtteranceTableTest(unittest.TestCase):
    # Verifies the per-utterance metric table written for paired statistical
    # analysis, including the empty cell a failed metric leaves behind.
    #
    # The table exists because a paired significance test needs every
    # utterance, not the reduced mean, so the row count, the row labels
    # that trace a row back to its batch, and the result-schema column
    # names are all part of its contract. The empty cell is asserted
    # explicitly: a failed metric must leave its column intact and
    # readable rather than writing a sentinel a later analysis could
    # mistake for a measurement.
    def setUp(self) -> None:
        # Seeds the waveform draws and opens a temporary directory to hold the
        # only file these tests write.
        torch.manual_seed(19)
        self._logger: RecordingLogger = RecordingLogger(Path(tempfile.gettempdir()))
        self._trainer: Trainer = Trainer(
            accelerator="cpu",
            strategy="single_device",
            enable_progress_bar=False,
            logger=self._logger
        )
        self._module: EvaluatedModule = EvaluatedModule(
            ProbeMelProtocol().create(),
            nn.Linear(4, 3)
        )
        self._temporary_directory: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory()
        self._table_path: Path = Path(self._temporary_directory.name) / "per_utterance.csv"

    def tearDown(self) -> None:
        # Removes the temporary directory and the table written inside it.
        self._temporary_directory.cleanup()

    def test_table_records_one_row_per_evaluated_utterance(self) -> None:
        # Paired statistics need every utterance, not the reduced mean.
        waveform: torch.Tensor = torch.randn(2, 256) * 0.1
        sequence: MetricSequence = MetricSequence(
            MetricSelection(names=("mel",)),
            self._table_path
        )
        sequence.on_test_start(self._trainer, self._module)
        sequence.on_test_batch_end(
            self._trainer,
            self._module,
            {"synthesized_waveform": waveform.clone()},
            SyntheticUtteranceBatch(waveform, 1000, 0).create(),
            0
        )
        sequence.on_test_end(self._trainer, self._module)
        rows: list[dict[str, str]] = self._read_table()
        self.assertEqual(len(rows), 2)

    def test_table_columns_follow_the_result_schema(self) -> None:
        # The table is keyed by result-column names, not by metric names.
        waveform: torch.Tensor = torch.randn(1, 256) * 0.1
        sequence: MetricSequence = MetricSequence(
            MetricSelection(names=("mel",)),
            self._table_path
        )
        sequence.on_test_start(self._trainer, self._module)
        sequence.on_test_batch_end(
            self._trainer,
            self._module,
            {"synthesized_waveform": waveform.clone()},
            SyntheticUtteranceBatch(waveform, 1000, 0).create(),
            0
        )
        sequence.on_test_end(self._trainer, self._module)
        rows: list[dict[str, str]] = self._read_table()
        self.assertEqual(list(rows[0]), ["utterance_index", "label", "mel_error"])

    def test_rows_are_labelled_by_batch_and_utterance_position(self) -> None:
        # A row must be traceable back to the batch that produced it.
        waveform: torch.Tensor = torch.randn(1, 256) * 0.1
        sequence: MetricSequence = MetricSequence(
            MetricSelection(names=("mel",)),
            self._table_path
        )
        sequence.on_test_start(self._trainer, self._module)
        sequence.on_test_batch_end(
            self._trainer,
            self._module,
            {"synthesized_waveform": waveform.clone()},
            SyntheticUtteranceBatch(waveform, 1000, 0).create(),
            3
        )
        sequence.on_test_end(self._trainer, self._module)
        rows: list[dict[str, str]] = self._read_table()
        self.assertEqual(rows[0]["label"], "batch=3;utterance=0")

    def test_failed_metric_leaves_an_empty_cell(self) -> None:
        # An empty cell records the failure without corrupting the column.
        waveform: torch.Tensor = torch.randn(2, 256) * 0.1
        candidate: torch.Tensor = waveform.clone()
        candidate[1] = torch.full((256,), float("nan"))
        sequence: MetricSequence = MetricSequence(
            MetricSelection(names=("mel",)),
            self._table_path
        )
        sequence.on_test_start(self._trainer, self._module)
        sequence.on_test_batch_end(
            self._trainer,
            self._module,
            {"synthesized_waveform": candidate},
            SyntheticUtteranceBatch(waveform, 1000, 0).create(),
            0
        )
        sequence.on_test_end(self._trainer, self._module)
        rows: list[dict[str, str]] = self._read_table()
        self.assertEqual(rows[0]["mel_error"], "0.0")
        self.assertEqual(rows[1]["mel_error"], "")

    def test_table_is_skipped_without_a_destination(self) -> None:
        # The table is optional; omitting the path writes nothing at all.
        waveform: torch.Tensor = torch.randn(1, 256) * 0.1
        sequence: MetricSequence = MetricSequence(MetricSelection(names=("mel",)))
        sequence.on_test_start(self._trainer, self._module)
        sequence.on_test_batch_end(
            self._trainer,
            self._module,
            {"synthesized_waveform": waveform.clone()},
            SyntheticUtteranceBatch(waveform, 1000, 0).create(),
            0
        )
        sequence.on_test_end(self._trainer, self._module)
        self.assertFalse(self._table_path.exists())

    def _read_table(self) -> list[dict[str, str]]:
        # Reads the written table back as rows keyed by column name, materialized
        # so the caller can index and count them.
        with self._table_path.open("r", newline="", encoding="utf-8") as table_file:
            return list(csv.DictReader(table_file))


if __name__ == "__main__":
    unittest.main()
