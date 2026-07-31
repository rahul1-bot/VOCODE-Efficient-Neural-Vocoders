# This module:
# 1. Verifies the RFWave loss configuration: the auxiliary weight, the
#    logarithm floor, the frozen record, and the rejection of unknown or
#    non-positive settings
# 2. Verifies the time balance: the shapes it preserves, the unit variance it
#    produces on the target, and its invariance to a common rescaling
# 3. Verifies the rectified-flow velocity term against its mean-square closed form
# 4. Verifies the magnitude term: its zero at identical spectra, its floor on
#    silent spectra, its growth with a magnitude discrepancy, and the even
#    channel split it requires
# 5. Verifies the overlap term against a hand-derived value on constant bands,
#    its zero at agreement, its zero at vanishing overlap widths, and its
#    normalization by the band count
#
# Design decisions:
# - The overlap anchor uses constant bands so the adjacency arithmetic is
#   derivable by hand: each interior boundary is charged from both sides, and
#   the accumulated value is divided by the number of bands
# - The time balance is asserted through the unit variance it targets and
#   through rescaling invariance, rather than by restating its formula
# - Silent spectra are used to expose the logarithm floor, which is the only
#   guard between this term and a negative infinity
#
# Author: Rahul Sawhney

import unittest

import torch
from pydantic import ValidationError

from vocode.losses.rfwave import RfwaveLoss, RfwaveLossConfig


class BandPredictionBuilder:
    # Builds the per-band spectral slabs the overlap term compares.
    def __init__(self, slab_width: int, frame_count: int) -> None:
        # Binds the constructor inputs into this component's state.
        self._slab_width: int = slab_width
        self._frame_count: int = frame_count

    def constant_bands(self, values: list[float]) -> list[torch.Tensor]:
        # Builds one constant slab per requested band value.
        bands: list[torch.Tensor] = [
            torch.full((1, self._slab_width, self._frame_count), value) for value in values
        ]
        return bands

    def constant_bands_requiring_gradient(self, values: list[float]) -> list[torch.Tensor]:
        # Builds constant slabs that participate in autograd.
        bands: list[torch.Tensor] = [
            torch.full((1, self._slab_width, self._frame_count), value, requires_grad=True)
            for value in values
        ]
        return bands

    def seeded_bands(self, seed: int, band_count: int) -> list[torch.Tensor]:
        # Builds seeded random slabs for the requested number of bands.
        torch.manual_seed(seed)
        bands: list[torch.Tensor] = [
            torch.randn(1, self._slab_width, self._frame_count) for band_index in range(band_count)
        ]
        return bands


class RfwaveLossConfigurationTest(unittest.TestCase):
    # Verifies the frozen settings record behind the RFWave composition.
    def setUp(self) -> None:
        # Constructs the record from its defaults with no arguments, so the
        # cases below assert the auxiliary weight and logarithm floor the
        # shipped configuration actually applies. This family's class never
        # applies the auxiliary weight itself, so the record is the only
        # place that value can be checked.
        self._configuration: RfwaveLossConfig = RfwaveLossConfig()

    def test_default_settings_match_the_reference_recipe(self) -> None:
        # The auxiliary terms enter at a hundredth of the velocity term.
        self.assertEqual(self._configuration.auxiliary_weight, 0.01)
        self.assertEqual(self._configuration.log_epsilon, 1e-8)

    def test_configuration_rejects_mutation_after_construction(self) -> None:
        # The record is frozen so a run cannot drift from its logged settings.
        with self.assertRaises(ValidationError):
            self._configuration.auxiliary_weight: float = 1.0

    def test_configuration_rejects_an_unknown_setting(self) -> None:
        # The phase term is not implemented, so naming one must fail loudly.
        with self.assertRaises(ValidationError):
            RfwaveLossConfig(phase_weight=1.0)

    def test_configuration_rejects_a_non_positive_setting(self) -> None:
        # A zero floor would reintroduce the infinite-loss failure mode.
        with self.assertRaises(ValidationError):
            RfwaveLossConfig(log_epsilon=0.0)

    def test_configuration_property_returns_the_injected_record(self) -> None:
        # The criterion exposes the exact record it was constructed with.
        configuration: RfwaveLossConfig = RfwaveLossConfig(auxiliary_weight=0.5)
        loss: RfwaveLoss = RfwaveLoss(configuration)
        self.assertIs(loss.configuration, configuration)


class RfwaveTimeBalanceTest(unittest.TestCase):
    # Verifies the per-frame variance normalization applied before the
    # rectified-flow velocity comparison.
    def setUp(self) -> None:
        # The target is scaled up by four relative to the prediction, so the
        # two tensors have visibly different spreads and a rescaling that
        # used the wrong operand, or none at all, would change the result.
        # The seed fixes both draws, making the balanced values reproducible
        # rather than merely statistically similar.
        self._loss: RfwaveLoss = RfwaveLoss(RfwaveLossConfig())
        torch.manual_seed(701)
        self._prediction: torch.Tensor = torch.randn(2, 8, 5)
        self._target: torch.Tensor = torch.randn(2, 8, 5) * 4.0

    def test_time_balance_preserves_both_shapes(self) -> None:
        # Balancing is an elementwise rescaling, not a reduction.
        balanced_prediction: torch.Tensor
        balanced_target: torch.Tensor
        balanced_prediction, balanced_target = self._loss.time_balance(self._prediction, self._target)
        self.assertEqual(balanced_prediction.shape, self._prediction.shape)
        self.assertEqual(balanced_target.shape, self._target.shape)

    def test_balanced_target_reaches_unit_variance(self) -> None:
        # The target variance is the scale the balance divides out.
        balanced_target: torch.Tensor
        _, balanced_target = self._loss.time_balance(self._prediction, self._target)
        variance: torch.Tensor = balanced_target.var(dim=1)
        self.assertTrue(
            torch.allclose(variance, torch.ones_like(variance), atol=1e-4),
            msg="Balancing must normalize the target to unit per-frame variance"
        )

    def test_time_balance_is_invariant_to_a_common_rescaling(self) -> None:
        # Scaling prediction and target together leaves the balanced pair fixed.
        plain_prediction: torch.Tensor
        plain_target: torch.Tensor
        plain_prediction, plain_target = self._loss.time_balance(self._prediction, self._target)
        scaled_prediction: torch.Tensor
        scaled_target: torch.Tensor
        scaled_prediction, scaled_target = self._loss.time_balance(
            self._prediction * 5.0,
            self._target * 5.0
        )
        self.assertTrue(torch.allclose(plain_prediction, scaled_prediction, atol=1e-4))
        self.assertTrue(torch.allclose(plain_target, scaled_target, atol=1e-4))

    def test_time_balance_preserves_the_prediction_to_target_ratio(self) -> None:
        # Both tensors are divided by the same scale, so their ratio is untouched.
        balanced_prediction: torch.Tensor
        balanced_target: torch.Tensor
        balanced_prediction, balanced_target = self._loss.time_balance(self._prediction, self._target)
        plain_ratio: torch.Tensor = self._prediction / self._target
        balanced_ratio: torch.Tensor = balanced_prediction / balanced_target
        self.assertTrue(torch.allclose(plain_ratio, balanced_ratio, atol=1e-3))


class RfwaveVelocityLossTest(unittest.TestCase):
    # Verifies the wave-mode rectified-flow term over the velocity error.
    def setUp(self) -> None:
        # Needs only the objective: the velocity term is a plain mean square
        # over whatever waveform it is handed, so each case supplies its own
        # tensor and no shared fixture would be reused.
        self._loss: RfwaveLoss = RfwaveLoss(RfwaveLossConfig())

    def test_velocity_loss_is_zero_for_a_vanishing_error(self) -> None:
        # A matched velocity field is the analytic optimum.
        value: torch.Tensor = self._loss.waveform_velocity_loss(torch.zeros(2, 16))
        self.assertAlmostEqual(float(value.item()), 0.0, places=6)

    def test_velocity_loss_is_the_mean_square_of_the_error(self) -> None:
        # The term is a plain mean square over the placed waveform difference.
        error: torch.Tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        value: torch.Tensor = self._loss.waveform_velocity_loss(error)
        self.assertAlmostEqual(float(value.item()), 7.5, places=5)

    def test_velocity_loss_ignores_the_sign_of_the_error(self) -> None:
        # A squared term charges the magnitude, not the direction.
        positive: torch.Tensor = self._loss.waveform_velocity_loss(torch.full((2, 8), 0.5))
        negative: torch.Tensor = self._loss.waveform_velocity_loss(torch.full((2, 8), -0.5))
        self.assertAlmostEqual(float(positive.item()), float(negative.item()), places=6)

    def test_velocity_loss_propagates_gradient_to_the_error_waveform(self) -> None:
        # The term must train the predicted velocity field.
        error: torch.Tensor = torch.full((2, 8), 0.5, requires_grad=True)
        value: torch.Tensor = self._loss.waveform_velocity_loss(error)
        value.backward()
        self.assertIsNotNone(error.grad, msg="The velocity error must receive gradient")
        self.assertTrue(torch.isfinite(error.grad).all().item())


class RfwaveMagnitudeLossTest(unittest.TestCase):
    # Verifies the auxiliary magnitude term over the implied endpoints.
    def setUp(self) -> None:
        # The eight-channel axis is what the magnitude term splits into its
        # real and imaginary halves, so the fixture carries four frequency
        # positions per component. A single shared target lets each case vary
        # only the prediction, isolating the term's response.
        self._loss: RfwaveLoss = RfwaveLoss(RfwaveLossConfig())
        torch.manual_seed(711)
        self._target_spectrum: torch.Tensor = torch.randn(2, 8, 5)

    def test_magnitude_loss_is_zero_for_identical_spectra(self) -> None:
        # Both the logarithmic and the convergence term vanish at identity.
        value: torch.Tensor = self._loss.magnitude_loss(self._target_spectrum, self._target_spectrum)
        self.assertAlmostEqual(
            float(value.item()),
            0.0,
            places=6,
            msg="Matching implied endpoints must carry no magnitude cost"
        )

    def test_magnitude_loss_stays_finite_on_silent_spectra(self) -> None:
        # The logarithm floor is the only guard against a negative infinity.
        value: torch.Tensor = self._loss.magnitude_loss(torch.zeros(2, 8, 5), torch.zeros(2, 8, 5))
        self.assertTrue(torch.isfinite(value).item())
        self.assertAlmostEqual(float(value.item()), 0.0, places=6)

    def test_magnitude_loss_grows_with_the_magnitude_discrepancy(self) -> None:
        # A larger endpoint error is strictly more expensive.
        near: torch.Tensor = self._loss.magnitude_loss(
            self._target_spectrum * 1.1,
            self._target_spectrum
        )
        far: torch.Tensor = self._loss.magnitude_loss(
            self._target_spectrum * 3.0,
            self._target_spectrum
        )
        self.assertGreater(float(far.item()), float(near.item()))

    def test_magnitude_loss_returns_a_finite_non_negative_scalar(self) -> None:
        # The term reduces the spectrum pair to one number.
        torch.manual_seed(712)
        value: torch.Tensor = self._loss.magnitude_loss(torch.randn(2, 8, 5), self._target_spectrum)
        self.assertEqual(value.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(value).item())
        self.assertGreater(float(value.item()), 0.0)

    def test_magnitude_loss_requires_an_even_channel_split(self) -> None:
        # The channel axis carries stacked real and imaginary halves.
        with self.assertRaises(RuntimeError):
            self._loss.magnitude_loss(torch.randn(2, 7, 5), torch.randn(2, 7, 5))

    def test_magnitude_loss_propagates_gradient_to_the_prediction(self) -> None:
        # The auxiliary term must train the predicted endpoint.
        prediction: torch.Tensor = (self._target_spectrum * 1.5).requires_grad_(True)
        value: torch.Tensor = self._loss.magnitude_loss(prediction, self._target_spectrum)
        value.backward()
        self.assertIsNotNone(prediction.grad, msg="The predicted spectrum must receive gradient")
        self.assertTrue(torch.isfinite(prediction.grad).all().item())


class RfwaveOverlapLossTest(unittest.TestCase):
    # Verifies the band-adjacency consistency term inside the shared overlaps.
    def setUp(self) -> None:
        # A slab width of twelve with two-bin margins on each side leaves a
        # core of eight, so a band's overlap regions and its interior are
        # clearly separable and a comparison that read the neighbour's margin
        # instead of its core would land on different values. Equal left and
        # right widths keep the two directions symmetric, so an asymmetric
        # result would indicate a directional bug rather than a fixture
        # artifact.
        self._loss: RfwaveLoss = RfwaveLoss(RfwaveLossConfig())
        self._builder: BandPredictionBuilder = BandPredictionBuilder(slab_width=12, frame_count=3)
        self._left_overlap: int = 2
        self._right_overlap: int = 2

    def test_overlap_loss_is_zero_for_agreeing_bands(self) -> None:
        # Bands that agree inside their shared region are the analytic optimum.
        bands: list[torch.Tensor] = self._builder.constant_bands([0.3, 0.3, 0.3])
        value: torch.Tensor = self._loss.overlap_loss(
            bands,
            bands,
            self._left_overlap,
            self._right_overlap
        )
        self.assertAlmostEqual(float(value.item()), 0.0, places=6)

    def test_overlap_loss_charges_each_boundary_from_both_sides(self) -> None:
        # Two constant bands one apart are charged twice, then halved by the band count.
        real_bands: list[torch.Tensor] = self._builder.constant_bands([0.0, 1.0])
        imaginary_bands: list[torch.Tensor] = self._builder.constant_bands([0.0, 0.0])
        value: torch.Tensor = self._loss.overlap_loss(
            real_bands,
            imaginary_bands,
            self._left_overlap,
            self._right_overlap
        )
        self.assertAlmostEqual(float(value.item()), 1.0, places=5)

    def test_overlap_loss_accumulates_the_imaginary_bands_as_well(self) -> None:
        # The real and imaginary disagreements are summed before normalization.
        matched: list[torch.Tensor] = self._builder.constant_bands([0.0, 0.0])
        offset: list[torch.Tensor] = self._builder.constant_bands([0.0, 1.0])
        real_only: torch.Tensor = self._loss.overlap_loss(
            offset,
            matched,
            self._left_overlap,
            self._right_overlap
        )
        both: torch.Tensor = self._loss.overlap_loss(
            offset,
            offset,
            self._left_overlap,
            self._right_overlap
        )
        self.assertAlmostEqual(float(both.item()), 2.0 * float(real_only.item()), places=5)

    def test_overlap_loss_is_normalized_by_the_band_count(self) -> None:
        # Three bands contribute four charged boundaries over three bands.
        real_bands: list[torch.Tensor] = self._builder.constant_bands([0.0, 1.0, 0.0])
        imaginary_bands: list[torch.Tensor] = self._builder.constant_bands([0.0, 0.0, 0.0])
        value: torch.Tensor = self._loss.overlap_loss(
            real_bands,
            imaginary_bands,
            self._left_overlap,
            self._right_overlap
        )
        self.assertAlmostEqual(float(value.item()), 4.0 / 3.0, places=5)

    def test_overlap_loss_is_zero_when_the_bands_do_not_overlap(self) -> None:
        # With no shared region there is no adjacency left to penalize.
        real_bands: list[torch.Tensor] = self._builder.seeded_bands(seed=721, band_count=3)
        imaginary_bands: list[torch.Tensor] = self._builder.seeded_bands(seed=722, band_count=3)
        value: torch.Tensor = self._loss.overlap_loss(real_bands, imaginary_bands, 0, 0)
        self.assertAlmostEqual(float(value.item()), 0.0, places=6)

    def test_overlap_loss_returns_a_finite_non_negative_scalar(self) -> None:
        # The term reduces the band lists to one number.
        real_bands: list[torch.Tensor] = self._builder.seeded_bands(seed=731, band_count=3)
        imaginary_bands: list[torch.Tensor] = self._builder.seeded_bands(seed=732, band_count=3)
        value: torch.Tensor = self._loss.overlap_loss(
            real_bands,
            imaginary_bands,
            self._left_overlap,
            self._right_overlap
        )
        self.assertEqual(value.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(value).item())
        self.assertGreater(float(value.item()), 0.0)

    def test_overlap_loss_propagates_gradient_to_the_band_predictions(self) -> None:
        # The consistency term must train every band it compares.
        real_bands: list[torch.Tensor] = self._builder.constant_bands_requiring_gradient([0.0, 1.0])
        imaginary_bands: list[torch.Tensor] = self._builder.constant_bands([0.0, 0.0])
        value: torch.Tensor = self._loss.overlap_loss(
            real_bands,
            imaginary_bands,
            self._left_overlap,
            self._right_overlap
        )
        value.backward()
        band: torch.Tensor
        for band in real_bands:
            self.assertIsNotNone(band.grad, msg="Every band prediction must receive gradient")
            self.assertTrue(torch.isfinite(band.grad).all().item())
