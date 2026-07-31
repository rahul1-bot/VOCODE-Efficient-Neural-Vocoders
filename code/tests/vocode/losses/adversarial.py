# This module:
# 1. Verifies the least-squares GAN objective shared by the adversarial
#    vocoder families: the generator term pulling fake logits toward one and
#    the discriminator term separating real from fake
# 2. Verifies that both terms sum over every ensemble member, return a finite
#    scalar, and carry gradient back into the logits that require it
# 3. Verifies the empty-ensemble boundary, where both terms return a zero
#    scalar instead of indexing an absent first member
#
# Design decisions:
# - The anchors are the analytic optima of the least-squares form rather than
#   fitted numbers: all-one fake logits zero the generator term, and one/zero
#   real-fake separation zeros the discriminator term
# - Logits are minimal constant tensors because the objective is a pure reduction
#   with no shape-dependent behavior; seeded random logits are used only where
#   the assertion is finiteness rather than an exact value
# - Ensemble scaling is asserted by comparing a one-member call against a
#   repeated-member call, which isolates the summation from the per-member value
#
# Author: Rahul Sawhney

import unittest

import torch

from vocode.losses.adversarial import LeastSquaresGanLoss


class LogitEnsembleBuilder:
    # Builds the constant and seeded logit ensembles the assertions consume.
    def __init__(self, batch_size: int, logit_count: int) -> None:
        # Binds the constructor inputs into this component's state.
        self._batch_size: int = batch_size
        self._logit_count: int = logit_count

    def constant(self, value: float, member_count: int) -> list[torch.Tensor]:
        # Builds one constant logit tensor per ensemble member.
        ensemble: list[torch.Tensor] = [
            torch.full((self._batch_size, self._logit_count), value)
            for member_index in range(member_count)
        ]
        return ensemble

    def seeded(self, seed: int, member_count: int) -> list[torch.Tensor]:
        # Builds one seeded random logit tensor per ensemble member.
        torch.manual_seed(seed)
        ensemble: list[torch.Tensor] = [
            torch.randn(self._batch_size, self._logit_count)
            for member_index in range(member_count)
        ]
        return ensemble

    def constant_requiring_gradient(self, value: float, member_count: int) -> list[torch.Tensor]:
        # Builds a constant ensemble whose members participate in autograd.
        ensemble: list[torch.Tensor] = [
            torch.full((self._batch_size, self._logit_count), value, requires_grad=True)
            for member_index in range(member_count)
        ]
        return ensemble


class LeastSquaresGeneratorObjectiveTest(unittest.TestCase):
    # Verifies the generator-side least-squares term, its optimum, its
    # ensemble summation, and its gradient path.
    def setUp(self) -> None:
        # Two ensemble members is the smallest count that distinguishes a sum
        # over sub-discriminators from a mean, which is what the summation
        # cases turn on; a one-member ensemble would satisfy both readings.
        # Four logits per tensor gives each inner mean a denominator greater
        # than one, so a reduction that forgot to average would not
        # coincidentally agree with one that did.
        self._loss: LeastSquaresGanLoss = LeastSquaresGanLoss()
        self._builder: LogitEnsembleBuilder = LogitEnsembleBuilder(batch_size=2, logit_count=4)

    def test_generator_loss_is_zero_when_every_fake_logit_equals_one(self) -> None:
        # A fully fooled discriminator is the analytic optimum of the term.
        fake_logits: list[torch.Tensor] = self._builder.constant(1.0, member_count=3)
        value: torch.Tensor = self._loss.generator_loss(fake_logits)
        self.assertAlmostEqual(
            float(value.item()),
            0.0,
            places=6,
            msg="All-one fake logits must zero the least-squares generator term"
        )

    def test_generator_loss_matches_the_closed_form_for_zero_logits(self) -> None:
        # Each zero-valued member contributes exactly (1 - 0) ** 2.
        fake_logits: list[torch.Tensor] = self._builder.constant(0.0, member_count=1)
        value: torch.Tensor = self._loss.generator_loss(fake_logits)
        self.assertAlmostEqual(float(value.item()), 1.0, places=6)

    def test_generator_loss_sums_over_ensemble_members(self) -> None:
        # Three identical members must cost exactly three times one member.
        single: torch.Tensor = self._loss.generator_loss(self._builder.constant(0.0, member_count=1))
        tripled: torch.Tensor = self._loss.generator_loss(self._builder.constant(0.0, member_count=3))
        self.assertAlmostEqual(
            float(tripled.item()),
            3.0 * float(single.item()),
            places=6,
            msg="The generator term must sum, not average, across sub-discriminators"
        )

    def test_generator_loss_decreases_as_fake_logits_approach_one(self) -> None:
        # The term is strictly decreasing in the logit value below one.
        far: torch.Tensor = self._loss.generator_loss(self._builder.constant(0.0, member_count=1))
        near: torch.Tensor = self._loss.generator_loss(self._builder.constant(0.5, member_count=1))
        optimal: torch.Tensor = self._loss.generator_loss(self._builder.constant(1.0, member_count=1))
        self.assertGreater(float(far.item()), float(near.item()))
        self.assertGreater(float(near.item()), float(optimal.item()))

    def test_generator_loss_returns_a_finite_scalar_for_seeded_logits(self) -> None:
        # The reduction collapses any ensemble to a finite zero-dimensional tensor.
        fake_logits: list[torch.Tensor] = self._builder.seeded(seed=17, member_count=4)
        value: torch.Tensor = self._loss.generator_loss(fake_logits)
        self.assertEqual(value.shape, torch.Size([]), msg="The objective must reduce to a scalar")
        self.assertTrue(torch.isfinite(value).item())
        self.assertGreaterEqual(float(value.item()), 0.0)

    def test_generator_loss_is_zero_for_an_empty_ensemble(self) -> None:
        # The empty-ensemble guard returns a zero scalar instead of indexing.
        empty_logits: list[torch.Tensor] = []
        value: torch.Tensor = self._loss.generator_loss(empty_logits)
        self.assertEqual(value.shape, torch.Size([]))
        self.assertAlmostEqual(float(value.item()), 0.0, places=6)

    def test_generator_loss_propagates_gradient_to_the_fake_logits(self) -> None:
        # Backpropagation reaches every member of the ensemble.
        fake_logits: list[torch.Tensor] = self._builder.constant_requiring_gradient(0.0, member_count=2)
        value: torch.Tensor = self._loss.generator_loss(fake_logits)
        value.backward()
        member: torch.Tensor
        for member in fake_logits:
            self.assertIsNotNone(member.grad, msg="Every sub-discriminator logit must receive gradient")
            self.assertTrue(torch.isfinite(member.grad).all().item())


class LeastSquaresDiscriminatorObjectiveTest(unittest.TestCase):
    # Verifies the discriminator-side least-squares term, its optimum, its
    # ensemble summation, and its gradient path.
    def setUp(self) -> None:
        # Two ensemble members is the smallest count that distinguishes a sum
        # over sub-discriminators from a mean, which is what the summation
        # cases turn on; a one-member ensemble would satisfy both readings.
        # Four logits per tensor gives each inner mean a denominator greater
        # than one, so a reduction that forgot to average would not
        # coincidentally agree with one that did.
        self._loss: LeastSquaresGanLoss = LeastSquaresGanLoss()
        self._builder: LogitEnsembleBuilder = LogitEnsembleBuilder(batch_size=2, logit_count=4)

    def test_discriminator_loss_is_zero_for_a_perfect_separation(self) -> None:
        # Real logits at one and fake logits at zero are the analytic optimum.
        real_logits: list[torch.Tensor] = self._builder.constant(1.0, member_count=3)
        fake_logits: list[torch.Tensor] = self._builder.constant(0.0, member_count=3)
        value: torch.Tensor = self._loss.discriminator_loss(real_logits, fake_logits)
        self.assertAlmostEqual(
            float(value.item()),
            0.0,
            places=6,
            msg="A perfectly separating discriminator must pay nothing"
        )

    def test_discriminator_loss_matches_the_closed_form_for_inverted_logits(self) -> None:
        # A fully inverted member costs (1 - 0) ** 2 plus 1 ** 2.
        real_logits: list[torch.Tensor] = self._builder.constant(0.0, member_count=1)
        fake_logits: list[torch.Tensor] = self._builder.constant(1.0, member_count=1)
        value: torch.Tensor = self._loss.discriminator_loss(real_logits, fake_logits)
        self.assertAlmostEqual(float(value.item()), 2.0, places=6)

    def test_discriminator_loss_sums_over_ensemble_members(self) -> None:
        # Two identical members must cost exactly twice one member.
        single: torch.Tensor = self._loss.discriminator_loss(
            self._builder.constant(0.0, member_count=1),
            self._builder.constant(1.0, member_count=1)
        )
        doubled: torch.Tensor = self._loss.discriminator_loss(
            self._builder.constant(0.0, member_count=2),
            self._builder.constant(1.0, member_count=2)
        )
        self.assertAlmostEqual(float(doubled.item()), 2.0 * float(single.item()), places=6)

    def test_discriminator_loss_returns_a_finite_scalar_for_seeded_logits(self) -> None:
        # The reduction collapses any ensemble to a finite zero-dimensional tensor.
        real_logits: list[torch.Tensor] = self._builder.seeded(seed=23, member_count=3)
        fake_logits: list[torch.Tensor] = self._builder.seeded(seed=29, member_count=3)
        value: torch.Tensor = self._loss.discriminator_loss(real_logits, fake_logits)
        self.assertEqual(value.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(value).item())
        self.assertGreaterEqual(float(value.item()), 0.0)

    def test_discriminator_loss_is_zero_for_an_empty_ensemble(self) -> None:
        # The empty-ensemble guard returns a zero scalar instead of indexing.
        empty_real: list[torch.Tensor] = []
        empty_fake: list[torch.Tensor] = []
        value: torch.Tensor = self._loss.discriminator_loss(empty_real, empty_fake)
        self.assertEqual(value.shape, torch.Size([]))
        self.assertAlmostEqual(float(value.item()), 0.0, places=6)

    def test_discriminator_loss_propagates_gradient_to_both_logit_sets(self) -> None:
        # Backpropagation reaches the real and the fake side of the objective.
        real_logits: list[torch.Tensor] = self._builder.constant_requiring_gradient(0.0, member_count=1)
        fake_logits: list[torch.Tensor] = self._builder.constant_requiring_gradient(1.0, member_count=1)
        value: torch.Tensor = self._loss.discriminator_loss(real_logits, fake_logits)
        value.backward()
        self.assertIsNotNone(real_logits[0].grad, msg="Real logits must receive gradient")
        self.assertIsNotNone(fake_logits[0].grad, msg="Fake logits must receive gradient")
        self.assertTrue(torch.isfinite(real_logits[0].grad).all().item())
        self.assertTrue(torch.isfinite(fake_logits[0].grad).all().item())
