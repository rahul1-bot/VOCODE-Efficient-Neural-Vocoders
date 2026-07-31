# This module:
# 1. Verifies the registry status vocabulary: the thirteen registered
#    architecture records, their readiness levels, capability flags, the
#    recorded HiFTNet blocker, and the unknown-name rejection of the
#    record query
# 2. Verifies canonical module construction routing for every registered
#    architecture, the recovery learning-rate scale, and the
#    published-weights support boundary between the nine architectures
#    with author-weight adapters and the four without
# 3. Verifies the frozen registry records themselves: implementation
#    record, build options, and built-module spec
#
# Design decisions:
# - Construction is exercised through the registry for all thirteen
#   architectures because every reference recipe builds on CPU in well
#   under a second; no forward pass runs here, since synthesis behavior
#   belongs to the per-network test modules
# - The weight-support boundary is asserted by constructing adapters and
#   reading their provenance records; no release is ever retrieved,
#   verified, or loaded, so the suite stays free of network access
# - Configuration values are read from the frozen configuration dump the
#   spec carries, because that dump is the record every evidence lane
#   consumes rather than the live configuration object
# - HiFTNet is built with the default options, which leave the F0
#   checkpoint path unset; the F0 loader returns early on an absent path,
#   so construction stays local and no author binary is touched
#
# Author: Rahul Sawhney

import unittest
from pathlib import Path

import torch
from pydantic import PositiveFloat, ValidationError

from syntheticmind.core.module import Module
from syntheticmind.utilities.exceptions import MisconfigurationError

from vocode.models.hifigan.hifigan import Hifigan, HifiganConfig
from vocode.models.melgan.melgan import Melgan, MelganConfig
from vocode.models.registry import (
    ArchitectureImplementationRecord,
    ArchitectureModuleSpec,
    ModelRegistry,
    ModuleBuildOptions,
)
from vocode.models.vocoder import ArchitectureName, PublishedWeights


class RegisteredArchitectureNames:
    # Names the study vocabulary registers, split into the sets the tests
    # assert against: the full registry order, the training-level cohort,
    # and the two sides of the author-weight support boundary.
    def __init__(self) -> None:
        # Records the vocabulary in registry declaration order alongside the
        # variant label and weight-support side expected for each name.
        # The expectations are transcribed here rather than derived from
        # the registry, so a change to the registry is reported as a test
        # failure instead of being mirrored into the assertions.
        self._all_names: tuple[str, ...] = (
            "hifigan_v1",
            "hifigan_v2",
            "hifigan_v3",
            "melgan",
            "vocos",
            "bigvgan",
            "apnet2",
            "freev",
            "hiftnet",
            "lpcnet",
            "rndvoc",
            "vocosformer",
            "rfwave"
        )
        self._weight_supported_names: tuple[str, ...] = (
            "hifigan_v1",
            "hifigan_v2",
            "hifigan_v3",
            "melgan",
            "vocos",
            "bigvgan",
            "apnet2",
            "freev",
            "hiftnet"
        )
        self._weight_unsupported_names: tuple[str, ...] = (
            "lpcnet",
            "rndvoc",
            "vocosformer",
            "rfwave"
        )
        self._variant_labels: dict[str, str] = {
            "hifigan_v1": "v1",
            "hifigan_v2": "v2",
            "hifigan_v3": "v3",
            "melgan": "seungwon",
            "vocos": "charactr_mel_24khz",
            "bigvgan": "base_24khz_100band",
            "apnet2": "redmist328_ljspeech",
            "freev": "official_ljspeech",
            "hiftnet": "yl4579_ljspeech",
            "lpcnet": "xiph_reference_ljspeech",
            "rndvoc": "andong_22k_ljspeech",
            "vocosformer": "matched_vocos_24khz_ljspeech",
            "rfwave": "bfs18_24khz_ljspeech"
        }

    @property
    def all_names(self) -> tuple[str, ...]:
        # Returns the whole vocabulary in registry declaration order.
        return self._all_names

    @property
    def variant_labels(self) -> dict[str, str]:
        # Returns a copy of the name-to-reference-recipe label mapping.
        return dict(self._variant_labels)

    @property
    def weight_supported_names(self) -> tuple[str, ...]:
        # Returns the architectures that carry an author-weight adapter.
        return self._weight_supported_names

    @property
    def weight_unsupported_names(self) -> tuple[str, ...]:
        # Returns the architectures without a validated author release.
        return self._weight_unsupported_names


class ArchitectureImplementationRecordValidationTest(unittest.TestCase):
    # Verifies the frozen implementation-status record accepts the
    # registered field set and rejects unknown names, unknown readiness
    # levels, extra fields, loose types, and mutation.
    def setUp(self) -> None:
        # Builds the accepted payload the rejection cases mutate one field at a time.
        self._valid_fields: dict[str, str | bool | None] = {
            "architecture_name": "melgan",
            "status": "training_level_ready",
            "local_network_available": True,
            "training_step_available": True,
            "validation_step_available": True,
            "test_step_available": True,
            "blocker": None
        }

    def test_record_accepts_the_registered_field_set(self) -> None:
        # A complete valid payload constructs and preserves every field.
        record: ArchitectureImplementationRecord = ArchitectureImplementationRecord(**self._valid_fields)
        self.assertEqual(record.architecture_name, "melgan")
        self.assertEqual(record.status, "training_level_ready")
        self.assertTrue(record.local_network_available)
        self.assertIsNone(record.blocker)

    def test_record_rejects_an_unregistered_architecture_name(self) -> None:
        # The closed architecture vocabulary refuses a name outside it.
        invalid_fields: dict[str, str | bool | None] = dict(self._valid_fields)
        invalid_fields["architecture_name"] = "griffinlim"
        with self.assertRaises(ValidationError):
            ArchitectureImplementationRecord(**invalid_fields)

    def test_record_rejects_an_unknown_status_level(self) -> None:
        # Readiness levels are a closed literal set.
        invalid_fields: dict[str, str | bool | None] = dict(self._valid_fields)
        invalid_fields["status"] = "almost_ready"
        with self.assertRaises(ValidationError):
            ArchitectureImplementationRecord(**invalid_fields)

    def test_record_rejects_an_extra_field(self) -> None:
        # extra="forbid" blocks undeclared payload keys.
        invalid_fields: dict[str, str | bool | None] = dict(self._valid_fields)
        invalid_fields["gpu_hours"] = "48"
        with self.assertRaises(ValidationError):
            ArchitectureImplementationRecord(**invalid_fields)

    def test_record_rejects_a_non_boolean_capability_flag(self) -> None:
        # strict=True refuses an integer where a capability flag is declared.
        invalid_fields: dict[str, str | bool | int | None] = dict(self._valid_fields)
        invalid_fields["training_step_available"] = 1
        with self.assertRaises(ValidationError):
            ArchitectureImplementationRecord(**invalid_fields)

    def test_record_rejects_mutation_after_construction(self) -> None:
        # frozen=True makes the record a value object.
        record: ArchitectureImplementationRecord = ArchitectureImplementationRecord(**self._valid_fields)
        with self.assertRaises(ValidationError):
            record.blocker: str | None = "blocked after construction"


class ModuleBuildOptionsValidationTest(unittest.TestCase):
    # Verifies the two construction-time degrees of freedom the registry
    # admits: their defaults, their type strictness, and their immutability.
    def test_default_options_leave_the_f0_path_unset_at_unit_scale(self) -> None:
        # Defaults must not alter any reference recipe.
        options: ModuleBuildOptions = ModuleBuildOptions()
        self.assertIsNone(options.hiftnet_f0_checkpoint_path)
        self.assertEqual(options.learning_rate_scale, 1.0)

    def test_options_accept_an_explicit_f0_checkpoint_path(self) -> None:
        # The HiFTNet F0 checkpoint path is carried as a pathlib path.
        checkpoint_path: Path = Path("weights/hiftnet/f0.pth")
        options: ModuleBuildOptions = ModuleBuildOptions(hiftnet_f0_checkpoint_path=checkpoint_path)
        self.assertEqual(options.hiftnet_f0_checkpoint_path, checkpoint_path)

    def test_options_reject_a_string_checkpoint_path(self) -> None:
        # strict=True refuses a string where a path is declared.
        with self.assertRaises(ValidationError):
            ModuleBuildOptions(hiftnet_f0_checkpoint_path="weights/hiftnet/f0.pth")

    def test_options_reject_a_zero_learning_rate_scale(self) -> None:
        # A zero scale would silently freeze training and is refused.
        with self.assertRaises(ValidationError):
            ModuleBuildOptions(learning_rate_scale=0.0)

    def test_options_reject_a_negative_learning_rate_scale(self) -> None:
        # A negative scale would invert the gradient step and is refused.
        with self.assertRaises(ValidationError):
            ModuleBuildOptions(learning_rate_scale=-0.5)

    def test_options_reject_mutation_after_construction(self) -> None:
        # frozen=True keeps a build request immutable once issued.
        options: ModuleBuildOptions = ModuleBuildOptions()
        with self.assertRaises(ValidationError):
            options.learning_rate_scale: PositiveFloat = 0.5


class ArchitectureModuleSpecValidationTest(unittest.TestCase):
    # Verifies the built-module spec binds an architecture name, variant
    # name, harness module, and configuration dump under strict validation.
    def setUp(self) -> None:
        # Builds one seeded MelGAN module to bind into the spec under test.
        torch.manual_seed(0)
        self._configuration: MelganConfig = MelganConfig.seungwon()
        self._module: Melgan = Melgan(self._configuration)

    def test_spec_binds_the_module_and_its_configuration_dump(self) -> None:
        # A valid spec carries the live module and its serialized configuration.
        spec: ArchitectureModuleSpec = ArchitectureModuleSpec(
            architecture_name="melgan",
            variant_name="seungwon",
            module=self._module,
            configuration_dump=self._configuration.model_dump(mode="json")
        )
        self.assertIs(spec.module, self._module)
        self.assertEqual(spec.variant_name, "seungwon")
        self.assertEqual(spec.configuration_dump["input_mel_channels"], 80)

    def test_spec_rejects_a_non_module_payload(self) -> None:
        # The module field must hold a harness module, not an arbitrary object.
        with self.assertRaises(ValidationError):
            ArchitectureModuleSpec(
                architecture_name="melgan",
                variant_name="seungwon",
                module="melgan-module",
                configuration_dump={}
            )

    def test_spec_rejects_an_extra_field(self) -> None:
        # extra="forbid" blocks undeclared payload keys.
        with self.assertRaises(ValidationError):
            ArchitectureModuleSpec(
                architecture_name="melgan",
                variant_name="seungwon",
                module=self._module,
                configuration_dump={},
                checkpoint_path="run/last.ckpt"
            )

    def test_spec_rejects_mutation_after_construction(self) -> None:
        # frozen=True prevents rebinding a built spec to another variant.
        spec: ArchitectureModuleSpec = ArchitectureModuleSpec(
            architecture_name="melgan",
            variant_name="seungwon",
            module=self._module,
            configuration_dump={}
        )
        with self.assertRaises(ValidationError):
            spec.variant_name: str = "other"


class RegistryRecordVocabularyTest(unittest.TestCase):
    # Verifies exact registry membership, the readiness split between the
    # training-level cohort and verification-only HiFTNet, and the record
    # query behavior on registered and unregistered names.
    def setUp(self) -> None:
        # Binds the registry under test against the expected name vocabulary.
        self._registry: ModelRegistry = ModelRegistry()
        self._expected_names: RegisteredArchitectureNames = RegisteredArchitectureNames()

    def test_registry_publishes_thirteen_records(self) -> None:
        # The study vocabulary is thirteen architectures wide.
        self.assertEqual(len(self._registry.all_records()), 13)

    def test_registered_names_match_the_study_vocabulary_in_order(self) -> None:
        # Membership and declaration order are both part of the record.
        registered_names: tuple[str, ...] = tuple(
            record.architecture_name for record in self._registry.all_records()
        )
        self.assertEqual(registered_names, self._expected_names.all_names)

    def test_each_architecture_is_registered_exactly_once(self) -> None:
        # Duplicate entries would make the record query order-dependent.
        registered_names: tuple[str, ...] = tuple(
            record.architecture_name for record in self._registry.all_records()
        )
        self.assertEqual(len(set(registered_names)), len(registered_names))

    def test_hiftnet_is_the_only_verification_only_entry(self) -> None:
        # HiFTNet is the single architecture excluded from Checkpoint 01.
        verification_only_names: list[str] = [
            record.architecture_name
            for record in self._registry.all_records()
            if record.status == "verification_only"
        ]
        self.assertEqual(verification_only_names, ["hiftnet"])

    def test_the_verification_only_entry_records_its_blocker(self) -> None:
        # The exclusion reason travels with the record rather than the prose.
        hiftnet_record: ArchitectureImplementationRecord = self._registry.get("hiftnet")
        self.assertIsNotNone(hiftnet_record.blocker)
        self.assertIn("Checkpoint 01", str(hiftnet_record.blocker))

    def test_training_level_records_carry_no_blocker(self) -> None:
        # A readiness claim and a recorded blocker are mutually exclusive.
        record: ArchitectureImplementationRecord
        for record in self._registry.all_records():
            if record.status == "training_level_ready":
                self.assertIsNone(
                    record.blocker,
                    msg=f"{record.architecture_name} claims training readiness while recording a blocker"
                )

    def test_every_record_reports_all_four_capabilities_available(self) -> None:
        # Every registered architecture implements the full harness surface.
        record: ArchitectureImplementationRecord
        for record in self._registry.all_records():
            self.assertTrue(record.local_network_available, msg=f"{record.architecture_name} network missing")
            self.assertTrue(record.training_step_available, msg=f"{record.architecture_name} training step missing")
            self.assertTrue(record.validation_step_available, msg=f"{record.architecture_name} validation step missing")
            self.assertTrue(record.test_step_available, msg=f"{record.architecture_name} test step missing")

    def test_training_level_architectures_are_the_twelve_executed_names(self) -> None:
        # The training-level cohort is the vocabulary minus HiFTNet.
        training_level_names: tuple[ArchitectureName, ...] = self._registry.training_level_architectures()
        expected_names: tuple[str, ...] = tuple(
            name for name in self._expected_names.all_names if name != "hiftnet"
        )
        self.assertEqual(tuple(training_level_names), expected_names)
        self.assertEqual(len(training_level_names), 12)

    def test_record_query_returns_the_entry_for_a_registered_name(self) -> None:
        # The query resolves each registered name to its own record.
        expected_name: str
        for expected_name in self._expected_names.all_names:
            record: ArchitectureImplementationRecord = self._registry.get(expected_name)
            self.assertEqual(record.architecture_name, expected_name)

    def test_record_query_rejects_an_unregistered_name(self) -> None:
        # An unknown name fails loudly instead of returning a default.
        with self.assertRaises(KeyError):
            self._registry.get("griffinlim")

    def test_record_query_names_the_rejected_architecture(self) -> None:
        # The failure text carries the offending name for debugging.
        with self.assertRaisesRegex(KeyError, "griffinlim"):
            self._registry.get("griffinlim")


class RegistryModuleConstructionTest(unittest.TestCase):
    # Verifies the registry builds a canonical module for every registered
    # architecture, routes each name to its own reference recipe and
    # variant label, and rejects unsupported names.
    def setUp(self) -> None:
        # Seeds construction and binds the registry, default options, and expected names.
        torch.manual_seed(0)
        self._registry: ModelRegistry = ModelRegistry()
        self._build_options: ModuleBuildOptions = ModuleBuildOptions()
        self._expected_names: RegisteredArchitectureNames = RegisteredArchitectureNames()

    def test_every_registered_architecture_builds_its_canonical_spec(self) -> None:
        # Construction routing covers the whole vocabulary, and each name
        # reaches its own reference recipe under its own variant label.
        expected_variant_names: dict[str, str] = self._expected_names.variant_labels
        self.assertEqual(set(expected_variant_names.keys()), set(self._expected_names.all_names))
        architecture_name: str
        expected_variant_name: str
        for architecture_name, expected_variant_name in expected_variant_names.items():
            spec: ArchitectureModuleSpec = self._registry.build_module_spec(
                architecture_name,
                self._build_options
            )
            self.assertEqual(spec.architecture_name, architecture_name)
            self.assertEqual(spec.variant_name, expected_variant_name)
            self.assertIsInstance(spec.module, Module, msg=f"{architecture_name} did not build a harness module")
            self.assertTrue(spec.configuration_dump, msg=f"{architecture_name} built without a configuration dump")

    def test_hifigan_names_route_to_the_three_reference_widths(self) -> None:
        # The three HiFi-GAN entries differ by generator width and resblock kind.
        expected_initial_channels: dict[str, int] = {
            "hifigan_v1": 512,
            "hifigan_v2": 128,
            "hifigan_v3": 256
        }
        architecture_name: str
        expected_channels: int
        for architecture_name, expected_channels in expected_initial_channels.items():
            spec: ArchitectureModuleSpec = self._registry.build_module_spec(
                architecture_name,
                self._build_options
            )
            self.assertIsInstance(spec.module, Hifigan)
            self.assertEqual(spec.configuration_dump["upsample_initial_channels"], expected_channels)

    def test_melgan_routes_to_the_seungwon_reference_recipe(self) -> None:
        # The MelGAN entry builds the reference generator width and factors.
        spec: ArchitectureModuleSpec = self._registry.build_module_spec("melgan", self._build_options)
        self.assertIsInstance(spec.module, Melgan)
        self.assertEqual(spec.configuration_dump["ngf"], 32)
        self.assertEqual(spec.configuration_dump["upsample_factors"], [8, 8, 2, 2])

    def test_unsupported_architecture_name_is_rejected(self) -> None:
        # Construction fails loudly rather than falling back to a default recipe.
        with self.assertRaises(MisconfigurationError):
            self._registry.build_module_spec("griffinlim", self._build_options)

    def test_construction_rejection_names_the_offending_architecture(self) -> None:
        # The failure text carries the offending name for debugging.
        with self.assertRaisesRegex(MisconfigurationError, "griffinlim"):
            self._registry.build_module_spec("griffinlim", self._build_options)

    def test_repeated_builds_return_independent_modules(self) -> None:
        # Each evidence lane must own its module rather than share one instance.
        first_spec: ArchitectureModuleSpec = self._registry.build_module_spec("melgan", self._build_options)
        second_spec: ArchitectureModuleSpec = self._registry.build_module_spec("melgan", self._build_options)
        self.assertIsNot(first_spec.module, second_spec.module)
        self.assertEqual(first_spec.configuration_dump, second_spec.configuration_dump)


class RegistryLearningRateScaleTest(unittest.TestCase):
    # Verifies the recovery learning-rate scale reaches the built module
    # through the fleet-uniform learning_rate field and touches nothing else.
    def setUp(self) -> None:
        # Seeds construction and binds the registry the scaled builds run through.
        torch.manual_seed(0)
        self._registry: ModelRegistry = ModelRegistry()

    def test_unit_scale_preserves_the_reference_learning_rate(self) -> None:
        # The default build must reproduce the published rate exactly.
        spec: ArchitectureModuleSpec = self._registry.build_module_spec(
            "hifigan_v1",
            ModuleBuildOptions()
        )
        self.assertEqual(spec.configuration_dump["learning_rate"], HifiganConfig.v1().learning_rate)

    def test_recovery_scale_reduces_the_hifigan_learning_rate(self) -> None:
        # Recovery fine-tuning obtains its reduced rate through the scale.
        spec: ArchitectureModuleSpec = self._registry.build_module_spec(
            "hifigan_v1",
            ModuleBuildOptions(learning_rate_scale=0.5)
        )
        expected_rate: float = HifiganConfig.v1().learning_rate * 0.5
        self.assertAlmostEqual(float(spec.configuration_dump["learning_rate"]), expected_rate, places=12)

    def test_recovery_scale_reduces_the_melgan_learning_rate(self) -> None:
        # The scale is family-uniform rather than a per-architecture code path.
        spec: ArchitectureModuleSpec = self._registry.build_module_spec(
            "melgan",
            ModuleBuildOptions(learning_rate_scale=0.25)
        )
        expected_rate: float = MelganConfig.seungwon().learning_rate * 0.25
        self.assertAlmostEqual(float(spec.configuration_dump["learning_rate"]), expected_rate, places=12)

    def test_scaling_leaves_every_other_configuration_field_untouched(self) -> None:
        # Only the learning rate may differ between a scaled and unscaled build.
        unscaled_dump: dict[str, object] = self._registry.build_module_spec(
            "melgan",
            ModuleBuildOptions()
        ).configuration_dump
        scaled_dump: dict[str, object] = self._registry.build_module_spec(
            "melgan",
            ModuleBuildOptions(learning_rate_scale=0.5)
        ).configuration_dump
        self.assertEqual(set(unscaled_dump.keys()), set(scaled_dump.keys()))
        differing_keys: list[str] = [
            field_name
            for field_name in unscaled_dump
            if unscaled_dump[field_name] != scaled_dump[field_name]
        ]
        self.assertEqual(differing_keys, ["learning_rate"])

    def test_the_generator_optimizer_receives_the_scaled_rate(self) -> None:
        # The scaled configuration must reach the live optimizer declaration.
        spec: ArchitectureModuleSpec = self._registry.build_module_spec(
            "melgan",
            ModuleBuildOptions(learning_rate_scale=0.5)
        )
        module: Melgan = spec.module
        expected_rate: float = MelganConfig.seungwon().learning_rate * 0.5
        self.assertAlmostEqual(module.configuration.learning_rate, expected_rate, places=12)


class RegistryPublishedWeightsSupportTest(unittest.TestCase):
    # Verifies the author-weight support boundary: nine architectures
    # construct a published-weights adapter carrying matching provenance,
    # and the four without a validated release fail loudly.
    def setUp(self) -> None:
        # Binds the registry against the two sides of the weight-support boundary.
        self._registry: ModelRegistry = ModelRegistry()
        self._expected_names: RegisteredArchitectureNames = RegisteredArchitectureNames()

    def test_nine_architectures_expose_an_author_weight_adapter(self) -> None:
        # The supported side of the boundary is exactly nine architectures.
        self.assertEqual(len(self._expected_names.weight_supported_names), 9)
        architecture_name: str
        for architecture_name in self._expected_names.weight_supported_names:
            adapter: PublishedWeights = self._registry.build_published_weights(architecture_name)
            self.assertTrue(
                callable(adapter.load),
                msg=f"{architecture_name} adapter does not expose a load surface"
            )

    def test_adapter_provenance_matches_the_requested_architecture(self) -> None:
        # An adapter must never be wired to another family's release.
        architecture_name: str
        for architecture_name in self._expected_names.weight_supported_names:
            adapter: PublishedWeights = self._registry.build_published_weights(architecture_name)
            self.assertEqual(adapter.provenance.architecture_name, architecture_name)

    def test_adapter_provenance_records_a_sha256_and_local_path(self) -> None:
        # Every reference anchor is traceable to exact release bytes.
        architecture_name: str
        for architecture_name in self._expected_names.weight_supported_names:
            adapter: PublishedWeights = self._registry.build_published_weights(architecture_name)
            self.assertEqual(
                len(adapter.provenance.expected_sha256),
                64,
                msg=f"{architecture_name} provenance does not record a full SHA-256"
            )
            self.assertIsInstance(adapter.provenance.local_relative_path, Path)

    def test_architectures_without_a_validated_release_fail_fast(self) -> None:
        # The four unsupported architectures must not silently return an adapter.
        self.assertEqual(len(self._expected_names.weight_unsupported_names), 4)
        architecture_name: str
        for architecture_name in self._expected_names.weight_unsupported_names:
            with self.assertRaises(MisconfigurationError, msg=f"{architecture_name} returned an adapter"):
                self._registry.build_published_weights(architecture_name)

    def test_unsupported_weight_rejection_directs_to_project_checkpoints(self) -> None:
        # The failure text explains the supported alternative.
        with self.assertRaisesRegex(MisconfigurationError, "no model-local author-weight loader"):
            self._registry.build_published_weights("rfwave")

    def test_unknown_architecture_name_has_no_weight_adapter(self) -> None:
        # A name outside the vocabulary falls into the same rejection path.
        with self.assertRaises(MisconfigurationError):
            self._registry.build_published_weights("griffinlim")


if __name__ == "__main__":
    unittest.main()
