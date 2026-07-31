# This module:
# 1. Registers the implementation status of every architecture in the
#    study vocabulary (training-level readiness, capability flags, and
#    recorded blockers) and exposes the status queries runners gate on
# 2. Constructs the canonical module for each architecture with its
#    authentic reference configuration, and constructs the registered
#    published-weights adapter where an author release exists
#
# Design decisions:
# - The registry is the single construction source for every evidence
#   lane, so Study 1 training, Study 2 interventions, and published
#   evaluations all measure identically built modules
# - Each architecture is built from its named reference configuration
#   factory; construction options are limited to the HiFTNet F0
#   checkpoint path and the recovery learning-rate scale, keeping the
#   build surface too narrow to smuggle in recipe drift
# - The learning-rate scale applies through the fleet-uniform
#   learning_rate field, which is how recovery fine-tuning obtains its
#   reduced rate without a per-family code path
#
# Author: Rahul Sawhney

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, PositiveFloat

from syntheticmind.core.module import Module
from syntheticmind.utilities.exceptions import MisconfigurationError

from vocode.models.apnet2.apnet2 import Apnet2, Apnet2Config
from vocode.models.apnet2.weights import Apnet2Weights
from vocode.models.bigvgan.bigvgan import Bigvgan, BigvganConfig
from vocode.models.bigvgan.weights import BigvganWeights
from vocode.models.freev.freev import Freev, FreevConfig
from vocode.models.freev.weights import FreevWeights
from vocode.models.hifigan.hifigan import Hifigan, HifiganConfig
from vocode.models.hifigan.weights import HifiganWeights
from vocode.models.hiftnet.hiftnet import Hiftnet, HiftnetConfig
from vocode.models.hiftnet.weights import HiftnetWeights
from vocode.models.lpcnet.lpcnet import Lpcnet, LpcnetConfig
from vocode.models.melgan.melgan import Melgan, MelganConfig
from vocode.models.melgan.weights import MelganWeights
from vocode.models.rfwave.rfwave import Rfwave, RfwaveConfig
from vocode.models.rndvoc.rndvoc import Rndvoc, RndvocConfig
from vocode.models.vocoder import ArchitectureName, PublishedWeights
from vocode.models.vocos.vocos import Vocos, VocosConfig
from vocode.models.vocos.weights import VocosWeights
from vocode.models.vocosformer.vocosformer import Vocosformer, VocosformerConfig

__all__: list[str] = [
    "ArchitectureImplementationRecord",
    "ArchitectureModuleSpec",
    "ModelRegistry",
    "ModuleBuildOptions"
]


# Closed readiness vocabulary of the registry, ordered from full
# reproduction readiness down to a declared-but-unexecutable family.
type ImplementationStatus = Literal[
    "training_level_ready",
    "verification_only",
    "scaffold_only"
]


class ArchitectureImplementationRecord(BaseModel):
    # Frozen implementation-status record for one architecture: readiness
    # level, per-capability flags, and the recorded blocker when one
    # exists.
    #
    # Fields:
    #     architecture_name: The architecture this record describes, drawn
    #         from the closed study vocabulary.
    #     status: Readiness level gating which evidence lanes may consume
    #         the architecture. ``"training_level_ready"`` admits it as a
    #         Project-Trained Configuration; ``"verification_only"`` means
    #         the implementation runs but produced no admissible Retained
    #         Project Checkpoint and result package, which makes the
    #         architecture a documented exclusion rather than a result
    #         row; ``"scaffold_only"`` is declared by the vocabulary and
    #         carried by no current record.
    #     local_network_available: Whether the synthesis network is
    #         implemented inside this repository rather than delegated to
    #         an external dependency.
    #     training_step_available: Whether the module implements the
    #         harness training step.
    #     validation_step_available: Whether the module implements the
    #         harness validation step.
    #     test_step_available: Whether the module implements the harness
    #         test step.
    #     blocker: Recorded reason the architecture falls short of full
    #         readiness. Every ``"training_level_ready"`` record carries
    #         ``None`` here, so a readiness claim and a blocker are
    #         mutually exclusive.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    architecture_name: ArchitectureName
    status: ImplementationStatus
    local_network_available: bool
    training_step_available: bool
    validation_step_available: bool
    test_step_available: bool
    blocker: str | None


class ModuleBuildOptions(BaseModel):
    # Frozen build options: the HiFTNet F0-extractor checkpoint path and
    # the recovery learning-rate scale, the only two construction-time
    # degrees of freedom the registry admits. Every other recipe value
    # comes from the architecture's named reference factory, which is what
    # keeps the build surface too narrow to smuggle in recipe drift.
    #
    # Fields:
    #     hiftnet_f0_checkpoint_path: Location of the published JDC
    #         pitch-model release forwarded into the HiFTNet reference
    #         factory. ``None`` leaves the pitch network randomly
    #         initialized, and the option is ignored by every other
    #         architecture. Default: ``None``.
    #     learning_rate_scale: Multiplier applied to the fleet-uniform
    #         ``learning_rate`` field of the selected reference
    #         configuration. It is how the recovery arm of the pruning
    #         intervention family obtains its reduced fine-tuning rate,
    #         the caller supplying the factor rather than this record
    #         fixing one. A value of ``1.0`` returns the published rate
    #         untouched. Default: ``1.0``.
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid", strict=True)
    hiftnet_f0_checkpoint_path: Path | None = None
    learning_rate_scale: PositiveFloat = 1.0


class ArchitectureModuleSpec(BaseModel):
    # Canonical built-module record binding one architecture to its authentic configuration.
    # The registry is the single construction source consumed by every evidence lane.
    # A spec is the constructed realization of one Project-Trained
    # Configuration before any training state is attached; the Retained
    # Project Checkpoint is loaded onto the module afterwards by the lane
    # that consumes the spec.
    #
    # Fields:
    #     architecture_name: The registered name this module was built for.
    #     variant_name: Label of the reference recipe the module was
    #         constructed from, for example ``"v1"`` for HiFi-GAN V1 or
    #         ``"seungwon"`` for the MelGAN release recipe. It identifies
    #         the recipe in evidence rows and run directories.
    #     module: The constructed harness module itself; the record allows
    #         arbitrary types so the live torch object survives validation.
    #     configuration_dump: JSON-mode dump of the configuration the
    #         module was built with. Evidence lanes serialize this dump
    #         rather than the live configuration object, so it is the
    #         value that must be read when comparing recipes across runs.
    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        arbitrary_types_allowed=True
    )
    architecture_name: ArchitectureName
    variant_name: str
    module: Module
    configuration_dump: dict[str, object]


class ModelRegistry:
    # The architecture authority: status records for the full vocabulary
    # and canonical construction of modules and published-weight adapters.
    # Runners and the CLI dispatch exclusively through this boundary. The
    # class carries no instance state: the record table is a class
    # variable and every method is a pure query or a fresh construction,
    # so a registry instance is interchangeable with any other.
    #
    # Integration: construction routing is a single match statement over
    # the architecture name. Each arm calls exactly one named reference
    # factory (HifiganConfig.v1, MelganConfig.seungwon,
    # VocosConfig.charactr_mel_24khz, and so on), passes the resulting
    # record through the recovery learning-rate scale, constructs the
    # module, and returns an ArchitectureModuleSpec carrying the module
    # and the JSON dump of the configuration it was built with. Because
    # Study 1 training, Study 2 interventions, and published evaluations
    # all enter through this one method, they measure identically built
    # modules; adding an architecture means adding a name to the closed
    # vocabulary, a status record, and one match arm, never a runtime
    # string.
    #
    # The author-weight support boundary is separate from the readiness
    # boundary and does not coincide with it. Nine architectures have a
    # registered published-weights adapter (the three HiFi-GAN widths,
    # MelGAN, Vocos, BigVGAN, APNet2, FreeV, and HiFTNet); the remaining
    # four (LPCNet, RndVoc, VocosFormer, RFWave) have no validated author
    # release and reach only the project-trained lane. Note that HiFTNet
    # sits on the supported side of the weight boundary while being the
    # sole verification_only entry of the readiness boundary, and
    # VocosFormer is training-level ready while having no author release
    # to load, because it is a project adaptation rather than a published
    # model.
    #
    # Example::
    #
    #     from vocode.models.registry import ModelRegistry, ModuleBuildOptions
    #
    #     registry: ModelRegistry = ModelRegistry()
    #
    #     # Canonical build at the published rate:
    #     spec = registry.build_module_spec("hifigan_v1", ModuleBuildOptions())
    #
    #     # Recovery build at a reduced rate; nothing else changes:
    #     recovery_spec = registry.build_module_spec(
    #         "hifigan_v1",
    #         ModuleBuildOptions(learning_rate_scale=0.5)
    #     )
    #
    #     # Published-evaluation lane, which needs both halves:
    #     weights = registry.build_published_weights("hifigan_v1")
    records: ClassVar[tuple[ArchitectureImplementationRecord, ...]] = (
        ArchitectureImplementationRecord(
            architecture_name="hifigan_v1",
            status="training_level_ready",
            local_network_available=True,
            training_step_available=True,
            validation_step_available=True,
            test_step_available=True,
            blocker=None
        ),
        ArchitectureImplementationRecord(
            architecture_name="hifigan_v2",
            status="training_level_ready",
            local_network_available=True,
            training_step_available=True,
            validation_step_available=True,
            test_step_available=True,
            blocker=None
        ),
        ArchitectureImplementationRecord(
            architecture_name="hifigan_v3",
            status="training_level_ready",
            local_network_available=True,
            training_step_available=True,
            validation_step_available=True,
            test_step_available=True,
            blocker=None
        ),
        ArchitectureImplementationRecord(
            architecture_name="melgan",
            status="training_level_ready",
            local_network_available=True,
            training_step_available=True,
            validation_step_available=True,
            test_step_available=True,
            blocker=None
        ),
        ArchitectureImplementationRecord(
            architecture_name="vocos",
            status="training_level_ready",
            local_network_available=True,
            training_step_available=True,
            validation_step_available=True,
            test_step_available=True,
            blocker=None
        ),
        ArchitectureImplementationRecord(
            architecture_name="bigvgan",
            status="training_level_ready",
            local_network_available=True,
            training_step_available=True,
            validation_step_available=True,
            test_step_available=True,
            blocker=None
        ),
        ArchitectureImplementationRecord(
            architecture_name="apnet2",
            status="training_level_ready",
            local_network_available=True,
            training_step_available=True,
            validation_step_available=True,
            test_step_available=True,
            blocker=None
        ),
        ArchitectureImplementationRecord(
            architecture_name="freev",
            status="training_level_ready",
            local_network_available=True,
            training_step_available=True,
            validation_step_available=True,
            test_step_available=True,
            blocker=None
        ),
        ArchitectureImplementationRecord(
            architecture_name="hiftnet",
            status="verification_only",
            local_network_available=True,
            training_step_available=True,
            validation_step_available=True,
            test_step_available=True,
            blocker=(
                "Checkpoint 01 excludes HiFTNet because no accepted project-trained "
                "checkpoint and three-seed result package were produced."
            )
        ),
        ArchitectureImplementationRecord(
            architecture_name="lpcnet",
            status="training_level_ready",
            local_network_available=True,
            training_step_available=True,
            validation_step_available=True,
            test_step_available=True,
            blocker=None
        ),
        ArchitectureImplementationRecord(
            architecture_name="rndvoc",
            status="training_level_ready",
            local_network_available=True,
            training_step_available=True,
            validation_step_available=True,
            test_step_available=True,
            blocker=None
        ),
        ArchitectureImplementationRecord(
            architecture_name="vocosformer",
            status="training_level_ready",
            local_network_available=True,
            training_step_available=True,
            validation_step_available=True,
            test_step_available=True,
            blocker=None
        ),
        ArchitectureImplementationRecord(
            architecture_name="rfwave",
            status="training_level_ready",
            local_network_available=True,
            training_step_available=True,
            validation_step_available=True,
            test_step_available=True,
            blocker=None
        )
    )

    def get(self, architecture_name: ArchitectureName) -> ArchitectureImplementationRecord:
        # Returns the implementation-status record for one architecture,
        # failing loudly on unregistered names. The lookup is a linear scan
        # that collects every matching record and returns the first, so a
        # duplicated registration would be shadowed here rather than
        # reported; uniqueness of the record table is asserted by the test
        # suite instead.
        #
        # Args:
        #     architecture_name: The registered name to resolve.
        #
        # Returns:
        #     The frozen ArchitectureImplementationRecord declaring that
        #     architecture's readiness level, capability flags, and
        #     recorded blocker.
        #
        # Raises:
        #     KeyError: If no record carries the requested name. The
        #         message names the offending architecture, because a
        #         silent default would let an unregistered name reach an
        #         evidence lane.
        matching_record_values: list[ArchitectureImplementationRecord] = []
        record: ArchitectureImplementationRecord
        for record in self.records:
            if record.architecture_name == architecture_name:
                matching_record_values.append(record)
        matching_records: tuple[ArchitectureImplementationRecord, ...] = tuple(
            matching_record_values
        )
        if not matching_records:
            raise KeyError(f"Unknown architecture_name={architecture_name}")
        return matching_records[0]

    def build_module_spec(
        self,
        architecture_name: ArchitectureName,
        build_options: ModuleBuildOptions
    ) -> ArchitectureModuleSpec:
        # Builds the canonical module with its authentic configuration for every evidence lane.
        # Each arm of the match resolves the architecture's named reference
        # factory, applies the recovery learning-rate scale to the record,
        # constructs the module from the scaled record, and assembles the
        # spec; the module and the configuration dump the spec carries are
        # therefore always built from the same record. The HiFTNet arm is
        # the only one that consumes the F0-checkpoint build option.
        #
        # Args:
        #     architecture_name: The registered architecture to construct.
        #     build_options: The two admitted construction-time degrees of
        #         freedom. Pass ``ModuleBuildOptions()`` to reproduce the
        #         published recipe exactly.
        #
        # Returns:
        #     A frozen ArchitectureModuleSpec binding the freshly
        #     constructed module to its variant label and the JSON dump of
        #     the configuration it was built with. Repeated calls return
        #     independent modules, so one spec is never shared between
        #     evidence lanes.
        #
        # Raises:
        #     MisconfigurationError: If the name reaches the fallthrough
        #         arm, which happens only for a name outside the closed
        #         vocabulary; construction never falls back to a default
        #         recipe.
        match architecture_name:
            case "hifigan_v1":
                hifigan_v1_configuration: HifiganConfig = self._scale_learning_rate(
                    HifiganConfig.v1(),
                    build_options
                )
                return self._assemble_spec(architecture_name, "v1", Hifigan(hifigan_v1_configuration), hifigan_v1_configuration)
            case "hifigan_v2":
                hifigan_v2_configuration: HifiganConfig = self._scale_learning_rate(
                    HifiganConfig.v2(),
                    build_options
                )
                return self._assemble_spec(architecture_name, "v2", Hifigan(hifigan_v2_configuration), hifigan_v2_configuration)
            case "hifigan_v3":
                hifigan_v3_configuration: HifiganConfig = self._scale_learning_rate(
                    HifiganConfig.v3(),
                    build_options
                )
                return self._assemble_spec(architecture_name, "v3", Hifigan(hifigan_v3_configuration), hifigan_v3_configuration)
            case "melgan":
                melgan_configuration: MelganConfig = self._scale_learning_rate(
                    MelganConfig.seungwon(),
                    build_options
                )
                return self._assemble_spec(architecture_name, "seungwon", Melgan(melgan_configuration), melgan_configuration)
            case "vocos":
                vocos_configuration: VocosConfig = self._scale_learning_rate(
                    VocosConfig.charactr_mel_24khz(),
                    build_options
                )
                return self._assemble_spec(architecture_name, "charactr_mel_24khz", Vocos(vocos_configuration), vocos_configuration)
            case "bigvgan":
                bigvgan_configuration: BigvganConfig = self._scale_learning_rate(
                    BigvganConfig.nvidia_base_24khz_100band(),
                    build_options
                )
                return self._assemble_spec(architecture_name, "base_24khz_100band", Bigvgan(bigvgan_configuration), bigvgan_configuration)
            case "apnet2":
                apnet2_configuration: Apnet2Config = self._scale_learning_rate(
                    Apnet2Config.redmist328(),
                    build_options
                )
                return self._assemble_spec(architecture_name, "redmist328_ljspeech", Apnet2(apnet2_configuration), apnet2_configuration)
            case "freev":
                freev_configuration: FreevConfig = self._scale_learning_rate(
                    FreevConfig.official(),
                    build_options
                )
                return self._assemble_spec(architecture_name, "official_ljspeech", Freev(freev_configuration), freev_configuration)
            case "hiftnet":
                hiftnet_configuration: HiftnetConfig = self._scale_learning_rate(
                    HiftnetConfig.yl4579(f0_checkpoint_path=build_options.hiftnet_f0_checkpoint_path),
                    build_options
                )
                return self._assemble_spec(architecture_name, "yl4579_ljspeech", Hiftnet(hiftnet_configuration), hiftnet_configuration)
            case "lpcnet":
                lpcnet_configuration: LpcnetConfig = self._scale_learning_rate(
                    LpcnetConfig.xiph_reference(),
                    build_options
                )
                return self._assemble_spec(architecture_name, "xiph_reference_ljspeech", Lpcnet(lpcnet_configuration), lpcnet_configuration)
            case "rndvoc":
                rndvoc_configuration: RndvocConfig = self._scale_learning_rate(
                    RndvocConfig.andong_22k(),
                    build_options
                )
                return self._assemble_spec(architecture_name, "andong_22k_ljspeech", Rndvoc(rndvoc_configuration), rndvoc_configuration)
            case "vocosformer":
                vocosformer_configuration: VocosformerConfig = self._scale_learning_rate(
                    VocosformerConfig.matched_vocos_24khz(),
                    build_options
                )
                return self._assemble_spec(architecture_name, "matched_vocos_24khz_ljspeech", Vocosformer(vocosformer_configuration), vocosformer_configuration)
            case "rfwave":
                rfwave_configuration: RfwaveConfig = self._scale_learning_rate(
                    RfwaveConfig.bfs18_24khz(),
                    build_options
                )
                return self._assemble_spec(architecture_name, "bfs18_24khz_ljspeech", Rfwave(rfwave_configuration), rfwave_configuration)
            case _:
                raise MisconfigurationError(f"Unsupported architecture_name={architecture_name}")

    def build_published_weights(self, architecture_name: ArchitectureName) -> PublishedWeights:
        # Constructs the registered published-weights adapter for one
        # architecture; families without an author release fail loudly.
        # Only the adapter object is built here: it carries its provenance
        # record immediately, while retrieval, SHA-256 verification, and
        # the strict load happen later inside the adapter's load method.
        # Construction therefore touches no network and no filesystem.
        #
        # Args:
        #     architecture_name: The registered architecture whose author
        #         release is requested.
        #
        # Returns:
        #     A PublishedWeights adapter whose provenance record anchors
        #     that architecture's release; the three HiFi-GAN widths each
        #     resolve to their own release record.
        #
        # Raises:
        #     MisconfigurationError: If the architecture has no
        #         model-local author-weight loader. This covers both the
        #         four registered architectures without a validated
        #         release and any name outside the vocabulary; the request
        #         fails closed rather than degrading to an untraceable
        #         anchor, and the message directs the caller to a Retained
        #         Project Checkpoint instead.
        match architecture_name:
            case "hifigan_v1":
                return HifiganWeights.v1()
            case "hifigan_v2":
                return HifiganWeights.v2()
            case "hifigan_v3":
                return HifiganWeights.v3()
            case "melgan":
                return MelganWeights()
            case "vocos":
                return VocosWeights()
            case "bigvgan":
                return BigvganWeights()
            case "apnet2":
                return Apnet2Weights()
            case "freev":
                return FreevWeights()
            case "hiftnet":
                return HiftnetWeights()
            case _:
                raise MisconfigurationError(
                    f"Architecture {architecture_name} has no model-local author-weight loader. "
                    f"Use a project-trained checkpoint or register a verified model-owned adapter."
                )

    def all_records(self) -> tuple[ArchitectureImplementationRecord, ...]:
        # Returns every registered architecture implementation record in
        # declaration order. The class-level tuple is handed back directly
        # rather than copied, which is safe because the tuple is immutable
        # and every record inside it is frozen.
        return self.records

    def training_level_architectures(self) -> tuple[ArchitectureName, ...]:
        # Returns only architectures marked ready for project-trained reproduction.
        # The filter reads the status field alone, so demoting an
        # architecture out of the reproduction cohort is a one-record edit
        # rather than a code change. The result preserves declaration
        # order and currently holds twelve of the thirteen names, which is
        # exactly the executed cohort of twelve Project-Trained
        # Configurations; HiFTNet is the sole documented exclusion and is
        # not a thirteenth result.
        architecture_names: list[ArchitectureName] = []
        record: ArchitectureImplementationRecord
        for record in self.records:
            if record.status == "training_level_ready":
                architecture_names.append(record.architecture_name)
        return tuple(architecture_names)

    def _scale_learning_rate[ConfigType: BaseModel](
        self,
        configuration: ConfigType,
        build_options: ModuleBuildOptions
    ) -> ConfigType:
        # Applies the recovery learning-rate scale through the fleet-uniform learning_rate field.
        # A unit scale short-circuits and returns the reference record
        # itself, so the default build path produces the published recipe
        # object unmodified. Any other scale produces a copy through
        # model_copy, leaving the frozen original intact and touching no
        # field except the rate. The helper is generic over the family
        # configuration type because every architecture configuration
        # declares learning_rate under that one name, which is what makes
        # recovery fine-tuning a single code path rather than thirteen.
        #
        # Args:
        #     configuration: The reference configuration record produced
        #         by the architecture's named factory.
        #     build_options: The build request whose learning_rate_scale
        #         is applied.
        #
        # Returns:
        #     The original record when the scale is ``1.0``, otherwise a
        #     copy whose learning_rate is the reference rate times the
        #     scale.
        if build_options.learning_rate_scale == 1.0:
            return configuration
        current_learning_rate: float = getattr(configuration, "learning_rate")
        return configuration.model_copy(
            update={"learning_rate": current_learning_rate * build_options.learning_rate_scale}
        )

    def _assemble_spec(
        self,
        architecture_name: ArchitectureName,
        variant_name: str,
        module: Module,
        configuration: BaseModel
    ) -> ArchitectureModuleSpec:
        # Assembles the canonical spec record from one constructed architecture case.
        # The configuration is serialized in JSON mode here, which is what
        # turns paths, tuples, and nested protocol records into the
        # primitive dump the evidence lanes persist and compare.
        return ArchitectureModuleSpec(
            architecture_name=architecture_name,
            variant_name=variant_name,
            module=module,
            configuration_dump=configuration.model_dump(mode="json")
        )
