# This module:
# 1. Defines the DataModule contract the trainer depends on: one-time data
#    preparation, stage-scoped setup and teardown, the four stage dataloader
#    factories, checkpointable state, exception handling, and the
#    batch-transfer hook trio
# 2. Provides DatasetDataModule, a concrete implementation that builds standard
#    dataloaders directly from supplied dataset objects
#
# Design decisions:
# - prepare_data is separated from setup because preparation (such as downloads
#   and one-time preprocessing) must run exactly once per node, while setup runs
#   in every process and may allocate per-process state
# - The dataloader factories raise NotImplementedError with instructions rather
#   than returning None, so requesting a stage that was never configured fails
#   with corrective guidance instead of a later attribute error inside a loop
# - The batch-transfer hook trio is duplicated here from the module-side data
#   hooks because the loops give datamodule overrides precedence over module
#   overrides, allowing data-representation concerns to live with the data
#   definition when both exist
# - state_dict and load_state_dict participate in checkpointing so stateful
#   data pipelines (for example samplers with position state) can survive
#   resumption
# - DatasetDataModule shuffles only map-style training datasets, because
#   iterable datasets manage their own ordering and the DataLoader forbids
#   shuffle for them; evaluation and prediction loaders never shuffle
#
# Author: Rahul Sawhney

from collections.abc import Callable

import torch

from syntheticmind.utilities.types import Batch, StateDict, TrainerStage

__all__: list[str] = ["DataModule", "DatasetDataModule"]


class DataModule:
    # Base data contract consumed by the trainer. Subclasses implement the
    # dataloader factories for the stages they support; every other member has
    # a working default so simple datamodules override only what they need.
    #
    # Integration: a client datamodule subclasses
    # syntheticmind.core.datamodule.DataModule, prepares external data once in
    # prepare_data, builds per-process dataset state in setup, and returns
    # one torch.utils.data.DataLoader per supported stage.
    #
    # Example::
    #
    #     import torch
    #
    #     from syntheticmind.core.datamodule import DataModule
    #     from syntheticmind.utilities.types import TrainerStage
    #
    #     class SpeechDataModule(DataModule):
    #         def __init__(self, corpus_root: Path) -> None:
    #             self._corpus_root: Path = corpus_root
    #             self._train_dataset: torch.utils.data.Dataset | None = None
    #             self._val_dataset: torch.utils.data.Dataset | None = None
    #
    #         def prepare_data(self) -> None:
    #             # Download or validate once; the trainer invokes this from
    #             # rank zero only.
    #             ...
    #
    #         def setup(self, stage: TrainerStage) -> None:
    #             # Build datasets and splits; runs in every process.
    #             ...
    #
    #         def train_dataloader(self) -> torch.utils.data.DataLoader:
    #             return torch.utils.data.DataLoader(self._train_dataset, shuffle=True)
    #
    #         def val_dataloader(self) -> torch.utils.data.DataLoader:
    #             return torch.utils.data.DataLoader(self._val_dataset, shuffle=False)
    #
    #         def teardown(self, stage: TrainerStage) -> None:
    #             # Release stage-local resources.
    #             ...
    def prepare_data(self) -> None:
        # Invoked once before dataloaders are requested, for work that must not
        # run in every process, such as downloading or one-time preprocessing.
        pass

    def setup(self, stage: TrainerStage) -> None:
        # Invoked in every process before the given stage executes, for
        # per-process state such as dataset object construction and splits.
        pass

    def teardown(self, stage: TrainerStage) -> None:
        # Invoked after the given stage completes, for releasing resources
        # acquired in setup.
        pass

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        # Factory for the training dataloader; mandatory for fit runs.
        raise NotImplementedError("train_dataloader() must be implemented by a DataModule subclass.")

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        # Factory for the validation dataloader; required when validation is
        # scheduled during fit or invoked standalone.
        raise NotImplementedError("val_dataloader() must be implemented by a DataModule subclass.")

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        # Factory for the test dataloader; required by the test entry point.
        raise NotImplementedError("test_dataloader() must be implemented by a DataModule subclass.")

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        # Factory for the prediction dataloader; required by the predict entry
        # point.
        raise NotImplementedError("predict_dataloader() must be implemented by a DataModule subclass.")

    def state_dict(self) -> StateDict:
        # Serializes datamodule state into checkpoints. The default contributes
        # nothing; stateful pipelines override this together with
        # load_state_dict.
        state_dict: StateDict = {}
        return state_dict

    def load_state_dict(self, state_dict: StateDict) -> None:
        # Restores datamodule state recorded by state_dict during resumption.
        pass

    def on_exception(self, exception: BaseException) -> None:
        # Invoked when a run terminates with an exception, so the datamodule
        # can release external resources before teardown.
        pass

    def on_before_batch_transfer(self, batch: Batch, dataloader_idx: int = 0) -> Batch:
        # Invoked on the host-side batch before device placement. Overrides
        # here take precedence over the module-side hook of the same name.
        return batch

    def transfer_batch_to_device(self, batch: Batch, device: torch.device, dataloader_idx: int = 0) -> Batch:
        # Places the batch on the target device. The default delegates to the
        # shared recursive transfer helper; overrides here take precedence over
        # the module-side hook of the same name.
        from syntheticmind.utilities.data_transfer import move_data_to_device

        return move_data_to_device(batch, device)

    def on_after_batch_transfer(self, batch: Batch, dataloader_idx: int = 0) -> Batch:
        # Invoked on the device-resident batch immediately before the step.
        # Overrides here take precedence over the module-side hook of the same
        # name.
        return batch

    @classmethod
    def from_datasets(
        cls,
        train_dataset: torch.utils.data.Dataset | None = None,
        val_dataset: torch.utils.data.Dataset | None = None,
        test_dataset: torch.utils.data.Dataset | None = None,
        predict_dataset: torch.utils.data.Dataset | None = None,
        batch_size: int = 1,
        num_workers: int = 0
    ) -> DataModule:
        # Factory that wraps plain dataset objects in a DatasetDataModule, for
        # callers that need standard dataloaders without a custom subclass.
        #
        # Args:
        #     train_dataset: Optional torch.utils.data.Dataset backing the
        #         training loader. Default: ``None``.
        #     val_dataset: Optional torch.utils.data.Dataset backing the
        #         validation loader. Default: ``None``.
        #     test_dataset: Optional torch.utils.data.Dataset backing the
        #         test loader. Default: ``None``.
        #     predict_dataset: Optional torch.utils.data.Dataset backing
        #         the prediction loader. Default: ``None``.
        #     batch_size: Batch size shared by every configured loader.
        #         Default: ``1``.
        #     num_workers: Worker processes shared by every configured
        #         loader; zero loads in the main process. Default: ``0``.
        #
        # Return:
        #     A DatasetDataModule with exactly the supplied stages
        #     configured; requesting an unconfigured stage raises with
        #     corrective guidance.
        return DatasetDataModule(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            predict_dataset=predict_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )

    def _configured_loader_stages(self) -> list[str]:
        # Determines which stages this datamodule supports by checking which
        # dataloader factories the subclass has overridden relative to this
        # base class; used by the representation only.
        configured_loaders: list[str] = []
        for stage in ("train", "val", "test", "predict"):
            method_name: str = f"{stage}_dataloader"
            class_method: Callable[..., torch.utils.data.DataLoader] | None = getattr(type(self), method_name, None)
            base_method: Callable[..., torch.utils.data.DataLoader] | None = getattr(DataModule, method_name, None)
            if class_method is not None and class_method is not base_method:
                configured_loaders.append(stage)
        return configured_loaders

    def __repr__(self) -> str:
        # Reports which stage dataloaders are configured, for logs and
        # interactive debugging.
        configured_loaders: list[str] = self._configured_loader_stages()
        class_name: str = type(self).__name__
        loaders_repr: str = ", ".join(f"{name}=True" for name in configured_loaders)
        return f"{class_name}({loaders_repr})"


class DatasetDataModule(DataModule):
    # Concrete datamodule over pre-constructed dataset objects. Each stage
    # factory builds a standard DataLoader with the shared batch size and
    # worker count, raising with guidance when its dataset was not supplied.
    def __init__(
        self,
        train_dataset: torch.utils.data.Dataset | None = None,
        val_dataset: torch.utils.data.Dataset | None = None,
        test_dataset: torch.utils.data.Dataset | None = None,
        predict_dataset: torch.utils.data.Dataset | None = None,
        batch_size: int = 1,
        num_workers: int = 0
    ) -> None:
        # Validates the loader parameters and stores the per-stage datasets;
        # absent datasets simply leave their stages unconfigured.
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {num_workers}")
        self._train_dataset: torch.utils.data.Dataset | None = train_dataset
        self._val_dataset: torch.utils.data.Dataset | None = val_dataset
        self._test_dataset: torch.utils.data.Dataset | None = test_dataset
        self._predict_dataset: torch.utils.data.Dataset | None = predict_dataset
        self._batch_size: int = batch_size
        self._num_workers: int = num_workers

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        # Builds the training dataloader. Map-style datasets are shuffled;
        # iterable datasets are not, because they own their ordering and the
        # DataLoader rejects shuffle for them.
        if self._train_dataset is None:
            raise NotImplementedError(
                "train_dataloader() is not configured. Override train_dataloader() "
                "or provide train_dataset when constructing DatasetDataModule."
            )
        train_shuffle: bool = not isinstance(self._train_dataset, torch.utils.data.IterableDataset)
        return torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=train_shuffle,
            num_workers=self._num_workers
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        # Builds the validation dataloader without shuffling, so evaluation
        # order is stable across epochs and runs.
        if self._val_dataset is None:
            raise NotImplementedError(
                "val_dataloader() is not configured. Override val_dataloader() "
                "or provide val_dataset when constructing DatasetDataModule."
            )
        return torch.utils.data.DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        # Builds the test dataloader without shuffling, so evaluation order is
        # stable across runs.
        if self._test_dataset is None:
            raise NotImplementedError(
                "test_dataloader() is not configured. Override test_dataloader() "
                "or provide test_dataset when constructing DatasetDataModule."
            )
        return torch.utils.data.DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        # Builds the prediction dataloader without shuffling, so outputs align
        # with the dataset order.
        if self._predict_dataset is None:
            raise NotImplementedError(
                "predict_dataloader() is not configured. Override predict_dataloader() "
                "or provide predict_dataset when constructing DatasetDataModule."
            )
        return torch.utils.data.DataLoader(
            self._predict_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers
        )

    def _configured_loader_stages(self) -> list[str]:
        # Reports configured stages from dataset presence rather than method
        # overrides, because this class configures stages through constructor
        # arguments instead of subclassing.
        configured_loaders: list[str] = []
        if self._train_dataset is not None:
            configured_loaders.append("train")
        if self._val_dataset is not None:
            configured_loaders.append("val")
        if self._test_dataset is not None:
            configured_loaders.append("test")
        if self._predict_dataset is not None:
            configured_loaders.append("predict")
        return configured_loaders
