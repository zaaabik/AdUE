from __future__ import annotations

from collections.abc import Callable
from typing import Any

import hydra.utils
from datasets import DatasetDict
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

from src.dataset.sequence_classification import BaseDataset
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class TextDataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        dataset: DictConfig,
        batch_size: int,
        collate_fn: DictConfig,
        tokenizer: DictConfig,
        num_labels: int,
        num_workers: int = 0,  # pylint: disable=unused-argument
        pin_memory: bool = False,  # pylint: disable=unused-argument
        max_length: int = 512,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        Args:
            dataset (DictConfig): config to instantiate dataset.
            batch_size (int): The batch size. Defaults to `64`.
            collate_fn (DictConfig): config to instantiate col.
            num_workers (int): The number of workers. Defaults to `0`.
            pin_memory (bool): Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.dataset: BaseDataset = hydra.utils.instantiate(dataset)
        self.tokenizer: PreTrainedTokenizer = hydra.utils.instantiate(
            tokenizer,
        )
        self.update_tokenizer_()

        self.collator_fn: Callable = hydra.utils.instantiate(collate_fn, tokenizer=self.tokenizer)

        self.max_length = max_length

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None
        self.loaded_datasets: DatasetDict | None = None

        self.batch_size_per_device = batch_size

    def update_tokenizer_(self):
        """Update tokenizer to work with padding."""
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        Returns:
            (int) The number of MNIST classes.
        """
        return self.hparams.num_labels

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True, max_length=self.max_length)

    def setup(self, stage: str | None = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.
        Args:
            stage (str | None): The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
            print("Batch size per device", self.batch_size_per_device)
            print("batch_size", self.hparams.batch_size)
            print("world_size", self.trainer.world_size)

        # Divide batch size by the number of devices.
        self.loaded_datasets = self.dataset.load()
        if "test" not in self.loaded_datasets:
            log.info("Dataset does not have test. Create automatic split")
            train_dataset = (
                self.loaded_datasets["train"]
                .map(self.tokenize_function, batched=True, load_from_cache_file=True)
                .remove_columns("text")
            )

            test_dataset = (
                self.loaded_datasets["test"]
                .map(self.tokenize_function, batched=True, load_from_cache_file=True)
                .remove_columns("text")
            )

            train_val_dataset = train_dataset.train_test_split(test_size=0.2, seed=1337)

            # train_dataset using for train model
            train_dataset = train_val_dataset["train"]
            # validation_dataset using for hyperparameters search
            validation_dataset = train_val_dataset["test"]
        else:
            log.info("Dataset has train/test/valid split")
            train_dataset = (
                self.loaded_datasets["train"]
                .map(self.tokenize_function, batched=True, load_from_cache_file=True)
                .remove_columns("text")
            )

            validation_dataset = (
                self.loaded_datasets["validation"]
                .map(self.tokenize_function, batched=True, load_from_cache_file=True)
                .remove_columns("text")
            )

            test_dataset = (
                self.loaded_datasets["test"]
                .map(self.tokenize_function, batched=True, load_from_cache_file=True)
                .remove_columns("text")
            )

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = train_dataset
            self.data_val = validation_dataset
            self.data_test = test_dataset
        assert self.dataset.num_classes() == self.hparams.num_labels, "Number of labels should match with " \
                                                                      "dataset num_classes"
        log.info(
            f"Train samples: {len(self.data_train)} Valid samples: {len(self.data_val)} "
            f"Test samples: {len(self.data_test)}"
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        Returns:
             (DataLoader) The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        Returns:
            The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        Returns:
            The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator_fn,
            shuffle=False,
        )

    def teardown(self, stage: str | None = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        Args:
            stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """

    def state_dict(self) -> dict[Any, Any]:
        """Called when saving a checkpoint.

        Implement to generate and save the datamodule state.
        Returns:
            A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        Args:
            state_dict: The datamodule state returned by `self.state_dict()`.
        """
