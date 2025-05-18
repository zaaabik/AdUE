from __future__ import annotations

from typing import Any, Tuple

import hydra.utils
import torch
from lightning import LightningModule
from omegaconf import DictConfig
from torch import Tensor
from torchmetrics import (
    AUROC,
    Accuracy,
    CalibrationError,
    MaxMetric,
    MeanMetric,
    Metric,
    MinMetric,
    MatthewsCorrCoef
)
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from src.uncertainty.metrics import ErrorAUROCMaxProb
from src.utils.instantiators import BaseModelInstantiate


class LLMLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model: DictConfig,
        optimizer: DictConfig,  # pylint: disable=unused-argument
        scheduler: DictConfig,  # pylint: disable=unused-argument
        optimize_metric_name: str,
        optimize_metric_mode: str,
        compile: bool = False,  # pylint: disable=redefined-builtin,unused-argument
    ) -> None:
        """Initialize a `MNISTLitModule`.

        Args:
            model: Cfg for model instantiate
            optimizer: The optimizer to use for training.
            scheduler: The learning rate scheduler to use for training.
            compile: torch compile model
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.model_instantiation: BaseModelInstantiate = hydra.utils.instantiate(model)
        self.net = self.model_instantiation.instantiate()
        self.num_labels = self.net.num_labels
        self.update_model_tokenizer_()

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_metrics: torch.nn.ParameterDict[str, Metric] = self.build_metric_dict(
            "train", num_classes=self.num_labels
        )

        self.val_metrics: torch.nn.ParameterDict[str, Metric] = self.build_metric_dict(
            "val", num_classes=self.num_labels
        )

        self.test_metrics: torch.nn.ParameterDict[str, Metric] = self.build_metric_dict(
            "test", num_classes=self.num_labels
        )

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.optimize_metric_name = optimize_metric_name

        # for tracking best so far validation accuracy
        if optimize_metric_mode == "max":
            self.choose_best_checkpoint_func = MaxMetric()
        elif optimize_metric_mode == "min":
            self.choose_best_checkpoint_func = MinMetric()
        else:
            raise ValueError(
                f"optimize_metric_mode should be min or max, but get {optimize_metric_mode}"
            )

    def update_model_tokenizer_(self):
        """Update model config to work proper with padding."""
        if not self.net.config.pad_token_id:
            self.net.config.pad_token_id = self.net.config.eos_token_id

    @staticmethod
    def build_metric_dict(
        dataset_name: str, num_classes: int = 2
    ) -> torch.nn.ParameterDict[str, Metric]:
        """Create classification metrics for dataset.

        Args:
            dataset_name (str): train/callback/valid/test
            num_classes (int): number of predicted classes
        Returns:
             (torch.nn.ParameterDict[str, Metric]) metric for each dataset
        """
        base_metric_params = {"task": "multiclass", "num_classes": num_classes}

        metric_dict = {
            "acc": Accuracy(**base_metric_params, top_k=1),
            "roc-auc": AUROC(**base_metric_params),
            "ece": CalibrationError(**base_metric_params),
            "matthews_corr": MatthewsCorrCoef(**base_metric_params),
            "error_roc_auc_max_prob": ErrorAUROCMaxProb()
        }

        return torch.nn.ParameterDict(
            {f"{dataset_name}/{name}": metric for name, metric in metric_dict.items()}
        )

    def compute_metrics_and_log(
        self,
        metric_dict: torch.nn.ParameterDict[str, Metric],
        preds: torch.Tensor,
        targets: torch.Tensor,
    ):
        """Compute all metrics
        Args:
            metric_dict (torch.nn.ParameterDict[str, Metric]): dict of all metrics for dataset
            preds (torch.Tensor): predictions of model
            targets (torch.Tensor): targets
        """
        for name, metric in metric_dict.items():
            metric(preds, targets)
            self.log(
                name, metric, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False
            )

    def forward(  # pylint: disable=arguments-differ
        self, x: dict
    ) -> SequenceClassifierOutputWithPast:
        """Perform a forward pass through the model `self.net`.

        Args:
            x: features
        Returns:
            SequenceClassifierOutputWithPast
        """
        return self.net(**x)

    def reset_val_metrics(self):
        """Reset all metrics for valid datasets."""
        for metric in self.val_metrics.values():
            metric.reset()

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.reset_val_metrics()
        self.val_loss.reset()
        self.choose_best_checkpoint_func.reset()

    def model_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.
        Args:
            batch (dict[str, torch.Tensor]): A batch of data (a tuple) containing the input tensor of images and target labels.

        Returns:
             A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        y = batch["labels"]
        output = self.forward(batch)

        loss = self.criterion(output.logits, y)
        preds = output.logits.detach()
        return loss, preds, y

    def model_step_predict(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[SequenceClassifierOutputWithPast, Tensor]:
        """Perform a single model step on a batch of data.
        Args:
            batch (dict[str, torch.Tensor]): A batch of data (a tuple) containing the input tensor of images and target labels.

        Returns:
             A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        y = batch["labels"]
        output = self.forward(batch)
        return output, y

    def training_step(  # pylint: disable=arguments-differ
        self, batch: dict[str, torch.Tensor], batch_idx: int  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        Args:
            batch (dict[str, Tensor]): A batch of data (a tuple) containing the input tensor of images and target
            labels.
            batch_idx (int): The index of the current batch.
        Return:
            (Tensor) A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.compute_metrics_and_log(self.train_metrics, preds, targets)
        return loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""

    def validation_step(  # pylint: disable=arguments-differ
        self, batch: dict[str, torch.Tensor], batch_idx: int  # pylint: disable=unused-argument
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        Args:
            batch (dict[str, Tensor]): A batch of data (a tuple) containing the input tensor of images and target
            labels.
            batch_idx (int): The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.compute_metrics_and_log(self.val_metrics, preds, targets)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        optimize_metric_value = self.val_metrics[self.optimize_metric_name].compute()

        self.choose_best_checkpoint_func(optimize_metric_value)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/optimize_metric",
            self.choose_best_checkpoint_func.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(  # pylint: disable=arguments-differ
        self, batch: dict[str, torch.Tensor], batch_idx: int  # pylint: disable=unused-argument
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        Args:
            batch (dict[str, Tensor]): A batch of data (a tuple) containing the input tensor of images and target
            labels.
            batch_idx (int): The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.compute_metrics_and_log(self.test_metrics, preds, targets)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""

    def predict_step(  # pylint: disable=arguments-differ
        self, batch: dict[str, torch.Tensor], **kwargs: Any
    ) -> dict:
        outputs, targets = self.model_step_predict(batch)
        outputs = dict(outputs)
        if 'hidden_states' in outputs:
            outputs['hidden_states'] = torch.stack(outputs['hidden_states'], axis=1)

        if 'attentions' in outputs:
            outputs['attentions'] = torch.stack(outputs['attentions'], axis=1)

        return {**outputs, "label_ids": targets}

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.
        Args:
            stage (str): Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        Returns:
            (dict) A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, params=self.trainer.model.parameters()
        )

        if self.hparams.scheduler is not None:
            scheduler = hydra.utils.instantiate(self.hparams.scheduler, optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.optimize_metric_name,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


class LLMLitModuleNuclearDiversityLoss(LLMLitModule):  # pylint: disable=too-many-ancestors
    def __init__(
        self, diversity_loss_lambda: float = -1e-3, **kwargs  # pylint: disable=unused-argument
    ) -> None:
        """Initialize a `MNISTLitModule`.

        Args:
            model: Cfg for model instantiate
            optimizer: The optimizer to use for training.
            scheduler: The learning rate scheduler to use for training.
            compile: torch compile model
        """
        super().__init__(**kwargs)
        self.train_diversity_loss = MeanMetric()
        self.val_diversity_loss = MeanMetric()
        self.test_diversity_loss = MeanMetric()

        self.train_cross_entropy_loss = MeanMetric()
        self.val_cross_entropy_loss = MeanMetric()
        self.test_cross_entropy_loss = MeanMetric()

    def model_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.
        Args:
            batch (dict[str, torch.Tensor]): A batch of data (a tuple) containing the input tensor of images and target labels.

        Returns:
             A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        y = batch["labels"]
        output = self.forward(batch)

        loss = self.criterion(output.logits, y)
        preds = output.logits.detach()
        return loss, preds, y

    def _calculate_diversity_loss(self):
        a_weights = {}
        for k, v in dict(self.net.named_modules()).items():
            if ".lora_A" in k and "seed" not in k and "merged" not in k:
                a_weights[k] = v

        loss_by_each_layer = []
        rank_by_each_layer = []
        for module_dict in a_weights.values():
            adapter_names = [name for name in module_dict.keys() if "seed" in name]
            weights = [module_dict[name].weight for name in adapter_names]
            a_nr_d = torch.cat(weights).float()
            for name in adapter_names:
                print(name, module_dict[name].weight)

            loss_on_layer = torch.linalg.norm(  # pylint: disable=not-callable
                a_nr_d, ord="nuc"
            ) / torch.linalg.norm(  # pylint: disable=not-callable
                a_nr_d, ord="fro"
            )
            loss_by_each_layer.append(loss_on_layer)
            rank_by_each_layer.append(
                torch.linalg.matrix_rank(a_nr_d).float()  # pylint: disable=not-callable
            )

        return torch.stack(loss_by_each_layer).mean(), torch.stack(rank_by_each_layer).mean()

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        Args:
            batch (dict[str, Tensor]): A batch of data (a tuple) containing the input tensor of images and target
            labels.
            batch_idx (int): The index of the current batch.
        Return:
            (Tensor) A tensor of losses between model predictions and targets.
        """
        cross_entropy_loss, preds, targets = self.model_step(batch)

        diversity_loss, mean_rank = self._calculate_diversity_loss()
        loss = cross_entropy_loss + diversity_loss * self.hparams.diversity_loss_lambda

        # update and log metrics
        self.log("train/mean_rank", mean_rank, on_step=False, on_epoch=True, prog_bar=True)

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.train_cross_entropy_loss(cross_entropy_loss)
        self.log(
            "train/cross_entropy_loss",
            self.train_cross_entropy_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.train_diversity_loss(diversity_loss)
        self.log(
            "train/diversity_loss",
            self.train_diversity_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.compute_metrics_and_log(self.train_metrics, preds, targets)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        Args:
            batch (dict[str, Tensor]): A batch of data (a tuple) containing the input tensor of images and target
            labels.
            batch_idx (int): The index of the current batch.
        """
        cross_entropy_loss, preds, targets = self.model_step(batch)

        diversity_loss, mean_rank = self._calculate_diversity_loss()
        loss = cross_entropy_loss + diversity_loss * self.hparams.diversity_loss_lambda

        # update and log metrics
        self.log("val/mean_rank", mean_rank, on_step=False, on_epoch=True, prog_bar=True)

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.val_cross_entropy_loss(cross_entropy_loss)
        self.log(
            "val/cross_entropy_loss",
            self.val_cross_entropy_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.val_diversity_loss(diversity_loss)
        self.log(
            "val/diversity_loss",
            self.val_diversity_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.compute_metrics_and_log(self.val_metrics, preds, targets)

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        Args:
            batch (dict[str, Tensor]): A batch of data (a tuple) containing the input tensor of images and target
            labels.
            batch_idx (int): The index of the current batch.
        """
        cross_entropy_loss, preds, targets = self.model_step(batch)

        diversity_loss, mean_rank = self._calculate_diversity_loss()
        loss = cross_entropy_loss + diversity_loss * self.hparams.diversity_loss_lambda

        # update and log metrics
        self.log("test/mean_rank", mean_rank, on_step=False, on_epoch=True, prog_bar=True)

        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.test_cross_entropy_loss(cross_entropy_loss)
        self.log(
            "test/cross_entropy_loss",
            self.test_cross_entropy_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.test_diversity_loss(diversity_loss)
        self.log(
            "test/diversity_loss",
            self.test_diversity_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.compute_metrics_and_log(self.test_metrics, preds, targets)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.reset_val_metrics()
        self.val_loss.reset()
        self.val_diversity_loss.reset()
        self.choose_best_checkpoint_func.reset()


class LLMLitModuleSingleTokenPrediction(LLMLitModule):  # pylint: disable=too-many-ancestors
    def __init__(self, last_token_idx: int, classes_indexes_in_logits: list[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_token_idx = torch.nn.Parameter(torch.tensor(last_token_idx), requires_grad=False)

        assert len(classes_indexes_in_logits) == self.num_labels
        self.classes_indexes_in_logits = torch.nn.Parameter(
            torch.tensor(classes_indexes_in_logits), requires_grad=False
        )

    def model_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.
        Args:
            batch (dict[str, torch.Tensor]): A batch of data (a tuple) containing the input tensor of images and target labels.

        Returns:
             A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        batch_size = batch["attention_mask"].shape[0]

        attention_mask = batch["attention_mask"]
        next_token_class = batch["mmlu_answer_label"]

        last_non_special_token_position = attention_mask.sum(dim=-1)
        answer_token_position_in_seq = last_non_special_token_position - self.last_token_idx - 1
        del batch["mmlu_answer_label"]
        output = self.forward(batch)
        shift_logits = output.logits[..., :-1, self.classes_indexes_in_logits].contiguous()

        # labels = batch["input_ids"]
        # sshift_labels = labels[..., 1:].contiguous()

        device = shift_logits.device

        answer_logits = shift_logits[
            torch.arange(batch_size, device=device), answer_token_position_in_seq, :
        ]
        # answer_labels = shift_labels[torch.arange(batch_size, device=device), answer_token_position_in_seq]
        loss = self.criterion(answer_logits, next_token_class)
        preds = answer_logits.detach()
        return loss, preds, next_token_class


class LLMLitModuleDropoutPredict(LLMLitModule):  # pylint: disable=too-many-ancestors
    def predict_step(self, batch: dict[str, torch.Tensor], **kwargs: Any) -> Any:
        self.net.train()
        return super().predict_step(batch=batch, **kwargs)
