import json
import os
import torchmetrics

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             roc_auc_score)
from torch.utils.data import DataLoader, TensorDataset


class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.w = nn.Linear(input_dim, 1)

    def forward(self, x, length):
        return self.w(x)[:,0]


class AttentionProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.q = nn.Linear(input_dim, 1, bias=False)
        self.w = nn.Linear(input_dim, 1, bias=False)

    @staticmethod
    def create_binary_mask(lengths, max_length=None):
        """
        Create a binary mask from lengths of each sample in a batch.

        Args:
            lengths: List or tensor of lengths for each sample in the batch
            max_length: Maximum length (if None, use max of lengths)

        Returns:
            Binary mask tensor of shape (batch_size, max_length) where:
              0 = real item (non-padding)
              1 = padding
        """
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.tensor(lengths)

        if max_length is None:
            max_length = lengths.max().item()

        batch_size = len(lengths)

        # Create range tensor: [0, 1, 2, ..., max_length-1]
        range_tensor = torch.arange(max_length, device=lengths.device).unsqueeze(0)

        # Expand range tensor to batch size
        range_tensor = range_tensor.expand(batch_size, max_length)

        # Expand lengths to compare with range
        lengths_expanded = lengths.unsqueeze(1).expand(batch_size, max_length)

        # Create mask: 1 where position >= length (padding), 0 otherwise (real item)
        mask = (range_tensor >= lengths_expanded)

        return mask

    def forward(self, x, length):
        mask = self.create_binary_mask(length, x.shape[1])
        attention_mask = torch.zeros((x.shape[0], x.shape[1]), device=x.device, dtype=x.dtype)
        attention_mask = torch.masked_fill(attention_mask, mask.to(device=x.device), float('-inf'))

        q_T_h = self.q(x)[:, :, 0]
        attention = torch.softmax(q_T_h + attention_mask, dim=-1)

        h_prime = torch.sum(attention[:, :, None] * x, dim=1)
        return self.w(h_prime)[:,0]


class ProbeLightningModule(L.LightningModule):
    def __init__(self, probe: nn.Module, lr: float = 3e-4, mode='max'):
        super().__init__()
        self.save_hyperparameters()
        self.probe = probe
        self.criterion = nn.BCEWithLogitsLoss()

        self.train_roc_auc = torchmetrics.AUROC(task='binary')
        self.val_roc_auc = torchmetrics.AUROC(task='binary')

        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        if mode == 'max':
            self.best_metric = torchmetrics.MaxMetric()
        elif mode == 'min':
            self.best_metric = torchmetrics.MinMetric()

    def forward(self, x, length):
        return self.probe(x, length)

    def training_step(self, batch, batch_idx):
        x, y, length = batch
        preds = self(x, length)
        loss = self.criterion(preds, y)
        self.train_loss(loss)
        self.train_roc_auc(preds, y)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/auc", self.train_roc_auc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, length = batch
        preds = self(x, length)
        loss = self.criterion(preds, y)
        self.val_loss(loss)
        self.val_roc_auc(preds, y)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/auc", self.val_roc_auc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        current_metric = self.val_roc_auc.compute()  # get current val acc
        prev_best_metric = self.best_metric.compute()
        self.best_metric(current_metric)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        if self.best_metric.compute() != prev_best_metric:
            self.log('val/roc_auc_best', self.val_roc_auc.compute(), prog_bar=True)
            self.log('train/roc_auc_best', self.train_roc_auc.compute(), prog_bar=True)
            self.log('best_epoch', self.trainer.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


def _make_loaders(train_X, train_y, tr_length, val_X, val_y, val_length, batch_size: int):
    train_ds = TensorDataset(train_X, train_y, tr_length)
    val_ds = TensorDataset(val_X, val_y, val_length)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_dl, val_dl


def train_probe_lightning(
    probe: nn.Module,
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    tr_length: torch.Tensor,
    val_X: torch.Tensor,
    val_y: torch.Tensor,
    val_length: torch.Tensor,
    lr: float,
    epochs: int,
    batch_size: int,
    log_params=None,
    accelerator: str = None,
):
    module = ProbeLightningModule(probe, lr)
    train_dl, val_dl = _make_loaders(train_X, train_y, tr_length, val_X, val_y, val_length, batch_size)

    os.makedirs('mlflow', exist_ok=True)
    mlflow_logger = L.pytorch.loggers.MLFlowLogger(
        experiment_name=f"{log_params['type']}_{log_params['dataset']}",
        tracking_uri='sqlite:///mlflow/database.db'
    )
    mlflow_logger.log_hyperparams(log_params)

    callbacks = [
        L.pytorch.callbacks.EarlyStopping(monitor="val/auc", mode="max", patience=5),
        L.pytorch.callbacks.ModelCheckpoint(monitor="val/auc", mode="max", save_weights_only=True),

    ]

    trainer = L.Trainer(
        max_epochs=epochs,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        enable_progress_bar=True,
        precision='bf16-mixed',
        accelerator='cpu',
        logger=[
            mlflow_logger
        ],
    )
    trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl)

    best_path = trainer.checkpoint_callback.best_model_path
    print(best_path)
    best_model = ProbeLightningModule.load_from_checkpoint(best_path)

    return best_model.probe


@torch.no_grad()
def evaluate_probe(
    probe: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    length: torch.Tensor,
    threshold: float = 0.5,
    device: str = "cpu",
):
    probe = probe.to(device=device, dtype=torch.float32).eval()
    dl = DataLoader(TensorDataset(features, labels, length), batch_size=128, shuffle=False)
    preds, targs = [], []
    for X, y, length in dl:
        X = X.to(device=device, dtype=torch.float32)
        length = length.to(device=device, dtype=torch.float32)
        out = probe(X, length)
        preds.extend(out.detach().cpu().numpy())
        targs.extend(y.cpu().numpy())
    auc = roc_auc_score(targs, preds)
    pred_bin = (np.array(preds) > threshold).astype(int)
    acc = accuracy_score(targs, pred_bin)
    f1 = f1_score(targs, pred_bin)
    rec = recall_score(targs, pred_bin)
    return auc, acc, f1, rec


def save_probe_model(probe: nn.Module, save_dir: str, metadata: dict):
    os.makedirs(save_dir, exist_ok=True)

    weights_path = os.path.join(save_dir, "probe.pt")
    torch.save(probe.state_dict(), weights_path)

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return weights_path
