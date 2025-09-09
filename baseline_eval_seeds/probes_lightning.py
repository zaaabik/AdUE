import json
import os
import shutil
import tempfile

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

    def forward(self, x):
        return torch.sigmoid(self.w(x).squeeze(-1))


class AttentionProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.q = nn.Linear(input_dim, 1, bias=False)
        self.w = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        q_T_h = self.q(x).squeeze(-1)
        attention = torch.softmax(q_T_h, dim=-1)
        h_prime = torch.sum(attention.unsqueeze(-1) * x, dim=1)
        return torch.sigmoid(self.w(h_prime).squeeze(-1))


class ProbeLightningModule(L.LightningModule):
    def __init__(self, probe: nn.Module, lr: float = 3e-4):
        super().__init__()
        self.save_hyperparameters()
        self.probe = probe
        self.criterion = nn.BCELoss()

    def forward(self, x):
        x = x.to(device=self.device, dtype=torch.float32)
        return self.probe(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        y = y.to(device=self.device, dtype=torch.float32)
        loss = self.criterion(preds, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        y = y.to(device=self.device, dtype=torch.float32)
        loss = self.criterion(preds, y)
        with torch.no_grad():
            preds_cpu = preds.detach().float().cpu().numpy()
            y_cpu = y.detach().cpu().numpy()
            try:
                auc = roc_auc_score(y_cpu, preds_cpu)
            except Exception:
                auc = float("nan")
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/auc", auc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


def _make_loaders(train_X, train_y, val_X, val_y, batch_size: int):
    train_ds = TensorDataset(train_X, train_y)
    val_ds = TensorDataset(val_X, val_y)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_dl, val_dl


def train_probe_lightning(
    probe: nn.Module,
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    val_X: torch.Tensor,
    val_y: torch.Tensor,
    lr: float,
    epochs: int,
    batch_size: int,
    accelerator: str = None,
):
    module = ProbeLightningModule(probe, lr)
    train_dl, val_dl = _make_loaders(train_X, train_y, val_X, val_y, batch_size)

    tmp_dir = tempfile.mkdtemp(prefix="probe_ckpt_")
    callbacks = [
        L.pytorch.callbacks.EarlyStopping(monitor="val/auc", mode="max", patience=3),
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=tmp_dir, monitor="val/auc", mode="max", save_weights_only=True
        ),
    ]

    trainer = L.Trainer(
        accelerator=accelerator or ("gpu" if torch.cuda.is_available() else "cpu"),
        max_epochs=epochs,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        enable_progress_bar=True,
        logger=False,
    )
    trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl)

    best_path = trainer.checkpoint_callback.best_model_path
    if best_path and os.path.exists(best_path):
        state = torch.load(best_path, map_location="cpu")
        module.load_state_dict(state["state_dict"], strict=False)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    return module.probe


@torch.no_grad()
def evaluate_probe(
    probe: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
    device: str = "cpu",
):
    probe = probe.to(device=device, dtype=torch.float32).eval()
    dl = DataLoader(TensorDataset(features, labels), batch_size=128, shuffle=False)
    preds, targs = [], []
    for X, y in dl:
        X = X.to(device=device, dtype=torch.float32)
        out = probe(X)
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
