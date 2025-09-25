import gc

import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers.models.electra.modeling_electra import ElectraClassificationHead
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
from torch.utils.data import WeightedRandomSampler
import lightning as L
import os


device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 256

generator = torch.Generator().manual_seed(42)


class EntropyClassifierHead(nn.Module):
    def __init__(self, original_lm_head: nn.Linear, config, num_classes, lam: float = 100.0, load_weights: str = 'cls'):
        super().__init__()
        if isinstance(original_lm_head, torch.nn.Linear):
            self.head = torch.nn.Linear(
                original_lm_head.in_features, original_lm_head.out_features,
                bias=(original_lm_head.bias is not None)
            )
            self.need_to_add_dim = False
        elif isinstance(original_lm_head, (RobertaClassificationHead, ElectraClassificationHead)):
            self.head = type(original_lm_head)(config)
            self.need_to_add_dim = True

        if load_weights == 'cls_head':
            self.head.load_state_dict(original_lm_head.state_dict())
        elif load_weights == 'cls_random':
            pass
        elif load_weights == 'linear_random':
            self.need_to_add_dim = False
            self.head = torch.nn.Linear(
                config.hidden_size, 1
            )
        else:
            raise ValueError(f'Load weight unknown {load_weights}')
        self.max_entropy = torch.nn.Parameter(torch.log2(torch.tensor(num_classes)), requires_grad=False)
        self.load_weights = load_weights

    def forward(self, cls_token):
        if self.need_to_add_dim:
            cls_token = cls_token[:, None, :]
        x = self.head(cls_token)
        if self.load_weights in ('cls_head', 'cls_random'):
            p = torch.softmax(x, dim=1)
            entropy = (-p * torch.log2(p + 1e-8)).sum(dim=-1) / self.max_entropy
            return torch.clamp(entropy, min=1e-9, max=1 - 1e-9)
        elif self.load_weights == 'linear_random':
            return torch.sigmoid(x[:, 0])


class SmoothMaxClassifierHead(nn.Module):
    def __init__(self, original_lm_head: nn.Linear, config, num_classes, lam: float = 100.0, load_weights: str = 'cls'):
        super().__init__()
        if isinstance(original_lm_head, torch.nn.Linear):
            self.head = torch.nn.Linear(
                original_lm_head.in_features, original_lm_head.out_features,
                bias=original_lm_head.bias
            )
            self.need_to_add_dim = False
        elif isinstance(original_lm_head, (RobertaClassificationHead, ElectraClassificationHead)):
            self.head = type(original_lm_head)(config)
            self.need_to_add_dim = True

        if load_weights == 'cls_head':
            self.head.load_state_dict(original_lm_head.state_dict())
        elif load_weights == 'cls_random':
            pass
        elif load_weights == 'linear_random':
            self.need_to_add_dim = False
            self.head = torch.nn.Linear(
                config.hidden_size, 1
            )
        else:
            raise ValueError(f'Load weight unknown {load_weights}')
        self.lam = torch.nn.Parameter(torch.tensor(lam), requires_grad=False)
        self.scaler = torch.nn.Parameter(torch.tensor(
            1 - 1 / num_classes
        ), requires_grad=False)
        self.load_weights = load_weights

    def forward(self, cls_token):
        if self.need_to_add_dim:
            cls_token = cls_token[:, None, :]
        x = self.head(cls_token)
        if self.load_weights in ('cls_head', 'cls_random'):
            p = torch.softmax(x, dim=1)
            smooth_max = (1 / self.lam) * torch.logsumexp(self.lam * p, dim=1)
            f_smooth = 1 - smooth_max
            eps = 1e-8
            return torch.clamp(f_smooth / self.scaler, min=eps, max=1 - eps)
        elif self.load_weights == 'linear_random':
            return torch.sigmoid(x[:, 0])


def evaluate_smooth_head(smooth_head, val_features, val_labels, device):
    smooth_head.eval().to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        feats = val_features.to(device, dtype=torch.float32)
        smooth_val = smooth_head(feats)
        all_preds.extend(smooth_val.float().cpu().numpy().tolist())
        all_labels.extend(val_labels.cpu().numpy().tolist())

    auc = roc_auc_score(np.array(all_labels), np.array(all_preds))
    return auc, np.array(all_preds), np.array(all_labels)


class AdueModel(L.LightningModule):
    def __init__(self, head, lr, reg_alpha, l2sp_alpha, mode='max'):
        super().__init__()
        self.save_hyperparameters(ignore=['head'], logger=False)
        self.head = head
        self.criterion = nn.BCELoss()

        self.initial_params = {
            name: p.detach().clone().requires_grad_(False)
            for name, p in head.named_parameters()
            if p.requires_grad
        }

        self.train_roc_auc = torchmetrics.AUROC(task='binary')
        self.val_roc_auc = torchmetrics.AUROC(task='binary')

        self.train_loss = torchmetrics.MeanMetric()

        self.train_bce_loss = torchmetrics.MeanMetric()
        self.train_reg_loss = torchmetrics.MeanMetric()
        self.train_reg_loss_with_alpha = torchmetrics.MeanMetric()

        self.train_l2sp_loss = torchmetrics.MeanMetric()
        self.train_l2sp_loss_with_alpha = torchmetrics.MeanMetric()

        self.valid_loss = torchmetrics.MeanMetric()
        self.valid_bce_loss = torchmetrics.MeanMetric()

        self.valid_reg_loss = torchmetrics.MeanMetric()
        self.valid_reg_loss_with_alpha = torchmetrics.MeanMetric()

        self.valid_l2sp_loss = torchmetrics.MeanMetric()
        self.valid_l2sp_loss_with_alpha = torchmetrics.MeanMetric()

        if mode == 'max':
            self.best_metric = torchmetrics.MaxMetric()
        elif mode == 'min':
            self.best_metric = torchmetrics.MinMetric()

    def on_fit_start(self):
        self.initial_params = {
            n: p.detach().clone().to(self.device).requires_grad_(False)
            for n, p in self.initial_params.items()
        }

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        feats, errors, base_probs = batch
        outputs = self.head(feats)
        loss_main = self.criterion(outputs, errors)

        reg_loss = torch.mean((outputs - base_probs) ** 2)
        l2sp_loss = torch.tensor(0.0, device=self.device)
        for name, param in self.head.named_parameters():
            if param.requires_grad:
                l2sp_loss += torch.sum((param - self.initial_params[name]) ** 2)

        reg_loss_with_alpha = self.hparams.reg_alpha * reg_loss

        self.train_reg_loss(reg_loss)
        self.log('train/reg_loss', self.train_reg_loss, on_step=True, on_epoch=True)
        self.train_reg_loss_with_alpha(reg_loss_with_alpha)
        self.log('train/reg_loss_with_alpha', self.train_reg_loss_with_alpha, on_step=True, on_epoch=True)

        l2sp_loss_with_alpha = self.hparams.l2sp_alpha * l2sp_loss

        self.train_l2sp_loss(l2sp_loss)
        self.log('train/l2sp_loss', self.train_l2sp_loss, on_step=True, on_epoch=True)

        self.train_l2sp_loss_with_alpha(l2sp_loss_with_alpha)
        self.log('train/l2sp_loss_with_alpha', self.train_l2sp_loss_with_alpha, on_step=True, on_epoch=True)
        loss = loss_main + reg_loss_with_alpha + l2sp_loss_with_alpha

        self.train_bce_loss(loss_main)
        self.log('train/bce_loss', self.train_bce_loss, on_step=True, on_epoch=True)

        self.train_loss(loss)
        self.log('train/loss', self.train_loss, on_step=True, on_epoch=True)

        self.train_roc_auc(outputs, errors)
        self.log('train/error_roc_auc', self.train_roc_auc, on_step=True, on_epoch=True, prog_bar=True)

        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", cur_lr, prog_bar=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        feats, errors, base_probs = batch
        outputs = self.head(feats)
        loss_main = self.criterion(outputs, errors)

        reg_loss = torch.mean((outputs - base_probs) ** 2)
        l2sp_loss = 0
        for name, param in self.head.named_parameters():
            if param.requires_grad:
                l2sp_loss += torch.sum((param - self.initial_params[name]) ** 2)

        reg_loss_with_alpha = self.hparams.reg_alpha * reg_loss

        self.valid_reg_loss(reg_loss)
        self.log('valid/reg_loss', self.valid_reg_loss, on_step=True, on_epoch=True)
        self.valid_reg_loss_with_alpha(reg_loss_with_alpha)
        self.log('valid/reg_loss_with_alpha', self.valid_reg_loss_with_alpha, on_step=True, on_epoch=True)

        l2sp_loss_with_alpha = self.hparams.l2sp_alpha * l2sp_loss

        self.valid_l2sp_loss(l2sp_loss)
        self.log('valid/l2sp_loss', self.valid_l2sp_loss, on_step=True, on_epoch=True)

        self.valid_l2sp_loss_with_alpha(l2sp_loss_with_alpha)
        self.log('valid/l2sp_loss_with_alpha', self.valid_l2sp_loss_with_alpha, on_step=True, on_epoch=True)
        loss = loss_main + reg_loss_with_alpha + l2sp_loss_with_alpha
        self.valid_bce_loss(loss_main)
        self.log('valid/bce_loss', self.valid_bce_loss, on_step=True, on_epoch=True)

        self.valid_loss(loss)
        self.log('valid/loss', self.valid_loss, on_step=True, on_epoch=True)

        self.val_roc_auc(outputs, errors)
        self.log('val/error_roc_auc', self.val_roc_auc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        current_metric = self.val_roc_auc.compute()  # get current val acc
        prev_best_metric = self.best_metric.compute()
        self.best_metric(current_metric)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        if self.best_metric.compute() != prev_best_metric:
            self.log('val/error_roc_auc_best', self.val_roc_auc.compute(), prog_bar=True)
            self.log('train/error_roc_auc_best', self.train_roc_auc.compute(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        print('Steps', self.trainer.estimated_stepping_batches)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            anneal_strategy="cos",
            pct_start=0.1,
            div_factor=25.0,
            final_div_factor=10000.0,
        )
        return {
            'optimizer': optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }


def train_smooth_head_lightning(
        smooth_head,

        train_features,
        train_errors,
        train_base_pred,

        val_features,
        val_errors,
        val_base_pred,

        lr=3e-4,
        reg_alpha=0.1,
        l2sp_alpha=0.01,
        smooth_batch_size=batch_size,
        log_params=dict()
):
    dataset = torch.utils.data.TensorDataset(train_features, train_errors, train_base_pred)
    def create_balanced_sampler(targets):
        """
        Create a WeightedRandomSampler for binary classification imbalance

        Args:
            targets: torch.Tensor of binary labels (0 and 1)

        Returns:
            WeightedRandomSampler object
        """
        # Count samples for each class
        class_counts = torch.bincount(targets.long())

        # Calculate weights for each sample
        class_weights = 1. / class_counts.float()

        # Assign weight to each sample based on its class
        sample_weights = class_weights[targets.long()]

        # Create sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(targets),
            replacement=True
        )

        return sampler
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=smooth_batch_size,
        shuffle=False,
        generator=generator,
        num_workers=0,
        pin_memory=False,
        sampler=create_balanced_sampler(train_errors)
    )

    val_dataset = torch.utils.data.TensorDataset(val_features, val_errors, val_base_pred)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=smooth_batch_size,
        shuffle=False,
        generator=generator,
        num_workers=0,
        pin_memory=False,
    )

    model = AdueModel(
        smooth_head,
        lr=lr, reg_alpha=reg_alpha, l2sp_alpha=l2sp_alpha
    )
    monitor = 'val/error_roc_auc'
    mode = 'max'

    early_stopping_callback = L.pytorch.callbacks.EarlyStopping(
        monitor=monitor,
        patience=5, mode=mode
    )
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(monitor=monitor, mode=mode, save_weights_only=True)
    os.makedirs('mlflow', exist_ok=True)

    mlflow_logger = L.pytorch.loggers.MLFlowLogger(
        experiment_name=f'{log_params["model"]}_{log_params["dataset"]}',
        tracking_uri='sqlite:///mlflow/database.db'
    )
    mlflow_logger.log_hyperparams(log_params)

    trainer = L.Trainer(
        # precision='16-mixed',
        num_sanity_val_steps=0,
        callbacks=[
            early_stopping_callback,
            checkpoint_callback
        ],
        logger=[
            mlflow_logger
        ],
        log_every_n_steps=25,
        max_epochs=50,
        enable_progress_bar=True
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    mlflow_logger.log_hyperparams({'best_path': trainer.checkpoint_callback.best_model_path})
    best_model = AdueModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, head=smooth_head)
    os.remove(trainer.checkpoint_callback.best_model_path)
    mlflow_logger.finalize("success")

    del model
    del trainer
    gc.collect()

    return best_model.head.cpu()


def train_smooth_head(
        smooth_head,
        train_features,
        train_errors,
        train_max_probs,
        device,
        num_epochs=5,
        lr=3e-4,
        reg_alpha=0.1,
        l2sp_alpha=0.01,
        warmup_steps_ratio=0.1,
        smooth_batch_size=batch_size,
):
    dataset = torch.utils.data.TensorDataset(train_features, train_errors, train_max_probs)

    def create_balanced_sampler(targets):
        """
        Create a WeightedRandomSampler for binary classification imbalance

        Args:
            targets: torch.Tensor of binary labels (0 and 1)

        Returns:
            WeightedRandomSampler object
        """
        # Count samples for each class
        class_counts = torch.bincount(targets)

        # Calculate weights for each sample
        class_weights = 1. / class_counts.float()

        # Assign weight to each sample based on its class
        sample_weights = class_weights[targets]

        # Create sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(targets),
            replacement=True
        )

        return sampler

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=smooth_batch_size,
        shuffle=False,
        generator=generator,
        num_workers=0,
        pin_memory=False,
        sampler=create_balanced_sampler(train_errors),
    )

    initial_params = {
        name: p.detach().clone()
        for name, p in smooth_head.named_parameters()
        if p.requires_grad
    }

    optimizer = torch.optim.Adam(smooth_head.parameters(), lr=lr)
    criterion = nn.BCELoss()

    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_steps_ratio)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        anneal_strategy="cos",
        pct_start=warmup_steps / total_steps,
        div_factor=25.0,
        final_div_factor=10000.0,
    )

    smooth_head.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for feats, labs, max_probs in train_loader:
            feats, labs, max_probs = (
                feats.to(device),
                labs.to(device),
                max_probs.to(device),
            )

            optimizer.zero_grad()
            outputs = smooth_head(feats.to(dtype=torch.float32))
            print(labs.mean())
            try:
                loss_main = criterion(outputs.float(), labs)
            except:
                print(outputs.float())
                print(outputs.max(), outputs.min())
                1 / 0

            reg_loss = torch.mean((outputs - max_probs) ** 2)

            l2sp_loss = 0.0
            for name, param in smooth_head.named_parameters():
                if param.requires_grad:
                    l2sp_loss += torch.sum((param - initial_params[name]) ** 2)

            reg_loss_with_alpha = reg_alpha * reg_loss
            # print(f'Reg loss: {reg_loss_with_alpha} before mult: {reg_loss}')
            l2sp_loss_with_alpha = l2sp_alpha * l2sp_loss
            # print(f'L2SP loss: {l2sp_loss_with_alpha} before mult: {l2sp_loss}')
            loss = loss_main + reg_loss_with_alpha + l2sp_loss_with_alpha
            # print(f'Bce loss: {loss_main}')

            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * feats.size(0)

    return smooth_head


def search_hyperparameters_lightning(
        model_cfg, original_head,
        train_features, train_logits, train_original_target,
        val_features, val_logits, val_original_targets,
        test_features, test_logits, test_original_targets,
        device, smooth_batch_size, adapter_name, dataset_name, cfg
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_val_auc = -1
    trial = 0
    grid = cfg.grid
    num_classes = test_logits.size(1)
    head_types = cfg.grid.head_type

    total_experiments = (
            len(grid.reg_alpha_candidates) *
            len(grid.lambda_candidates) *
            len(grid.learning_rates) *
            len(grid.reg_l2sp_candidates)
    )
    best_state = None
    all_states = []

    test_errors = test_logits.argmax(dim=-1) != test_original_targets
    base_eu_metrics = calculate_base_metrics(test_logits, test_errors)

    test_errors = test_original_targets != test_logits.argmax(dim=-1)
    val_errors = val_original_targets != val_logits.argmax(dim=-1)

    test_acc = (
            test_logits.argmax(dim=1) == test_original_targets
    ).float().mean().item()

    test_max_probs = 1 - torch.softmax(test_logits, dim=-1).amax(dim=-1)
    test_max_prob_auc = roc_auc_score(test_errors, test_max_probs)

    test_max_prob_auc_v2 = roc_auc_score(
        (test_logits.argmax(dim=1) != test_original_targets),
        1 - (torch.softmax(test_logits, dim=1).amax(dim=1))
    )
    assert np.allclose(test_max_prob_auc, test_max_prob_auc_v2)

    val_acc = (val_logits.argmax(dim=-1) == val_original_targets).float().mean().item()
    val_max_probs = 1 - torch.softmax(val_logits, dim=-1).amax(dim=-1)

    max_prob_val = roc_auc_score(val_errors, val_max_probs)
    max_prob_val_v2 = roc_auc_score(
        val_logits.argmax(dim=-1) != val_original_targets,
        1 - torch.softmax(val_logits, dim=1).amax(dim=-1)
    )
    assert np.allclose(max_prob_val, max_prob_val_v2)

    with tqdm(total=total_experiments, desc="Hyperparameter search") as pbar_main:
        for head_type in head_types:
            for reg_alpha in grid.reg_alpha_candidates:
                for lam in grid.lambda_candidates:
                    for lr in grid.learning_rates:
                        for l2sp_alpha in grid.reg_l2sp_candidates:
                            for load_weights in grid.load_weights:
                                pbar_main.set_postfix({
                                    **base_eu_metrics,
                                    'best_auc': f"{best_val_auc:.4f}",
                                    'trial': trial + 1,
                                    'reg_α': reg_alpha,
                                    'λ': lam,
                                    'lr': lr,
                                    'l2sp_α': l2sp_alpha,
                                    'load_weights': load_weights,
                                })
                                train_errors, train_base_pred = get_data_for_training(
                                    train_logits, train_original_target, head_type
                                )
                                test_errors, test_base_pred = get_data_for_training(
                                    test_logits, test_original_targets, head_type
                                )
                                val_errors, val_base_pred = get_data_for_training(
                                    val_logits, val_original_targets, head_type
                                )
                                base_eu_metrics = calculate_base_metrics(val_logits, val_errors)

                                if head_type == 'sr':
                                    candidate_head = SmoothMaxClassifierHead(
                                        original_head, config=model_cfg,
                                        num_classes=num_classes, lam=lam,
                                        load_weights=load_weights,
                                    ).to(device=device, dtype=torch.float32)
                                elif head_type == 'entropy':
                                    candidate_head = EntropyClassifierHead(
                                        original_head, config=model_cfg,
                                        num_classes=num_classes, lam=lam,
                                        load_weights=load_weights,
                                    ).to(device=device, dtype=torch.float32)
                                else:
                                    raise ValueError(f'Wrong head type {head_type}')

                                candidate_head = train_smooth_head_lightning(
                                    candidate_head,

                                    train_features,
                                    train_errors,
                                    train_base_pred,

                                    val_features=val_features,
                                    val_errors=val_errors,
                                    val_base_pred=val_base_pred,

                                    lr=lr,
                                    reg_alpha=reg_alpha,
                                    l2sp_alpha=l2sp_alpha,
                                    smooth_batch_size=smooth_batch_size,
                                    log_params={
                                        'model': model_cfg._name_or_path,
                                        'adapter': adapter_name,
                                        'train_on_dataset': cfg.train_on_dataset,
                                        'dataset': dataset_name,
                                        'load_weights' : load_weights,
                                        'trial_number': trial,
                                        'grid': cfg.grid.name,
                                        'seed': cfg.seed,
                                        'head_type': head_type,
                                        'lr': lr,
                                        'reg_alpha': reg_alpha,
                                        'l2sp_alpha': l2sp_alpha,
                                    }
                                )
                                val_auc, predicted_val_pred, predicted_val_target = evaluate_smooth_head(
                                    candidate_head, val_features, val_errors, device
                                )

                                if val_auc > best_val_auc:
                                    best_lr = lr
                                    best_lam = lam
                                    best_reg_alpha = reg_alpha
                                    best_l2sp_alpha = l2sp_alpha
                                    test_auc, predicted_test_pred, predicted_test_target = evaluate_smooth_head(
                                        candidate_head, test_features, test_errors, device
                                    )

                                    current_state = {
                                        'model': model_cfg._name_or_path,
                                        'adapter': adapter_name,
                                        'dataset': dataset_name,
                                        'train_on_dataset': cfg.train_on_dataset,
                                        'seed': cfg.seed,
                                        'grid': cfg.grid.name,
                                        'lam': best_lam,
                                        'reg_alpha': best_reg_alpha,
                                        'l2sp_alpha': best_l2sp_alpha,
                                        'load_weights': load_weights,
                                        'lr': best_lr,
                                        'head_type': head_type,
                                        'trial_number': trial,

                                        'valid-acc': val_acc,
                                        'valid-base-max-prob-roc-auc': max_prob_val,
                                        'valid-fine-tune-max-prob-roc-auc': val_auc,

                                        'test-acc': test_acc,
                                        'test-base-max-prob-roc-auc': test_max_prob_auc,
                                        'test-fine-tune-max-prob-roc-auc': test_auc,
                                    }
                                    metric_df = pd.DataFrame([current_state])
                                    result_dict = {
                                        'metric_df': metric_df,
                                        'original_model_scores': {
                                            'logits': np.array(test_logits),
                                            'targets': np.array(test_original_targets),
                                        },
                                        'adue_uncertainty_head_scores': {
                                            'logits': np.array(predicted_test_pred),
                                            'targets': np.array(predicted_test_target)
                                        },
                                        'layer_state': candidate_head.cpu().state_dict()
                                    }

                                    best_state = result_dict
                                    best_val_auc = val_auc

                                # all_states.append(result_dict)
                                trial += 1
                                pbar_main.update(1)
    return best_state, all_states


def search_hyperparameters(
        model_cfg, original_head, train_features, train_labels, train_max_probs,
        val_features, val_labels, val_max_probs, val_logits, val_original_targets,
        test_features, test_labels, test_max_probs, test_logits, test_original_targets,
        device, smooth_batch_size, adapter_name, dataset_name, cfg
):
    best_val_auc = -1
    trial = 0
    grid = cfg.grid
    num_classes = test_logits.size(1)
    head_type = cfg.grid.get('head_type', 'entropy')

    total_experiments = (
            len(grid.reg_alpha_candidates) *
            len(grid.lambda_candidates) *
            len(grid.learning_rates) *
            len(grid.reg_l2sp_candidates) *
            len(grid.epoch_candidates)
    )
    best_state = None
    all_states = []

    test_errors = test_original_targets != test_logits.argmax(dim=-1)

    test_acc = (
            test_logits.argmax(dim=1) == test_original_targets
    ).float().mean().item()

    test_max_prob_auc = roc_auc_score(test_errors, test_max_probs)
    test_max_prob_auc_v2 = roc_auc_score(
        (test_logits.argmax(dim=1) != test_original_targets),
        1 - (torch.softmax(test_logits, dim=1).amax(dim=1))
    )
    assert np.allclose(test_max_prob_auc, test_max_prob_auc_v2)
    assert torch.all(test_errors == test_labels)

    val_acc = (val_logits.argmax(dim=-1) == val_original_targets).float().mean().item()
    max_prob_val = roc_auc_score(val_labels, val_max_probs)
    max_prob_val_v2 = roc_auc_score(
        val_logits.argmax(dim=-1) != val_original_targets,
        1 - torch.softmax(val_logits, dim=1).amax(dim=-1)
    )
    assert np.allclose(max_prob_val, max_prob_val_v2)

    with tqdm(total=total_experiments, desc="Hyperparameter search") as pbar_main:
        for reg_alpha in grid.reg_alpha_candidates:
            for lam in grid.lambda_candidates:
                for lr in grid.learning_rates:
                    for l2sp_alpha in grid.reg_l2sp_candidates:
                        for ep in grid.epoch_candidates:
                            for load_weights in grid.load_weights:
                                pbar_main.set_postfix({
                                    'trial': trial + 1,
                                    'reg_α': reg_alpha,
                                    'λ': lam,
                                    'lr': lr,
                                    'l2sp_α': l2sp_alpha,
                                    'epochs': ep,
                                    'load_weights': load_weights,
                                    # 'base_sr_response_auc': f"{test_max_prob_auc_v2:.4f}",
                                    # 'best_auc': f"{best_val_auc:.4f}"
                                })
                                if head_type == 'sr':
                                    candidate_head = SmoothMaxClassifierHead(
                                        original_head, config=model_cfg,
                                        num_classes=num_classes, lam=lam,
                                        load_weights=load_weights,
                                    ).to(device=device, dtype=torch.float32)
                                elif head_type == 'entropy':
                                    candidate_head = EntropyClassifierHead(
                                        original_head, config=model_cfg,
                                        num_classes=num_classes, lam=lam,
                                        load_weights=load_weights,
                                    ).to(device=device, dtype=torch.float32)
                                else:
                                    raise ValueError(f'Wrong head type {head_type}')

                                candidate_head = train_smooth_head(
                                    candidate_head,
                                    train_features,
                                    train_labels,
                                    train_max_probs,
                                    device,
                                    num_epochs=ep,
                                    lr=lr,
                                    reg_alpha=reg_alpha,
                                    l2sp_alpha=l2sp_alpha,
                                    smooth_batch_size=smooth_batch_size,
                                )
                                val_auc, predicted_val_pred, predicted_val_target = evaluate_smooth_head(
                                    candidate_head, val_features, val_labels, device
                                )

                                best_lr = lr
                                best_epochs = ep
                                best_lam = lam
                                best_reg_alpha = reg_alpha
                                best_l2sp_alpha = l2sp_alpha
                                test_auc, predicted_test_pred, predicted_test_target = evaluate_smooth_head(
                                    candidate_head, test_features, test_labels, device
                                )

                                current_state = {
                                    'model': model_cfg._name_or_path,
                                    'adapter': adapter_name,
                                    'dataset': dataset_name,
                                    'train_on_dataset': cfg.train_on_dataset,
                                    'seed': cfg.seed,
                                    'grid': cfg.grid.name,
                                    'lam': best_lam,
                                    'reg_alpha': best_reg_alpha,
                                    'l2sp_alpha': best_l2sp_alpha,
                                    'load_weights': load_weights,
                                    'lr': best_lr,
                                    'epoch': best_epochs,
                                    'trial_number': trial,

                                    'valid-acc': val_acc,
                                    'valid-base-max-prob-roc-auc': max_prob_val,
                                    'valid-fine-tune-max-prob-roc-auc': val_auc,

                                    'test-acc': test_acc,
                                    'test-base-max-prob-roc-auc': test_max_prob_auc,
                                    'test-base-max-prob-roc-auc-v2': test_max_prob_auc_v2,
                                    'test-fine-tune-max-prob-roc-auc': test_auc,
                                }
                                metric_df = pd.DataFrame([current_state])
                                result_dict = {
                                    'metric_df': metric_df,
                                    'original_model_scores': {
                                        'logits': np.array(test_logits),
                                        'targets': np.array(test_original_targets),
                                    },
                                    'adue_uncertainty_head_scores': {
                                        'logits': np.array(predicted_test_pred),
                                        'targets': np.array(predicted_test_target)
                                    },
                                    'layer_state': candidate_head.cpu().state_dict()
                                }
                                if val_auc > best_val_auc:
                                    best_state = result_dict
                                    best_val_auc = val_auc

                                all_states.append(result_dict)
                                trial += 1
                                pbar_main.update(1)
    return best_state, all_states


def get_data_for_training(logits, original_targets, head_type):
    errors = logits.argmax(dim=-1) != original_targets
    if head_type == 'sr':
        num_classes = logits.shape[1]
        scale_factor = 1 - (1 / num_classes)
        p = torch.softmax(logits, dim=-1)
        base_prediction = (1 - p.amax(dim=1)) / scale_factor
    elif head_type == 'entropy':
        num_classes = logits.shape[1]
        scale_factor = torch.log2(torch.tensor(num_classes))
        p = torch.softmax(logits, dim=-1)
        base_prediction = ((-p * torch.log2(p + 1e-6)).sum(dim=1) / scale_factor)
    else:
        raise ValueError(f'head_type is {head_type}')

    return errors.float(), base_prediction


def calculate_base_metrics(logits, errors):
    p = torch.softmax(logits, dim=-1)
    entropy = (-p * torch.log2(p + 1e-8)).sum(dim=1)
    sr = 1 - p.amax(dim=-1)

    return {
        'roc_auc_entropy': f"{roc_auc_score(errors, entropy):.4f}",
        'roc_auc_sr': f"{roc_auc_score(errors, sr):.4f}",
    }


def search_hyperparameters_v2(
        model_cfg, original_head,
        train_features, train_logits, train_original_target,
        val_features, val_logits, val_original_targets,
        test_features, test_logits, test_original_targets,
        device, smooth_batch_size, adapter_name, dataset_name, cfg
):
    best_val_auc = -1
    trial = 0
    grid = cfg.grid
    num_classes = test_logits.size(1)
    head_types = cfg.grid.head_type

    total_experiments = (
            len(grid.reg_alpha_candidates) *
            len(grid.lambda_candidates) *
            len(grid.learning_rates) *
            len(grid.reg_l2sp_candidates) *
            len(grid.epoch_candidates)
    )
    best_state = None
    all_states = []

    test_errors = test_logits.amax(dim=1) != test_original_targets
    base_eu_metrics = calculate_base_metrics(test_logits, test_errors)

    test_errors = test_original_targets != test_logits.argmax(dim=-1)
    val_errors = val_original_targets != val_logits.argmax(dim=-1)

    test_acc = (
            test_logits.argmax(dim=1) == test_original_targets
    ).float().mean().item()

    test_max_probs = 1 - torch.softmax(test_logits, dim=-1).amax(dim=-1)
    test_max_prob_auc = roc_auc_score(test_errors, test_max_probs)

    test_max_prob_auc_v2 = roc_auc_score(
        (test_logits.argmax(dim=1) != test_original_targets),
        1 - (torch.softmax(test_logits, dim=1).amax(dim=1))
    )
    assert np.allclose(test_max_prob_auc, test_max_prob_auc_v2)

    val_acc = (val_logits.argmax(dim=-1) == val_original_targets).float().mean().item()
    val_max_probs = 1 - torch.softmax(val_logits, dim=-1).amax(dim=-1)

    max_prob_val = roc_auc_score(val_errors, val_max_probs)
    max_prob_val_v2 = roc_auc_score(
        val_logits.argmax(dim=-1) != val_original_targets,
        1 - torch.softmax(val_logits, dim=1).amax(dim=-1)
    )
    assert np.allclose(max_prob_val, max_prob_val_v2)

    with tqdm(total=total_experiments, desc="Hyperparameter search") as pbar_main:
        for head_type in head_types:
            for reg_alpha in grid.reg_alpha_candidates:
                for lam in grid.lambda_candidates:
                    for lr in grid.learning_rates:
                        for l2sp_alpha in grid.reg_l2sp_candidates:
                            for ep in grid.epoch_candidates:
                                for load_weights in grid.load_weights:
                                    pbar_main.set_postfix({
                                        **base_eu_metrics,
                                        'best_auc': f"{best_val_auc:.4f}",
                                        'trial': trial + 1,
                                        'reg_α': reg_alpha,
                                        'λ': lam,
                                        'lr': lr,
                                        'l2sp_α': l2sp_alpha,
                                        'epochs': ep,
                                        'load_weights': load_weights,
                                    })
                                    train_errors, train_base_pred = get_data_for_training(
                                        train_logits, train_original_target, head_type
                                    )
                                    test_errors, test_base_pred = get_data_for_training(
                                        test_logits, test_original_targets, head_type
                                    )
                                    val_errors, val_base_pred = get_data_for_training(
                                        val_logits, val_original_targets, head_type
                                    )
                                    base_eu_metrics = calculate_base_metrics(val_logits, val_errors)

                                    if head_type == 'sr':
                                        candidate_head = SmoothMaxClassifierHead(
                                            original_head, config=model_cfg,
                                            num_classes=num_classes, lam=lam,
                                            load_weights=load_weights,
                                        ).to(device=device, dtype=torch.float32)
                                    elif head_type == 'entropy':
                                        candidate_head = EntropyClassifierHead(
                                            original_head, config=model_cfg,
                                            num_classes=num_classes, lam=lam,
                                            load_weights=load_weights,
                                        ).to(device=device, dtype=torch.float32)
                                    else:
                                        raise ValueError(f'Wrong head type {head_type}')

                                    candidate_head = train_smooth_head(
                                        candidate_head,
                                        train_features,
                                        train_errors,
                                        train_base_pred,
                                        device,
                                        num_epochs=ep,
                                        lr=lr,
                                        reg_alpha=reg_alpha,
                                        l2sp_alpha=l2sp_alpha,
                                        smooth_batch_size=smooth_batch_size,
                                    )
                                    val_auc, predicted_val_pred, predicted_val_target = evaluate_smooth_head(
                                        candidate_head, val_features, val_errors, device
                                    )

                                    best_lr = lr
                                    best_epochs = ep
                                    best_lam = lam
                                    best_reg_alpha = reg_alpha
                                    best_l2sp_alpha = l2sp_alpha
                                    test_auc, predicted_test_pred, predicted_test_target = evaluate_smooth_head(
                                        candidate_head, test_features, test_errors, device
                                    )

                                    current_state = {
                                        'model': model_cfg._name_or_path,
                                        'adapter': adapter_name,
                                        'dataset': dataset_name,
                                        'train_on_dataset': cfg.train_on_dataset,
                                        'seed': cfg.seed,
                                        'grid': cfg.grid.name,
                                        'lam': best_lam,
                                        'reg_alpha': best_reg_alpha,
                                        'l2sp_alpha': best_l2sp_alpha,
                                        'load_weights': load_weights,
                                        'lr': best_lr,
                                        'head_type': head_type,
                                        'epoch': best_epochs,
                                        'trial_number': trial,

                                        'valid-acc': val_acc,
                                        'valid-base-max-prob-roc-auc': max_prob_val,
                                        'valid-fine-tune-max-prob-roc-auc': val_auc,

                                        'test-acc': test_acc,
                                        'test-base-max-prob-roc-auc': test_max_prob_auc,
                                        'test-fine-tune-max-prob-roc-auc': test_auc,
                                    }
                                    metric_df = pd.DataFrame([current_state])
                                    result_dict = {
                                        'metric_df': metric_df,
                                        'original_model_scores': {
                                            'logits': np.array(test_logits),
                                            'targets': np.array(test_original_targets),
                                        },
                                        'adue_uncertainty_head_scores': {
                                            'logits': np.array(predicted_test_pred),
                                            'targets': np.array(predicted_test_target)
                                        },
                                        'layer_state': candidate_head.cpu().state_dict()
                                    }
                                    if val_auc > best_val_auc:
                                        best_state = result_dict
                                        best_val_auc = val_auc

                                    all_states.append(result_dict)
                                    trial += 1
                                    pbar_main.update(1)
    return best_state, all_states
