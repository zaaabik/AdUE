import pandas as pd

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers.models.electra.modeling_electra import ElectraClassificationHead
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 256

generator = torch.Generator().manual_seed(42)


class EntropyClassifierHead(nn.Module):
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
                config.hidden_size, config.num_labels
            )
        else:
            raise ValueError(f'Load weight unknown {load_weights}')
        self.max_entropy = torch.nn.Parameter(torch.log2(torch.tensor(num_classes)), requires_grad=False)

    def forward(self, cls_token):
        if self.need_to_add_dim:
            cls_token = cls_token[:, None, :]
        x = self.head(cls_token)
        p = torch.softmax(x, dim=1)
        entropy = (-p * torch.log2(p + 1e-8)).sum(dim=-1) / self.max_entropy
        return torch.clamp(entropy, min=1e-8, max=1-1e-8)


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
                config.hidden_size, config.num_labels
            )
        else:
            raise ValueError(f'Load weight unknown {load_weights}')
        self.lam = torch.nn.Parameter(torch.tensor(lam), requires_grad=False)
        self.scaler = torch.nn.Parameter(torch.tensor(
            1 - 1 / num_classes
        ), requires_grad=False)

    def forward(self, cls_token):
        if self.need_to_add_dim:
            cls_token = cls_token[:, None, :]
        x = self.head(cls_token)
        p = torch.softmax(x, dim=1)
        smooth_max = (1 / self.lam) * torch.logsumexp(self.lam * p, dim=1)
        f_smooth = 1 - smooth_max
        eps = 1e-8
        return torch.clamp(f_smooth / self.scaler, min=eps, max=1 - eps)


def evaluate_smooth_head(smooth_head, val_features, val_labels, device):
    smooth_head.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        feats = val_features.to(device, dtype=torch.float32)
        smooth_val = smooth_head(feats)
        all_preds.extend(smooth_val.float().cpu().numpy().tolist())
        all_labels.extend(val_labels.cpu().numpy().tolist())

    auc = roc_auc_score(np.array(all_labels), np.array(all_preds))
    return auc, np.array(all_preds), np.array(all_labels)


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
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=smooth_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
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

            try:
                loss_main = criterion(outputs.float(), labs)
            except:
                print(outputs.float())
                print(outputs.max(), outputs.min())
                1/0

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
                                    'base_sr_response_auc': f"{test_max_prob_auc_v2:.4f}",
                                    'best_auc': f"{best_val_auc:.4f}"
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
        base_prediction = 1 - ((-p * torch.log2(p + 1e-6)).sum(dim=1) / scale_factor)
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
    head_types = cfg.grid.get('head_type', ['sr'])


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
