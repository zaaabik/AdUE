import hydra
import pickle
import pandas as pd
from omegaconf import DictConfig

# pylint: disable=wrong-import-position
import rootutils

import os
import random

import numpy as np
import peft
import torch
import torch.nn as nn
import transformers
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.models.electra.modeling_electra import ElectraClassificationHead
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

generator = torch.Generator().manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from src.uncertainty.estimators import max_prob

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_device = torch.bfloat16
batch_size = 256


class SmoothMaxClassifierHead(nn.Module):
    def __init__(self, original_lm_head: nn.Linear, config, lam: float = 10.0, load_weights: str = 'cls'):
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

    def forward(self, cls_token):
        if self.need_to_add_dim:
            cls_token = cls_token[:, None, :]
        x = self.head(cls_token)
        p = torch.softmax(x, dim=1)
        smooth_max = (1 / self.lam) * torch.logsumexp(self.lam * p, dim=1)
        f_smooth = 1 - smooth_max
        return torch.clamp(f_smooth, min=1e-7, max=1 - 1e-7)


def extract_features(model, dataloader, device, pooling):
    model.eval()
    features_list, labels_list, maxprob_list, logits_list, original_targets_list = [], [], [], [], []
    with torch.autocast(device_type=device):
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch, output_hidden_states=True)
                cls_token = pooling(
                    out.hidden_states,
                    input_ids=batch['input_ids'], model=model
                )

                preds = out.logits.argmax(dim=1)
                errors = (preds != batch["labels"]).float()
                features_list.append(cls_token.cpu())
                labels_list.append(errors.cpu())
                logits_list.append(out.logits.cpu())
                original_targets_list.append(batch["labels"].cpu())

                probs = torch.softmax(out.logits, dim=-1)
                maxprob_np = max_prob(probs[None].float().cpu().numpy())
                maxprob_tensor = torch.from_numpy(maxprob_np).squeeze(0)
                maxprob_list.append(maxprob_tensor)

    X = torch.cat(features_list, dim=0)
    Y = torch.cat(labels_list, dim=0)
    logits = torch.cat(logits_list, dim=0)
    max_probs_all = torch.cat(maxprob_list, dim=0)
    original_targets = torch.cat(original_targets_list, dim=0)

    return X, Y, max_probs_all, logits, original_targets


def train_smooth_head(
        smooth_head,
        train_features,
        train_labels,
        train_max_probs,
        device,
        num_epochs=5,
        lr=3e-4,
        reg_alpha=0.1,
        l2sp_alpha=0.01,
        warmup_steps_ratio=0.1,
        smooth_batch_size=batch_size,
):
    dataset = torch.utils.data.TensorDataset(train_features, train_labels, train_max_probs)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=smooth_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )

    initial_params = {
        name: p.clone().detach()
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
            with torch.autocast(device_type=device, dtype=torch.float32):
                outputs = smooth_head(feats)
            loss_main = criterion(outputs.float(), labs)
            reg_loss = torch.mean((outputs - max_probs) ** 2)

            l2sp_loss = 0.0
            for name, param in smooth_head.named_parameters():
                if param.requires_grad:
                    l2sp_loss += torch.sum((param - initial_params[name]) ** 2)

            loss = loss_main + reg_alpha * reg_loss + l2sp_alpha * l2sp_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * feats.size(0)

    return smooth_head


def evaluate_smooth_head(smooth_head, val_features, val_labels, device):
    smooth_head.eval()

    all_preds = []
    all_labels = []

    with torch.autocast(device_type=device, dtype=torch.float32):
        with torch.no_grad():
            feats = val_features.to(device)
            smooth_val = smooth_head(feats)
            all_preds.extend(smooth_val.float().cpu().numpy().tolist())
            all_labels.extend(val_labels.cpu().numpy().tolist())

    auc = roc_auc_score(np.array(all_labels), np.array(all_preds))
    return auc, np.array(all_preds), np.array(all_labels)


def search_hyperparameters(
        model, original_head, train_features, train_labels, train_max_probs,
        val_features, val_labels, val_max_probs, val_logits, val_original_targets,
        test_features, test_labels, test_max_probs, test_logits, test_original_targets,
        device, smooth_batch_size, adapter_name, dataset_name, cfg
):
    best_lr = None
    best_epochs = None
    best_lam = None
    best_reg_alpha = None
    best_val_auc = -1
    best_state = None
    best_l2sp_alpha = None
    trial = 0
    grid = cfg.grid

    total_experiments = (
        len(grid.reg_alpha_candidates) *
        len(grid.lambda_candidates) *
        len(grid.learning_rates) *
        len(grid.reg_l2sp_candidates) *
        len(grid.epoch_candidates)
    )

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
                                    'best_auc': f"{best_val_auc:.4f}"
                                })
                                candidate_head = SmoothMaxClassifierHead(
                                    original_head, config=model.config, lam=lam, load_weights=load_weights
                                ).to(device=device, dtype=torch_device)
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

                                if val_auc > best_val_auc:
                                    best_val_auc = val_auc
                                    best_lr = lr
                                    best_epochs = ep
                                    best_lam = lam
                                    best_reg_alpha = reg_alpha
                                    best_l2sp_alpha = l2sp_alpha
                                    # best_state = candidate_head.state_dict()
                                    best_load_weights = load_weights

                                    test_auc, predicted_test_pred, predicted_test_target = evaluate_smooth_head(
                                        candidate_head, test_features, test_labels, device
                                    )

                                    # Calculate accuracy and max_prob metrics for logging
                                    with torch.autocast(device_type=device):
                                        with torch.no_grad():
                                            # print(test_original_targets.shape, test_original_targets.shape)
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

                                    current_state = {
                                        'model': model.config._name_or_path,
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
                                    full_save_path = os.path.join(
                                        cfg.save_dir,
                                        f"{adapter_name}_{dataset_name}_smooth_classifier_one_cycle_lr_l2sp_probe.csv"
                                    )
                                    print(full_save_path)
                                    metric_df.to_csv(
                                        full_save_path,
                                        index=False
                                    )

                                    result_dict = {
                                        'metric_df': metric_df,
                                        'original_model_scores': {
                                            'logits': np.array(test_logits),
                                            'targets': np.array(test_original_targets),
                                        },
                                        'adue_uncertainty_head_scores': {
                                            'logits': np.array(predicted_test_pred),
                                            'targets': np.array(predicted_test_target)
                                        }
                                    }

                                    with open(
                                            os.path.join(
                                                cfg.save_dir,
                                                f"{adapter_name}_{dataset_name}_smooth_classifier_one_cycle_lr_l2sp_probe.pkl"
                                            ), 'wb'
                                    ) as f:
                                        pickle.dump(result_dict, f)
                                trial += 1
                                pbar_main.update(1)


def train(cfg):
    dataset = hydra.utils.instantiate(cfg.data.dataset)
    dataset_name = dataset.name()
    num_classes = dataset.num_classes()
    dataset = dataset.load()
    cfg.model.num_labels = num_classes
    os.makedirs(cfg.save_dir, exist_ok=True)

    adapter_path = cfg.adapter.path
    model = peft.AutoPeftModelForSequenceClassification.from_pretrained(
        adapter_path,
        **hydra.utils.instantiate(cfg.model)
    )

    adapter_name = model.peft_config['default'].__class__.__name__
    model = model.eval().merge_and_unload().to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model.config._name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=cfg.data.max_length)

    if cfg.train_on_dataset == 'train':
        train_split = (
            dataset["train"]
            .map(tokenize_function, batched=False, load_from_cache_file=True)
            .remove_columns("text")
        )
        val_split = (
            dataset["validation"]
            .map(tokenize_function, batched=False, load_from_cache_file=True)
            .remove_columns("text")
        )
        test = (
            dataset["test"]
            .map(tokenize_function, batched=False, load_from_cache_file=True)
            .remove_columns("text")
        )
    elif cfg.train_on_dataset == 'valid':
        valid = (
            dataset["validation"]
            .map(tokenize_function, batched=False, load_from_cache_file=True)
            .remove_columns("text")
        )
        test = (
            dataset["test"]
            .map(tokenize_function, batched=False, load_from_cache_file=True)
            .remove_columns("text")
        )
        split = valid.train_test_split(test_size=0.15, seed=42)
        train_split = split["train"]
        val_split = split["test"]
    elif cfg.train_on_dataset == 'valid_30_percent':
        valid = (
            dataset["validation"]
            .map(tokenize_function, batched=False, load_from_cache_file=True)
            .remove_columns("text")
        )
        test = (
            dataset["test"]
            .map(tokenize_function, batched=False, load_from_cache_file=True)
            .remove_columns("text")
        )
        split = valid.train_test_split(test_size=0.30, seed=42)
        train_split = split["train"]
        val_split = split["test"]

    collate_fn = transformers.DataCollatorWithPadding(
        return_tensors="pt", padding="longest", tokenizer=tokenizer
    )
    train_dataloader = DataLoader(
        train_split,
        batch_size=cfg.data.llm_batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_split,
        batch_size=cfg.data.llm_batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test,
        batch_size=cfg.data.llm_batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
    )

    original_head = model.__getattr__(cfg.classifier_name)
    pooling = hydra.utils.instantiate(cfg.pooling)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    # Extract features once for all datasets
    train_features, train_labels, train_max_probs, train_logits, train_original_target = extract_features(
        model, train_dataloader, device, pooling
    )
    val_features, val_labels, val_max_probs, val_logits, val_original_target = extract_features(
        model, val_dataloader, device, pooling
    )
    test_features, test_labels, test_max_probs, test_logits, test_original_targets = extract_features(
        model, test_dataloader, device, pooling
    )

    search_hyperparameters(
        model=model,
        original_head=original_head,
        train_features=train_features,
        train_labels=train_labels,
        train_max_probs=train_max_probs,
        val_features=val_features.float(),
        val_labels=val_labels,
        val_max_probs=val_max_probs.float(),
        val_logits=val_logits.float(),
        val_original_targets=val_original_target,
        test_features=test_features.float(),
        test_labels=test_labels.float(),
        test_max_probs=test_max_probs.float(),
        test_logits=test_logits.float(),
        test_original_targets=test_original_targets,
        device=device,
        smooth_batch_size=cfg.data.smooth_head_batch_size,
        adapter_name=adapter_name,
        dataset_name=dataset_name,
        cfg=cfg
    )

    torch.cuda.empty_cache()


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Main entry point for inference.

    Args:
        cfg: cfg: DictConfig configuration composed by Hydra.
    """

    train(cfg)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()