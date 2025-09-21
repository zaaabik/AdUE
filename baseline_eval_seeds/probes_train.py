import os
import pickle
import random

import hydra
import lightning as L
import numpy as np
import pandas as pd
import peft
import rootutils
import torch
import transformers
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = rootutils.setup_root(
    __file__, indicator=".project-root", pythonpath=True
)

from baseline_eval_seeds.probes_lightning import (AttentionProbe, LinearProbe,
                                                  evaluate_probe,
                                                  save_probe_model,
                                                  train_probe_lightning)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@torch.no_grad()
def extract_cls_features(model, dataloader, pooling_cfg, layer_num, device):
    model.eval().to(device)
    features, labels, logits, length = [], [], [], []
    pooling = hydra.utils.instantiate(pooling_cfg, layer_number=layer_num)
    for batch in tqdm(dataloader, desc=f"Extract CLS L{layer_num}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        fwd_labels = batch["labels"]
        del batch["labels"]

        out = model(**batch, output_hidden_states=True)
        cls = pooling(out.hidden_states, batch['input_ids'], model)

        if len(out.logits.shape) == 2:
            logits.append(out.logits.cpu())
        elif len(out.logits.shape) == 3:
            hs = out.logits
            bs = out.logits.shape[0]

            non_pad_mask = (batch['input_ids'] != model.config.pad_token_id).to(
                device=out.logits.device, dtype=torch.int32)
            token_indices = torch.arange(batch['input_ids'].shape[-1], device=hs.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)

            pooled_logits = hs[torch.arange(bs, device=hs.device), last_non_pad_token]
            logits.append(pooled_logits.cpu())

        features.append(cls.cpu())
        labels.append(fwd_labels.cpu())
        # logits.append(out.logits.cpu())
        length.append(batch['attention_mask'].sum(axis=1))
    return torch.cat(features), torch.cat(labels), torch.cat(logits), torch.cat(length)


@torch.no_grad()
def extract_token_hidden_states(
    model, dataloader, layer_num, num_layers_add, max_length, device
):
    model.eval().to(device)
    hs_list, labels, logits, length = [], [], [], []
    for batch in tqdm(dataloader, desc=f"Extract HS L{layer_num} +{num_layers_add}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        fwd_labels = batch["labels"]
        del batch["labels"]
        out = model(**batch, output_hidden_states=True)
        total_layers = len(out.hidden_states)
        if layer_num < 0:
            base_idx = total_layers + layer_num
        else:
            base_idx = layer_num
        start = max(0, base_idx - num_layers_add)
        end = base_idx + 1
        hs = out.hidden_states[start:end]
        hs_combined = torch.cat(hs, dim=-1)
        length.append(batch['attention_mask'].sum(axis=1))

        hs_combined = hs_combined * batch['attention_mask'][:, :, None]
        padded = torch.zeros(
            (hs_combined.size(0), max_length, hs_combined.size(2)),
            device=hs_combined.device,
        )
        padded[:, : hs_combined.size(1), :] = hs_combined

        if len(out.logits.shape) == 2:
            logits.append(out.logits.cpu())
        elif len(out.logits.shape) == 3:
            hs = out.logits
            bs = out.logits.shape[0]

            non_pad_mask = (batch['input_ids'] != model.config.pad_token_id).to(
                device=out.logits.device, dtype=torch.int32)
            token_indices = torch.arange(batch['input_ids'].shape[-1], device=hs.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)

            pooled_logits = hs[torch.arange(bs, device=hs.device), last_non_pad_token]
            logits.append(pooled_logits.cpu())


        hs_list.append(padded.cpu())
        labels.append(fwd_labels.cpu())
    return torch.cat(hs_list), torch.cat(labels), torch.cat(logits), torch.cat(length)


def _build_probe_save_dir(
    project_root, model_name, dataset_name, seed, probe_type, layer, layers_plus=None
):
    base = os.path.join(
        project_root,
        "data",
        "models",
        f"{model_name}_probes_seeds",
        dataset_name,
        str(seed),
        probe_type,
    )
    if layers_plus is None:
        sub = f"layer_{layer}"
    else:
        sub = f"layer_{layer}_plus_{layers_plus}"
    return os.path.join(base, sub)


def search_hyperparameters_attention_pooling(model, train_loader, val_loader, test_loader, pooling_cfg, adapter_path, dataset_name, cfg):
    results = {}
    results['prediction'] = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for layer in cfg.probes.layers:
        for num_add in cfg.probes.attention.num_layers_add:
            tr_H, tr_y, tr_logits, tr_length = extract_token_hidden_states(
                model, train_loader, layer, num_add, cfg.data.max_length, device
            )
            va_H, va_y, va_logits, va_length = extract_token_hidden_states(
                model, val_loader, layer, num_add, cfg.data.max_length, device
            )

            errors_tr = (tr_logits.argmax(dim=1) != tr_y).float()
            errors_va = (va_logits.argmax(dim=1) != va_y).float()

            probe = AttentionProbe(tr_H.shape[-1])
            probe = train_probe_lightning(
                probe,
                tr_H,
                errors_tr,
                tr_length,
                va_H,
                errors_va,
                va_length,
                lr=cfg.probes.attention.lr,
                epochs=cfg.probes.attention.epochs,
                batch_size=cfg.probes.batch_size,
                accelerator=("gpu" if torch.cuda.is_available() else "cpu"),
                log_params={
                    'layer': layer,
                    'lr': cfg.probes.linear.lr,
                    'dataset': cfg.data.name,
                    'type': 'attention',
                    'num_add': num_add,
                },
            )

            save_dir = _build_probe_save_dir(
                project_root,
                cfg.model_name,
                dataset_name,
                cfg.seed,
                "attention",
                layer,
                layers_plus=num_add,
            )
            metadata = {
                "probe": "attention",
                "layer": int(layer),
                "layers_plus": int(num_add),
                "model_name": cfg.model_name,
                "dataset": dataset_name,
                "seed": int(cfg.seed),
                "train_on_dataset": cfg.train_on_dataset,
                "adapter_path": adapter_path,
                "hyperparams": {
                    "lr": float(cfg.probes.attention.lr),
                    "epochs": int(cfg.probes.attention.epochs),
                    "batch_size": int(cfg.probes.batch_size),
                },
            }
            save_probe_model(probe, save_dir, metadata)

            te_H, te_y, te_logits, te_length = extract_token_hidden_states(
                model, test_loader, layer, num_add, cfg.data.max_length, device
            )
            errors_te = (te_logits.argmax(dim=1) != te_y).float()
            auc, _, _, _ = evaluate_probe(
                probe,
                te_H,
                errors_te,
                te_length,
                device=("cuda" if torch.cuda.is_available() else "cpu"),
            )

            result_row = {
                "model_name": cfg.model_name,
                "dataset": dataset_name,
                "seed": int(cfg.seed),
                "train_on_dataset": cfg.train_on_dataset,
                "adapter_path": adapter_path,
                "probe": "attention",
                "layer": int(layer),
                "layers_plus": int(num_add),
                "roc_auc": float(auc),
                "lr": float(cfg.probes.attention.lr),
                "epochs": int(cfg.probes.attention.epochs),
                "batch_size": int(cfg.probes.batch_size),
            }
            os.makedirs(cfg.save_dir, exist_ok=True)
            csv_path = os.path.join(cfg.save_dir, "probes_rocauc.csv")
            pd.DataFrame([result_row]).to_csv(
                csv_path, mode="a", header=not os.path.exists(csv_path), index=False
            )

            with torch.no_grad():
                probe_cpu = probe.to(device="cpu").eval()
                attention_scores = (
                    probe_cpu(
                        te_H.to(dtype=torch.float32),
                        te_length.to(dtype=torch.float32),
                    ).detach().cpu()
                )

            results['prediction']['targets'] = te_y
            results['prediction']['logits'] = te_logits
            results['prediction'][f'attention_pooling_layer_{layer}_num_add_layers_{num_add}_prediction'] = attention_scores
    return results


def search_hyperparameters_linear_probe(model, train_loader, val_loader, test_loader, pooling_cfg, adapter_path, dataset_name, cfg):
    results = {}
    results['prediction'] = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for layer in cfg.probes.layers:
        tr_X, tr_y, tr_logits, tr_length = extract_cls_features(model, train_loader, pooling_cfg, layer, device)
        print('ACC:', (tr_logits.argmax(dim=-1) == tr_y).float().mean())
        print(
            'Unique: ', torch.unique(
                tr_logits.argmax(dim=-1),
                return_counts=True
            )
        )
        print(
            'Unique tgt: ', torch.unique(
                tr_y,
                return_counts=True
            )
        )
        va_X, va_y, va_logits, va_length = extract_cls_features(model, val_loader, pooling_cfg, layer, device)

        errors_tr = (tr_logits.argmax(dim=1) != tr_y).float()
        errors_va = (va_logits.argmax(dim=1) != va_y).float()

        probe = LinearProbe(tr_X.shape[-1])
        probe = train_probe_lightning(
            probe,
            tr_X,
            errors_tr,
            tr_length,

            va_X,
            errors_va,
            va_length,

            lr=cfg.probes.linear.lr,
            epochs=cfg.probes.linear.epochs,
            batch_size=cfg.probes.batch_size,
            log_params={
                'layer': layer,
                'lr': cfg.probes.linear.lr,
                'dataset': cfg.data.name,
                'type': 'linear_probe'
            },

        )

        save_dir = _build_probe_save_dir(
            project_root, cfg.model_name, dataset_name, cfg.seed, "linear", layer
        )
        metadata = {
            "probe": "linear",
            "layer": int(layer),
            "model_name": cfg.model_name,
            "dataset": dataset_name,
            "seed": int(cfg.seed),
            "train_on_dataset": cfg.train_on_dataset,
            "adapter_path": adapter_path,
            "hyperparams": {
                "lr": float(cfg.probes.linear.lr),
                "epochs": int(cfg.probes.linear.epochs),
                "batch_size": int(cfg.probes.batch_size),
            },
        }
        save_probe_model(probe, save_dir, metadata)

        te_X, te_y, te_logits, te_length = extract_cls_features(model, test_loader, pooling_cfg, layer, device)
        errors_te = (te_logits.argmax(dim=1) != te_y).float()
        auc, _, _, _ = evaluate_probe(
            probe,
            te_X,
            errors_te,
            te_length,
            device=("cuda" if torch.cuda.is_available() else "cpu"),
        )

        # current_state = {
        #     'model': model.config._name_or_path,
        #     'adapter': adapter_name,
        #     'dataset': dataset_name,
        #     'train_on_dataset': cfg.train_on_dataset,
        #     'normalization': cfg.normalization,
        #     'seed': cfg.seed,

        os.makedirs(cfg.save_dir, exist_ok=True)

        with torch.no_grad():
            probe_cpu = probe.to(device="cpu").eval()
            linear_scores = probe_cpu(te_X.to(dtype=torch.float32), te_length).detach().cpu()

        results['prediction']['targets'] = te_y
        results['prediction']['logits'] = te_logits
        results['prediction'][f'linear_prob_layer_{layer}_prediction'] = linear_scores
    return results


def run(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dataset = hydra.utils.instantiate(cfg.data.dataset)

    dataset_name = base_dataset.name()
    num_classes = base_dataset.num_classes()
    dataset = base_dataset.load()
    cfg.model.num_labels = num_classes
    os.makedirs(cfg.save_dir, exist_ok=True)

    adapter_path = cfg.adapter.path
    if cfg.get('base_model', False):
        del cfg.model.num_labels
        model = transformers.AutoModelForCausalLM.from_pretrained(
            adapter_path,
            **hydra.utils.instantiate(cfg.model)
        )
        adapter_name = 'base_full'
        tokenizer = transformers.AutoTokenizer.from_pretrained(adapter_path)
        model = model.eval().to(device)
    else:
        model = peft.AutoPeftModelForSequenceClassification.from_pretrained(
            adapter_path, **hydra.utils.instantiate(cfg.model)
        )
        adapter_name = model.peft_config["default"].__class__.__name__
        model = model.eval().merge_and_unload().to(device)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    tokenizer = transformers.AutoTokenizer.from_pretrained(model.config._name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=cfg.data.max_length
        )

    if cfg.train_on_dataset == "train":
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
        test_split = (
            dataset["test"]
            .map(tokenize_function, batched=False, load_from_cache_file=True)
            .remove_columns("text")
        )
    elif cfg.train_on_dataset == "valid":
        valid = (
            dataset["validation"]
            .map(tokenize_function, batched=False, load_from_cache_file=True)
            .remove_columns("text")
        )
        test_split = (
            dataset["test"]
            .map(tokenize_function, batched=False, load_from_cache_file=True)
            .remove_columns("text")
        )
        split = valid.train_test_split(test_size=0.15, seed=cfg.seed)
        train_split, val_split = split["train"], split["test"]
    else:
        raise ValueError("Unsupported train_on_dataset option")

    collate_fn = transformers.DataCollatorWithPadding(
        return_tensors="pt", padding="longest", tokenizer=tokenizer
    )

    if cfg.get('map_targets', False):
        choices = base_dataset.get_choices()
        if cfg.get('add_space', None):
            mapping = dict([
                (idx, tokenizer(' ' + choice, add_special_tokens=False)['input_ids'][0]) for choice, idx in zip(choices, range(512))
            ])
        else:
            mapping = dict([
                (idx, tokenizer(choice, add_special_tokens=False)['input_ids'][0]) for choice, idx in zip(choices, range(512))
            ])
        print('before', train_split['label'])
        train_split = train_split.map(lambda x: {
            'label': mapping.get(x['label'], x['label'])
        }, batched=False)
        print('after', train_split['label'])
        val_split = val_split.map(lambda x: {
            'label': mapping.get(x['label'], x['label'])
        }, batched=False)
        test_split = test_split.map(lambda x: {
            'label': mapping.get(x['label'], x['label'])
        }, batched=False)



    train_loader = DataLoader(
        train_split,
        batch_size=cfg.data.llm_batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_split,
        batch_size=cfg.data.llm_batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_split,
        batch_size=cfg.data.llm_batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
    )
    # model, train_loader, val_loader, test_loader, adapter_path, dataset_name, cfg
    linear_results = search_hyperparameters_linear_probe(
        model, train_loader, val_loader, test_loader, cfg.pooling, adapter_path, dataset_name, cfg
    )

    attention_pooling_results = {}
    attention_pooling_results = search_hyperparameters_attention_pooling(
        model, train_loader, val_loader, test_loader, cfg.pooling, adapter_path, dataset_name, cfg
    )


    metric_df = {
        "model_name": cfg.model_name,
        "dataset": dataset_name,
        "seed": int(cfg.seed),
        "train_on_dataset": cfg.train_on_dataset,
        "adapter_path": adapter_path,
    }
    total_results = {
        'metric_df': metric_df,
        'prediction': {
            **linear_results['prediction'],
            **attention_pooling_results['prediction']
        }
    }

    scores_path = os.path.join(
        cfg.save_dir,
        f"{cfg.model_name}_{dataset_name}_seed_{cfg.seed}_{cfg.train_on_dataset}_prob_baselines.pkl",
    )
    with open(scores_path, "wb") as f:
        pickle.dump(total_results, f)


@hydra.main(version_base="1.3", config_path="configs", config_name="train_probes.yaml")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
