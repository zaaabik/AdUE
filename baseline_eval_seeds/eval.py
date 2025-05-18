import hydra
import pandas as pd
from omegaconf import DictConfig

# pylint: disable=wrong-import-position
import lightning as L
import rootutils
from torch.nn import functional as F
from transformers.activations import GELUActivation
from transformers.models.electra.modeling_electra import ElectraClassificationHead
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
from torch.nn.utils import spectral_norm as pt_spectral_norm
from tqdm import tqdm

import os
import random

import numpy as np
import peft
import torch
import transformers
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

generator = torch.Generator().manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from src.utils import distances_torch as distances
from src.utils.distances import rde_distance


def eval(cfg):
    L.seed_everything(42, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

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

    collate_fn = transformers.DataCollatorWithPadding(
        return_tensors="pt", padding="longest", tokenizer=tokenizer
    )
    train_dataloader = DataLoader(
        train_split,
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
    embedding_head = get_embedding_head(original_head, model.config)
    device = model.device
    embedding_head.to(device=device, dtype=torch.float32)

    raw_train_features, train_logits, train_targets = extraxt_features(model, train_dataloader, pooling)
    raw_test_features, test_logits, test_targets = extraxt_features(model, test_dataloader, pooling)
    with torch.no_grad():
        embedding_head.train()
        embedding_head(raw_train_features.to(device))

        embedding_head.eval()
        train_features = embedding_head(raw_train_features.to(device))
        test_features = embedding_head(raw_test_features.to(device))

    train_features = train_features.double().to(device)
    train_targets = train_targets.to(device)
    test_features = test_features.double().to(device)



    with torch.no_grad():
        if cfg.normalization:
            train_features = F.normalize(train_features, p=2, dim=1)
            test_features = F.normalize(test_features, p=2, dim=1)
        md_relative = distances.mahalanobis_distance_relative(
            train_features,
            train_targets,
            test_features,
        ).cpu()

        md = distances.mahalanobis_distance(
            train_features,
            train_targets,
            test_features,
        ).cpu()

        md_marginal = distances.mahalanobis_distance_marginal(
            train_features,
            train_targets,
            test_features,
        ).cpu()

        errors = test_targets != test_logits.argmax(dim=1)

        rde_roc_auc = {}
        for n_components in cfg.rde_n_components:
            _rde_dist = rde_distance(
                train_features, test_features, n_components=n_components
            )

            rde_roc_auc[f'rde_n_components_{n_components}_roc_auc'] = roc_auc_score(
                errors, _rde_dist
            )

        probs = torch.softmax(test_logits, dim=-1)
        md_roc_auc = roc_auc_score(errors, md)
        md_marginal_roc_auc = roc_auc_score(errors, md_marginal)
        md_relative_roc_auc = roc_auc_score(errors, md_relative)
        test_acc = (test_targets == test_logits.argmax(dim=1)).to(dtype=torch.float32).mean().item()
        test_max_prob_val = roc_auc_score(
            errors, 1 - probs.amax(dim=-1)
        )

        current_state = {
            'model': model.config._name_or_path,
            'adapter': adapter_name,
            'dataset': dataset_name,
            'train_on_dataset': cfg.train_on_dataset,
            'normalization': cfg.normalization,
            'seed': cfg.seed,

            'test-acc': test_acc,

            'md_roc_auc': md_roc_auc,
            'md_marginal_roc_auc': md_marginal_roc_auc,
            'md_relative_roc_auc': md_relative_roc_auc,

            'test-base-max-prob-roc-auc': test_max_prob_val,
        }
        current_state.update(rde_roc_auc)

        metric_df = pd.DataFrame([current_state])

        metric_df.to_csv(
            os.path.join(cfg.save_dir, f"{adapter_name}_{dataset_name}_{cfg.seed}_{cfg.normalization}_baselines.csv"),
            index=False
        )


def extraxt_features(model, dataloader, pooling):
    logits = []
    targets = []
    features = []
    device = 'cuda'

    model.eval().cuda()
    with torch.autocast(device_type='cuda:0', dtype=torch.bfloat16):
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch, output_hidden_states=True)
                cls_token = pooling(
                    out.hidden_states, input_ids=batch['input_ids'], model=model
                )
                logits.append(out.logits.cpu())
                targets.append(batch['labels'].cpu())
                features.append(cls_token.cpu())

    targets = torch.concat(targets, dim=0).float()
    logits = torch.concat(logits, dim=0).float()
    features = torch.concat(features, dim=0).float()
    return features, logits, targets


def get_embedding_head(cls_head, config):
    if isinstance(cls_head, RobertaClassificationHead):
        cls = RobertaClassificationHead(config)
        cls.load_state_dict(cls_head.state_dict())
        return torch.nn.Sequential(
            pt_spectral_norm(cls.dense, n_power_iterations=64),
            torch.nn.Tanh()
        )
    elif isinstance(cls_head, ElectraClassificationHead):
        cls = ElectraClassificationHead(config)
        cls.load_state_dict(cls_head.state_dict())
        return torch.nn.Sequential(
            pt_spectral_norm(cls.dense, n_power_iterations=64),
            GELUActivation()
        )
    elif isinstance(cls_head, torch.nn.Linear):
        return torch.nn.Identity()


@hydra.main(version_base="1.3", config_path="configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    """Main entry point for inference.

    Args:
        cfg: cfg: DictConfig configuration composed by Hydra.
    """

    eval(cfg)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
