import hydra
import pandas as pd
from omegaconf import DictConfig
import pickle

# pylint: disable=wrong-import-position
import lightning as L
import rootutils
from torch.nn import functional as F

import os
import random

import numpy as np
import peft
import torch
from sklearn.metrics import roc_auc_score

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from baseline_eval_seeds.eval import get_embedding_head

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
    os.makedirs(cfg.save_dir, exist_ok=True)

    dataset = hydra.utils.instantiate(cfg.data.dataset)
    dataset_name = dataset.name()
    num_classes = dataset.num_classes()
    cfg.model.num_labels = num_classes

    adapter_path = cfg.adapter.path
    model = peft.AutoPeftModelForSequenceClassification.from_pretrained(
        adapter_path,
        **hydra.utils.instantiate(cfg.model)
    )

    adapter_name = 'LoRA'
    model = model.eval().merge_and_unload().to(device)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    original_head = model.__getattr__(cfg.classifier_name)

    embedding_head = get_embedding_head(original_head, model.config)
    device = model.device
    embedding_head.to(device=device, dtype=torch.float32)
    state = pd.read_pickle(cfg.embedding_path)
    raw_train_features, train_targets = state['train_features'], state['train_original_target']
    raw_test_features, test_targets = state['test_features'], state['test_original_targets']
    train_logits = state['train_logits']
    test_logits = state['test_logits']

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
            os.path.join(
                cfg.save_dir,
                f"{adapter_name}_{dataset_name}_seed_{cfg.seed}_normalization_{cfg.normalization}_baselines.csv")
            ,
            index=False
        )

        final_state = {
            'metric_df': metric_df,
            'prediction': {
                'targets': test_targets,
                'logits': test_logits,
                'md': md,
                'md_marginal': md_marginal,
                'md_relative': md_relative,
                'sr': 1 - probs.amax(dim=-1)
            }
        }

        with open(
                os.path.join(
                    cfg.save_dir,
                    f"{adapter_name}_{dataset_name}_seed_{cfg.seed}_normalization_{cfg.normalization}_baselines_state"
                    f".pkl"
                ), 'wb') as f:
            pickle.dump(final_state, f)


@hydra.main(version_base="1.3", config_path="configs", config_name="apply_to_embeddings.yaml")
def main(cfg: DictConfig):
    """Main entry point for inference.

    Args:
        cfg: cfg: DictConfig configuration composed by Hydra.
    """

    eval(cfg)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
