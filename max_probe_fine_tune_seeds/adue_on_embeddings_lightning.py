import pickle
import hydra
import random
import os

import numpy as np
import torch

import pandas as pd
from omegaconf import DictConfig

# pylint: disable=wrong-import-position
import rootutils

project_root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from max_probe_fine_tune_seeds.core import search_hyperparameters_lightning

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

generator = torch.Generator().manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(cfg):
    os.makedirs(cfg.save_dir, exist_ok=True)

    state = pd.read_pickle(cfg.embedding_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state['device'] = device
    state['cfg']['grid'] = cfg.grid
    state['cfg']['save_dir'] = cfg.save_dir

    del state['train_labels']
    del state['test_labels']
    del state['val_labels']
    del state['train_max_probs']
    del state['test_max_probs']
    del state['val_max_probs']

    def map_targets(targets, unique_targets):
        replace_dict = {
            val: idx for idx, val in enumerate(unique_targets)
        }

        new_targets = torch.ones_like(targets) * -1
        for original_value, new_value in replace_dict.items():
            new_targets[targets == original_value] = new_value
        return new_targets

    if cfg.grid.get('restricted_classes', False):
        print('Train with restricted classes')
        unique_targets = state['train_original_target'].unique()

        print('Test acc', (state['test_logits'].argmax(dim=-1) == state['test_original_targets']).float().mean().item())

        state['train_original_target'] = map_targets(state['train_original_target'], unique_targets)
        state['val_original_targets'] = map_targets(state['val_original_targets'], unique_targets)
        state['test_original_targets'] = map_targets(state['test_original_targets'], unique_targets)

        state['train_logits'] = state['train_logits'][:, unique_targets]
        state['val_logits'] = state['val_logits'][:, unique_targets]
        state['test_logits'] = state['test_logits'][:, unique_targets]

        print('Min max tgt', min(state['train_original_target']), max(state['train_original_target']))

        original_head = state['original_head']
        new_head = torch.nn.Linear(original_head.in_features, len(unique_targets))
        new_head.weight.data = original_head.weight.data[unique_targets, :]
        new_head = new_head.float()
        state['original_head'] = new_head

        print(
            'Test acc after mapping',
            (state['test_logits'].argmax(dim=-1) == state['test_original_targets']).float().mean().item()
        )
        with torch.autocast(
                device_type='cpu',
                dtype=torch.float32
        ):
            print(
                'Test acc after new layer',
                (
                        new_head(
                            state['test_features'].float()
                        ).argmax(dim=-1) == state['test_original_targets']).float().mean().item()
            )
        return

    best_run, all_runs = search_hyperparameters_lightning(**state)

    current_output_dir = os.path.join(cfg.save_dir)
    print(f'Saving results to: {current_output_dir}')
    os.makedirs(current_output_dir, exist_ok=True)
    with open(os.path.join(current_output_dir, 'best_run.pickle'), 'wb') as f:
        pickle.dump(best_run, f)

    best_run['metric_df'].to_csv(
        os.path.join(current_output_dir, 'best_run_metric_df.csv')
    )

    with open(os.path.join(current_output_dir, 'all_runs.pickle'), 'wb') as f:
        pickle.dump(all_runs, f)


@hydra.main(version_base="1.3", config_path="configs", config_name="apply_to_embeddings.yaml")
def main(cfg: DictConfig):
    """Main entry point for inference.

    Args:
        cfg: cfg: DictConfig configuration composed by Hydra.
    """

    train(cfg)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
