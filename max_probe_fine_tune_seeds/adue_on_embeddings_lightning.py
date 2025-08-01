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

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

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

    best_run, all_runs = search_hyperparameters_lightning(**state)

    current_output_dir = os.path.join(cfg.save_dir)
    print(f'Saving results to: {current_output_dir}')
    os.makedirs(current_output_dir, exist_ok=True)
    with open(os.path.join(current_output_dir, 'best_run.pickle'), 'wb') as f:
        pickle.dump(best_run, f)

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
