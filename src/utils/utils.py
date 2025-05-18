from __future__ import annotations
import warnings

from collections.abc import Callable
from importlib.util import find_spec
from typing import Any

import numpy as np
import torch.nn
import os
from omegaconf import DictConfig
from peft import PeftModel
from peft.tuners.lora import LoraLayer

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb  # pylint: disable=import-outside-toplevel

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_trainable_parameters(model: torch.nn.Module) -> dict:
    """Training function based on configuration dictionary
    Args:
        model: torch module
    Returns:
        dict with trainable params and all params
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
    return {"trainable params": trainable_params, "all params": all_param}


def get_model_name_without_username(hf_model_name: str) -> str:
    """Remove user prefix from model name
    Args:
        hf_model_name: model name from hugging face
    Returns:
        model name without username
    """
    model_name = hf_model_name.split("/")[-1]
    return model_name


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def count_all_parameters(model):
    model_parameters = model.parameters()
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def find_best_path_from_ckpt(ckpt: dict) -> str:
    callbacks = ckpt['callbacks']
    callbacks_names = list(callbacks.keys())

    for callbacks_name in callbacks_names:
        if 'ModelCheckpoint' in callbacks_name:
            return callbacks[callbacks_name]['best_model_path']


def get_adapter_path_from_ckpt_path(ckpt_path: str) -> str:
    folder, file = os.path.split(ckpt_path)
    file = 'adapter_' + file.replace('.ckpt', '')
    adapter_path = os.path.join(folder, file)
    return adapter_path

