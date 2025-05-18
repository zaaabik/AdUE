from __future__ import annotations

from typing import Any

# pylint: disable=wrong-import-position
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

import hydra
import torch

torch.set_float32_matmul_precision('medium')
import lightning as L
from lightning import Callback, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from src.data.text_module import TextDataModule
from src.model.llm_module import LLMLitModule
from src.utils.instantiators import (
    instantiate_callbacks,
    instantiate_loggers,
    instantiate_omega_conf_resolvers,
)
from src.utils.logging_utils import log_hyperparameters
from src.utils.pylogger import RankedLogger
from src.utils.utils import task_wrapper, get_adapter_path_from_ckpt_path, extras

log = RankedLogger(__name__, rank_zero_only=True)
instantiate_omega_conf_resolvers()


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Training function based on configuration dictionary
    Args:
        cfg: DictConfig hydra configuration file
    Returns:
        metric on validation dataset, it could be helpful in case of optimization
    """

    # set seed to better reproducing
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: TextDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LLMLitModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    ckpt_path = trainer.checkpoint_callback.best_model_path
    if ckpt_path == "":
        raise ValueError('Best model path does not exists')
    adapter_path = get_adapter_path_from_ckpt_path(ckpt_path)

    if cfg.get("test"):
        log.info("Starting testing!")
        model: LLMLitModule = hydra.utils.instantiate(
            cfg.model_eval,
            model=dict(
                pretrained_path=adapter_path
            )
        )
        trainer.test(model=model, datamodule=datamodule)

        log.info(f"Best ckpt path: {ckpt_path}")
        log.info(f"Best adapter path: {adapter_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}
    return metric_dict, OmegaConf.to_container(object_dict["cfg"], resolve=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Main entry point for training.

    Args:
        cfg: cfg: DictConfig configuration composed by Hydra.
    """
    extras(cfg)

    return train(cfg)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
