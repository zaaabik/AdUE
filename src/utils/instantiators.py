from __future__ import annotations

from abc import ABC, abstractmethod

import hydra.utils
import peft
import uuid
from hydra.utils import instantiate
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from peft import LoraModel, PeftConfig, PeftModel, get_peft_model
from transformers import (
    LlamaForSequenceClassification,
    MistralForSequenceClassification,
    MistralPreTrainedModel,
    PreTrainedModel, AutoModelForSequenceClassification,
)

from src.utils import pylogger
from src.utils.utils import count_trainable_parameters, count_all_parameters

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    """Instantiates callbacks from config.

    Args:
        callbacks_cfg (DictConfig): A DictConfig object containing callback configurations.
    Returns:
        (list[Callback]) A list of instantiated callbacks.
    """
    callbacks: list[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> list[Logger]:
    """Instantiates loggers from config.

    Args:
        logger_cfg (DictConfig): A DictConfig object containing logger configurations.
    Returns:
        (list[Logger]) A list of instantiated loggers.
    """
    logger: list[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


class BaseModelInstantiate(ABC):
    @abstractmethod
    def instantiate(self) -> PreTrainedModel | PeftModel | LoraModel:
        """Base method for instantiate model."""
        raise NotImplementedError

    def get_name(self):
        return 'base'


class PartModelTrainableInstantiate(BaseModelInstantiate):
    def __init__(self, model: BaseModelInstantiate, trainable_params: list[str]):
        self.model: MistralForSequenceClassification | LlamaForSequenceClassification = hydra.utils.instantiate(
            model).instantiate()

        if not trainable_params:
            raise ValueError('There is not trainable parameters')

        self.trainable_params = trainable_params

    def instantiate(self) -> PreTrainedModel | PeftModel | LoraModel:
        """Init peft model with multiple adapters."""
        self.model.requires_grad_(False)
        named_parameters = dict(self.model.named_parameters())
        for trainable_param in self.trainable_params:
            named_parameters[trainable_param].requires_grad_(True)

        trainable_params_count = count_trainable_parameters(self.model)
        params_count = count_all_parameters(self.model)
        print(f"Nb_trainable_parameters: {trainable_params_count}")
        print(f"Nb_parameters: {params_count}")
        print(f'Percent of trainable params {trainable_params_count / params_count * 100}')

        return self.model


class PeftModelInstantiate(BaseModelInstantiate):
    def __init__(self, model: DictConfig, peft_config: list[DictConfig]):
        self.model: MistralForSequenceClassification | LlamaForSequenceClassification = (
            instantiate_model(model)
        )
        self.peft_config: PeftConfig = instantiate(peft_config)

    def instantiate(self) -> PreTrainedModel | PeftModel | LoraModel:
        """Init peft model with multiple adapters."""
        peft_model = peft.get_peft_model(self.model, self.peft_config)

        print('Model structure:')
        print(peft_model)
        print(f"Nb_trainable_parameters: {peft_model.get_nb_trainable_parameters()}")
        return peft_model


class PeftPreTrainedModelInstantiate(BaseModelInstantiate):
    def __init__(self, model: DictConfig, pretrained_path: str):
        self.model: MistralForSequenceClassification | LlamaForSequenceClassification = instantiate(model)

        self.pretrained_path = pretrained_path

    def instantiate(self) -> PreTrainedModel | PeftModel | LoraModel:
        """Init peft model with multiple adapters."""
        peft_model = peft.PeftModel.from_pretrained(
            self.model, self.pretrained_path, inference_mode=True, is_trainable=False
        ).eval()

        print('Model structure:')
        print(peft_model)
        print(f"Nb_trainable_parameters: {peft_model.get_nb_trainable_parameters()}")
        if hasattr(peft_model, 'merge_and_unload'):
            return peft_model.merge_and_unload()
        else:
            return peft_model

    def get_name(self):
        return self.pretrained_path


class PreTrainedModelInstantiate(BaseModelInstantiate):
    def __init__(self, model: DictConfig, pretrained_path: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_path)
        # self.model: MistralForSequenceClassification | LlamaForSequenceClassification = instantiate(model)

        self.pretrained_path = pretrained_path

    def instantiate(self) -> PreTrainedModel | PeftModel | LoraModel:
        """Init peft model with multiple adapters."""
        model = self.model
        print('Model structure:')
        print(model)
        return model

    def get_name(self):
        return self.pretrained_path


def instantiate_model(
        model_config: DictConfig) -> MistralPreTrainedModel | MistralPreTrainedModel | LlamaForSequenceClassification:
    """Init model from configuration dict
    Args:
        model_config (`omegaconf.DictConfig`): hydra model config
    Returns:
        (`transformers.PreTrainedModel`) initialized model
    """
    model = instantiate(model_config)
    return model


def instantiate_peft_model(
        model: PreTrainedModel, peft_config: DictConfig
) -> tuple[PeftModel, str]:
    """Calculate uncertainty estimation
    Args:
        model (`transformers.PreTrainedModel`): model prediction result
        peft_config (`omegaconf.DictConfig`): configuration for adapter model
    Returns:
        Tuple(
            (`peft.PeftModel`),
            (`str`)
        ) peft model and name for logging
    """
    config: PeftConfig = instantiate(peft_config, task_type=peft.TaskType.SEQ_CLS)
    peft_model = get_peft_model(model, config)
    return peft_model, config.peft_type.title() + f"_r-{peft_config.r}"

def instantiate_omega_conf_resolvers():
    """Add to omega conf resolvers for custom functions."""
    OmegaConf.register_new_resolver("div", lambda x, y: x // y)
    OmegaConf.register_new_resolver("mult", lambda x, y: x * y)
    OmegaConf.register_new_resolver("minus", lambda x, y: x - y)
    OmegaConf.register_new_resolver("uuid", lambda: uuid.uuid4().hex)
