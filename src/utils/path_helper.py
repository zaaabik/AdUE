from omegaconf import DictConfig

from src.utils.utils import get_model_name_without_username


def build_run_name(model_name: str, task_name: str, seed: int) -> str:
    """Build run name based on experiment parameters.

    Args:
        model_name: str name of model
        task_name: str name of training task
        seed: int global seed using for reproduce
    """

    return f"arch-{model_name}_task-{task_name}_seed-{seed}"


def get_model_name(cfg: DictConfig, adapter_name: str = "") -> str:
    """Training function based on configuration dictionary
    Args:
        cfg: DictConfig hydra configuration file
        adapter_name: str name of adapter
    Returns:
        model named that includes some parameters of training
    """
    lr = cfg.trainer.args.learning_rate
    batch_size = cfg.total_batch_size
    model_name: str = cfg.model.base_architecture
    model_name = get_model_name_without_username(model_name)
    if adapter_name:
        model_name += f"_{adapter_name}"
    return f"{model_name}_lr-{lr}_bs-{batch_size}"
