import os
from collections import OrderedDict
from typing import Any

import lightning as L
from lightning.pytorch import Callback
from peft import PeftModel

MERGED_ADAPTER_NAME = "merged"


class SavePeftModel(Callback):
    """For peft model save only adapter weights."""

    def on_save_checkpoint(
        self, trainer: L.Trainer, pl_module: L.LightningModule, checkpoint: dict[str, Any]
    ) -> None:
        """Remove all model weight from checkpoint file and save only adapters."""
        checkpoint["state_dict"] = OrderedDict()
        checkpoint['optimizer_states'] = OrderedDict()
        model: PeftModel = trainer.model.net
        base_save_path = trainer.checkpoint_callback.dirpath
        epoch = trainer.current_epoch
        adapters_save_path = os.path.join(base_save_path, f"adapter_epoch_{epoch:03}")
        model.save_pretrained(adapters_save_path)
