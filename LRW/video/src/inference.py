from __future__ import annotations

import sys
import warnings

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data import create_dataloaders
from lightning import TransformerLightningModule, DCTCNLightningModule

warnings.filterwarnings("ignore")


def main(config: DictConfig):
    _, _, test_dataloader = create_dataloaders(config)
    checkpoint = ModelCheckpoint(
        monitor="val/accuracy_top1", mode="max", save_weights_only=True
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=config.devices,
        precision=config.train.precision,
        amp_backend="native",
        strategy="ddp",
        log_every_n_steps=config.train.log_every_n_steps,
        max_epochs=config.train.get("epochs", -1),
        max_steps=config.optim.scheduler.get("num_training_steps", -1),
        gradient_clip_val=config.train.gradient_clip_val,
        accumulate_grad_batches=config.train.accumulate_grad_batches,
        val_check_interval=config.train.validation_interval,
        logger=WandbLogger(
            name=config.train.name,
            project="cross-modal-sync",
        ),
        callbacks=[checkpoint, LearningRateMonitor("step")],
    )

    trainer.test(
        model=TransformerLightningModule(config) if config.model.name=='transformer' else \
            DCTCNLightningModule(config),
        ckpt_path=config.evaluate.ckpt_path,
        dataloaders=[test_dataloader],
    )


if __name__ == "__main__":
    main(OmegaConf.merge(OmegaConf.load(sys.argv[1]), OmegaConf.from_cli()))