from __future__ import annotations

import shutil
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
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(config)
    checkpoint = ModelCheckpoint(
        monitor="val/accuracy_top1", mode="max", save_weights_only=True
    )

    trainer = Trainer(
        accelerator="gpu",
        devices="auto",
        precision=16,
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

    if config.model.name == 'dc-tcn' :
        trainer.fit(DCTCNLightningModule(config), train_dataloader, val_dataloader)
    else :
        trainer.fit(TransformerLightningModule(config), train_dataloader, val_dataloader)
    trainer.test(ckpt_path=checkpoint.best_model_path, dataloaders=[test_dataloader])

    shutil.copy(checkpoint.best_model_path, ".")

if __name__ == "__main__":
    main(OmegaConf.merge(OmegaConf.load(sys.argv[1]), OmegaConf.from_cli()))
