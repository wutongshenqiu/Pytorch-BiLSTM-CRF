import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.pl_models import BiLSTMCRFModel
from src.config import base_config


if __name__ == "__main__":
    gradient_clip_val = 5.0
    check_val_every_n_epoch = 1
    
    model = BiLSTMCRFModel(
        batch_size=128,
        max_epochs=100
    )

    checkpoint_dir_path = (
        base_config.checkpoints_dir_path /
        f"{model.name}"
    )

    every_epoch_callback = ModelCheckpoint(
        dirpath=checkpoint_dir_path,
        filename="BiLSTM-CRF-{epoch}"
    )
    
    logger = TensorBoardLogger(
        save_dir="tb_logs",
        name=f"{model.name}",
    )

    trainer = Trainer(
        callbacks=[every_epoch_callback],
        max_epochs=model.hparams.max_epochs,
        gpus=1,
        auto_select_gpus=True,
        gradient_clip_val=gradient_clip_val,
        check_val_every_n_epoch=check_val_every_n_epoch,
        logger=logger,
        flush_logs_every_n_steps=10
    )

    trainer.fit(
        model=model,
        datamodule=model.datamodule
    )