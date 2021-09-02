from typing import Any, Tuple, Optional, List

from torch import Tensor
from torch import optim

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from .utils import get_loss_function
from src.networks import BiLSTMCRF
from src.utils.type import (
    FILE_PATH_TYPE,
)
from src.utils import (
    read_corpus,
    Vocab,
    load_vocabs
)
from src.data import get_datamodule
from .config.bilstm_crf import Config

_config = Config()

__all__ = ["BiLSTMCRFModel"]


class BiLSTMCRFModel(pl.LightningModule):
    name: str = "BiLSTM-CRF"
    
    def __init__(
        self, *,
        train_dataset_path: FILE_PATH_TYPE = _config.train_dataset_path,
        test_dataset_path: FILE_PATH_TYPE = _config.test_dataset_path,
        vocabs_dir: FILE_PATH_TYPE = _config.vocabs_dir,
        loss_name: str = _config.loss_name,
        lr: float = _config.lr,
        dropout_rate: float = _config.dropout_rate,
        embedding_dim: int = _config.embedding_dim,
        hidden_size: int = _config.hidden_size,
        max_epochs: int = _config.max_epochs,
        datamodule_name: str = _config.datamodule_name,
        **datamodule_kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self._loss_function = get_loss_function(loss_name=loss_name)
        
        sentences_vocab, tags_vocab = load_vocabs(vocabs_dir)
        self._sentences_vocab = sentences_vocab
        self._tags_vocab = tags_vocab
        self._model = BiLSTMCRF(
            sentences_vocab=sentences_vocab,
            tags_vocab=tags_vocab,
            dropout_rate=dropout_rate,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size  
        )
        self._datamodule = get_datamodule(
            datamodule_name=datamodule_name,
            train_dataset_path=train_dataset_path,
            test_dataset_path=test_dataset_path,
            vocabs_dir=vocabs_dir,
            **datamodule_kwargs
        )
        
    def configure_optimizers(self) -> Any:
        return optim.Adam(
            params=self._model.parameters(),
            lr=self.hparams.lr
        )
    
    def training_step(
        self,
        batch: Tuple[Tensor, Tensor, List[int]],
        batch_idx: int
    ) -> STEP_OUTPUT:
        batch_loss = self._step_next(batch)
        average_batch_loss = batch_loss.mean()
        
        self.log("training_loss", average_batch_loss, prog_bar=True, on_step=True, on_epoch=False)
        
        return average_batch_loss
    
    def validation_step(
        self,
        batch: Tuple[Tensor, Tensor, List[int]],
        batch_idx: int    
    ) -> Optional[STEP_OUTPUT]:
        batch_loss = self._step_next(batch)
        average_batch_loss = batch_loss.mean()
        
        self.log_dict({
            "validation_loss": average_batch_loss
        }, on_step=True)
        
        return average_batch_loss
        
    def _step_next(self, batch: Tuple[Tensor, Tensor, List[int]]) -> Tensor:
        pad_sentences, pad_tags, sentences_lengths = batch
        
        batch_loss: Tensor = self._model(
            sentences=pad_sentences,
            tags=pad_tags,
            sentences_lengths=sentences_lengths
        )
        
        return batch_loss
    
    @property
    def datamodule(self) -> pl.LightningDataModule:
        return self._datamodule
