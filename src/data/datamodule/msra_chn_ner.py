from typing import Optional, Dict, Any
import functools

import pytorch_lightning as pl

from torch.utils.data import random_split, DataLoader

from src.data.dataset import WordDataset
from src.utils import (
    read_corpus,
    load_vocabs
)
from src.utils.type import FILE_PATH_TYPE


class MsraCHNNERDataModule(pl.LightningDataModule):

    def __init__(
        self, *,
        train_dataset_path: FILE_PATH_TYPE = "./data/train.txt",
        test_dataset_path: FILE_PATH_TYPE = "./data/test.txt",
        vocabs_dir: FILE_PATH_TYPE = "./data/vocabs",
        validation_rate: float = 0.2,
        batch_size: int = 32,
        num_workers: int = 4,
        train_shuffle: bool = True,
        test_shuffle: bool = False,
        **kwargs
    ) -> None:
        self._train_dataset_path = train_dataset_path
        self._test_dataset_path = test_dataset_path
        self._validation_rate = validation_rate
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._train_shuffle = train_shuffle
        self._test_shuffle = test_shuffle
        
        self._sentences_vocab, self._tags_vocab = load_vocabs(vocabs_dir)
        
        super().__init__(**kwargs)
        
    def prepare_data(self) -> None:
        import os
        if not os.path.exists(self._train_dataset_path):
            raise ValueError(f"train dataset `{self._train_dataset_path}` not exists!")
        if not os.path.exists(self._test_dataset_path):
            raise ValueError(f"test dataset `{self._test_dataset_path}` not exists!")

    def setup(self, stage: Optional[str] = None) -> None:
        def prepare_arguments(dataset_path: FILE_PATH_TYPE) -> Dict[str, Any]:
            sentences, tags = read_corpus(dataset_path)
            return {
                "sentences": sentences,
                "tags": tags,
            }
        
        self._collate_fn = functools.partial(
            WordDataset.collate_fn, 
            sentence_pad_idx=self._sentences_vocab.pad_idx,
            tag_pad_idx=self._tags_vocab.pad_idx
        )
        
        partial_dataset = functools.partial(
            WordDataset, 
            sentences_vocab=self._sentences_vocab,
            tags_vocab=self._tags_vocab
        )
          
        self._test_dataset = partial_dataset(
            **prepare_arguments(self._test_dataset_path)
        )
        
        dataset = partial_dataset(
            **prepare_arguments(self._train_dataset_path)
        )
        total_len = len(dataset)
        val_len = int(total_len * self._validation_rate)
        self._train_dataset, self._val_dataset = random_split(
            dataset, [total_len - val_len, val_len]
        )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._train_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=self._train_shuffle,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._val_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=self._test_shuffle,
            collate_fn=self._collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._test_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=self._test_shuffle,
            collate_fn=self._collate_fn
        )
        
        
if __name__ == "__main__":
    msra_ner = MsraCHNNERDataModule(
        train_dataset_path="./data/train.txt",
        test_dataset_path="./data/test.txt",
        vocabs_dir="./data/vocabs"
    )
    msra_ner.prepare_data()
    msra_ner.setup()
    
    print(len(msra_ner.val_dataloader()) * 32)
    print(len(msra_ner.test_dataloader()) * 32)
    print(len(msra_ner.train_dataloader()) * 32)
    
    for x, y, seq_len in msra_ner.train_dataloader():
        print(f"type x: {type(x)}, shape x: {x.shape}")
        print(f"type y: {type(y)}, shape y: {y.shape}")
        print(f"type seq_len: {type(seq_len)}")