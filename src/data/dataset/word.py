from typing import Tuple, List

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.utils import Vocab
from src.utils.type import (
    SENTENCES_TYPE,
    TAGS_TYPE,
    SpecialTokens as ST
)

_ITEM_TYPE = Tuple[List[int], List[int]]
_BATCH_TYPE = List[_ITEM_TYPE]
_COLLATE_BATCH_TYPE = Tuple[Tensor, Tensor, List[int]]

__all__ = ["WordDataset"]


class WordDataset(Dataset):

    def __init__(
        self, *,
        sentences: SENTENCES_TYPE,
        tags: TAGS_TYPE,
        sentences_vocab: Vocab,
        tags_vocab: Vocab
    ) -> None:
        sentences_indices = sentences_vocab.sentences_to_indices(sentences)
        tags_indices = tags_vocab.sentences_to_indices(tags)

        self._data = sentences_indices
        self._targets = tags_indices

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> _ITEM_TYPE:
        return (
            self._data[idx],
            self._targets[idx]
        )

    # TODO
    # ugly, must be used with partial functools to fill `sentence_pad_idx` and `tag_pad_idx`
    @staticmethod
    def collate_fn(
        batch: _BATCH_TYPE,
        sentence_pad_idx: int,
        tag_pad_idx: int
    ) -> _COLLATE_BATCH_TYPE:
        batch = sorted(batch, key=lambda x : len(x[0]), reverse=True)
        sentences, tags = zip(*batch)
        sentences_lengths = [len(s) for s in sentences]
        max_len = sentences_lengths[0]

        padded_sentences = [
            s + [sentence_pad_idx] * (max_len - len(s))
            for s in sentences
        ]
        padded_tags = [
            t + [tag_pad_idx] * (max_len - len(t))
            for t in tags
        ]

        return torch.tensor(padded_sentences), \
            torch.tensor(padded_tags), sentences_lengths


if __name__ == "__main__":
    from src.utils import Vocab, read_corpus
    from torch.utils.data import DataLoader, random_split
    from torch.nn.utils import rnn
    from functools import partial

    data_path = "./data/train.txt"
    sents, tags = read_corpus(data_path)
    sents_vocab = Vocab.build_from_sentences(
        sents, False
    )
    tags_vocab = Vocab.build_from_sentences(
        tags, True
    )

    dataset = WordDataset(
        sentences=sents,
        tags=tags,
        sentences_vocab=sents_vocab,
        tags_vocab=tags_vocab
    )
    d1, d2 = random_split(dataset, [10000, len(dataset) - 10000])

    collate_fn = partial(
        dataset.collate_fn,
        sentence_pad_idx=sents_vocab.pad_idx,
        tag_pad_idx=tags_vocab.pad_idx
    )
    dataloader = DataLoader(
        dataset=d1,
        batch_size=32,
        collate_fn=collate_fn
    )

    for pad_x, pad_y, seq_len in dataloader:
        print(f"pad_x shape: {pad_x.shape}")
        pack_pad_x = rnn.pack_padded_sequence(
            pad_x, batch_first=True, lengths=seq_len, enforce_sorted=True)
        pad_pack_pad_x, _ = rnn.pad_packed_sequence(
            pack_pad_x, batch_first=True, padding_value=sents_vocab.pad_idx)
        print(f"pad_pack_pad_x shape: {pad_pack_pad_x.shape}")
        continue
