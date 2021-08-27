from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.utils import Vocab
from src.utils.type import (
    SENTENCES_TYPE,
    TAGS_TYPE
)


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

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return (
            torch.tensor(self._data[idx]),
            torch.tensor(self._targets[idx])
        )

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


if __name__ == "__main__":
    from src.utils import Vocab, read_corpus
    from torch.utils.data import DataLoader, random_split
    from torch.nn.utils import rnn

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

    dataloader = DataLoader(
        dataset=d1,
        batch_size=32,
        collate_fn=dataset.collate_fn
    )

    for x, y in dataloader:
        seq_len = [xs.size(0) for xs in x]
        pad_x = rnn.pad_sequence(x, batch_first=True, padding_value=sents_vocab.word_to_idx("<PAD>"))
        # pad_x = torch.unsqueeze(pad_x, -1)
        print(f"pad_x shape: {pad_x.shape}")
        pack_pad_x = rnn.pack_padded_sequence(pad_x, batch_first=True, lengths=seq_len, enforce_sorted=False)
        pad_pack_pad_x, _ = rnn.pad_packed_sequence(pack_pad_x, batch_first=True, padding_value=sents_vocab.word_to_idx("<PAD>"))
        print(f"pad_pack_pad_x shape: {pad_pack_pad_x.shape}")
        continue
