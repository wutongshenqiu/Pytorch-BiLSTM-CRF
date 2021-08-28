from __future__ import annotations

from typing import Dict, List, Union
from pathlib import PurePath
from collections import Counter
import itertools
import json

from .type import (
    SpecialTokens,
    FILE_PATH_TYPE,
    SENTENCES_TYPE,
    INDICES_TYPE
)

ST = SpecialTokens

__all__ = ["Vocab"]


class Vocab:

    def __init__(
        self, *,
        word2idx: Dict[str, int],
        idx2word: List[int]
    ) -> None:
        self._word2idx = word2idx
        self._idx2word = idx2word

    def word_to_idx(self, word: str) -> int:
        if (idx := self._word2idx.get(word)) is not None:
            return idx
        if ST.UNK in self._word2idx:
            return self._word2idx[ST.UNK]
        raise ValueError(f"Word `{word}` not in directories")
    
    def sentences_to_indices(self, sentences: SENTENCES_TYPE) -> INDICES_TYPE:
        return [
            [self.word_to_idx(w) for w in sentence] 
            for sentence in sentences
        ]

    def idx_to_word(self, idx: int) -> str:
        return self._idx2word[idx]

    def save_to_json(self, json_path: FILE_PATH_TYPE) -> None:
        with open(json_path, "w", encoding="utf8") as f:
            json.dump({
                "word2idx": self._word2idx,
                "idx2word": self._idx2word
            }, f, ensure_ascii=False)

    @staticmethod
    def load_from_json(json_path: FILE_PATH_TYPE) -> Vocab:
        with open(json_path, "r", encoding="utf8") as f:
            entry = json.load(f)

        return Vocab(
            word2idx=entry["word2idx"],
            idx2word=entry["idx2word"]
        )

    @staticmethod
    def build_from_sentences(
        sentences: SENTENCES_TYPE,
        is_tags: bool,
        max_dict_size: int = 5000,
        freq_cutoff: int = 2
    ) -> Vocab:
        """build vocab from given sentences

        Args:
            sentences (List[List[str]]): List of sentences, each sentence is a list of str
            max_dict_size (int): The maximum size of dict
                                 If the number of valid words exceeds dict_size, only the most frequently-occurred
                                 max_dict_size words will be kept
            freq_cutoff (int): If a word occurs less than freq_size times, it will be dropped
            is_tags (bool): whether this Vocab is for tags
        """
        word_count = Counter(itertools.chain(*sentences))

        valid_words = [w for w, d in word_count.items() if d >= freq_cutoff]
        valid_words.sort(key=lambda x: word_count[x], reverse=True)
        valid_words = valid_words[:max_dict_size]
        valid_words.append(ST.PAD)

        if not is_tags:
            valid_words.append(ST.UNK)

        word2idx = {w: idx for idx, w in enumerate(valid_words)}

        return Vocab(
            word2idx=word2idx,
            idx2word=valid_words
        )
    
    def __len__(self) -> int:
        return len(self._word2idx)


if __name__ == "__main__":
    from .utils import read_corpus

    sents, tags = read_corpus("./data/train.txt")

    sent_vocab = Vocab.build_from_sentences(
        sentences=sents,
        max_dict_size=5000,
        freq_cutoff=2,
        is_tags=False
    )
    sent_vocab.save_to_json("sent.json")

    sents = Vocab.load_from_json("sent.json")
    print(sents.idx_to_word(12))
    print(sents.word_to_idx("人"))
