from typing import List

import torch
from torch import Tensor
from torch.nn import (
    Module,
    Embedding,
    Dropout,
    LSTM,
    Linear,
    Parameter
)
from torch.nn.utils import rnn

from src.utils import Vocab
from src.utils.type import (
    SpecialTokens as ST,
    INDICES_TYPE
)


class BiLSTMCRF(Module):

    def __init__(
        self, *,
        sentences_vocab: Vocab,
        tags_vocab: Vocab,
        dropout_rate: float = 0.5,
        embedding_dim: int = 256,
        hidden_size: int = 256
    ) -> None:
        super().__init__()

        self._dropout_rate = dropout_rate
        self._embedding_dim = embedding_dim
        self._hidden_size = hidden_size
        self._sentences_vocab = sentences_vocab
        self._tags_vocab = tags_vocab

        # usage of embedding
        # https://www.jianshu.com/p/63e7acc5e890
        self._embedding = Embedding(
            num_embeddings=len(sentences_vocab),
            embedding_dim=embedding_dim
        )
        self._dropout = Dropout(p=dropout_rate)
        self._encoder = LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            bidirectional=True
        )
        self._hidden2emit_score = Linear(
            in_features=hidden_size*2,
            out_features=len(tags_vocab)
        )
        self._transition = Parameter(
            torch.randn(len(tags_vocab), len(tags_vocab))
        )

    def forward(
        self, *,
        sentences: Tensor,
        tags: Tensor,
        sentences_lengths: List[int]
    ) -> Tensor:
        """
        Args:
            sentences (Tensor): padded sentences, shape (batch_size, max_len)
            tags (Tensor): corresponding tags, shape (batch_size, max_len)
            sentences_lengths (List[int]): actually length of every sentence

        Returns:
            Tensor: loss
        """
        # (batch_size, max_len) tensor, 0 for pad
        mask = (sentences != self._sentences_vocab.word_to_idx(ST.PAD))
        # (batch_size, max_len) => (max_len, batch_size)
        sentences = sentences.transpose(0, 1)
        # (max_len, batch_size, embedding_dim)
        sentences = self._embedding(sentences)
        # use BiLSTM to estimate emit score
        emit_score = self._encode(
            sentences=sentences,
            sentences_lengths=sentences_lengths
        )
        # use emit score matrix and transition matrix to calculate CRF loss
        loss = self._crf_loss(
            tags=tags,
            mask=mask,
            emit_score=emit_score
        )

        return loss

    def _encode(
        self, *,
        sentences: Tensor,
        sentences_lengths: List[int]
    ) -> Tensor:
        """Estimate emit score using BiLSTM

        Args:
            sentences (Tensor): sentences with word embeddings, \
                                shape (max_len, batch_size, embedding_dim)
            sentences_lengths (List[int]): actually length of every sentence

        Returns:
            Tensor: emit score matrix, shape (batch_size, max_len, K), \
                    K is number of tags
        """
        # get packed sentences
        packed_sentences = rnn.pack_padded_sequence(
            input=sentences, lengths=sentences_lengths, enforce_sorted=False
        )
        hidden_states, _ = self._encoder(packed_sentences)
        # padding for matrix transformation
        # padding value should be `0` and not PAD
        hidden_states, _ = rnn.pad_packed_sequence(
            sequence=hidden_states,
            batch_first=True,
        )

        emit_score = self._hidden2emit_score(hidden_states)
        emit_score = self._dropout(emit_score)

        return emit_score

    # TODO
    # not understand clearly
    def _crf_loss(
        self, *,
        tags: Tensor,
        mask: Tensor,
        emit_score: Tensor
    ) -> Tensor:
        """Calculate Conditional Random Field loss

        Args:
            tags (Tensor): a batch of tags, shape (batch_size, max_len)
            mask (Tensor): mask for tags, 0 for padding, shape (batch_size, max_len)
            emit_score (Tensor): emit matrix, shape (batch_size, max_len, K)

        Returns:
            Tensor: Conditional Random Field loss for batch
        """
        # shape (batch_size, max_len)
        score = torch.gather(
            emit_score, dim=2, index=tags.unsqueeze(2)
        ).squeeze(2)
        score[:, 1:] += self._transition[tags[:, :-1], tags[:, 1:]]
        # shape (batch_size)
        total_score = (score * mask.type(torch.float)).sum(dim=1)

        d = torch.unsqueeze(emit_score[:, 0], dim=1)  # shape: (b, 1, K)
        for i in range(1, tags.shape[1]):
            n_unfinished = mask[:, i].sum()
            d_uf = d[: n_unfinished]  # shape: (uf, 1, K)
            emit_and_transition = emit_score[: n_unfinished, i].unsqueeze(dim=1) \
                + self._transition  # shape: (uf, K, K)
            log_sum = d_uf.transpose(
                1, 2) + emit_and_transition  # shape: (uf, K, K)
            max_v = log_sum.max(dim=1)[0].unsqueeze(dim=1)  # shape: (uf, 1, K)
            log_sum = log_sum - max_v  # shape: (uf, K, K)
            # shape: (uf, 1, K)
            d_uf = max_v + torch.logsumexp(log_sum, dim=1).unsqueeze(dim=1)
            d = torch.cat((d_uf, d[n_unfinished:]), dim=0)
        d = d.squeeze(dim=1)  # shape: (b, K)
        max_d = d.max(dim=-1)[0]  # shape: (b,)
        # shape: (b,)
        d = max_d + torch.logsumexp(d - max_d.unsqueeze(dim=1), dim=1)
        llk = total_score - d  # shape: (b,)
        loss = -llk  # shape: (b,)

        return loss

    def predict(
        self, *,
        sentences: Tensor,
        sentences_lengths: List[int]
    ) -> INDICES_TYPE:
        """predict tags for given batch sentences

        Args:
            sentences (Tensor): sentences, shape (batch_size, max_len)
            sentences_lengths (List[int]): length of every sentence

        Returns:
            List[List[str]]: predicted tags
        """
        batch_size = sentences.shape[0]
        # shape: (b, len)
        mask = (sentences != self._sentences_vocab.word_to_idx(ST.PAD))
        # shape: (len, b)
        sentences = sentences.transpose(0, 1)
        # shape: (len, b, e)
        sentences = self._embedding(sentences)
        # shape: (b, len, K)
        emit_score = self._encode(
            sentences=sentences,
            sentences_lengths=sentences_lengths
        )
        # list, shape: (b, K, 1)
        tags = [[[i] for i in range(len(self._tags_vocab))]] * batch_size
        # shape: (b, 1, K)
        d = torch.unsqueeze(emit_score[:, 0], dim=1)
        for i in range(1, sentences_lengths[0]):
            n_unfinished = mask[:, i].sum()
            # shape: (uf, 1, K)
            d_uf = d[: n_unfinished]
            # shape: (uf, K, K)
            emit_and_transition = self._transition + \
                emit_score[: n_unfinished, i].unsqueeze(dim=1)
            # shape: (uf, K, K)
            new_d_uf = d_uf.transpose(1, 2) + emit_and_transition
            d_uf, max_idx = torch.max(new_d_uf, dim=1)
            max_idx = max_idx.tolist()  # list, shape: (nf, K)
            tags[: n_unfinished] = [
                [tags[b][k] + [j]
                 for j, k in enumerate(max_idx[b])] for b in range(n_unfinished)
            ]
            # shape: (b, 1, K)
            d = torch.cat((torch.unsqueeze(d_uf, dim=1),
                          d[n_unfinished:]), dim=0)
        d = d.squeeze(dim=1)  # shape: (b, K)
        _, max_idx = torch.max(d, dim=1)  # shape: (b,)
        max_idx = max_idx.tolist()
        tags = [tags[b][k] for b, k in enumerate(max_idx)]

        return tags


if __name__ == "__main__":
    from pprint import pprint
    from torch.nn.utils import rnn
    from src.data import MsraCHNNERDataModule
    from src.utils import Vocab, read_corpus

    train_file = "./data/train.txt"
    test_file = "./data/test.txt"
    datamodule = MsraCHNNERDataModule(
        train_dataset_path=train_file,
        test_dataset_path=test_file
    )
    datamodule.setup()

    sentences, tags = read_corpus(train_file)
    sent_vocab = Vocab.build_from_sentences(sentences, is_tags=False)
    tags_vocab = Vocab.build_from_sentences(tags, is_tags=True)

    model = BiLSTMCRF(
        sentences_vocab=sent_vocab,
        tags_vocab=tags_vocab
    )
    print(model)

    for sents, tags in datamodule.train_dataloader():
        pad_sents = rnn.pad_sequence(
            sents, batch_first=True, padding_value=sent_vocab.word_to_idx(ST.PAD))
        pad_tags = rnn.pad_sequence(
            tags, batch_first=True, padding_value=tags_vocab.word_to_idx(ST.PAD))
        print(f"pad sents shape: {pad_sents.shape}")
        print(f"pad tags shape: {pad_sents.shape}")
        sent_lens = [s.size(0) for s in sents]

        loss = model(
            sentences=pad_sents,
            tags=pad_tags,
            sentences_lengths=sent_lens
        )
        print(f"loss shape: {loss.shape}")
        print(f"loss mean: {loss.mean()}")

        results = model.predict(sentences=pad_sents, sentences_lengths=sent_lens)
        print(f"results length: {len(results)}")
        
        break
