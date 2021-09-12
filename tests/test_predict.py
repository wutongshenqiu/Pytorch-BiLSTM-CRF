from typing import TextIO
from pathlib import PurePath

import torch
from torch.nn import Module
from torch import Tensor
from torch.utils.data import DataLoader

from src.pl_models import BiLSTMCRFModel
from src.utils import read_corpus, Vocab
from src.data import get_datamodule
from src.utils.type import (
    INDICES_TYPE,
    FILE_PATH_TYPE,
)


def write_to_file(
    model: Module,
    dataloader: DataLoader,
    sentences_vocab: Vocab,
    tags_vocab: Vocab,
    result_file_path: FILE_PATH_TYPE
) -> None:
    def write_batch(
        predicted_tags: INDICES_TYPE,
        real_tags: INDICES_TYPE,
        sentences: INDICES_TYPE,
        fp: TextIO
    ) -> None:
        for sent, real_tag, pred_tag in zip(sentences, real_tags, predicted_tags):
            # skip <start> and <end>
            sent, real_tag, pred_tag = sent[1:-1], real_tag[1:-1], pred_tag[1:-1]
            for sent_token, real_tag_token, pred_tag_token in zip(
                sent, real_tag, pred_tag
            ):
                fp.write(
                    f"{sentences_vocab.idx_to_word(sent_token)} "
                    f"{tags_vocab.idx_to_word(real_tag_token)} "
                    f"{tags_vocab.idx_to_word(pred_tag_token)}\n"
                )
            fp.write("\n")

    with torch.no_grad():
        with open(result_file_path, "w", encoding="utf8") as fp:
            for padded_sentences, padded_tags, sentences_lengths in dataloader:
                predicted_tags = model.predict(
                    sentences=padded_sentences,
                    sentences_lengths=sentences_lengths
                )
                real_tags = [
                    pt[:tag_len] for pt, tag_len in zip(
                        padded_tags.tolist(), sentences_lengths
                    )
                ]
                sentences = [
                    pt[:sent_len] for pt, sent_len in zip(
                        padded_sentences.tolist(), sentences_lengths
                    )
                ]
                write_batch(predicted_tags, real_tags, sentences, fp)


def evaluate_result_file(
    result_file_path: FILE_PATH_TYPE
) -> None:
    def parse_line(prefix: str, total_num: int, correct_num: int) -> str:
        return f"{prefix} total tokens: {total_num}, correct tokens: {correct_num}," \
               f"accuracy: {correct_num / total_num:.2f}\n"

    with open(result_file_path, "r", encoding="utf8") as f:
        total_token, total_correct = 0, 0
        o_total, o_correct = 0, 0
        loc_total, loc_correct = 0, 0
        per_total, per_correct = 0, 0
        org_total, org_correct = 0, 0
        for line in f:
            if line == "\n":
                continue
            line = line.strip()
            total_token += 1
            real_tag, pred_tag = line.split(" ")[1:]
            real_tag: str; pred_tag: str
            if real_tag == pred_tag:
                total_correct += 1
                if real_tag == "O":
                    o_correct += 1
                elif real_tag.endswith("LOC"):
                    loc_correct += 1
                elif real_tag.endswith("ORG"):
                    org_correct += 1
                elif real_tag.endswith("PER"):
                    per_correct += 1
            
            if real_tag == "O":
                o_total += 1
            elif real_tag.endswith("LOC"):
                loc_total += 1
            elif real_tag.endswith("ORG"):
                org_total += 1
            elif real_tag.endswith("PER"):
                per_total += 1

        print(
            parse_line("", total_token, total_correct),
            parse_line("LOC", loc_total, loc_correct),
            parse_line("PER", per_total, per_correct),
            parse_line("ORG", org_total, org_correct)
        )


if __name__ == "__main__":
    ckpt_path = "checkpoints/BiLSTM-CRF/BiLSTM-CRF-epoch=99.ckpt"
    train_corpus_path = "data/train.txt"
    test_corpus_path = "data/test.txt"
    vocabs_dir = PurePath("data/vocabs")

    pl_model = BiLSTMCRFModel.load_from_checkpoint(ckpt_path)
    model = pl_model._model
    model.eval()

    datamodule = get_datamodule("msra_chn", batch_size=2, test_shuffle=True)
    datamodule.setup()

    train_sents_vocab = Vocab.load_from_json(
        vocabs_dir / "sentences_vocab.json")
    train_tags_vocab = Vocab.load_from_json(vocabs_dir / "tags_vocab.json")

    write_to_file(
        model=model,
        dataloader=datamodule.test_dataloader(),
        sentences_vocab=train_sents_vocab,
        tags_vocab=train_tags_vocab,
        result_file_path="tmp/result.txt"
    )

    evaluate_result_file("tmp/result.txt")
