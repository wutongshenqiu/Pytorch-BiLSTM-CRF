from typing import Tuple
from .type import (
    SpecialTokens,
    FILE_PATH_TYPE,
    SENTENCES_TYPE,
    TAGS_TYPE
)

ST = SpecialTokens

__all__ = ["read_corpus"]


def read_corpus(file_path: FILE_PATH_TYPE) -> Tuple[SENTENCES_TYPE, TAGS_TYPE]:
    sentences = []
    tags = []
    
    sent = [ST.START]
    tag = [ST.START]
    with open(file_path, "r", encoding="utf8") as f:
        for line in f:
            if line == '\n':
                if len(sent) > 1:
                    sentences.append(sent + [ST.END])
                    tags.append(tag + [ST.END])
                    
                    sent = [ST.START]
                    tag = [ST.START]
            else:
                line = line.split()
                sent.append(line[0])
                tag.append(line[1])

    return sentences, tags
