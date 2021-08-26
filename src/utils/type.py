from typing import Union, List
from pathlib import PurePath


# meaning of special tokens below
# https://github.com/nicolas-ivanov/tf_seq2seq_chatbot/issues/15
class SpecialTokens:
    UNK: str = "<UNK>"
    PAD: str = "<PAD>"
    START: str = "<START>"
    END: str = "<END>"


FILE_PATH_TYPE = Union[PurePath, str]
SENTENCE_TYPE = List[str]
TAG_TYPE = List[str]
SENTENCES_TYPE = List[SENTENCE_TYPE]
TAGS_TYPE = List[TAG_TYPE]
