from src.utils import Vocab, read_corpus


if __name__ == "__main__":
    file_path = "./data/train.txt"
    
    sents, tags = read_corpus(file_path)
    sents_vocab = Vocab.build_from_sentences(
        sentences=sents, is_tags=False
    )
    tags_vocab = Vocab.build_from_sentences(
        sentences=tags, is_tags=True
    )
    
    sents_vocab.save_to_json("./data/vocabs/sentences_vocab.json")
    tags_vocab.save_to_json("./data/vocabs/tags_vocab.json")