from pydantic import BaseModel


class Config(BaseModel):
    train_dataset_path: str = "./data/train.txt"
    test_dataset_path: str = "./data/test.txt"
    vocabs_dir: str = "./data/vocabs"
    
    loss_name: str = "CrossEntropyLoss"
    lr: float = 0.001
    max_epochs: int = 20
    
    dropout_rate: float = 0.5    
    embedding_dim: int = 256
    hidden_size: int = 256
    
    datamodule_name: str = "msra_chn"