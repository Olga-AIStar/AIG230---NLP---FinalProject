from dataclasses import dataclass

@dataclass
class Config:
    file_name: str = 'complaints_with_concent.csv'
    target_sample_size: int = 100000
    product_target_size: int = 5000

    min_freq: int = 3
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10




cfg = Config()
SPECIAL = {"BOS": "<bos>", "EOS": "<eos>", "UNK": "<unk>"}