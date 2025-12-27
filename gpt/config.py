# All POD structures

from dataclasses import dataclass


@dataclass
class DatasetConfig:
    seq_len: int = 10
    filename: str = ''
    batch_size: int = 64


class ModelConfig:
    embed_dim: int = 10
    seq_len: int = 10
