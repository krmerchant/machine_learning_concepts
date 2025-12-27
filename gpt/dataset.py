import torch
from torch.utils.data import Dataset
import tiktoken
from config import DatasetConfig


class TinyShakespereDataset():
    def __init__(self, config: DatasetConfig = DatasetConfig(filename='./data/input.txt')):
        super().__init__()
        self.config = config
        with open(config.filename, 'r', encoding='utf-8') as f:
            self.corpus = f.read()
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.corpus = torch.tensor(self.tokenizer.encode(self.corpus))

    def length(self):
        return len(self.corpus)

    def get_batch(self):
        # Generate random starting positions
        ix = torch.randint(self.length() - self.config.seq_len,
                           (self.config.batch_size,))
        x = torch.stack([self.corpus[i:i+self.config.seq_len] for i in ix])
        y = torch.stack([self.corpus[i+1:i+self.config.seq_len+1] for i in ix])
        return x, y
