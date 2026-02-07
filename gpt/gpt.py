from transformer_elements import Block, FFN
from config import ModelConfig
import torch
import torch.nn as nn
import math
from torch.nn import functional as F


class GPT2(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict('tf': nn.ModuleList([TransformerBlock(config.embed_dim) for i in range(config.num_blocks])))

    def forward(self, input, target=None):
        pass
