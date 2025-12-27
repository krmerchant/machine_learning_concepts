import torch
import torch.nn as nn
import math
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self,  n_embed):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embed)
        self.att_block = SelfAttention(n_embed)
        self.ln_2 = nn.LayerNorm(n_embed)
        self.ffn = SelfAttention(n_embed)

    def forward(self, x):
        o = self.ln_1(self.att_block(x) + x)
        f = self.ffn(o)
        return self.ln_2(f + o)


class GELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class FFN(nn.Module):
    def __init__(self,  n_embed):
        super().__init__()
        self.layer1 = nn.Linear(n_embed, 4 * n_embed)
        self.act = nn.GLEU()  # @todo make this configurable
        self.layer2 = nn.Linear(4*n_embed, n_embed)

    def forward(self, x):
        o = self.layer1(x)
        o = self.act(o)
        o = self.layer2(o)
        return o


class SelfAttention(nn.Module):
    """ TODO: Make this multihead """

    def __init__(self, n_embed):
        super().__init__()
        self.W_key = nn.Linear(n_embed, n_embed)
        self.W_query = nn.Linear(n_embed, n_embed)
        self.W_value = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        B, C, N = x.shape
        k = self.W_key(x)
        q = self.W_query(x)
        v = self.W_value(x)
        att = q @ k.transpose(-2, -1) / math.sqrt(C)
        att = F.softmax(att, dim=-1)
        output = att @ v
        return output
