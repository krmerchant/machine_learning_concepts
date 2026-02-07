import torch
import torch.nn as nn
import math
from torch.nn import functional as F

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("debug_tf")


class TransformerBlock(nn.Module):
    def __init__(self,  n_embed, seq_len):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embed)
        self.att_block = SelfAttention(n_embed, seq_len)
        self.ln_2 = nn.LayerNorm(n_embed)
        self.ffn = FFN(n_embed)

    def forward(self, x):
        o = self.ln_1(self.att_block(x) + x)
        f = self.ffn(o)
        return self.ln_2(f + o)


class GELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415

    Straight-copy from min-gpt
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class FFN(nn.Module):
    def __init__(self,  n_embed):
        super().__init__()
        self.layer1 = nn.Linear(n_embed, 4 * n_embed)
        self.act = GELU()  # @todo make this configurable
        self.layer2 = nn.Linear(4*n_embed, n_embed)

    def forward(self, x):
        o = self.layer1(x)
        o = self.act(o)
        o = self.layer2(o)
        return o


class SelfAttention(nn.Module):
    """ TODO: Make this multihead """

    def __init__(self, n_embed, seq_len):
        super().__init__()
        self.W_key = nn.Linear(n_embed, n_embed)
        self.W_query = nn.Linear(n_embed, n_embed)
        self.W_value = nn.Linear(n_embed, n_embed)
        self.W_proj = nn.Linear(n_embed, n_embed)

        mask = torch.triu(torch.ones(seq_len, seq_len),
                          diagonal=1).bool()
        self.register_buffer('causal_mask', mask)

    def forward(self, x):

        B, C, N = x.shape
        k = self.W_key(x)
        q = self.W_query(x)
        v = self.W_value(x)
        att_scores = q @ k.transpose(-2, -1) / math.sqrt(C)
        att_scores.masked_fill(self.causal_mask, float('-inf'))
        att_weights = F.softmax(att_scores, dim=-1)
        logger.debug(f" c: {att_scores[0,:,:]}")
        o = att_weights @ v
        output = self.W_proj(o)
        return output
