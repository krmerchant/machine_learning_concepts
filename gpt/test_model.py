from model import Block
import torch

B = 3
T = 100
C = 256

value = torch.ones(B, T, C)

x = Block(C)
x(value)
