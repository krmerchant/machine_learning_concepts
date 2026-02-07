from transformer_elements import TransformerBlock
from dataset import TinyShakespereDataset
from config import DatasetConfig
import torch

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("test_model")


def basic_tf_block_test():
    B = 3
    T = 10
    C = 5
    value = torch.ones(B, T, C)
    dataset = TinyShakespereDataset()
    dataset.get_batch()

    logger.debug(
        f"TransformerBlock forward pass with tensor of size {value.shape}")
    x = TransformerBlock(C, T)
    y = x(value)
    logger.debug(f"TransformerBlock run produce tensor of size {y.shape}")


def main():
    basic_tf_block_test()


if __name__ == "__main__":
    main()
