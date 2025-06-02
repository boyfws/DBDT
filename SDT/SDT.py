from torch import nn
import torch

from typing import Optional

from .Node import Node
from .BaseSplitter import BaseSplitter


class SDT(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            depth: int,
            regularization: bool = True,
            split_function: Optional[nn.Module] = None,
            t: float = 1,
    ) -> None:
        super().__init__()

        if split_function is None:
            split_function_arg = BaseSplitter(input_dim, t)
        else:
            split_function_arg = split_function

        self.start_node = Node(
            output_dim=output_dim,
            remaining_depth=depth,
            depth=0,
            split_function=split_function_arg,
            regularization=regularization,
        )

    def forward(self, x):
        return self.start_node(x)