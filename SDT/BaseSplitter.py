import torch.nn.functional as F
from torch import nn
import torch


class BaseSplitter(nn.Module):
    def __init__(self, input_dim: int, depth: int):
        super().__init__()
        self.raw_t = nn.Parameter(torch.empty(1, 2**depth - 1))
        nn.init.xavier_uniform_(self.raw_t)

        self.log_reg = nn.Linear(input_dim, 2**depth - 1)

    def forward(self, x):
        t = F.softplus(self.raw_t)  # always > 0
        x = self.log_reg(x)
        x = t * x
        return torch.sigmoid(x)