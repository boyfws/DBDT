import torch.nn as nn
import torch.nn.functional as F


class BaseSplitter(nn.Module):
    def __init__(self, input_dim: int, t: float):
        super().__init__()
        self.t = t

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.GELU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.mlp(x)
        x = self.t * x

        return F.sigmoid(x)  # [Batch_size, 1]