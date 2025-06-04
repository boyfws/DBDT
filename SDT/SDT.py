from torch import nn
import torch

from typing import Optional

from .BaseSplitter import BaseSplitter
import copy


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

        assert depth >= 1

        self.depth = depth
        self.regularization = regularization


        self.splitters = nn.ModuleList(
            [
                copy.deepcopy(split_function_arg) for _ in range(2 ** depth - 1)
            ]
        )

        self.value = nn.Parameter(
                torch.empty(2 ** depth, output_dim)
        )
        nn.init.xavier_uniform_(self.value)

    def forward(self, x):
        value = self.value.unsqueeze(0).expand(x.size(0), -1, -1).clone()  # (batch_size, a, b)

        device = x.device

        reg_term = torch.tensor(0.0, device=device)

        for d in range(self.depth):  # Iterate through depth
            start = 2 ** d - 1
            end = 2 ** (d + 1) - 1

            num_models = end - start  # It is always a power of 2

            split_size = self.value.size(0) / num_models

            # This operation returns int, as self.value.size(0) == 2 ** depth
            # and num_models is a power of 2
            split_size = int(split_size)

            slice_start = torch.arange(num_models, dtype=torch.long) * split_size
            slice_half = slice_start + split_size // 2
            slice_end = torch.arange(1, num_models + 1, dtype=torch.long) * split_size

            for i, model_idx in enumerate(range(start, end)):
                p = self.splitters[model_idx](x)  # [batch_size, 1]

                if self.regularization:
                    reg_term += -0.5 * (
                        2 ** (-d) * torch.log(
                            torch.clamp(p * (1 - p), min=1e-5)
                        )
                    ).mean()

                p = p.unsqueeze(-1)  # [batch_size, 1, 1]

                s = slice_start[i]
                h = slice_half[i]
                e = slice_end[i]

                chunk_left = value[:, s:h]  # shape: [B, W, D]
                chunk_right = value[:, h:e]  # shape: [B, W, D]

                value[:, s:h] = chunk_left.clone() * p
                value[:, h:e] = chunk_right.clone() * (1 - p)

        value = value.sum(dim=1)

        if not self.regularization:
            reg_term = None

        return value, reg_term

    def eval(self):
        self.regularization = False
        super().eval()




