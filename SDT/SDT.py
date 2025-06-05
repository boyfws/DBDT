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
            split_function_arg = BaseSplitter(input_dim, t, depth)
        else:
            split_function_arg = split_function(input_dim, t, depth)

        assert depth >= 1

        self.depth = depth
        self.regularization = regularization

        self.splitter = split_function_arg

        self.value = nn.Parameter(
                torch.empty(2 ** depth, output_dim)
        )
        nn.init.xavier_uniform_(self.value)

        self.model_idx_start =[
            2 ** d - 1 for d in range(self.depth)
        ]

        self.model_idx_end = [
            2 ** (d + 1) - 1 for d in range(self.depth)
        ]

        num_models = [
            self.model_idx_end[d] - self.model_idx_start[d]
            for d in range(self.depth)
        ]

        split_size = [
            self.value.size(0) // num_models[d] for d in range(self.depth)
        ]

        self.slice_start = [
            [i * split_size[d] for i in range(num_models[d])]
            for d in range(self.depth)
        ]

        self.slice_half = [
            [self.slice_start[d][i] + split_size[d] // 2 for i in range(num_models[d])]
            for d in range(self.depth)
        ]

        self.slice_end = [
            [i * split_size[d] for i in range(1, num_models[d] + 1)]
            for d in range(self.depth)
        ]

    def forward(self, x):
        device = x.device

        predicted_probs = self.splitter(x)

        accum_probs = torch.zeros(
            (x.size(0), self.value.size(0)),
            device=device,
            dtype=torch.float,
        )

        reg_term = torch.tensor(0.0, device=device)

        for d in range(self.depth):  # Iterate through depth
            start = self.model_idx_start[d]
            end = self.model_idx_end[d]

            slice_start = self.slice_start[d]
            slice_half = self.slice_half[d]
            slice_end = self.slice_end[d]

            for i, model_idx in enumerate(range(start, end)):
                p = predicted_probs[:, model_idx].unsqueeze(-1)  # [batch_size, 1]

                if self.regularization:
                    reg_term += -0.5 * (
                        2 ** (-d) * torch.log(
                            torch.clamp(p * (1 - p), min=1e-5)
                        )
                    ).mean()

                s = slice_start[i]
                h = slice_half[i]
                e = slice_end[i]

                accum_probs[:, s:h] += torch.log(torch.clamp(p, min=1e-5))
                accum_probs[:, h:e] += torch.log(torch.clamp(1 - p, min=1e-5))

        accum_probs = accum_probs.unsqueeze(-1) # [B, L, 1]

        sign = torch.sign(self.value)  # [L, D]
        logval = torch.log(torch.clamp(self.value.abs().clone(), min=1e-10))  # [B, L, D]

        ret = logval + accum_probs  # [B, L, D]

        pos_mask = (sign > 0).float()
        neg_mask = (sign < 0).float()

        log_pos = torch.logsumexp(ret + torch.log(pos_mask + 1e-12), dim=1)
        log_neg = torch.logsumexp(ret + torch.log(neg_mask + 1e-12), dim=1)

        value = torch.exp(log_pos) - torch.exp(log_neg)

        if not self.regularization:
            reg_term = None

        return value, reg_term

    def eval(self):
        self.regularization = False
        super().eval()




