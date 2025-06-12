import torch
from torch import nn

from .BaseSplitter import BaseSplitter


class SDT(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int,
        regularization: bool = True,
    ) -> None:
        super().__init__()
        assert depth >= 1

        self.depth = depth
        self.regularization = regularization

        self.splitter = BaseSplitter(input_dim, depth)

        self.value = nn.Parameter(torch.empty(2**depth, output_dim))
        nn.init.xavier_uniform_(self.value)

        d = (
            torch.arange(self.depth)
            .repeat_interleave(2 ** torch.arange(self.depth))
        )
        self.register_buffer("d", d)

    def forward(self, x, left_mask, right_mask):
        device = x.device

        predicted_probs = self.splitter(x)  # [Batch_size, 2 ** depth - 1]

        if self.regularization:
            reg_term = (
                -0.5
                * (
                    0.5 ** self.d
                    * torch.log(
                        torch.clamp(predicted_probs * (1 - predicted_probs), min=1e-5)
                    )
                )
                .mean(dim=1)
                .sum()
            )

        else:
            reg_term = None

        log_p = torch.log(torch.clamp(predicted_probs, min=1e-5))  # [B, N]
        log_1mp = torch.log(torch.clamp(1 - predicted_probs, min=1e-5))  # [B, N]

        # left_mask  [L, N]
        # right_mask  [L, N]

        accum_probs = log_p @ left_mask + log_1mp @ right_mask  # [B, L]
        accum_probs = accum_probs.unsqueeze(-1)  # [B, L, 1]

        sign = torch.sign(self.value)  # [L, D]
        logval = torch.log(torch.clamp(self.value.abs(), min=1e-10))  # [L, D]
        ret = logval + accum_probs  # [B, L, D]

        pos_mask = (sign > 0).float()
        neg_mask = (sign < 0).float()

        log_pos = torch.logsumexp(ret + torch.log(pos_mask + 1e-12), dim=1)
        log_neg = torch.logsumexp(ret + torch.log(neg_mask + 1e-12), dim=1)

        value = torch.exp(log_pos) - torch.exp(log_neg)

        return value, reg_term


