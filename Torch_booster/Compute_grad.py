import torch
import torch.nn as nn


def compute_loss_gradient(
        F: torch.Tensor,
        y: torch.Tensor,
        loss_fn: nn.Module,
        ) -> torch.Tensor:

    F_var = torch.empty_like(F, requires_grad=True)
    F_var.data.copy_(F.detach())

    loss = loss_fn(F_var, y)
    loss.backward()

    grad = F_var.grad.detach()

    return grad
