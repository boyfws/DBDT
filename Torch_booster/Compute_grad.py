import torch
import torch.nn as nn


def compute_loss_gradient(
        F: torch.Tensor,
        y: torch.Tensor,
        loss_fn: nn.Module,
        ) -> torch.Tensor:
    F = F.clone().detach()
    F.requires_grad = True

    loss = loss_fn(F, y)
    loss.backward()

    grad = F.grad.detach()

    return grad
