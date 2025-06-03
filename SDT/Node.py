import torch.nn as nn
import torch

from copy import deepcopy


class Node(nn.Module):
    def __init__(
            self,
            output_dim: int,
            depth: int,
            remaining_depth: int,
            split_function: nn.Module,
            regularization: bool = True
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.depth = depth
        self.regularization = regularization

        if remaining_depth == 0:
            self.leaf = True
            a = (6 / output_dim) ** 0.5  # Xavier uniform
            self.value = nn.Parameter(
                torch.empty(output_dim).uniform_(-a, a)
            )

        else:
            self.leaf = False
            self.split = deepcopy(split_function)

            self.left_leaf = Node(
                output_dim=output_dim,
                depth=depth + 1,
                remaining_depth=remaining_depth - 1,
                split_function=split_function,
                regularization=regularization,
            )

            self.right_leaf = Node(
                output_dim=output_dim,
                depth=depth + 1,
                remaining_depth=remaining_depth - 1,
                split_function=split_function,
                regularization=regularization,
            )

    def forward(
            self,
            x: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
    ]:
        if self.leaf:

            reg_term = torch.tensor(0.0) if self.regularization else None

            return self.value.expand(x.size(0), self.output_dim), reg_term

        else:
            p = self.split(x)  # [Batch_size]

            left_pred, reg_left = self.left_leaf(x)
            # left_pred | [Batch_size, output_dim]

            right_pred, reg_right = self.right_leaf(x)
            # right_pred | [Batch_size, output_dim]

            if self.regularization:
                reg_term = -0.5 * (
                        2 ** (-self.depth) * torch.log(
                    torch.clamp(p * (1 - p), min=1e-5)
                )
                ).mean() + reg_right + reg_left
            else:
                reg_term = None

            return (1 - p) * left_pred + p * right_pred, reg_term

    @torch._dynamo.disable
    def set_regularization(self, regularization: bool):
        self.regularization = regularization
        if hasattr(self, 'left_leaf'):
            self.left_leaf.set_regularization(regularization)

        if hasattr(self, 'right_leaf'):
            self.right_leaf.set_regularization(regularization)

    @torch._dynamo.disable
    def eval(self):
        self.regularization = False
        super().eval()
