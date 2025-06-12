import copy

import torch
import torch.nn.functional as F
from torch import nn

from SDT import SDT


class Booster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        depth: int,
        n_estimators: int,
        learning_rate: float,
        regularization_coef: float = 0.0,
    ) -> None:

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.regularization_coef = regularization_coef
        self.n_estimators = n_estimators

        self._create_models(
            regularization_coef=regularization_coef,
        )
        self.learning_rate = learning_rate
        self._build_masks(depth=depth)

    def _build_masks(self, depth):
        num_leaves = 2**depth
        num_nodes = 2**depth - 1

        left_mask = torch.zeros((num_nodes, num_leaves))
        right_mask = torch.zeros((num_nodes, num_leaves))

        for d in range(depth):
            splits_per_node = num_leaves // (2**d)
            for i in range(2**d):
                node_idx = (2**d - 1) + i
                start = i * splits_per_node
                mid = start + splits_per_node // 2
                end = start + splits_per_node

                left_mask[node_idx, start:mid] = 1.0
                right_mask[node_idx, mid:end] = 1.0

        self.register_buffer("left_mask", left_mask)
        self.register_buffer("right_mask", right_mask)

    def _create_models(self, regularization_coef: float) -> None:
        regularization = regularization_coef != 0.0

        estimator = SDT(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            depth=self.depth,
            regularization=regularization,
        )
        self.models = nn.ModuleList(
            [copy.deepcopy(estimator) for _ in range(self.n_estimators)]
        )

    def create_pred(self, X):
        device = next(self.parameters()).device
        return torch.zeros(
            (X.size(0), self.output_dim),
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )

    def update_pred(self, pred: torch.Tensor, update: torch.Tensor):
        pred = (pred + self.learning_rate * update).detach().requires_grad_(True)
        return pred

    def fit_forward(self, X: torch.Tensor, y: torch.Tensor, criterion):
        pred = self.create_pred(X)

        for m in range(self.n_estimators):
            loss = criterion(pred, y)

            grad = torch.autograd.grad(loss, pred, create_graph=False)[0]

            update, reg_term = self.models[m](X, self.left_mask, self.right_mask)

            loss = F.mse_loss(update, -grad)

            if reg_term is not None and self.regularization_coef != 0.0:
                loss += self.regularization_coef * reg_term

            loss.backward()

            pred = self.update_pred(pred, update)

        return pred

    def forward(self, X: torch.Tensor):
        output = 0.0
        for model in self.models:
            update, _ = model(X, self.left_mask, self.right_mask)
            output += self.learning_rate * update

        return output