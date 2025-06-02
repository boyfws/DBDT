import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

from typing import Optional
from SDT import SDT

from .Compute_grad import compute_loss_gradient


class Booster(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 depth: int,
                 n_estimators: int,
                 learning_rate: float,
                 learning_rate_decay: float,
                 regularization_coef: float = 0.0,
                 split_function: Optional[nn.Module] = None,
                 t: float = 1,
    ) -> None:

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.regularization_coef = regularization_coef
        self.n_estimators = n_estimators

        self._create_models(
            regularization_coef=regularization_coef,
            split_function=split_function,
            t=t,
        )

        lr_values = learning_rate / (1 + learning_rate_decay * torch.arange(n_estimators))
        self.register_buffer("lr", lr_values)

    def _create_models(
            self,
            regularization_coef: float,
            split_function: Optional[nn.Module] = None,
            t: float = 1

    ) -> None:
        regularization = regularization_coef != 0.0

        estimator = SDT(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            depth=self.depth,
            regularization=regularization,
            split_function=split_function,
            t=t,

        )
        self.models = nn.ModuleList([copy.deepcopy(estimator) for _ in range(self.n_estimators)])

    def fit_forward(self, X: torch.Tensor, y: torch.Tensor, criterion):
        pred = torch.zeros(
            (X.size(0), self.output_dim)
        ).float().to(next(self.parameters()).device)

        for m in range(self.n_estimators):
            grad = compute_loss_gradient(pred, y, criterion).detach()

            update, reg_term = self.models[m](X)

            loss = F.mse_loss(update, -grad)

            if reg_term is not None and self.regularization_coef != 0.0:
                loss += self.regularization_coef * reg_term

            loss.backward()

            pred = (pred + self.lr[m].item() * update).detach()

        return pred

    def forward(self, X: torch.Tensor):
        return sum(
            self.lr[i] * model(X)[0]
            for i, model in enumerate(self.models)
        )