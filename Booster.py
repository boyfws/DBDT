import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import numpy.typing as npt

from sklearn.base import BaseEstimator

from Torch_booster import Booster

from typing import Optional

from tqdm import tqdm

import sys


class BoosterWrapper(BaseEstimator):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            depth: int,
            n_estimators: int,
            booster_learning_rate: float,
            booster_learning_rate_decay: float,
            regularization_coef: float,

            epochs: int,
            batch_size: int,
            learning_rate: float,
            loss: nn.Module,
            verbose: bool = False,

            split_function: Optional[nn.Module] = None,
            t: float = 1,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.n_estimators = n_estimators
        self.booster_learning_rate = booster_learning_rate
        self.booster_learning_rate_decay = booster_learning_rate_decay

        self.regularization_coef = regularization_coef

        self.split_function = split_function
        self.t = t

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.verbose = verbose

        self.loss = loss

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _build_model(
            self,
            input_dim: int
    ) -> None:
        self.base = Booster(
            input_dim=input_dim,
            output_dim=self.output_dim,
            depth=self.depth,
            n_estimators=self.n_estimators,
            learning_rate=self.booster_learning_rate,
            learning_rate_decay=self.booster_learning_rate_decay,
            regularization_coef=self.regularization_coef,
            split_function=self.split_function,
            t=self.t,
        )
        self.base.compile(fullgraph=True)
        self.base.to(self.device)

    def fit(
            self,
            X: npt.NDArray,
            y: npt.NDArray
    ) -> None:
        X = torch.tensor(X).float()
        y = torch.tensor(y).float()

        self._build_model(X.size(1))

        dataset = TensorDataset(
            X, y
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=self.device.type == "cuda",
            prefetch_factor=4,
        )

        self.optim = torch.optim.Adam(self.base.parameters(), lr=self.learning_rate)
        epoch_len = len(str(self.epochs))

        for epoch in range(self.epochs):

            if self.verbose:
                pbar = tqdm(
                    loader,
                    desc=f"Epoch {str(epoch + 1).rjust(epoch_len)}/{self.epochs}",
                    file=sys.stdout,
                    leave=True,
                )
            else:
                pbar = loader

            cum_loss = 0
            size = 0

            for i, (batch, target) in enumerate(pbar):
                self.optim.zero_grad()

                batch, target = batch.to(self.device), target.to(self.device)
                bs = batch.size(0)

                pred = self.base.fit_forward(batch, target, self.loss)

                with torch.no_grad():
                    loss = self.loss(pred, target)

                self.optim.step()

                cum_loss += loss.item() * bs
                size += bs

                if self.verbose:
                    if i != len(loader) - 1:
                        pbar.set_postfix({
                            "Loss": f"{(loss.item()):.4f}",
                        })
                    else:
                        pbar.set_postfix({
                            "Epoch Loss": f"{(cum_loss / size):.4f}",
                        })

        self.base.eval()

    def predict(
            self,
            X: npt.NDArray
    ) -> np.ndarray:
        X = torch.tensor(X).float()

        dataset = TensorDataset(X)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=self.device.type == "cuda",
            prefetch_factor=4,
        )

        with torch.no_grad():
            res = []
            for (batch,) in loader:
                batch = batch.to(self.device)
                pred = self.base(batch)
                res.append(pred.cpu())

            return torch.cat(res, dim=0).numpy()
