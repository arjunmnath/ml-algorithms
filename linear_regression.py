"""
linear_regression.py — linear regression using the autodiff Tensor
=================================================================

A minimal linear regression model built on the custom autodiff `Tensor`.
Forward passes, MSE loss, and gradient updates all run through Tensor ops
and the optimizer framework.

Features:
- mean squared error loss via Tensor operations
- training with any Optimizer subclass (`fit`)
- simple Tensor-based prediction
- 1-D demo that visualizes the fitted line (`run_demo`)
- optimizer benchmarking (`benchmark_optimizers`) with loss plots
"""

from typing import Type

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from first_order_optimizers import *
from tensors import Tensor


class LinearRegression:
    def __init__(self, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def _mse(self, y_predicted: Tensor, y_true: Tensor, n_samples: int):
        loss = ((y_predicted - y_true) ** 2).sum() / n_samples
        return loss

    def fit(self, X, y, optimizer: Optimizer):
        n_samples, n_features = X.shape
        self.weights = Tensor(np.zeros((n_features, 1)))
        self.bias = Tensor(np.zeros(1))
        _optimizer = optimizer(
            [self.weights, self.bias],
            learning_rate=self.lr,
        )
        losses = []
        X_tensor = Tensor(X)
        y_tensor = Tensor(y.reshape(-1, 1))
        for epoch in range(self.epochs):
            y_hat = self.predict(X_tensor)
            loss = self._mse(y_hat, y_tensor, n_samples)
            losses.append(loss.data.item())
            _optimizer.zero_grad()
            loss.backward()
            _optimizer.step()
        return losses

    def predict(self, X: Tensor) -> Tensor:
        return X @ self.weights + self.bias

    @classmethod
    def run_demo(
        cls,
        *,
        optimizer_cls,
        lr=0.01,
        epochs=200,
        n_samples=200,
        n_features=1,
        test_size=0.2,
        data_random_state=42,
        split_random_state=123,
        plot=True,
    ):
        assert (
            n_features == 1
        ), "run_demo plotting supports only 1 feature (straight line)."

        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=15.0,
            random_state=data_random_state,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=split_random_state
        )

        model = cls(lr=lr, epochs=epochs)
        losses = model.fit(X_train, y_train, optimizer_cls)

        x_all = X.reshape(-1, 1)
        x_tensor = Tensor(x_all.astype(float))
        y_pred_all = model.predict(x_tensor).data.reshape(-1)

        sort_idx = np.argsort(x_all[:, 0])
        x_sorted = x_all[sort_idx, 0]
        y_sorted_pred = y_pred_all[sort_idx]

        if plot:
            cmap = plt.get_cmap("viridis")
            plt.figure(figsize=(7, 5))
            plt.scatter(X_train[:, 0], y_train, color=cmap(0.8), s=30, label="train")
            plt.scatter(
                X_test[:, 0], y_test, color=cmap(0.4), s=40, label="test", edgecolor="k"
            )
            plt.plot(x_sorted, y_sorted_pred, color="black", linewidth=2, label="fit")
            plt.xlabel("feature")
            plt.ylabel("target")
            plt.title("Linear regression — data & fitted line")
            plt.legend()
            plt.tight_layout()
            plt.show()

        return model, losses

    @classmethod
    def benchmark_optimizers(
        cls,
        *,
        optimizer_list: List[Type[Optimizer]],
        step_sizes: List[float],
        epochs=200,
        n_samples=500,
        n_features=1,
        data_random_state=42,
        plot=True,
    ):
        assert len(optimizer_list) == len(step_sizes)
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=20.0,
            random_state=data_random_state,
        )
        X_train, y_train = X, y

        results = {}
        models = {}
        for opt_cls, lr in zip(optimizer_list, step_sizes):
            label = f"{getattr(opt_cls, '__name__', str(opt_cls))}@{lr}"
            model = cls(lr=lr, epochs=epochs)
            losses = model.fit(X_train, y_train, opt_cls)
            results[label] = list(losses)
            models[label] = model

        if plot:
            plt.figure(figsize=(9, 5))
            for label, losses in results.items():
                steps = list(range(len(losses)))
                plt.plot(steps, losses, linewidth=2, label=label)
            plt.xlabel("Epoch")
            plt.ylabel("Training loss (MSE)")
            plt.title("Optimizer benchmark (LinearRegression) — training loss")
            plt.legend(framealpha=0.9)
            plt.grid(alpha=0.12)
            plt.tight_layout()
            plt.show()

        return results, models
