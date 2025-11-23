"""
logistic_regression.py — logistic regression using the autodiff Tensor
=====================================================================

Implements a small logistic regression model built entirely on top of the
custom `Tensor` class and the optimizer framework. Forward passes, loss
computations, and training updates all run through Tensor ops and backprop.

The model supports:
- sigmoid activation via Tensor ops
- binary cross-entropy loss
- training with any Optimizer subclass (`fit`)
- probability predictions and hard labels (`predict_proba`, `predict`)
- a 2-D visualization demo (`run_demo`) showing decision regions
- optimizer comparison (`benchmark_optimizers`) plotting loss curves
"""

from typing import List, Type

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from first_order_optimizers import Optimizer
from tensors import Tensor


class LogisticRegression:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def _sigmoid(self, z: Tensor) -> Tensor:
        """The sigmoid activation function using Tensor operations."""
        return 1 / (1 + (-z).exp())

    def _bce(self, y_predicted: Tensor, y_true: Tensor, n_samples: int, epsilon=1e-7):
        loss = (
            -1
            / n_samples
            * (
                (y_true * (y_predicted + epsilon).log())
                + ((1 - y_true) * (1 - y_predicted + epsilon).log())
            ).sum()
        )
        return loss

    def fit(self, X, y, optimizer: Optimizer):
        n_samples, n_features = X.shape
        self.weights = Tensor(np.zeros((n_features, 1)))
        self.bias = Tensor(np.zeros(1))
        _optimizer = optimizer([self.weights, self.bias], learning_rate=self.lr, f=None)
        losses = []
        X_tensor = Tensor(X)
        y_tensor = Tensor(y.reshape(-1, 1))
        for epoch in range(self.epochs):
            linear_output = X_tensor @ self.weights + self.bias
            y_hat = self._sigmoid(linear_output)
            loss = self._bce(y_hat, y_tensor, n_samples)
            losses.append(loss.data.item())
            _optimizer.zero_grad()
            loss.backward()
            _optimizer.step()
        return losses

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts probabilities for raw numpy arrays."""
        X_tensor = Tensor(X)
        linear_output = X_tensor @ self.weights + self.bias
        y_hat_tensor = self._sigmoid(linear_output)
        return y_hat_tensor.data

    def predict(self, X: np.ndarray, threshold=0.5) -> np.ndarray:
        """Predicts class labels (0 or 1)."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    @classmethod
    def run_demo(
        cls,
        *,
        optimizer_cls,
        lr=0.1,
        epochs=1000,
        n_samples=100,
        n_features=1,
        test_size=0.2,
        data_random_state=42,
        split_random_state=123,
        plot=True,
    ):
        assert n_features <= 2, "PCA not implemented for dimension > 2"
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=min(n_features, 2),
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=data_random_state,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=split_random_state
        )
        model = cls(lr=lr, epochs=epochs)
        model.fit(X_train, y_train, optimizer_cls)
        if plot:
            plt.figure(figsize=(6, 5))
            x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
            y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300)
            )
            grid = np.c_[xx.ravel(), yy.ravel()]
            probs = model.predict_proba(grid).reshape(xx.shape)
            plt.contourf(xx, yy, probs, levels=25, alpha=0.35)
            cs = plt.contour(xx, yy, probs, levels=[0.5], colors="k", linewidths=1.2)

            label = "decision boundary (p=0.5)"
            try:
                cs.collections[0].set_label(label)
            except Exception:
                proxy = Line2D([0], [0], color="k", lw=1.2, label=label)
                handles, labels = plt.gca().get_legend_handles_labels()
                handles.append(proxy)
                plt.legend(handles=handles)

            train_probs = model.predict_proba(X_train).ravel()
            train_pred = (train_probs >= 0.5).astype(int)
            mis_mask = train_pred != y_train
            plt.scatter(
                X_train[:, 0],
                X_train[:, 1],
                c=y_train,
                cmap="viridis",
                s=40,
                edgecolor="k",
                alpha=0.9,
                label="train (true label)",
            )

            # overlay misclassified as black X's (prominent)
            if mis_mask.any():
                plt.scatter(
                    X_train[mis_mask, 0],
                    X_train[mis_mask, 1],
                    marker="x",
                    color="k",
                    s=100,
                    linewidths=2,
                    label="misclassified (train)",
                )

            plt.xlabel("feature 0")
            plt.ylabel("feature 1")
            plt.title(
                "LogisticRegression: Decision region (shaded). Train colored by true label; misclassified marked"
            )
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.legend(loc="best")
            plt.tight_layout()
            plt.show()
        return model

    @classmethod
    def benchmark_optimizers(
        cls,
        *,
        optimizer_list: List[Type[Optimizer]],
        step_sizes: List[float],
        epochs=1000,
        n_samples=100,
        n_features=1,
        data_random_state=42,
        plot=True,
    ):
        assert len(optimizer_list) == len(step_sizes)
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=2,
            random_state=data_random_state,
        )
        X_train, y_train = X, y
        results = {}
        models = {}
        for opt_cls, lrate in zip(optimizer_list, step_sizes):
            label = f"{getattr(opt_cls, '__name__', str(opt_cls))}@{lrate}"
            model = cls(lr=lrate, epochs=epochs)
            losses = model.fit(X_train, y_train, opt_cls)
            losses = list(losses)
            results[label] = losses
            models[label] = model

        if plot:
            plt.figure(figsize=(8, 5))
            for label, losses in results.items():
                steps = list(range(len(losses)))
                plt.plot(steps, losses, label=label, linewidth=2)
            plt.xlabel("Update step")
            plt.ylabel("Training loss")
            plt.title("Optimizer benchmark (LogisticRegression) — training loss vs update step")
            plt.legend(framealpha=0.9)
            plt.grid(alpha=0.15)
            plt.tight_layout()
            plt.show()
        return results, models
