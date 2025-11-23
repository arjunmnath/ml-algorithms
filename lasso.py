"""
lasso.py — 2-D Lasso subgradient optimizer (demo)
=============================================================
Provides a compact helper for running subgradient descent on a 2-D Lasso
objective:

    0.5 * mean((y - X @ w)**2) + factor * ||w||_1

Contents:
- LassoSubgradientOptimizer: builds MSE + L1 subgradients, runs optimization via
  SubgradientDescent, and offers contour + surface visualizations of the loss
  and optimization path.
"""

from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from subgradient import SubgradientDescent


class LassoSubgradientOptimizer:
    """
    Subgradient descent helper specialized for 2-D Lasso regression weight space.

    Optimizes the objective:
        0.5 * mean((y - X @ w)**2) + factor * ||w||_1
    """

    def __init__(
        self,
        X_data: np.ndarray,
        y_true: np.ndarray,
        factor: float = 1e-3,
        step_size: float = 1e-2,
        n_epochs: int = 1000,
        tolerance: float = 1e-4,
        rng: Optional[np.random.Generator] = None,
    ):
        self.X = np.asarray(X_data, dtype=float)
        self.y = np.asarray(y_true, dtype=float).reshape(-1)
        if self.X.ndim != 2 or self.X.shape[1] != 2:
            raise ValueError("This optimizer expects X_data with exactly two features.")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X_data and y_true must have matching sample counts.")

        self.factor = float(factor)
        self.step_size = float(step_size)
        self.n_epochs = int(n_epochs)
        self.tolerance = float(tolerance)
        self.rng = rng or np.random.default_rng()
        self.num_vars = self.X.shape[1]

    def model(self, w: np.ndarray) -> np.ndarray:
        return self.X @ w

    def mse_loss(self, w: np.ndarray) -> float:
        residuals = self.y - self.model(w)
        return 0.5 * np.mean(residuals**2)

    def mse_grad(self, w: np.ndarray) -> np.ndarray:
        residuals = self.y - self.model(w)
        return -(self.X.T @ residuals) / self.X.shape[0]

    def l1_subgradient(self, w: np.ndarray) -> np.ndarray:
        grad = np.sign(w)
        zero_mask = np.isclose(w, 0.0)
        if zero_mask.any():
            grad[zero_mask] = self.rng.uniform(-1.0, 1.0, size=zero_mask.sum())
        return grad

    def objective(self, w: np.ndarray) -> float:
        return self.mse_loss(w) + self.factor * np.linalg.norm(w, 1)

    def objective_subgradient(self, w: np.ndarray) -> np.ndarray:
        return self.mse_grad(w) + self.factor * self.l1_subgradient(w)

    def optimize(
        self, w_seed: Optional[Union[np.ndarray, Sequence[float]]] = None
    ) -> List[np.ndarray]:
        if w_seed is None:
            start = self.rng.uniform(low=-1.0, high=1.0, size=self.num_vars)
        else:
            start = np.asarray(w_seed, dtype=float).reshape(self.num_vars)

        descent = SubgradientDescent(
            func=self.objective,
            rng=self.rng,
            subgradient_fn=self.objective_subgradient,
        )
        history = descent.optimize(
            start=start,
            step_size=self.step_size,
            n_epochs=self.n_epochs,
            grad_tol=self.tolerance,
        )
        return history

    def _meshgrid(
        self, bounds: Tuple[float, float], resolution: int = 200
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.linspace(bounds[0], bounds[1], resolution)
        y = np.linspace(bounds[0], bounds[1], resolution)
        X, Y = np.meshgrid(x, y)
        vectorized = np.vectorize(
            lambda a, b: self.objective(np.array([a, b], dtype=float)), otypes=[float]
        )
        Z = vectorized(X, Y)
        return X, Y, Z

    def _bounds_from_path(self, path: Sequence[np.ndarray]) -> Tuple[float, float]:
        pts = np.asarray(path)
        min_val = np.min(pts) * 1.5
        max_val = np.max(pts) * 1.5
        if np.isclose(min_val, max_val):
            max_val = min_val + 1.0
        return (min(min_val, max_val), max(min_val, max_val))

    def plot_loss_contours(
        self,
        path: Sequence[np.ndarray],
        resolution: int = 200,
    ) -> None:
        bounds = self._bounds_from_path(path)
        X, Y, Z = self._meshgrid(bounds, resolution)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        contour = ax.contour(X, Y, Z, 40, cmap="viridis")
        plt.colorbar(contour, ax=ax)
        pts = np.asarray(path)
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color="red",
            marker="o",
            markersize=3,
            label="Optimization Path",
        )
        ax.legend()
        ax.set_xlabel("w0")
        ax.set_ylabel("w1")
        ax.set_title("Lasso Objective Contours")
        plt.show()

    def plot_loss_surface(
        self,
        path: Sequence[np.ndarray],
        resolution: int = 200,
        elev: float = 30,
        azim: float = 45,
    ) -> None:
        bounds = self._bounds_from_path(path)
        X, Y, Z = self._meshgrid(bounds, resolution)
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.6)
        pts = np.asarray(path)
        zs = np.array([self.objective(p) for p in pts])
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            zs,
            color="black",
            marker="o",
            markersize=4,
            linewidth=2,
            label="Optimization Path",
        )
        ax.scatter(
            pts[-1, 0],
            pts[-1, 1],
            zs[-1],
            color="red",
            marker="x",
            s=60,
            label="Final Point",
        )
        ax.legend()
        ax.set_xlabel("w0")
        ax.set_ylabel("w1")
        ax.set_zlabel("Objective")
        ax.set_title("Lasso Objective Surface")
        ax.view_init(elev=elev, azim=azim)
        plt.show()
