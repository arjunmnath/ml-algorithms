"""
constrained_optimizers.py — Minimal equality & inequality constrained solvers
============================================================================

Implements simple 2-D Lagrange and primal–dual gradient methods for constrained
optimization. These solvers use finite-difference gradients, fixed step sizes,
and lightweight multiplier updates suitable for demos and visualization.

Classes:
- EqualityConstrainedOptimizer
      Solves  min f(x)  s.t. h_i(x)=0  via Lagrange-multiplier gradient steps.
- InequalityConstrainedOptimizer
      Solves  min f(x)  s.t. g_i(x)<=0 using projected primal–dual updates.

Features:
- central finite-difference gradients
- KKT-style multiplier updates
- built-in 3D and contour plotting of optimization paths
- simple API for small educational examples
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

ConstraintFn = Callable[[np.ndarray], float]
ObjectiveFn = Callable[[np.ndarray], float]

__all__ = [
    "EqualityConstrainedOptimizer",
    "InequalityConstrainedOptimizer",
]


@dataclass
class OptimizationResult:
    """Lightweight container for the optimization trajectory."""

    path: np.ndarray
    multipliers: np.ndarray
    stop_iter: int


class _BaseConstrainedOptimizer:
    """
    Shared utilities for the equality/inequality constrained optimizers.

    The optimizers follow the "call -> optimize -> optional plot"
    """

    def __init__(
        self,
        objective: ObjectiveFn,
        constraints: Optional[Sequence[ConstraintFn]] = None,
        *,
        num_vars: int = 2,
        tol: float = 1e-4,
        delta_fd: float = 1e-4,
        step_size_x: float = 5e-3,
        step_size_lambda: float = 5e-3,
        n_epochs: int = 2000,
        random_state: Optional[int] = None,
    ) -> None:
        self.objective = objective
        self.constraints = list(constraints or [])
        self.num_vars = num_vars
        self.tol = tol
        self.delta_fd = delta_fd
        self.step_size_x = step_size_x
        self.step_size_lambda = step_size_lambda
        self.n_epochs = n_epochs
        if random_state is not None:
            np.random.seed(random_state)

    def _finite_diff_grad(self, func: Callable[[np.ndarray], float]):
        delta = self.delta_fd

        def grad_fn(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float).ravel()
            grad = np.zeros(self.num_vars, dtype=float)
            for i in range(self.num_vars):
                e = np.zeros(self.num_vars, dtype=float)
                e[i] = delta
                grad[i] = (func(x + e) - func(x - e)) / (2.0 * delta)
            return grad

        return grad_fn

    @staticmethod
    def _stack_constraint_grads(
        grad_list: Sequence[Callable[[np.ndarray], np.ndarray]], x: np.ndarray
    ) -> np.ndarray:
        if not grad_list:
            return np.zeros((0, x.size), dtype=float)
        return np.stack([grad_fn(x) for grad_fn in grad_list])

    def plot_descent_path(
        self,
        points: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        *,
        elev: float = 30,
        azim: float = 60,
        title: str = "Gradient Descent on Objective",
    ):
        points = np.asarray(points, dtype=float)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7)
        z_vals = np.array([self.objective(pt) for pt in points])
        ax.plot(
            points[:, 0],
            points[:, 1],
            z_vals,
            color="b",
            marker="o",
            markersize=3,
            label="Optimization Path",
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x, y)")
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def plot_contours(
        self,
        points: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        *,
        title: str = "Optimization Path (Contours)",
    ):
        points = np.asarray(points, dtype=float)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)
        contour = ax.contour(X, Y, Z, 50, cmap="viridis")
        ax.clabel(contour, inline=True, fontsize=8)
        ax.plot(
            points[:, 0],
            points[:, 1],
            color="b",
            marker="o",
            markersize=3,
            label="Optimization Path",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def plot_path(
        self,
        path: np.ndarray,
        *,
        x_bounds: Optional[Tuple[float, float]] = None,
        y_bounds: Optional[Tuple[float, float]] = None,
        grid_res: int = 200,
        elev: float = 30,
        azim: float = 60,
        title: str = "Constrained Optimization",
        show_contours: bool = False,
    ):
        """
        Visualize the objective surface f(x, y) alongside the optimization path.
        """
        path = np.asarray(path, dtype=float)
        assert path.ndim == 2 and path.shape[1] == 2, "Expect path with shape (k, 2)"

        def _auto_bounds(coords, default=(-3.0, 3.0)):
            if coords.size == 0:
                return default
            lo = np.min(coords) - 0.5
            hi = np.max(coords) + 0.5
            if np.isclose(lo, hi):
                lo -= 1.0
                hi += 1.0
            return float(lo), float(hi)

        x_bounds = x_bounds or _auto_bounds(path[:, 0])
        y_bounds = y_bounds or _auto_bounds(path[:, 1])

        x = np.linspace(*x_bounds, grid_res)
        y = np.linspace(*y_bounds, grid_res)
        X, Y = np.meshgrid(x, y)
        grid = np.array([X, Y])
        Z = self.objective(grid)

        fig3d = self.plot_descent_path(path, X, Y, Z, elev=elev, azim=azim, title=title)
        fig2d = None
        if show_contours:
            fig2d = self.plot_contours(
                path,
                X,
                Y,
                Z,
                title=f"{title} (Contours)",
            )

        plt.show()


class EqualityConstrainedOptimizer(_BaseConstrainedOptimizer):
    """
    Lagrange multiplier based solver for equality-constrained problems:

        min  f(x)
        s.t. h_i(x) = 0
    """

    def optimize(
        self,
        *,
        seed_pos: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        x = (
            np.asarray(seed_pos, dtype=float).ravel()
            if seed_pos is not None
            else np.random.uniform(-1.0, 1.0, size=self.num_vars)
        )
        m = len(self.constraints)
        lam = np.zeros(m, dtype=float)
        steps: List[np.ndarray] = [x.copy()]

        grad_f = self._finite_diff_grad(self.objective)
        grad_h_list = [self._finite_diff_grad(h) for h in self.constraints]

        for k in range(self.n_epochs):
            g_f = grad_f(x)
            h_vals = np.array([h(x) for h in self.constraints], dtype=float)
            h_grads = self._stack_constraint_grads(grad_h_list, x)

            grad_L_x = g_f + (lam @ h_grads if m else 0.0)

            if (
                np.linalg.norm(grad_L_x) < self.tol
                and np.linalg.norm(h_vals) < self.tol
            ):
                break

            x = x - self.step_size_x * grad_L_x
            lam = lam + self.step_size_lambda * h_vals

            steps.append(x.copy())

        return OptimizationResult(
            path=np.asarray(steps, dtype=float),
            multipliers=lam.copy(),
            stop_iter=len(steps) - 1,
        )


class InequalityConstrainedOptimizer(_BaseConstrainedOptimizer):
    """
    Primal–dual gradient method for inequality constraints (KKT conditions):
        min  f(x)
        s.t. g_i(x) <= 0
    """

    def optimize(
        self,
        *,
        seed_pos: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        x = (
            np.asarray(seed_pos, dtype=float).ravel()
            if seed_pos is not None
            else np.random.uniform(-1.0, 1.0, size=self.num_vars)
        )
        m = len(self.constraints)
        lam = np.zeros(m, dtype=float)
        steps: List[np.ndarray] = [x.copy()]

        grad_f = self._finite_diff_grad(self.objective)
        grad_g_list = [self._finite_diff_grad(g) for g in self.constraints]

        for k in range(self.n_epochs):
            g_f = grad_f(x)
            g_vals = np.array([g(x) for g in self.constraints], dtype=float)
            g_grads = self._stack_constraint_grads(grad_g_list, x)

            grad_L_x = g_f + (g_grads.T @ lam if m else 0.0)

            if np.linalg.norm(grad_L_x) < self.tol and np.all(g_vals <= self.tol):
                break

            x = x - self.step_size_x * grad_L_x
            lam = np.maximum(lam + self.step_size_lambda * g_vals, 0.0)

            steps.append(x.copy())

        return OptimizationResult(
            path=np.asarray(steps, dtype=float),
            multipliers=lam.copy(),
            stop_iter=len(steps) - 1,
        )
