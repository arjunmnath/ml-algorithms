"""
subgradient.py — Subgradient descent utilities for non-smooth 2-D functions
============================================================================

Encapsulates the logic required to run a subgradient-based optimizer on
functions of the form f(x, y) → R that may include non-differentiable points
such as absolute-value ridges. Provides:

- `SubgradientDescent.optimize` to iterate subgradient updates with simple
  backtracking when progress stalls.
- `plot_contours` and `plot_surface` helpers to visualize both the loss
  landscape and the optimization trajectory in 2-D and 3-D respectively.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

Function2D = Callable[[np.ndarray], float]


class SubgradientDescent:
    def __init__(
        self,
        func: Function2D,
        delta: float = 1e-3,
        nondiff_tol: float = 1e-4,
        rng: Optional[np.random.Generator] = None,
        subgradient_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        """
        Args:
            func: Callable that takes a 2-element np.ndarray `[x, y]` and returns f(x, y).
            delta: Finite-difference step used to approximate directional slopes.
            nondiff_tol: Threshold between left/right slopes used to detect kinks.
            rng: Optional numpy random generator for deterministic subgradient sampling.
            subgradient_fn: Optional callable returning a custom subgradient at `point`.
        """
        self.func = func
        self.delta = float(delta)
        self.nondiff_tol = float(nondiff_tol)
        self.rng = rng or np.random.default_rng()
        self._subgradient_fn = subgradient_fn

    def _directional_slopes(self, point: np.ndarray, axis: int) -> Tuple[float, float]:
        """Compute forward and backward finite-difference slopes along `axis`."""
        shift = np.zeros_like(point)
        shift[axis] = self.delta
        forward = (self.func(point + shift) - self.func(point)) / self.delta
        backward = (self.func(point) - self.func(point - shift)) / self.delta
        return backward, forward  # left, right

    def subgradient(self, point: np.ndarray) -> np.ndarray:
        """
        Estimate a subgradient at `point` using one-sided slopes.

        If the slopes disagree beyond `nondiff_tol`, pick a value inside the
        valid subgradient interval [min(left, right), max(left, right)].
        """
        point = np.asarray(point, dtype=float).reshape(2)
        if self._subgradient_fn is not None:
            return np.asarray(self._subgradient_fn(point), dtype=float).reshape(point.shape)

        grad = np.zeros_like(point)
        for axis in range(point.size):
            left, right = self._directional_slopes(point, axis)
            if abs(left - right) <= self.nondiff_tol:
                grad[axis] = 0.5 * (left + right)
            else:
                lo, hi = sorted((left, right))
                grad[axis] = self.rng.uniform(lo, hi)
        return grad

    def optimize(
        self,
        start: Sequence[float],
        step_size: float = 1e-2,
        n_epochs: int = 1000,
        grad_tol: float = 1e-6,
        min_step_size: float = 1e-6,
        backtrack: float = 0.5,
    ) -> List[np.ndarray]:
        """
        Run subgradient descent with simple backtracking.

        Returns:
            A list of visited points (including the start) representing
            the accepted optimization trajectory.
        """
        x = np.asarray(start, dtype=float).reshape(2)
        history: List[np.ndarray] = [x.copy()]
        best_val = self.func(x)
        alpha = float(step_size)

        for _ in range(n_epochs):
            g = self.subgradient(x)
            if np.linalg.norm(g) <= grad_tol:
                break

            updated = False
            local_alpha = alpha
            while local_alpha >= min_step_size:
                candidate = x - local_alpha * g
                candidate_val = self.func(candidate)
                if candidate_val <= best_val:
                    x = candidate
                    best_val = candidate_val
                    history.append(x.copy())
                    alpha = local_alpha  # remember useful step size
                    updated = True
                    break
                local_alpha *= backtrack

            if not updated:
                # No successful step — shrink baseline step size and continue
                alpha *= backtrack
                if alpha < min_step_size:
                    break

        return history

    def _compute_grid(
        self, bounds: Tuple[float, float], resolution: int = 200
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Utility to evaluate `func` on a meshgrid over `bounds`."""
        x = np.linspace(bounds[0], bounds[1], resolution)
        y = np.linspace(bounds[0], bounds[1], resolution)
        X, Y = np.meshgrid(x, y)
        vectorized = np.vectorize(
            lambda a, b: self.func(np.array([a, b])), otypes=[float]
        )
        Z = vectorized(X, Y)
        return X, Y, Z

    def plot_contours(
        self,
        bounds: Tuple[float, float],
        path: Optional[Sequence[np.ndarray]] = None,
        resolution: int = 200,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Plot contour lines for the objective and optionally overlay a trajectory.
        """
        X, Y, Z = self._compute_grid(bounds, resolution)
        ax = ax or plt.figure(figsize=(8, 6)).add_subplot(111)
        contour = ax.contour(X, Y, Z, 40, cmap="viridis")
        plt.colorbar(contour, ax=ax)
        if path is not None and len(path) > 1:
            pts = np.asarray(path)
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                color="red",
                marker="o",
                markersize=3,
                label="Path",
            )
            ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Objective Contours with Subgradient Path")
        plt.show()

    def plot_surface(
        self,
        bounds: Tuple[float, float],
        path: Optional[Sequence[np.ndarray]] = None,
        resolution: int = 200,
        ax: Optional[plt.Axes] = None,
        elev: float = 30,
        azim: float = 60,
    ) -> plt.Axes:
        """
        Plot the 3-D surface of the objective and optionally overlay a trajectory.
        """
        X, Y, Z = self._compute_grid(bounds, resolution)
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            X, Y, Z, cmap=cm.viridis, alpha=0.6, linewidth=0, antialiased=False
        )
        if path is not None and len(path) > 1:
            pts = np.asarray(path)
            zs = np.array([self.func(p) for p in pts])
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                zs,
                color="black",
                marker="o",
                markersize=4,
                linewidth=2,
                label="Path",
            )
            ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x, y)")
        ax.set_title("3D Surface with Subgradient Trajectory")
        ax.view_init(elev=elev, azim=azim)
        plt.show()

