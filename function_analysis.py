"""
function_analysis.py — visualize and compare optimizer descent paths
====================================================================

Utility class for analyzing how different optimizers move across a
2-D loss landscape. Takes a function f(x, y) (using Tensor ops), walks
each optimizer from a common starting point, and records the parameter
trajectory.

Features:
- run multiple optimizers in parallel (`walk`)
- capture descent paths as numpy arrays
- plot 3D surface trajectories (`plot_descent_path`)
- plot 2D contour paths (`plot_contours`)

A small tool for understanding optimizer behavior on hand-crafted
functions such as Rosenbrock.
"""

from types import FunctionType
from typing import Dict, List, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from first_order_optimizers import Optimizer
from tensors import Tensor


class FunctionAnalysis:
    def __init__(self, f: FunctionType):
        self._f = f

    def __call__(self, x: Union[np.ndarray, Tensor], y: Union[np.ndarray, Tensor]):
        return self._f(x, y)

    def walk(
        self,
        start: Tuple[float, float],
        n_steps: int,
        step_sizes: List[float],
        optimizer_list: List[Type[Optimizer]],
        clip_gradient: bool = False,
        bounds: Tuple[float, float] = [-2, 2],
    ):
        assert len(step_sizes) == len(optimizer_list)
        x, y = start
        current_position = {
            optim.__name__: [Tensor(x), Tensor(y)] for optim in optimizer_list
        }
        _optimizers = [
            optim(
                current_position[optim.__name__], learning_rate=step_sizes[i], f=self._f
            )
            for i, optim in enumerate(optimizer_list)
        ]
        points = {optim.__class__.__name__: [[x, y]] for optim in _optimizers}
        for step in tqdm(range(n_steps)):
            for optim in _optimizers:
                _x, _y = current_position[optim.__class__.__name__]
                optim.zero_grad()
                f = self(_x, _y)
                f.backward()
                if clip_gradient:
                    optim.clip_grad()
                optim.step()
                points[optim.__class__.__name__].append([_x.item(), _y.item()])
        return {
            key: np.array(val, dtype=np.float64).clip(*bounds)
            for key, val in points.items()
        }

    def plot_descent_path(self, points_dict: Dict[str, np.ndarray], X, Y, Z):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7)
        colors = ["b", "g", "r", "c", "m", "y", "k", "orange", "purple"]
        for idx, (name, points) in enumerate(points_dict.items()):
            ax.plot(
                points[:, 0],
                points[:, 1],
                self(points[:, 0], points[:, 1]),
                color=colors[idx],
                marker="o",
                markersize=3,
                label=name,
            )

        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_title("Gradient Descent on given Function")
        ax.legend()
        plt.show()

    def plot_contours(self, points_dict: Dict[str, np.ndarray], X, Y, Z):
        """Plots the contours and the gradient descent path."""
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)

        ax.contour(X, Y, Z, 50, cmap="viridis")
        colors = ["b", "g", "r", "c", "m", "y", "k", "orange", "purple"]
        for idx, (name, points) in enumerate(points_dict.items()):
            ax.plot(
                points[:, 0],
                points[:, 1],
                color=colors[idx % len(colors)],
                marker="o",
                markersize=3,
                label=name,
            )

        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_title("Contours of Given Function with Descent Path")
        ax.legend()
        plt.show()
