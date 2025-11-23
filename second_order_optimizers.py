"""
second_order_optimizers.py — Quasi-Newton optimization using the Function API
==========================================================
A lightweight BFGS optimizer built on the project's `Function` abstraction.
The optimizer performs unconstrained minimization using numerical gradients,
a backtracking Armijo line search, and the standard BFGS Hessian update.

Features:
- works with any `Function` subclass providing value/gradient access
- numerical gradient evaluation through the `Function` API
- backtracking Armijo line search for stable descent
- automatic curvature checking to maintain a valid inverse Hessian
- returns the optimization trajectory for visualization or analysis

Typical usage:
    opt = BFGS(f)
    history = opt.optimize(max_iter=200)

The result is a NumPy array containing the path of iterates, clipped to the
function’s domain bounds as defined by `Function.bound`.
"""

from typing import List, Tuple

import numpy as np

from function import Function


class BFGS:
    """
    BFGS optimizer that works with the provided Function API.
    Usage:
        opt = BFGS(f)
        history = opt.optimize(max_iter=200)
    """

    def __init__(
        self,
        func: Function,
        eps: float = 1e-6,
    ):
        self.f = func
        self.eps = eps

    def _get_value_and_grad(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Return (f(x), grad(x)) using numeric."""
        fx = float(self.f(x))
        g = np.asarray(self.f.gradient(), dtype=float).ravel()
        return fx, g

    @staticmethod
    def _armijo_line_search(
        value_fn,
        x: np.ndarray,
        p: np.ndarray,
        g: np.ndarray,
        beta: float = 0.5,
        c: float = 1e-4,
        max_iters: int = 30,
    ) -> float:
        """
        Simple backtracking Armijo line search.
        """
        fx = float(value_fn(x))
        alpha = 1
        g_dot_p = float(np.dot(g, p))
        if g_dot_p >= 0:
            return 0.0
        for _ in range(max_iters):
            x_new = x + alpha * p
            fx_new = float(value_fn(x_new))
            if fx_new <= fx + c * alpha * g_dot_p:
                return alpha
            alpha *= beta
        return 0.0

    def optimize(self, max_iter) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        x = np.asarray(self.f.start, dtype=float).ravel()
        n = x.size
        _, g = self._get_value_and_grad(x)
        history: List[List[float]] = [x.copy().tolist()]
        Hk = np.eye(n, dtype=float)
        for _ in range(max_iter):
            if np.linalg.norm(g) < self.eps:
                break
            p = -Hk.dot(g)
            # Ensure descent direction (guard for numerical issues)
            if float(np.dot(g, p)) >= 0:
                p = -g

            # Line search (Armijo)
            alpha = self._armijo_line_search(
                self.f.value,
                x,
                p,
                g,
            )
            if alpha == 0.0:
                # line search failed: stop
                break

            x_new = x + alpha * p  #  % Update the correct point
            _, g_new = self._get_value_and_grad(x_new)  # % Calculate the new gradient
            s = x_new - x  # % Compute the difference in position
            y = g_new - g  #  % Compute the difference in gradient
            rho = 1 / float(np.dot(y, s))  # rho = 1 / (y^T s)
            if rho > 0.0:
                I = np.eye(n, dtype=float)
                Vy = I - rho * np.outer(s, y)  # rho(k) * delta(k) * y(k) ^ T
                Hk = Vy.dot(Hk).dot(Vy.T) + rho * np.outer(s, s)
            else:
                # curvature condition failed -> keep Hk unchanged
                pass

            # step forward
            x = x_new
            g = g_new
            history.append(x.tolist())
        return np.array(history, dtype=np.float64).clip(*self.f.bound)
