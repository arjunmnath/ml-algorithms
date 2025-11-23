"""
function.py — Base Function API with numerical gradients and Hessians
====================================================================

This module defines the abstract `Function` class used throughout the project
for optimization experiments. It provides a uniform interface for evaluating
scalar functions, computing numerical gradients and Hessians, and tracking
cached values for optimizers.

Features:
- abstract `Function` class with:
    • value(x)     → scalar function value
    • gradient()   → numerical gradient via central differences
    • hessian()    → numerical Hessian (dense matrix)
    • hessian_vector_product() for second-order methods
    • caching of the last evaluation point for efficient reuse
- optional `start` and `bound` attributes for optimizers and visualizers
- several benchmark test functions commonly used in optimization:
    • Rosenbrock
    • QuadraticBowl
    • Beale
    • Goldstein–Price
    • Himmelblau
    • Three-Hump Camel

These functions provide consistent 2-D testbeds for verifying gradient-based
and Newton-type optimizers across convex and non-convex landscapes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np


class Function(ABC):
    def __init__(self):
        self._last_x: Optional[np.ndarray] = None
        self._last_value: Optional[np.ndarray] = None
        self._last_grad: Optional[np.ndarray] = None
        self._last_hess: Optional[np.ndarray] = None
        self.eps = 1e-6
        self.start: Tuple[float, float] = None
        self.bound: Tuple[float, float] = None

    @abstractmethod
    def value(self, x_: np.ndarray) -> np.ndarray:
        """Return scalar function value at x (1-D numpy array)."""
        raise NotImplementedError

    def _grad(self, x_: np.ndarray) -> np.ndarray:
        n = x_.size
        grad = np.zeros(n, dtype=np.float64)
        for i in range(n):
            e = np.zeros(n, dtype=float)
            e[i] = self.eps
            grad[i] = (self.value(x_ + e) - self.value(x_ - e)) / (2 * self.eps)
        return grad

    def _hess(self, x_: np.ndarray) -> np.ndarray:
        n = x_.size
        H = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i, n):
                ei = np.zeros(n, dtype=float)
                ej = np.zeros(n, dtype=float)
                ei[i] = self.eps
                ej[j] = self.eps
                f_pp = self.value(x_ + ei + ej)
                f_pm = self.value(x_ + ei - ej)
                f_mp = self.value(x_ - ei + ej)
                f_mm = self.value(x_ - ei - ej)
                H_ij = (f_pp - f_pm - f_mp + f_mm) / (4 * self.eps * self.eps)
                H[i, j] = H_ij
                H[j, i] = H_ij
        return H

    def __call__(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float).ravel()
        self._last_x = x.copy()
        self._last_value = self.value(x)
        self._last_grad = None
        self._last_hess = None
        return self._last_value

    def gradient(self) -> np.ndarray:
        x = self._last_x
        assert x is not None
        self._last_grad = self._grad(x)
        return self._last_grad

    def hessian(self) -> np.ndarray:
        """
        Return (dense) Hessian at x. Uses analytic _hess if implemented; otherwise numeric second
        derivatives (central differences). Caches result.
        """
        x = self._last_x
        assert x is not None
        self._last_hess = self._hess(x)
        return self._last_hess

    def hessian_vector_product(self) -> np.ndarray:
        if self._last_grad is None:
            self.gradient()
        if self._last_hess is None:
            self.hessian()
        H = self._last_hess
        g = self._last_grad
        assert isinstance(H, np.ndarray) and isinstance(
            g, np.ndarray
        )  # asserts the gradient and hessian is properly set
        assert H.shape[0] == H.shape[1] and g.shape[0] == H.shape[0]
        return H @ g

    def invalidate_cache(self):
        self._last_x = None
        self._last_value = None
        self._last_grad = None
        self._last_hess = None


class Rosenbrock(Function):
    """
    Standard 2D Rosenbrock function:
        f(x, y) = (a - x)^2 + b * (y - x^2)^2
    Default: a = 1, b = 100
    """

    def __init__(self, a: float = 1.0, b: float = 100.0):
        super().__init__()
        self.a = float(a)
        self.b = float(b)
        self.start: Tuple[float, float] = (-1.2, 1.0)
        self.bound: Tuple[float, float] = (-2, 2)
        self.f = lambda x, y: (a - x) ** 2 + b * (y - x**2) ** 2

    def value(self, x_: np.ndarray) -> float:
        assert x_.size == 2
        x, y = x_
        return self.f(x, y)


class QuadraticBowl(Function):
    def __init__(self):
        super().__init__()
        self.start: Tuple[float, float] = (-2, -2)
        self.bound: Tuple[float, float] = (-2, 2)
        self.f = lambda x, y: 4 * x**2 + y**2 - 2 * x * y

    def value(self, x_: np.ndarray) -> float:
        assert x_.size == 2
        x, y = x_
        return self.f(x, y)


class Beale(Function):
    def __init__(self):
        super().__init__()
        self.start: Tuple[float, float] = (-4.0, 4.0)
        self.bound: Tuple[float, float] = (-4.5, 4.5)
        self.f = (
            lambda x, y: (1.5 - x + x * y) ** 2
            + ((2.5 - x + x * y**2) ** 2)
            + ((2.625 - x + x * y**3) ** 2)
        )

    def value(self, x_: np.ndarray) -> float:
        assert x_.size == 2
        x, y = x_
        return self.f(x, y)


class GoldsteinPrice(Function):
    def __init__(self):
        super().__init__()
        self.start: Tuple[float, float] = (-2.0, 2.0)
        self.bound: Tuple[float, float] = (-2.0, 2.0)
        self.f = lambda x, y: (
            1
            + (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)
            * (x + y + 1) ** 2
        ) * (
            30
            + (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2)
            * (2 * x - 3 * y) ** 2
        )

    def value(self, x_: np.ndarray) -> float:
        assert x_.size == 2
        x, y = x_
        return self.f(x, y)


class Himmelblau(Function):
    def __init__(self):
        super().__init__()
        self.start: Tuple[float, float] = (5.0, 5.0)
        self.bound: Tuple[float, float] = (-5.0, 5.0)
        self.f = lambda x, y: (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

    def value(self, x_: np.ndarray) -> float:
        assert x_.size == 2
        x, y = x_
        return self.f(x, y)


class ThreeHumpCamel(Function):
    def __init__(self):
        super().__init__()
        self.start: Tuple[float, float] = (5.0, 5.0)
        self.bound: Tuple[float, float] = (-5.0, 5.0)
        self.f = lambda x, y: (
            2 * (x**2) - 1.05 * (x**4) + ((x**6) / 6) + (x * y) + (y**2)
        )

    def value(self, x_: np.ndarray) -> float:
        assert x_.size == 2
        x, y = x_
        return self.f(x, y)
