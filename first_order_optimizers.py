"""
optimizers.py — small collection of gradient-based optimizers
=============================================================

Defines a lightweight optimizer framework (PyTorch inspired pattern) for Tensor-based autodiff systems.
Each optimizer updates a list of trainable `Tensor` parameters in place using
its own update rule. The base `Optimizer` handles parameter storage, gradient
clearing, clipping, and iteration flow; subclasses implement `_update()`.

Provided optimizers:
- Sgd: plain stochastic gradient descent.
- Momentum: SGD with exponential moving average of gradients.
- Adagrad: per-parameter adaptive learning rates.
- RMSProp: exponentially decayed squared-gradient normalization.
- Adam: momentum + RMSProp with bias correction.
- ArmijoGD: gradient descent with Armijo backtracking line search.
- NesterovGD: Nesterov accelerated gradient (computes look-ahead gradient).

All optimizers operate directly on `Tensor.data` and expect `Tensor.grad` to
be populated by backprop before `step()` is called.
"""

import copy
from abc import ABC, abstractmethod
from types import FunctionType
from typing import List, Optional

import numpy as np

from tensors import Tensor


class Optimizer(ABC):
    """
    Abstract base for all optimizers.
    """

    def __init__(
        self, params: List[Tensor], learning_rate: float, f: Optional[FunctionType]
    ):
        """
        Args:
            params (List[Tensor]): A list of tensors to be optimized.
            learning_rate (float): The learning rate.
        """
        self.f = f
        self.params = [p for p in params if p.requires_grad]
        self.learning_rate = learning_rate

    def step(self) -> None:
        """
        Performs a single optimization step (parameter update).
        """
        for param in self.params:
            self._update(param)

    def zero_grad(self) -> None:
        """
        Clears the gradients of all optimized parameters.
        """
        for param in self.params:
            param.grad.fill(0)

    def clip_grad(self, min: float = -5.0, max: float = 5.0) -> None:
        """
        Clips gradient values to be in a certain range.
        """
        for param in self.params:
            param.grad = np.clip(param.grad, min, max)

    @abstractmethod
    def _update(self, param: Tensor) -> None:
        """
        The specific update rule for an optimizer.
        This method should be implemented by concrete class.
        """
        pass


class Sgd(Optimizer):
    def __init__(self, params: List[Tensor], learning_rate: float = 0.01, f=None):
        super().__init__(params, learning_rate, f)

    def _update(self, param: Tensor) -> None:
        param.data -= self.learning_rate * param.grad


class Momentum(Optimizer):

    def __init__(
        self, params: List[Tensor], learning_rate: float, beta: float = 0.9, f=None
    ):
        super().__init__(params, learning_rate, f)
        self.beta = beta
        self.velocities = [Tensor(np.zeros_like(p.data)) for p in self.params]

    def _update(self, param: Tensor) -> None:
        idx = self.params.index(param)
        velocity = self.velocities[idx]
        velocity.data = self.beta * velocity.data + (1 - self.beta) * param.grad
        param.data -= self.learning_rate * velocity.data


class Adagrad(Optimizer):
    def __init__(self, params: List[Tensor], learning_rate: float, f=None):
        super().__init__(params, learning_rate, f)
        self.eps = 1e-8
        self.g = [Tensor(np.zeros_like(p.data)) for p in self.params]

    def _update(self, param: Tensor) -> None:
        idx = self.params.index(param)
        g = self.g[idx]
        g.data = g.data + param.grad**2
        param.data -= (self.learning_rate / (np.sqrt(g.data + self.eps))) * param.grad


class RMSProp(Optimizer):
    def __init__(
        self, params: List[Tensor], learning_rate: float, beta: float = 0.9, f=None
    ):
        super().__init__(params, learning_rate, f)
        self.eps = 1e-8
        self.beta = beta
        self.g = [Tensor(np.zeros_like(p.data)) for p in self.params]

    def _update(self, param: Tensor) -> None:
        idx = self.params.index(param)
        g = self.g[idx]
        g.data = self.beta * g.data + (1 - self.beta) * param.grad**2
        param.data -= (self.learning_rate / (np.sqrt(g.data + self.eps))) * param.grad


class Adam(Optimizer):
    def __init__(
        self,
        params: List[Tensor],
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.99,
        f=None,
    ):
        super().__init__(params, learning_rate, f)
        self.eps = 1e-8
        self.beta2 = beta2
        self.beta1 = beta1
        self.g = [Tensor(np.zeros_like(p.data)) for p in self.params]
        self.m = [Tensor(np.zeros_like(p.data)) for p in self.params]
        self.k = 1

    def _update(self, param: Tensor) -> None:
        idx = self.params.index(param)
        G = self.g[idx]
        M = self.m[idx]
        M.data = self.beta1 * M.data + (1 - self.beta1) * param.grad
        G.data = self.beta2 * G.data + (1 - self.beta2) * param.grad**2

        M_corrected = M.data / (1 - self.beta1**self.k)
        G_corrected = G.data / (1 - self.beta2**self.k)
        param.data -= (self.learning_rate * M_corrected) / (
            np.sqrt(G_corrected) + self.eps
        )
        self.k += 1


class ArmijoGD(Optimizer):
    def __init__(
        self,
        params: List[Tensor],
        learning_rate: float = 0.01,
        beta: float = 0.5,
        c1: float = 1e-4,
        f=None,
    ):
        super().__init__(params, learning_rate, f)
        self.beta = beta
        self.c1 = c1
        self.optimal_alpha = None

    def _find_alpha(self):
        if self.f is None:
            raise ValueError("ArmijoGD requires an objective function f.")

        alpha = 1
        f_current = self.f(*[param.data for param in self.params])
        x = self.params.copy()
        g_sq_sum = 0
        for param in self.params:
            g_sq_sum += param * param
        while True:
            f_new = self.f(*[_x.data - alpha * _x.grad for _x in x])
            threadshold = self.c1 * alpha * g_sq_sum.data
            alpha *= self.beta
            if f_new < f_current - threadshold:
                break
        self.optimal_alpha = alpha

    def _update(self, param: Tensor) -> None:
        self._find_alpha()
        param.data -= self.optimal_alpha * param.grad


class NesterovGD(Optimizer):
    def __init__(
        self,
        params: List[Tensor],
        learning_rate: float = 0.01,
        gamma: float = 0.75,
        f: Optional[FunctionType] = None,
        fd_eps: float = 1e-6,
    ):
        super().__init__(params, learning_rate, f)
        self.gamma = gamma
        self.velocities = [Tensor(np.zeros_like(p.data)) for p in self.params]
        self.fd_eps = fd_eps
        self._look_ahead_grad = None

    def _compute_lookahead_grad(self) -> List[np.ndarray]:
        if self.f is None:
            raise ValueError("NesterovGD requires an objective function f.")
        # x_lookahead = x(k) - gamma * v(k-1)
        x_lookahead = [
            copy.deepcopy(param) - self.gamma * self.velocities[i]
            for i, param in enumerate(self.params)
        ]
        for param in x_lookahead:
            param.grad.fill(0)  # zeroing out the old gradient before accumulated
        f_lookahead = self.f(*x_lookahead)
        f_lookahead.backward()
        self._look_ahead_grad = [param.grad for param in x_lookahead]

    def _update(self, param: Tensor) -> None:
        self._compute_lookahead_grad()

        idx = self.params.index(param)
        self.velocities[idx] = (
            self.gamma * self.velocities[idx]
            + self.learning_rate * self._look_ahead_grad[idx]
        )
        param.data -= self.velocities[idx].data
