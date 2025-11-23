"""
tensor.py — A small wrapper for numpy to enable automatic differentiation (Pytorch Inspired) based on computational graphs
=========================================================================================================

Core Design Motivation: Generalize function optimization to any arbitary python function.

The Tensor class builds computational graphs storing operation history, performs topological sort
and applies chain rule to compute gradients. The emphasis here is simplicity and readability .

The class wraps a NumPy array (`.data`) and optionally tracks gradients (`requires_grad=True`).
Each operation creates a new Tensor with a link to its parents, and calling `.backward()` walks
this graph in reverse, executing the stored gradient functions to populate `.grad`.

- Basic example
>>> a = Tensor(2.0, requires_grad=True)
>>> b = Tensor(3.0, requires_grad=True)
>>> c = a * b + a
>>> c.backward()
>>> a.grad   # b + 1
>>> array([4.])
>>>  b.grad   # a
>>>  array([2.])

- Graph example
>>> x = Tensor([1.0, 2.0], requires_grad=True)
>>> y = x.exp().sum()
>>> dot = y.draw_dot().view() // shows the DAG of arithemetic (requires graphviz binary)
"""

import numpy as np
from graphviz import Digraph


class Tensor:
    def __init__(
        self, data, _children=(), op_="", dtype=np.float64, requires_grad=True
    ):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=dtype)
        if data.ndim == 0:
            data = data.reshape(1)

        self.data = data
        self.shape = data.shape
        self.requires_grad = requires_grad

        # Gradient and graph-related attributes are only set if the tensor requires gradients.
        self.grad = np.zeros_like(data, dtype=dtype) if requires_grad else None
        self.prev_ = set(_children)
        self._backward = lambda: None
        self.op_ = op_

    def __repr__(self):
        return f"Tensor(data={self.data}, grad_shape={self.grad.shape if self.grad is not None else None}, requires_grad={self.requires_grad})"

    def _unbroadcast(self, target_shape, grad):
        """Helper to sum gradients back to their original shape before broadcasting."""
        while len(grad.shape) > len(target_shape):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(target_shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def item(self):
        return self.data.item() if self.data.size == 1 else self.data

    def __add__(self, other):
        # Constants don't require gradients.
        other = (
            other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        )

        out_requires_grad = self.requires_grad or other.requires_grad
        children = [t for t in (self, other) if t.requires_grad]
        out = Tensor(
            self.data + other.data, children, "+", requires_grad=out_requires_grad
        )

        if out_requires_grad:

            def _backward():
                if self.requires_grad:
                    self.grad += self._unbroadcast(self.shape, out.grad)
                if other.requires_grad:
                    other.grad += self._unbroadcast(other.shape, out.grad)

            out._backward = _backward

        return out

    def __mul__(self, other):
        other = (
            other if isinstance(other, Tensor) else Tensor(other, requires_grad=False)
        )

        out_requires_grad = self.requires_grad or other.requires_grad
        children = [t for t in (self, other) if t.requires_grad]
        out = Tensor(
            self.data * other.data, children, "*", requires_grad=out_requires_grad
        )

        if out_requires_grad:

            def _backward():
                if self.requires_grad:
                    self.grad += self._unbroadcast(self.shape, other.data * out.grad)
                if other.requires_grad:
                    other.grad += self._unbroadcast(other.shape, self.data * out.grad)

            out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(
            other, (int, float)
        ), "Only supporting int/float powers for now"

        children = [self] if self.requires_grad else []
        out = Tensor(
            self.data**other, children, f"**{other}", requires_grad=self.requires_grad
        )

        if self.requires_grad:

            def _backward():
                self.grad += (other * self.data ** (other - 1)) * out.grad

            out._backward = _backward

        return out

    def __matmul__(self, other):
        assert isinstance(
            other, Tensor
        ), "Matrix multiplication requires another Tensor"

        out_requires_grad = self.requires_grad or other.requires_grad
        children = [t for t in (self, other) if t.requires_grad]
        out = Tensor(
            self.data @ other.data, children, "@", requires_grad=out_requires_grad
        )

        if out_requires_grad:

            def _backward():
                if self.requires_grad:
                    self.grad += out.grad @ other.data.T
                if other.requires_grad:
                    other.grad += self.data.T @ out.grad

            out._backward = _backward

        return out

    def transpose(self, axes=None):
        children = [self] if self.requires_grad else []
        out = Tensor(
            np.transpose(self.data, axes=axes),
            children,
            "T",
            requires_grad=self.requires_grad,
        )

        if self.requires_grad:

            def _backward():
                inv_axes = None if axes is None else tuple(np.argsort(axes))
                self.grad += self._unbroadcast(
                    self.shape, np.transpose(out.grad, axes=inv_axes)
                )

            out._backward = _backward

        return out

    def sum(self):
        children = [self] if self.requires_grad else []
        out = Tensor(
            np.sum(self.data), children, "sum", requires_grad=self.requires_grad
        )

        if self.requires_grad:

            def _backward():
                self.grad += np.ones_like(self.data) * out.grad

            out._backward = _backward

        return out

    def exp(self):
        children = [self] if self.requires_grad else []
        out = Tensor(
            np.exp(self.data), children, "exp", requires_grad=self.requires_grad
        )

        if self.requires_grad:

            def _backward():
                self.grad += self._unbroadcast(self.shape, out.grad * np.exp(self.data))

            out._backward = _backward

        return out

    def log(self):
        children = [self] if self.requires_grad else []
        out = Tensor(
            np.log(self.data), children, "log", requires_grad=self.requires_grad
        )

        if self.requires_grad:

            def _backward():
                self.grad += self._unbroadcast(self.shape, out.grad / self.data)

            out._backward = _backward

        return out

    def backward(self):
        """Performs backpropagation starting from this tensor."""
        if not self.requires_grad:
            raise RuntimeError(
                "Cannot call .backward() on a tensor that does not require gradients"
            )

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev_:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def trace(self):
        nodes, edges = set(), set()

        def _build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v.prev_:
                    edges.add((child, v))
                    _build(child)

        _build(self)
        return nodes, edges

    def get_variable_name(self, obj):
        variable_name = [name for name, value in locals().items() if value is obj]
        return variable_name[0] if variable_name else None

    def draw_dot(self):
        dot = Digraph(format="png", graph_attr={"rankdir": "LR"})
        nodes, edges = self.trace()
        for n in nodes:
            uid = str(id(n))
            var_name = self.get_variable_name(n)
            dot.node(
                name=uid,
                label="{ %s | data %s | grad %s}"
                % (
                    var_name,
                    str(n.data),
                    str(n.grad),
                ),
                shape="record",
            )
            if n.op_:
                dot.node(name=uid + n.op_, label=n.op_)
                dot.edge(uid + n.op_, uid)

        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2.op_)

        return dot
