# Optimization Techniques for Data Science

## Overview

This project provides a complete framework for studying and comparing optimization techniques commonly used in data science and machine learning. It includes implementations of:

- **First-order optimizers**: SGD, Momentum, Adagrad, RMSProp, Adam, Armijo line search, Nesterov accelerated gradient
- **Second-order optimizers**: BFGS (quasi-Newton method)
- **Subgradient methods**: For non-smooth optimization problems
- **Constrained optimization**: Equality and inequality constrained solvers using Lagrange multipliers and KKT conditions
- **Automatic differentiation**: Custom Tensor class with computational graph support
- **Machine learning models**: Linear regression, logistic regression, Lasso, and Ridge regression

---

## Features

### 🧮 Automatic Differentiation
- Custom `Tensor` class inspired by PyTorch
- Computational graph construction and automatic backpropagation
- Support for common operations (addition, multiplication, matrix operations, exponentials, logarithms, etc.)
- Visualization of computational graphs using Graphviz

### 📈 Optimization Algorithms

#### First-Order Methods
- **SGD**: Standard stochastic gradient descent
- **Momentum**: SGD with exponential moving average of gradients
- **Adagrad**: Adaptive learning rates per parameter
- **RMSProp**: Exponentially decayed squared-gradient normalization
- **Adam**: Combines momentum and RMSProp with bias correction
- **ArmijoGD**: Gradient descent with Armijo backtracking line search
- **NesterovGD**: Nesterov accelerated gradient with look-ahead

#### Second-Order Methods
- **BFGS**: Quasi-Newton method with backtracking line search

#### Specialized Methods
- **Subgradient Descent**: For non-smooth optimization problems
- **Constrained Optimization**: 
  - Equality constraints using Lagrange multipliers
  - Inequality constraints using primal-dual KKT method

### 🤖 Machine Learning Models

- **Linear Regression**: MSE loss with various optimizers
- **Logistic Regression**: Binary classification with BCE loss
- **Lasso Regression**: L1 regularization with subgradient optimization
- **Ridge Regression**: L2 regularization with subgradient optimization

### 📊 Visualization Tools

- Contour plots of optimization landscapes
- 3D surface plots
- Optimization path visualization
- Loss curve comparisons
- Decision boundary visualization (for classification)

---

## Installation

### Prerequisites

- Python 3.7+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ml-algorithms
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

- `numpy`: Numerical computations
- `matplotlib`: Plotting and visualization
- `scikit-learn`: Dataset generation utilities
- `graphviz`: Computational graph visualization (optional, for Tensor visualization)
- `tqdm`: Progress bars (if used)

---

## Project Structure

```
ima-project/
├── main.py                      # Main demo script
├── tensors.py                   # Custom Tensor class with autodiff
├── function.py                  # Base Function API with test functions
├── function_analysis.py         # Function analysis and visualization
├── first_order_optimizers.py   # First-order optimization algorithms
├── second_order_optimizers.py  # BFGS optimizer
├── subgradient.py              # Subgradient descent implementation
├── constrained.py              # Constrained optimization solvers
├── linear_regression.py        # Linear regression model
├── logistic_regression.py     # Logistic regression model
├── lasso.py                    # Lasso regression optimizer
├── ridge.py                    # Ridge regression optimizer
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Usage

### Running the Main Demo

The `main.py` file contains comprehensive demonstrations of all optimization techniques:

```bash
python main.py
```

This will run:
1. Comparative analysis of optimizers on test functions (Rosenbrock, Beale, Goldstein-Price, Himmelblau, etc.)
2. Linear regression benchmarks
3. Logistic regression benchmarks
4. Subgradient optimization examples
5. Lasso and Ridge regression demonstrations
6. Constrained optimization examples

### Using Individual Components

#### Test Function Optimization

```python
from function import Rosenbrock
from first_order_optimizers import Adam
from function_analysis import FunctionAnalysis

# Create a test function
rosenbrock = Rosenbrock()

# Analyze and optimize
function = FunctionAnalysis(rosenbrock.f)
points_dict = function.walk(
    start=rosenbrock.start,
    n_steps=30,
    step_sizes=[0.2],
    optimizer_list=[Adam],
    bounds=rosenbrock.bound
)
function.plot_contours(points_dict, X, Y, Z)
```

#### Linear Regression

```python
from linear_regression import LinearRegression
from first_order_optimizers import Adam

# Run a demo
model, losses = LinearRegression.run_demo(
    optimizer_cls=Adam,
    n_samples=1000,
    n_features=1
)

# Benchmark multiple optimizers
results, models = LinearRegression.benchmark_optimizers(
    optimizer_list=[Sgd, Momentum, Adam],
    step_sizes=[0.01, 0.01, 0.01],
    epochs=200
)
```

#### Logistic Regression

```python
from logistic_regression import LogisticRegression
from first_order_optimizers import Adam

# Run a demo with visualization
model = LogisticRegression.run_demo(
    optimizer_cls=Adam,
    n_samples=1000,
    n_features=2
)

# Benchmark optimizers
results, models = LogisticRegression.benchmark_optimizers(
    optimizer_list=[Sgd, Adam, RMSProp],
    step_sizes=[0.1, 0.1, 0.1],
    epochs=1000
)
```

#### Lasso Regression

```python
from lasso import LassoSubgradientOptimizer
from sklearn.datasets import make_regression

# Generate data
X, y = make_regression(n_samples=100, n_features=2, noise=10.0, random_state=42)

# Optimize
lasso = LassoSubgradientOptimizer(X, y, factor=1e-3)
path = lasso.optimize()
lasso.plot_loss_contours(path)
lasso.plot_loss_surface(path)
```

#### Constrained Optimization

```python
from constrained import EqualityConstrainedOptimizer
import numpy as np

# Define objective and constraint
def objective(x):
    return x[0] * x[1]

def constraint(x):
    return x[0]**2 / 8.0 + x[1]**2 / 2.0 - 1.0

# Solve
solver = EqualityConstrainedOptimizer(
    objective,
    [constraint],
    n_epochs=2000,
    step_size_x=5e-3,
    step_size_lambda=5e-3
)
result = solver.optimize(seed_pos=np.array([2.0, 1.0]))
solver.plot_path(result.path, title="Equality-Constrained Optimization")
```

#### Custom Tensor Operations

```python
from tensors import Tensor
import numpy as np

# Create tensors with gradient tracking
x = Tensor([1.0, 2.0], requires_grad=True)
y = Tensor([3.0, 4.0], requires_grad=True)

# Perform operations
z = (x * y).sum()
z.backward()

# Access gradients
print(x.grad)  # [3., 4.]
print(y.grad)  # [1., 2.]

# Visualize computational graph (requires graphviz)
# z.draw_dot().view()
```

---

## Test Functions

The project includes several benchmark optimization functions:

- **Quadratic Bowl**: Simple convex function
- **Rosenbrock**: Classic non-convex function with narrow valley
- **Beale**: Multi-modal function
- **Goldstein-Price**: Complex multi-modal landscape
- **Himmelblau**: Function with four equal minima
- **Three-Hump Camel**: Function with multiple local minima

---

## Key Algorithms Implemented

### Gradient-Based Optimizers

All first-order optimizers follow a unified interface:

```python
optimizer = OptimizerClass(params, learning_rate=0.01)
for epoch in range(epochs):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### BFGS (Quasi-Newton)

Uses numerical gradients and backtracking line search:

```python
from second_order_optimizers import BFGS
from function import Rosenbrock

opt = BFGS(Rosenbrock())
history = opt.optimize(max_iter=200)
```

### Subgradient Methods

Handles non-smooth functions with absolute value terms:

```python
from subgradient import SubgradientDescent

def f(x):
    return x[0]**2 + abs(x[1]) - abs(x[0])

descent = SubgradientDescent(f)
path = descent.optimize([5, 5])
```

### Constrained Optimization

- **Equality Constraints**: Lagrange multiplier method
- **Inequality Constraints**: Primal-dual KKT method with projection

---

## Visualization Features

The project provides extensive visualization capabilities:

1. **Contour Plots**: Show optimization landscapes and paths
2. **3D Surface Plots**: Visualize loss surfaces
3. **Loss Curves**: Compare optimizer performance
4. **Decision Boundaries**: For classification models
5. **Computational Graphs**: Visualize Tensor operation graphs

---

## Performance Considerations

- Numerical gradients are computed using central differences
- BFGS uses backtracking line search for stability
- Subgradient methods handle non-differentiable points
- All optimizers support gradient clipping
- Tensor operations use NumPy for efficiency

---

## Educational Value

This project serves as a comprehensive learning resource for:

- Understanding optimization algorithms from first principles
- Implementing automatic differentiation
- Comparing optimizer performance on different landscapes
- Applying optimization to real ML problems
- Visualizing optimization dynamics

---

## Notes

- The project focuses on 2D optimization for visualization purposes
- Some functions (like ArmijoGD) can be computationally intensive
- Step sizes may need tuning for different functions
- The Tensor implementation prioritizes clarity over performance

---

## References

- Test functions from [Wikipedia: Test functions for optimization](https://en.wikipedia.org/wiki/Test_functions_for_optimization)
- Optimization algorithms based on standard textbooks and research papers
- Tensor implementation inspired by PyTorch's design

---
