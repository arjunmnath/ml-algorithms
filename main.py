import numpy as np
from sklearn.datasets import make_regression

from constrained import EqualityConstrainedOptimizer, InequalityConstrainedOptimizer
from first_order_optimizers import (
    Adagrad,
    Adam,
    ArmijoGD,
    Momentum,
    NesterovGD,
    RMSProp,
    Sgd,
)
from function import *
from function_analysis import FunctionAnalysis
from lasso import LassoSubgradientOptimizer
from linear_regression import LinearRegression
from logistic_regression import LogisticRegression
from ridge import RidgeSubgradientOptimizer
from second_order_optimizers import BFGS
from subgradient import SubgradientDescent

# ====================================================================================================
# Test Functions (taken from: https://en.wikipedia.org/wiki/Test_functions_for_optimization)
# ====================================================================================================
quadratic_bowl = QuadraticBowl()
beale = Beale()
goldstein_price = GoldsteinPrice()
himmelblau = Himmelblau()
three_hump_camel = ThreeHumpCamel()
rosenbrock = Rosenbrock()

# ====================================================================================================
# Comparative Analysis: gradient and hessian based optimizers
# ====================================================================================================
selected_function = quadratic_bowl  # change this function from one of the above to test different functions
bounds = selected_function.bound
x = np.linspace(*bounds, 400)
y = np.linspace(*bounds, 400)
X, Y = np.meshgrid(x, y)
function = FunctionAnalysis(selected_function.f)
Z = function(X, Y)
points_dict = function.walk(
    start=selected_function.start,
    n_steps=30,
    step_sizes=[
        0.2,
        0.2,
        0.01,
        0.2,
        0.2,
        0.2,
        0.2,
    ],  # eeds tweaking for function other than quadratic_bowl
    optimizer_list=[
        Sgd,
        ArmijoGD,
        NesterovGD,
        Momentum,
        Adagrad,
        RMSProp,
        Adam,
    ],  # Note: armijio is computationally intensive for most of the functions except quadratic_bowl
    clip_gradient=True,
    bounds=bounds,
)
opt = BFGS(selected_function)
hist = opt.optimize(
    max_iter=200
)  # returns a np array with x points taken during optimization
points_dict["bfgs"] = hist
function.plot_contours(points_dict, X, Y, Z)
function.plot_descent_path(points_dict, X, Y, Z)

# ================================================================================
# comparative analysis of optimizers on LinearRegression model
# ================================================================================
LinearRegression.run_demo(optimizer_cls=Sgd, n_features=1, n_samples=1000)
LinearRegression.benchmark_optimizers(
    optimizer_list=[Sgd, Momentum, Adagrad, RMSProp, Adam],
    step_sizes=[0.2, 0.2, 0.2, 0.2, 0.2],
    epochs=200,
    n_features=2,
    n_samples=1000,
)

# ================================================================================
# comparative analysis of optimizers on LogisticRegression model
# ================================================================================
LogisticRegression.run_demo(optimizer_cls=Sgd, n_features=2, n_samples=1000)
LogisticRegression.benchmark_optimizers(
    optimizer_list=[Sgd, Momentum, Adagrad, RMSProp, Adam],
    step_sizes=[0.2, 0.2, 0.2, 0.2, 0.2],
    epochs=200,
    n_features=2,
    n_samples=1000,
)


# ========================================
# Sub Gradient Optimization
# ========================================
def f(args: np.ndarray) -> float:
    res = args[0] ** 2 + np.abs(args[1]) - np.abs(args[0])
    return res


subgradient = SubgradientDescent(f)
path = subgradient.optimize([5, 5])
subgradient.plot_contours((-5, 5), path)
subgradient.plot_surface((-5, 5), path)

# ========================================
# Lasso Regularisation
# ========================================
X, y = make_regression(n_samples=100, n_features=2, noise=10.0, random_state=42)
lasso = LassoSubgradientOptimizer(X, y)
path = lasso.optimize()
lasso.plot_loss_contours(path)
lasso.plot_loss_surface(path)

# ========================================
# Ridge Regularisation
# ========================================
ridge = RidgeSubgradientOptimizer(X, y)
path = ridge.optimize()
ridge.plot_loss_contours(path)
ridge.plot_loss_surface(path)


# ========================================
# Constraint Optimization (Equalitiy)
# ========================================
def f_eq(args: np.ndarray) -> float:
    x, y = args
    return x * y


def h_eq(args: np.ndarray) -> float:
    x, y = args
    return x**2 / 8.0 + y**2 / 2.0 - 1.0


eq_solver = EqualityConstrainedOptimizer(
    f_eq,
    [h_eq],
    n_epochs=2000,
    step_size_x=5e-3,
    step_size_lambda=5e-3,
)
eq_result = eq_solver.optimize(seed_pos=np.array([2.0, 1.0]))
eq_solver.plot_path(
    eq_result.path,
    title="Equality-Constrained Optimization using Lagrange Multipliers",
)
# ========================================
# Constraint Optimization (InEqualitiy)
# ========================================


def f_ineq(args: np.ndarray) -> float:
    x, y = args
    return (x - 2.0) ** 2 + 2.0 * (y - 1.0) ** 2


def g1_ineq(args: np.ndarray) -> float:
    x, y = args
    return x + 4.0 * y - 3.0


def g2_ineq(args: np.ndarray) -> float:
    x, y = args
    return y - x


ineq_solver = InequalityConstrainedOptimizer(
    f_ineq,
    [g1_ineq, g2_ineq],
    n_epochs=5000,
    step_size_x=5e-3,
    step_size_lambda=5e-3,
)
ineq_result = ineq_solver.optimize(seed_pos=np.array([2.0, 1.0]))
ineq_solver.plot_path(
    ineq_result.path,
    title="Inequality-Constrained Optimization using Primal–Dual KKT Method",
)
