"""
Part g) Lasso regression with the gradient descent methods of parts e) and f).
Written with Claude (Sept 2026).

The Lasso cost (Eq. 3.57) is not differentiable at theta_j = 0. We use its
subgradient, (2/n) X^T (X theta - y) + lamb * sgn(theta), in the optimisers
of part f) (plain, momentum, AdaGrad, RMSprop, Adam), unchanged.
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

from utils import *

jax.config.update("jax_enable_x64", True)
warnings.simplefilter("ignore", ConvergenceWarning)  # sklearn at very small lambda

# Same data and split as part e, same optimiser settings as part f
n = 100
noise = 0.1
seed = 2026
methods = ["plain", "momentum", "adagrad", "rmsprop", "adam"]
gamma = 0.01
num_iters = 10000 #Making it consistent with other parts, specifically e and f

x, y = MakeData(n, noise, seed + n)


def prepare(d):
    X = MakeDesignMatrix(x, d)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
    X_train_scaled, X_test_scaled, y_train_centered = scaleData(X_train, X_test, y_train)
    return X_train_scaled, X_test_scaled, y_train_centered, y_train, y_test

# Learning rate for plain GD, as in part e: 0.5 * 2 / (largest eigenvalue of the Hessian)
# Hessian of the smooth (OLS) part of the Lasso cost: (2/n) X^T X
def gamma_plain(X):
    H = (2.0 / X.shape[0]) * X.T @ X
    return 0.5 * 2.0 / np.max(np.linalg.eigvalsh(H))
#y=0.5y_max for plain gradient descent, and y=0.01y_max for the others

# =====================================================================
# 1. Automatic differentiation of |theta| at theta = 0
# =====================================================================
d_abs = jax.grad(jnp.abs)
print("1. Derivative of |theta| at theta = 0")
print(f"   JAX (jax.grad):  {float(d_abs(0.0)):+.1f}")
print(f"   numpy.sign(0):   {float(np.sign(0.0)):+.1f}")
print("   Both lie in the subdifferential [-1, 1] (Eq. 3.65): valid subgradients.")

# Check the analytic subgradient against JAX away from zero (as in part e)
GradLassoAD = jax.grad(CostLasso)
X_tr, X_te, y_c, y_tr, y_te = prepare(10)
theta_check = np.random.default_rng(seed).normal(size=X_tr.shape[1])
ad_diff = np.max(np.abs(np.asarray(GradLassoAD(theta_check, X_tr, y_c, 1e-2))
                        - GradLassoAnalytic(theta_check, X_tr, y_c, 1e-2)))
assert ad_diff < 1e-12, f"Lasso gradient mismatch: {ad_diff}"
print(f"   Analytic subgradient = JAX gradient for theta != 0 (max diff {ad_diff:.1e})\n")


# =====================================================================
# 2. Lasso vs scikit-learn as a function of lambda, d = 10
# =====================================================================
d_fixed = 5 #making it consistent with other parts again. 
X_tr, X_te, y_c, y_tr, y_te = prepare(d_fixed)
p = X_tr.shape[1]
theta_init = np.zeros(p)
lambdas = np.logspace(-4, 0, 9)
results_lambda = []

for lamb in lambdas:
    theta_sk = sklearnLasso(X_tr, y_c, lamb)
    cost_sk = float(CostLasso(theta_sk, X_tr, y_c, lamb))

    def grad(theta, lamb=lamb):
        return GradLassoAnalytic(theta, X_tr, y_c, lamb)

    for method in methods:
        1r = gamma_plain(X_tr) if method == "plain" else gamma
        theta = optimise(grad, theta_init, method, 1r, num_iters=num_iters)[-1]
        results_lambda.append({
            "method": method, "lambda": lamb,
            "param_diff_sklearn": np.max(np.abs(theta - theta_sk)),
            "cost_diff_sklearn": float(CostLasso(theta, X_tr, y_c, lamb)) - cost_sk,
            "n_zeros": int(np.sum(theta == 0)),
            "n_small": int(np.sum(np.abs(theta) < 1e-3)),
            "n_zeros_sklearn": int(np.sum(theta_sk == 0)),
            "theta": theta,
        })

    # As soft thresholding is build on plain GD
    theta = theta_init.copy()
    1r = gamma_plain(X_tr)
    for _ in range(num_iters):
        theta = soft_threshold(theta - 1r * GradOLSAnalytic(theta, X_tr, y_c), lamb * 1r)
    results_lambda.append({
        "method": "soft_threshold", "lambda": lamb,
        "param_diff_sklearn": np.max(np.abs(theta - theta_sk)),
        "cost_diff_sklearn": float(CostLasso(theta, X_tr, y_c, lamb)) - cost_sk,
        "n_zeros": int(np.sum(theta == 0)),
        "n_small": int(np.sum(np.abs(theta) < 1e-3)),
        "n_zeros_sklearn": int(np.sum(theta_sk == 0)),
        "theta": theta,
    })
    results_lambda.append({
        "method": "sklearn", "lambda": lamb, "param_diff_sklearn": 0.0,
        "cost_diff_sklearn": 0.0, "n_zeros": int(np.sum(theta_sk == 0)),
        "n_small": int(np.sum(np.abs(theta_sk) < 1e-3)),
        "n_zeros_sklearn": int(np.sum(theta_sk == 0)), "theta": theta_sk,
    })

print(f"2. Lasso vs scikit-learn, d = {d_fixed}: max coefficient difference (exact zeros)")
for lamb in lambdas:
    rows = [r for r in results_lambda if r["lambda"] == lamb]
    print(f"   lambda={lamb:6.0e} sklearn zeros={rows[-1]['n_zeros']:2d} | " +
          "  ".join(f"{r['method'][:6]}: {r['param_diff_sklearn']:.0e} ({r['n_zeros']})"
                    for r in rows[:-1]))

writeToFile({"n": n, "noise": noise, "d": d_fixed, "type": "lambda_sweep", "exercise": "g"},
            results_lambda)


# =====================================================================
# 3. OLS, Ridge and Lasso as functions of the polynomial degree
# =====================================================================
lamb_cmp = 1e-2  # same penalty as used for Ridge in part e and f
degrees = np.arange(1, 16)
results_degree = []

for d in degrees:
    X_tr, X_te, y_c, y_tr, y_te = prepare(d)
    p = X_tr.shape[1]

    exact = {"OLS": closedForm(X_tr, y_c),
             "Ridge": closedForm(X_tr, y_c, lamb_cmp),
             "Lasso": sklearnLasso(X_tr, y_c, lamb_cmp)}
    grads = {"OLS": lambda th: GradOLSAnalytic(th, X_tr, y_c),
             "Ridge": lambda th: GradRidgeAnalytic(th, X_tr, y_c, lamb_cmp),
             "Lasso": lambda th: GradLassoAnalytic(th, X_tr, y_c, lamb_cmp)}

    for model in ["OLS", "Ridge", "Lasso"]:
        thetas = {"exact": exact[model]}
        for method in methods:
            1r = gamma_plain(X_tr) if method == "plain" else gamma
            thetas[method] = optimise(grads[model], np.zeros(p), method, 1r,
                                      num_iters=num_iters)[-1]
        for method, theta in thetas.items():
            y_pred = X_te @ theta + y_tr.mean()
            results_degree.append({
                "model": model, "method": method, "d": d,
                "lambda": 0.0 if model == "OLS" else lamb_cmp,
                "MSE": MSE(y_pred, y_te), "R2": R2Score(y_pred, y_te),
                "param_diff_exact": np.max(np.abs(theta - exact[model])),
                "n_zeros": int(np.sum(theta == 0)),
            })
    print(f"3. d = {d:2d} done")

writeToFile({"n": n, "noise": noise, "lambda": lamb_cmp, "type": "degree_sweep",
             "exercise": "g"}, results_degree)
