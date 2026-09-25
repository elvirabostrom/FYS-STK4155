# -----------------------------------------------------------------------------------
# For part i)
# In this part, we use Adam optimizer for Lasso regression for all polynomial degrees
# -----------------------------------------------------------------------------------

import numpy as np
from sklearn.model_selection import train_test_split
from utils import *

# Parameters
n = 100
noise = 0.1
seed = 2026
degrees = np.arange(1, 16, 1)

# Optimal lambda from cross-validation
lamb = 1e-10

x, y = MakeData(n, noise, seed + n)

# Convergence parameters
target_tol = 1e-8
max_iters_cap = 10000

# Learning rate for Adam
gamma = 0.55

# -----------------------------------------------------------------------------------
# Run Lasso with Adam for every polynomial degree
# -----------------------------------------------------------------------------------

results = []

for d_val in degrees:
    X = MakeDesignMatrix(x, d_val)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=69
    )

    X_train_scaled, X_test_scaled, y_train_centered = scaleData(
        X_train, X_test, y_train
    )

    n_train, p = X_train_scaled.shape
    theta_init = np.zeros(p)

    # Analytic Lasso gradient
    def grad_d(theta):
        return GradLassoAnalytic(
            theta, X_train_scaled, y_train_centered, lamb
        )

    # Run Adam
    history = optimise(
        grad=grad_d,
        theta0=theta_init,
        method="adam",
        gamma=gamma,
        num_iters=max_iters_cap,
        tol=target_tol,
    )

    iters_to_accuracy = len(history) - 1
    theta_lasso = history[-1]

    # Compute prediction and MSE
    y_pred = X_test_scaled @ theta_lasso + y_train.mean()
    mse = MSE(y_pred, y_test)

    # Save results
    results.append(
        {
            "model": "Lasso",
            "degree": d_val,
            "method": "adam",
            "learning_rate": gamma,
            "lambda": lamb,
            "iters_needed": iters_to_accuracy,
            "MSE": mse,
        }
    )

naming = {
    "n": n,
    "noise": noise,
    "type": "adam_vs_degree",
    "exercise": "i_Lasso"
}
writeToFile(naming, results)

