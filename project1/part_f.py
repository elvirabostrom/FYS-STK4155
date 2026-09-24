# -----------------------------------------------------------------------------------
# In this part, we test the various optimezers and compare them with plain gradient descent
# (using closed form as a benchmark), we do it for both OLS and Ridge
# -----------------------------------------------------------------------------------
import numpy as np
from sklearn.model_selection import train_test_split
from utils import *

# Parameters
n = 100
noise = 0.1
seed = 2026
methods = ["plain", "momentum", "adagrad", "rmsprop", "adam"] # optimizers to test

# Models to test (OLS and Ridge)
models = [
    {"name": "OLS", "lambda": 0.0},
    {"name": "Ridge", "lambda": 0.01}
]

x, y = MakeData(n, noise, seed + n)

# Convergence parameters
target_tol = 1e-8  
max_iters_cap = 10000

# ------------------------------------------------------------------------------------------
# 1. Here we generate data to plot how many iterations it takes for each method to converge
# to closed form (based on a criterion we define ourselves), for plain GD we use adaptive
# learning rate (like in part e), for other optimizers we use (initial) learning rate gamma = 0.01,
# we do this for every degree, and we never use more than 10,000 iterations
# ------------------------------------------------------------------------------------------
degrees = np.arange(1, 16, 1)
deg_sweep_results = []

for model_cfg in models:
    model_name = model_cfg["name"]
    lamb = model_cfg["lambda"]

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

        # Closed form for testing convergence
        theta_cf = closedForm(X_train_scaled, y_train_centered, lamb=lamb)

        # Pre-compute Hessian and optimal adaptive learning rate for Plain GD
        H_base = (2.0 / n_train) * (X_train_scaled.T @ X_train_scaled)
        H_matrix = H_base + (2.0 * lamb * np.eye(p))
        lambda_max = np.max(np.linalg.eigvalsh(H_matrix))
        eta_adaptive_plain = 0.5 * (2.0 / lambda_max)

        # Use analytic gradients
        if lamb == 0.0:
            def grad_d(theta):
                return GradOLSAnalytic(theta, X_train_scaled, y_train_centered)
        else:
            def grad_d(theta):
                return GradRidgeAnalytic(theta, X_train_scaled, y_train_centered, lamb=lamb)

        # Loop over the different optimizers
        for method in methods:
            gamma_used = eta_adaptive_plain if method == "plain" else 0.01

            history = optimise(
                grad=grad_d,
                theta0=theta_init,
                method=method,
                gamma=gamma_used,
                num_iters=max_iters_cap,
                tol=target_tol,
            )

            iters_to_accuracy = len(history) - 1 # iterations used
            final_param_error = np.max(np.abs(history[-1] - theta_cf)) # final parameter error for plotting

            # Save results
            deg_sweep_results.append(
                {
                    "model": model_name,
                    "degree": d_val,
                    "method": method,
                    "learning_rate": gamma_used,
                    "lambda": lamb,
                    "iters_needed": iters_to_accuracy,
                    "final_param_error": final_param_error,
                }
            )

naming_deg = {"n": n, "type": "iters_vs_degree", "exercise": "f"}
writeToFile(naming_deg, deg_sweep_results)


# ---------------------------------------------------------------------------------------
# 2. Here we generate data to plot learning rate sensitivity, we use a fixed degree d = 5
# ---------------------------------------------------------------------------------------
fixed_degree = 5
learning_rates = np.logspace(-6, 0, 25)
lr_sensitivity_results = []

for model_cfg in models:
    model_name = model_cfg["name"]
    lamb = model_cfg["lambda"]

    X = MakeDesignMatrix(x, fixed_degree)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=69
    )
    X_train_scaled, X_test_scaled, y_train_centered = scaleData(
        X_train, X_test, y_train
    )

    n_train, p = X_train_scaled.shape
    theta_init = np.zeros(p)

    # Closed form for testing convergence
    theta_cf = closedForm(X_train_scaled, y_train_centered, lamb=lamb)

    # Analytic gradients
    if lamb == 0.0:
        def grad_sens(theta):
            return GradOLSAnalytic(theta, X_train_scaled, y_train_centered)
    else:
        def grad_sens(theta):
            return GradRidgeAnalytic(theta, X_train_scaled, y_train_centered, lamb=lamb)

    # Loop over the different optimizers
    for method in methods:
        for lr in learning_rates:
            history = optimise(
                grad=grad_sens,
                theta0=theta_init,
                method=method,
                gamma=lr,
                num_iters=max_iters_cap,
                tol=target_tol,
            )

            iters_to_accuracy = len(history) - 1 # iterations used to reach convergence
            final_param_error = np.max(np.abs(history[-1] - theta_cf)) # final parameter error for plotting

            # Save results
            lr_sensitivity_results.append(
                {
                    "model": model_name,
                    "degree": fixed_degree,
                    "method": method,
                    "learning_rate": lr,
                    "lambda": lamb,
                    "iters_needed": iters_to_accuracy,
                    "final_param_error": final_param_error,
                }
            )

naming_sens = {"n": n, "type": "lr_sensitivity", "exercise": "f"}
writeToFile(naming_sens, lr_sensitivity_results)