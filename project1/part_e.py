import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils import *

# Enforce 64-bit floating point precision in JAX
jax.config.update("jax_enable_x64", True)

GradOLSad = jax.grad(CostOLS)
GradRidgead = jax.grad(CostRidge)

# Set fixed parameters
n = 100
noise = 0.1
seed = 2026
degrees = np.arange(1, 16, 1)
n_punishers = 10
punishers = np.logspace(-6, 6, n_punishers)

x, y = MakeData(n, noise, seed + n)  # Matching Part B punishers
naming = {"n": n, "noise": noise, "exercise": "e"}
results = []

for d in degrees:
    X = MakeDesignMatrix(x, d)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=69
    )

    X_train_scaled, X_test_scaled, y_train_centered = scaleData(
        X_train, X_test, y_train
    )

    n_train, p = X_train_scaled.shape
    theta_init = np.zeros(p)

    # -------------------------------------------------------------
    # OLS Gradient Descent
    # -------------------------------------------------------------
    H_ols = (2.0 / n_train) * (X_train_scaled.T @ X_train_scaled)
    lambda_max_ols = np.max(np.linalg.eigvalsh(H_ols))
    eta_max_ols = 2.0 / lambda_max_ols
    eta_ols = 0.5 * eta_max_ols

    params_ols_cf = closedForm(X_train_scaled, y_train_centered)
    params_ols_gd, iters_ols = GradientDescent(
        X_train_scaled,
        y_train_centered,
        GradOLSAnalytic,
        theta_init,
        eta=eta_ols,
    )

    ad_diff_ols = np.max(
        np.abs(
            GradOLSad(theta_init, X_train_scaled, y_train_centered)
            - GradOLSAnalytic(theta_init, X_train_scaled, y_train_centered)
        )
    )

    # Automated verification check
    assert ad_diff_ols < 1e-15, f"OLS gradient mismatch at d={d}: {ad_diff_ols}"

    y_pred_ols = X_test_scaled @ params_ols_gd + y_train.mean()
    cf_diff_ols = np.max(np.abs(params_ols_gd - params_ols_cf))

    results.append(
        {
            "model": "OLS",
            "d": d,
            "lambda": 0.0,
            "eta_max": eta_max_ols,
            "eta_used": eta_ols,
            "iters": iters_ols,
            "max_diff_AD_analytic": ad_diff_ols,
            "max_diff_GD_ClosedForm": cf_diff_ols,
            "MSE": MSE(y_pred_ols, y_test),
            "R2": R2Score(y_pred_ols, y_test),
        }
    )

    # -------------------------------------------------------------
    # Ridge Gradient Descent
    # -------------------------------------------------------------
    for lamb in punishers:
        H_ridge = H_ols + 2.0 * lamb * np.eye(p)
        lambda_max_ridge = np.max(np.linalg.eigvalsh(H_ridge))
        eta_max_ridge = 2.0 / lambda_max_ridge
        eta_ridge = 0.5 * eta_max_ridge
        
        params_ridge_cf = closedForm(
            X_train_scaled, y_train_centered, lamb
        )
        params_ridge_gd, iters_ridge = GradientDescent(
            X_train_scaled,
            y_train_centered,
            GradRidgeAnalytic,
            theta_init,
            eta=eta_ridge,
            lamb=lamb,
        )

        ad_diff_ridge = np.max(
            np.abs(
                GradRidgead(theta_init, X_train_scaled, y_train_centered, lamb)
                - GradRidgeAnalytic(theta_init, X_train_scaled, y_train_centered, lamb)
            )
        )

        # Automated verification check
        assert ad_diff_ridge < 1e-15, f"Ridge gradient mismatch at d={d}, lambda={lamb}: {ad_diff_ridge}"

        y_pred_ridge = X_test_scaled @ params_ridge_gd + y_train.mean()
        cf_diff_ridge = np.max(np.abs(params_ridge_gd - params_ridge_cf))

        results.append(
            {
                "model": "Ridge",
                "d": d,
                "lambda": lamb,
                "eta_max": eta_max_ridge,
                "eta_used": eta_ridge,
                "iters": iters_ridge,
                "max_diff_AD_analytic": ad_diff_ridge,
                "max_diff_GD_ClosedForm": cf_diff_ridge,
                "MSE": MSE(y_pred_ridge, y_test),
                "R2": R2Score(y_pred_ridge, y_test),
            }
        )

writeToFile(naming, results)

