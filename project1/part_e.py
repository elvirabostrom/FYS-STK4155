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

# Unified target tolerance for gradient norm stopping criterion
target_tol = 1e-8  
max_iters_cap = 10000

x, y = MakeData(n, noise, seed + n)  
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
    
    # Capture y_train mean if scaleData centers it, for proper test evaluation
    y_train_mean = np.mean(y_train)

    n_train, p = X_train_scaled.shape
    theta_init = np.zeros(p)

    # -------------------------------------------------------------
    # OLS Gradient Descent & Closed-Form
    # -------------------------------------------------------------
    H_ols = (2.0 / n_train) * (X_train_scaled.T @ X_train_scaled)
    lambda_max_ols = np.max(np.linalg.eigvalsh(H_ols))
    eta_max_ols = 2.0 / lambda_max_ols
    eta_ols = 0.5 * eta_max_ols

    params_ols_cf = closedForm(X_train_scaled, y_train_centered)
    
    history_ols = optimise(
        grad=lambda th: GradOLSAnalytic(th, X_train_scaled, y_train_centered),
        theta0=theta_init,
        method="plain",
        gamma=eta_ols,
        num_iters=max_iters_cap,
        tol=target_tol,
    )
    
    iters_ols = len(history_ols) - 1
    params_ols_gd = history_ols[-1]

    # Compute predictions and metrics for OLS
    y_pred_ols_gd = X_test_scaled @ params_ols_gd + y_train_mean
    y_pred_ols_cf = X_test_scaled @ params_ols_cf + y_train_mean

    mse_gd_ols = MSE(y_pred_ols_gd, y_test)
    mse_cf_ols = MSE(y_pred_ols_cf, y_test)
    r2_gd_ols = R2Score(y_pred_ols_gd, y_test)
    r2_cf_ols = R2Score(y_pred_ols_cf, y_test)

    ad_diff_ols = np.max(
        np.abs(
            GradOLSad(theta_init, X_train_scaled, y_train_centered)
            - GradOLSAnalytic(theta_init, X_train_scaled, y_train_centered)
        )
    )
    assert ad_diff_ols < 1e-15, f"OLS gradient mismatch at d={d}: {ad_diff_ols}"

    results.append(
        {
            "model": "OLS",
            "d": d,
            "lambda": 0.0,
            "eta_max": eta_max_ols,
            "eta_used": eta_ols,
            "iters_needed": iters_ols,
            "MSE_GD": mse_gd_ols,
            "MSE_CF": mse_cf_ols,
            "R2_GD": r2_gd_ols,
            "R2_CF": r2_cf_ols,
            "max_diff_AD_analytic": ad_diff_ols,
            "max_diff_GD_ClosedForm": np.max(np.abs(params_ols_gd - params_ols_cf)),
        }
    )

    # -------------------------------------------------------------
    # Ridge Gradient Descent & Closed-Form
    # -------------------------------------------------------------
    for lamb in punishers:
        H_ridge = H_ols + 2.0 * lamb * np.eye(p)
        lambda_max_ridge = np.max(np.linalg.eigvalsh(H_ridge))
        eta_max_ridge = 2.0 / lambda_max_ridge
        eta_ridge = 0.5 * eta_max_ridge
        
        params_ridge_cf = closedForm(X_train_scaled, y_train_centered, lamb)
        
        history_ridge = optimise(
            grad=lambda th: GradRidgeAnalytic(th, X_train_scaled, y_train_centered, lamb=lamb),
            theta0=theta_init,
            method="plain",
            gamma=eta_ridge,
            num_iters=max_iters_cap,
            tol=target_tol,
        )
        
        iters_ridge = len(history_ridge) - 1
        params_ridge_gd = history_ridge[-1]

        # Compute predictions and metrics for Ridge
        y_pred_ridge_gd = X_test_scaled @ params_ridge_gd + y_train_mean
        y_pred_ridge_cf = X_test_scaled @ params_ridge_cf + y_train_mean

        mse_gd_ridge = MSE(y_pred_ridge_gd, y_test)
        mse_cf_ridge = MSE(y_pred_ridge_cf, y_test)
        r2_gd_ridge = R2Score(y_pred_ridge_gd, y_test)
        r2_cf_ridge = R2Score(y_pred_ridge_cf, y_test)

        ad_diff_ridge = np.max(
            np.abs(
                GradRidgead(theta_init, X_train_scaled, y_train_centered, lamb)
                - GradRidgeAnalytic(theta_init, X_train_scaled, y_train_centered, lamb)
            )
        )
        assert ad_diff_ridge < 1e-15, f"Ridge gradient mismatch at d={d}, lambda={lamb}: {ad_diff_ridge}"

        results.append(
            {
                "model": "Ridge",
                "d": d,
                "lambda": lamb,
                "eta_max": eta_max_ridge,
                "eta_used": eta_ridge,
                "iters_needed": iters_ridge,
                "MSE_GD": mse_gd_ridge,
                "MSE_CF": mse_cf_ridge,
                "R2_GD": r2_gd_ridge,
                "R2_CF": r2_cf_ridge,
                "max_diff_AD_analytic": ad_diff_ridge,
                "max_diff_GD_ClosedForm": np.max(np.abs(params_ridge_gd - params_ridge_cf)),
            }
        )

writeToFile(naming, results)