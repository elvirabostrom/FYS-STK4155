# --------------------------------------------------------------------------
# COMMENTS FOR MYSELF: 
# I should always explain the choice of learning rate in report
# I should explain the choice of lambda in the report
# When using a fixed degree, I should explain my choice in the report
# I NEED TO MAKE SURE I PLOT THE FINAL PARAMETER ERROR IN PART E
# --------------------------------------------------------------------------




# ------------------------------------------------------------------------------------
# In this code we implement plain gradient descent with OLS and Ridge for all degrees,
# we use adaptive learning rate based on the Hessian matrix (depends on degree),
# we test convergence to closed form by plotting MSE and R2 score, BUT I THINK I SHOULD 
# PLOT FINAL PARAMETER ERROR INSTEAD. In addition we look into how many iterations it takes 
# to reach convergence (but the maximum we run is 10,000 iterations)
# ------------------------------------------------------------------------------------
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils import *

# Enforce 64-bit floating point precision in JAX
jax.config.update("jax_enable_x64", True)

# Gradients via automatic differentiation
GradOLSad = jax.grad(CostOLS)
GradRidgead = jax.grad(CostRidge)

# Set parameters
n = 100
noise = 0.1
seed = 2026
degrees = np.arange(1, 16, 1)
n_punishers = 10
punishers = np.logspace(-6, 6, n_punishers) # run the code for different punishers (just in case), but in the end plot only for lambda = 0.01

# Target tolerance for gradient norm
target_tol = 1e-8  
max_iters_cap = 10000 # we do 10,000 iterations (or less if converged early)

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
    

    n_train, p = X_train_scaled.shape
    theta_init = np.zeros(p)

    # -------------------------------------------------------------
    # OLS GD vs CF
    # -------------------------------------------------------------

    # Determine optimal learning rate
    H_ols = (2.0 / n_train) * (X_train_scaled.T @ X_train_scaled)
    lambda_max_ols = np.max(np.linalg.eigvalsh(H_ols))
    eta_max_ols = 2.0 / lambda_max_ols # largest learning rate
    eta_ols = 0.5 * eta_max_ols # learning rate we use

    # Closed form for comparison
    params_ols_cf = closedForm(X_train_scaled, y_train_centered)
    
    history_ols = optimise(
        grad=lambda th: GradOLSAnalytic(th, X_train_scaled, y_train_centered),
        theta0=theta_init,
        method="plain",
        gamma=eta_ols,
        num_iters=max_iters_cap,
        tol=target_tol,
    )
    
    iters_ols = len(history_ols) - 1 # iterations used
    params_ols_gd = history_ols[-1] # final parameters

    # Compute predictions and metrics for OLS
    y_pred_ols_gd = X_test_scaled @ params_ols_gd + y_train.mean()
    y_pred_ols_cf = X_test_scaled @ params_ols_cf + y_train.mean()

    mse_gd_ols = MSE(y_pred_ols_gd, y_test)
    mse_cf_ols = MSE(y_pred_ols_cf, y_test)
    r2_gd_ols = R2Score(y_pred_ols_gd, y_test)
    r2_cf_ols = R2Score(y_pred_ols_cf, y_test)

    # Compare analytic and AD gradient
    ad_diff_ols = np.max(
        np.abs(
            GradOLSad(theta_init, X_train_scaled, y_train_centered)
            - GradOLSAnalytic(theta_init, X_train_scaled, y_train_centered)
        )
    )
    assert ad_diff_ols < 1e-15, f"OLS gradient mismatch at d={d}: {ad_diff_ols}"

    # Save results
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
    # Ridge GD vs CF
    # -------------------------------------------------------------

    # Loop over punishers (plot only for lambda = 0.01)
    for lamb in punishers:
        H_ridge = H_ols + 2.0 * lamb * np.eye(p)
        lambda_max_ridge = np.max(np.linalg.eigvalsh(H_ridge))
        eta_max_ridge = 2.0 / lambda_max_ridge
        eta_ridge = 0.5 * eta_max_ridge
        
        # Closed form for comparison
        params_ridge_cf = closedForm(X_train_scaled, y_train_centered, lamb)
        
        history_ridge = optimise(
            grad=lambda th: GradRidgeAnalytic(th, X_train_scaled, y_train_centered, lamb=lamb),
            theta0=theta_init,
            method="plain",
            gamma=eta_ridge,
            num_iters=max_iters_cap,
            tol=target_tol,
        )
        
        iters_ridge = len(history_ridge) - 1 # iterations used
        params_ridge_gd = history_ridge[-1] # final parameters

        # Compute predictions and metrics for Ridge
        y_pred_ridge_gd = X_test_scaled @ params_ridge_gd + y_train.mean()
        y_pred_ridge_cf = X_test_scaled @ params_ridge_cf + y_train.mean()

        mse_gd_ridge = MSE(y_pred_ridge_gd, y_test)
        mse_cf_ridge = MSE(y_pred_ridge_cf, y_test)
        r2_gd_ridge = R2Score(y_pred_ridge_gd, y_test)
        r2_cf_ridge = R2Score(y_pred_ridge_cf, y_test)

        # Compare analytic gradient and AD gradient
        ad_diff_ridge = np.max(
            np.abs(
                GradRidgead(theta_init, X_train_scaled, y_train_centered, lamb)
                - GradRidgeAnalytic(theta_init, X_train_scaled, y_train_centered, lamb)
            )
        )
        assert ad_diff_ridge < 1e-15, f"Ridge gradient mismatch at d={d}, lambda={lamb}: {ad_diff_ridge}"

        # Save results
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