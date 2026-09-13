import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import *

# Enforce 64-bit floating point precision in JAX
jax.config.update("jax_enable_x64", True)


grad_ols_ad = jax.grad(CostOLS)
grad_ridge_ad = jax.grad(CostRidge)


num_points = np.array((50, 100, 250, 500))
noise = 0.1
degrees = np.arange(1, 16, 1)
n_punishers = 10
punishers = np.logspace(-1, 10, n_punishers)

for n in num_points:
    x, y = MakeData(n, noise)
    naming = {"n": n, "noise": noise, "exercise": "e"}
    results = []

    for d in degrees:
        X = MakeDesignMatrix(x, d)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=69
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        y_train_centered = y_train - y_train.mean()

        n_train, p = X_train_scaled.shape
        theta_init = np.zeros(p)

        # -------------------------------------------------------------
        # OLS Gradient Descent vs Closed Form
        # -------------------------------------------------------------
        H_ols = (2.0 / n_train) * (X_train_scaled.T @ X_train_scaled)
        lambda_max_ols = np.max(np.linalg.eigvalsh(H_ols))
        eta_max_ols = 2.0 / lambda_max_ols
        eta_ols = 0.5 * eta_max_ols

        params_ols_cf = closedForm(X_train_scaled, y_train_centered)
        params_ols_gd, iters_ols = gradient_descent(
            X_train_scaled,
            y_train_centered,
            grad_ols_analytic,
            theta_init,
            eta=eta_ols,
        )

        ad_diff_ols = np.max(
            np.abs(
                grad_ols_ad(theta_init, X_train_scaled, y_train_centered)
                - grad_ols_analytic(
                    theta_init, X_train_scaled, y_train_centered
                )
            )
        )

        y_pred_ols = X_test_scaled @ params_ols_gd + y_train.mean()
        mse_ols = MSE(y_pred_ols, y_test)
        r2_ols = R2Score(y_pred_ols, y_test)
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
                "MSE": mse_ols,
                "R2": r2_ols,
            }
        )

        # -------------------------------------------------------------
        # Ridge Gradient Descent vs Closed Form
        # -------------------------------------------------------------
        for lamb in punishers:
            H_ridge = H_ols + 2.0 * lamb * np.eye(p)
            lambda_max_ridge = np.max(np.linalg.eigvalsh(H_ridge))
            eta_max_ridge = 2.0 / lambda_max_ridge
            eta_ridge = 0.5 * eta_max_ridge

            params_ridge_cf = closedForm(
                X_train_scaled, y_train_centered, lamb
            )
            params_ridge_gd, iters_ridge = gradient_descent(
                X_train_scaled,
                y_train_centered,
                grad_ridge_analytic,
                theta_init,
                eta=eta_ridge,
                lamb=lamb,
            )

            ad_diff_ridge = np.max(
                np.abs(
                    grad_ridge_ad(
                        theta_init, X_train_scaled, y_train_centered, lamb
                    )
                    - grad_ridge_analytic(
                        theta_init, X_train_scaled, y_train_centered, lamb
                    )
                )
            )

            y_pred_ridge = X_test_scaled @ params_ridge_gd + y_train.mean()
            mse_ridge = MSE(y_pred_ridge, y_test)
            r2_ridge = R2Score(y_pred_ridge, y_test)
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
                    "MSE": mse_ridge,
                    "R2": r2_ridge,
                }
            )

    writeToFile_by_folder(naming, results)