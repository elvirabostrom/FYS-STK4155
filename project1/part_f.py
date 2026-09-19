from pathlib import Path
from sklearn.model_selection import train_test_split
from utils import *

# Define optimization methods and lambdas to evaluate
methods = ["plain", "momentum", "adagrad", "rmsprop", "adam"]
lambdas = [0.0, 1e-3]  # 0.0 for OLS, >0 for Ridge

# Generate synthetic dataset
x, y = MakeData(n=100, noise=0.1, seed=42)

# =====================================================================
# 1. Iterations vs. Polynomial Degree
# =====================================================================
degrees_sweep = list(range(1, 15))
target_tol = 1e-4
max_iters_cap = 50000
deg_sweep_results = []

for lamb in lambdas:
    for d_val in degrees_sweep:
        # Build design matrix and split
        X_d = MakeDesignMatrix(x, d_val)
        X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
            X_d, y, test_size=0.2, random_state=42
        )
        X_tr_scaled_d, _, y_tr_centered_d = scaleData(X_train_d, X_test_d, y_train_d)

        n_tr_d, p_d = X_tr_scaled_d.shape
        theta0_d = np.zeros(p_d)

        # Reference analytical closed-form solution
        theta_cf_d = closedForm(X_tr_scaled_d, y_tr_centered_d, lamb=lamb)

        # Gradient function closure with penalty
        def grad_d(theta):
            return GradRidgeAnalytic(theta, X_tr_scaled_d, y_tr_centered_d, lamb=lamb)

        for method in methods:
            history = optimise(
                grad=grad_d,
                theta0=theta0_d,
                method=method,
                gamma=0.01,
                num_iters=max_iters_cap,
                tol=1e-8,
            )

            iters_to_accuracy = max_iters_cap
            for step, th in enumerate(history):
                if np.max(np.abs(th - theta_cf_d)) < target_tol:
                    iters_to_accuracy = step
                    break

            deg_sweep_results.append(
                {
                    "degree": d_val,
                    "method": method,
                    "learning_rate": 0.01,
                    "lambda": lamb,
                    "iters_needed": iters_to_accuracy,
                    "reached_target": iters_to_accuracy < max_iters_cap,
                }
            )

naming_deg = {"n": 100, "type": "iters_vs_degree", "exercise": "f"}
writeToFile(naming_deg, deg_sweep_results)


# =====================================================================
# 2. Learning Rate Sensitivity Sweep (with Divergence Guard)
# =====================================================================
learning_rates_sens = np.logspace(-5, 0, 15)
d_sens = 10
sensitivity_results = []

# Build design matrix and split for d=10
X_sens = MakeDesignMatrix(x, d_sens)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_sens, y, test_size=0.2, random_state=42
)
X_tr_scaled_s, _, y_tr_centered_s = scaleData(X_train_s, X_test_s, y_train_s)

n_tr_s, p_s = X_tr_scaled_s.shape
theta0_s = np.zeros(p_s)

for lamb in lambdas:
    # Reference analytical closed-form solution
    theta_cf_s = closedForm(X_tr_scaled_s, y_tr_centered_s, lamb=lamb)

    # Gradient function closure with penalty
    def grad_sens(theta):
        return GradRidgeAnalytic(theta, X_tr_scaled_s, y_tr_centered_s, lamb=lamb)

    for method in methods:
        for lr in learning_rates_sens:
            theta, state = np.array(theta0_s, dtype=float), {}
            diverged = False

            for t in range(1, 20001):
                g = grad_sens(theta)
                theta, state = optimiser_step(method, theta, g, state, t, lr)

                # Divergence guard: halt early before matrix math overflows
                if np.any(np.isnan(theta)) or np.any(np.abs(theta) > 1e10):
                    diverged = True
                    break

                if np.linalg.norm(g) < 1e-8:
                    break

            if diverged:
                err = 1e2
            else:
                err = np.max(np.abs(theta - theta_cf_s))
                if err > 100:
                    err = 1e2
                    diverged = True

            sensitivity_results.append(
                {
                    "degree": d_sens,
                    "method": method,
                    "learning_rate": lr,
                    "lambda": lamb,
                    "final_param_error": err,
                    "diverged": diverged,
                }
            )

naming_sens = {"n": 100, "type": "lr_sensitivity", "exercise": "f"}
writeToFile(naming_sens, sensitivity_results)