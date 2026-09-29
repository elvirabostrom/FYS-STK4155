# -----------------------------------------------------------------------------------
# In this part, we use Adam optimizer for Lasso regression for all polynomial degrees 
# to find the optimal lambda
# Run with data with new seed for OLS, Ridge and lambda, with their optimal
# parameters found through cross-validation
# -----------------------------------------------------------------------------------

import numpy as np
from sklearn.model_selection import train_test_split
from utils import *


# -----------------------------------------------------------------------------------
# Find best lambda by cross-validation
# -----------------------------------------------------------------------------------

seed = 2026

def CrossValidationLasso(num_points, noise, degrees, punishers):
# Lasso, compute and save MSE using k-fold cross-validation
    for n in num_points:
        x, y = MakeData(n, noise, seed + n)
        x = x.reshape(-1, 1)

        naming = {"n": n, "noise": noise, "exercise": "i_Lasso"}
        results = []

        # Test 5-fold and 10-fold cross-validation
        for k in [5, 10]:
            kfold = KFold(n_splits=k, shuffle=True, random_state=1)

            for d in degrees:
                for lamb in punishers:
                    # Lasso model
                    model = make_pipeline(
                        PolynomialFeatures(degree=d, include_bias=False),
                        StandardScaler(),
                        Lasso(
                            alpha=lamb / 2.0,
                            fit_intercept=True,
                            max_iter=100000,
                            tol=1e-10
                        )
                    )

                    # Cross-validation
                    scores = -cross_val_score(
                        model,
                        x,
                        y,
                        cv=kfold,
                        scoring="neg_mean_squared_error"
                    )

                    mse = np.mean(scores)

                    results.append({
                        "k": k,
                        "d": d,
                        "lambda": lamb,
                        "MSE": mse
                    })

        writeToFile(naming, results)

# Run cross-validation Lasso, run for k=5 and k=10
num_points = np.array((100,))
noise = 0.1
degrees = np.arange(1, 16, 1)
n_punishers = 100
punishers = np.concatenate([[0], np.logspace(-12, -2, n_punishers)])

CrossValidationLasso(num_points, noise, degrees, punishers)


from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# -----------------------------------------------------------------------------------
# Final test: OLS, Ridge and Lasso fitted once with optimal parameters from CV,
# trained and tested on the same split of new, independent data
# -----------------------------------------------------------------------------------
# Written by Claude (Sept, 2026) w/ prompt to use optimal results from tables and complete
# a last fit with new seed

# Optimal parameters from cross-validation (n = 100, k = 5) - fill in from results
d_OLS = 11                          # from Fig. 7b
d_Ridge, lamb_Ridge = 12, 5.59e-8    # Tab. III
d_Lasso, lamb_Lasso = ..., ...       # from Lasso CV results

n = 100
noise = 0.1
test_seed = 123  # NEW SEED FOR FINAL TEST

x_new, y_new = MakeData(n, noise, test_seed)
x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size=0.2, random_state=69)
n_train = len(x_train)

results = []

# ---------------- OLS (closed form) ----------------
model = make_pipeline(
    PolynomialFeatures(d_OLS, include_bias=False),
    StandardScaler(),
    LinearRegression(fit_intercept=True)
)
model.fit(x_train.reshape(-1, 1), y_train)
mse = mean_squared_error(y_test, model.predict(x_test.reshape(-1, 1)))
results.append({"model": "OLS", "degree": d_OLS, "lambda": 0, "MSE": mse})

# ---------------- Ridge (closed form) ----------------
model = make_pipeline(
    PolynomialFeatures(d_Ridge, include_bias=False),
    StandardScaler(),
    Ridge(alpha=n_train * lamb_Ridge)   # same scaling as in CV
)
model.fit(x_train.reshape(-1, 1), y_train)
mse = mean_squared_error(y_test, model.predict(x_test.reshape(-1, 1)))
results.append({"model": "Ridge", "degree": d_Ridge, "lambda": lamb_Ridge, "MSE": mse})

# ---------------- Lasso (Adam) ----------------
target_tol = 1e-8
max_iters_cap = 10000
gamma = 0.55

X_train = MakeDesignMatrix(x_train, d_Lasso)
X_test = MakeDesignMatrix(x_test, d_Lasso)
X_train_scaled, X_test_scaled, y_train_centered = scaleData(X_train, X_test, y_train)

theta_init = np.zeros(X_train_scaled.shape[1])

def grad_d(theta):
    return GradLassoAnalytic(theta, X_train_scaled, y_train_centered, lamb_Lasso)

history = optimise(
    grad=grad_d,
    theta0=theta_init,
    method="adam",
    gamma=gamma,
    num_iters=max_iters_cap,
    tol=target_tol,
)

iters_needed = len(history) - 1
if iters_needed >= max_iters_cap:
    print(f"WARNING: Adam did not converge within {max_iters_cap} iterations")

theta_lasso = history[-1]
y_pred = X_test_scaled @ theta_lasso + y_train.mean()
mse = MSE(y_pred, y_test)
results.append({"model": "Lasso (Adam)", "degree": d_Lasso, "lambda": lamb_Lasso,
                "iters_needed": iters_needed, "MSE": mse})


# ---------------- Sanity check and save data ----------------
model = make_pipeline(
    PolynomialFeatures(d_Lasso, include_bias=False),
    StandardScaler(),
    Lasso(alpha=lamb_Lasso / 2.0, fit_intercept=True, max_iter=100000, tol=1e-10)
)
model.fit(x_train.reshape(-1, 1), y_train)
mse_sk = mean_squared_error(y_test, model.predict(x_test.reshape(-1, 1)))
print(f"Lasso test-MSE: Adam = {mse:.5f}, sklearn = {mse_sk:.5f}")

naming = {"n": n, "noise": noise, "type": "final_test", "exercise": "i"}
writeToFile(naming, results)


# -----------------------------------------------------------------------------------
# Save predictions for final fit
# -----------------------------------------------------------------------------------

x_plot = np.linspace(x_new.min(), x_new.max(), 200)

# OLS and Ridge
pred_OLS = model_OLS.predict(x_plot.reshape(-1, 1))
pred_Ridge = model_Ridge.predict(x_plot.reshape(-1, 1))

# Lasso (Adam), scale grid with the TRAINING mean/std, same as the test set
X_plot = MakeDesignMatrix(x_plot, d_Lasso)
_, X_plot_scaled, _ = scaleData(X_train, X_plot, y_train)
pred_Lasso = X_plot_scaled @ theta_lasso + y_train.mean()

predictions = [
    {"x": xi, "OLS": p1, "Ridge": p2, "Lasso": p3}
    for xi, p1, p2, p3 in zip(x_plot, pred_OLS, pred_Ridge, pred_Lasso)
]
naming = {"n": n, "noise": noise, "type": "final_fit_predictions", "exercise": "final"}
writeToFile(naming, predictions)

# Data points (train/test) for scatter in the same plot
datapoints = (
    [{"x": xi, "y": yi, "set": "train"} for xi, yi in zip(x_train, y_train)]
    + [{"x": xi, "y": yi, "set": "test"} for xi, yi in zip(x_test, y_test)]
)
naming = {"n": n, "noise": noise, "type": "final_fit_data", "exercise": "final"}
writeToFile(naming, datapoints)

