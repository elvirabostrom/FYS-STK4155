# -----------------------------------------------------------------------------------
# In this part, we use Adam optimizer for Lasso regression for all polynomial degrees 
# to find the optimal lambda
# Run with data with new seed for OLS, Ridge and lambda, with their optimal
# parameters found through cross-validation
# -----------------------------------------------------------------------------------

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import warnings
from sklearn.exceptions import ConvergenceWarning
from utils import *


# -----------------------------------------------------------------------------------
# Find best lambda by cross-validation
# -----------------------------------------------------------------------------------

seed = 2026

import warnings
from sklearn.exceptions import ConvergenceWarning

def CrossValidationLasso(num_points, noise, degrees, punishers):
# Lasso, k-fold CV with warm starts along the lambda path (large -> small)
# with the help of Claude (2026) to include the warm start
    lambdas = np.sort(punishers)[::-1]

    for n in num_points:
        x, y = MakeData(n, noise, seed + n)
        x = x.reshape(-1, 1)

        naming = {"n": n, "noise": noise, "exercise": "i_Lasso"}
        results = []

        for k in [5, 10]:
            kfold = KFold(n_splits=k, shuffle=True, random_state=1)
            print(k)

            for d in degrees:
                print(d)
                mse_folds = np.zeros((k, len(lambdas)))
                converged = np.ones(len(lambdas), dtype=bool)

                for i, (tr, va) in enumerate(kfold.split(x)):
                    poly = PolynomialFeatures(degree=d, include_bias=False)
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(poly.fit_transform(x[tr]))
                    X_va = scaler.transform(poly.transform(x[va]))

                    lasso = Lasso(fit_intercept=True, max_iter=100000, tol=1e-6, warm_start=True)

                    for j, lamb in enumerate(lambdas):
                        lasso.set_params(alpha = lamb / 2.0)
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always", ConvergenceWarning)
                            lasso.fit(X_tr, y[tr])
                        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                            converged[j] = False
                        mse_folds[i, j] = np.mean((lasso.predict(X_va) - y[va])**2)

                for j, lamb in enumerate(lambdas):
                    results.append({
                        "k": k,
                        "d": d,
                        "lambda": lamb,
                        "MSE": mse_folds[:, j].mean(), 
                        "MSE_std": mse_folds[:, j].std(ddof=1),
                        "converged": bool(converged[j])
                    })

        writeToFile(naming, results)
        return results

# Run cross-validation Lasso, run for k=5 and k=10
num_points = np.array((100,))
noise = 0.1
degrees = np.arange(1, 16, 1)
n_punishers = 100
punishers = np.logspace(-10, -2, n_punishers)

results_cv = CrossValidationLasso(num_points, noise, degrees, punishers)

chosen_k = 10
candidates = [r for r in results_cv if r["k"] == chosen_k]
best = min(candidates, key=lambda r: r["MSE"])
d_Lasso, lamb_Lasso = int(best["d"]), best["lambda"]
print(f"Lasso optimum: d = {d_Lasso}, lambda = {lamb_Lasso:.3e}, CV-MSE = {best['MSE']:.5f}")

# Lasso optimum: d = 12, lambda = 5.337e-10, CV-MSE = 0.01302

# -----------------------------------------------------------------------------------
# Final test: OLS, Ridge and Lasso fitted once with optimal parameters from CV,
# trained and tested on the same split of new, independent data
# -----------------------------------------------------------------------------------

# Compute error with standard error
def MSEwithSE(y_true, y_pred):
    MSE = (y_true - y_pred)**2
    return MSE.mean(), MSE.std(ddof=1) / np.sqrt(len(MSE))

# Optimal parameters from cross-validation (n = 100, k = 10) filled in from results
d_OLS = 10                        
d_Ridge, lamb_Ridge = 12, 8.11e-8    
d_Lasso, lamb_Lasso = 12, 6.428e-10

n = 100
noise = 0.1
test_seed = 123  # NEW SEED FOR FINAL TEST

x_new, y_new = MakeData(n, noise, test_seed)
x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size=0.2, random_state=69)
n_train = len(x_train)

results = []

# OLS
model_OLS = make_pipeline(
    PolynomialFeatures(d_OLS, include_bias=False),
    StandardScaler(),
    LinearRegression(fit_intercept=True)
)
model_OLS.fit(x_train.reshape(-1, 1), y_train)
mse, mse_se = MSEwithSE(y_test, model_OLS.predict(x_test.reshape(-1, 1)))
results.append({"model": "OLS", "degree": d_OLS, "lambda": 0, "MSE": mse, "SE": mse_se})

# Ridge
model_Ridge = make_pipeline(
    PolynomialFeatures(d_Ridge, include_bias=False),
    StandardScaler(),
    Ridge(alpha=n_train * lamb_Ridge)   # same scaling as in CV
)
model_Ridge.fit(x_train.reshape(-1, 1), y_train)
mse, mse_se = MSEwithSE(y_test, model_Ridge.predict(x_test.reshape(-1, 1)))
results.append({"model": "Ridge", "degree": d_Ridge, "lambda": lamb_Ridge, "MSE": mse, "SE": mse_se})


# Lasso (scikit-learn, same solver as in the cross-validation), fitted at the optimal lambda
# Written by Claude (Sept, 2026) w/ prompt to do the same for Lasso, as done for OLS and Ridge above
model_Lasso = make_pipeline(
    PolynomialFeatures(d_Lasso, include_bias=False),
    StandardScaler(),
    Lasso(alpha=lamb_Lasso / 2.0, fit_intercept=True, max_iter=100000, tol=1e-6)
)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", ConvergenceWarning)
    model_Lasso.fit(x_train.reshape(-1, 1), y_train)

lasso_converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
if not lasso_converged:
    print("WARNING: Lasso did not converge at the optimal lambda")

mse, mse_se = MSEwithSE(y_test, model_Lasso.predict(x_test.reshape(-1, 1)))
results.append({"model": "Lasso", "degree": d_Lasso, "lambda": lamb_Lasso,
                "converged": lasso_converged, "MSE": mse, "SE": mse_se})


# check and save data
model = make_pipeline(
    PolynomialFeatures(d_Lasso, include_bias=False),
    StandardScaler(),
    Lasso(alpha=lamb_Lasso / 2.0, fit_intercept=True, max_iter=100000, tol=1e-10)
)
model.fit(x_train.reshape(-1, 1), y_train)
mse_sk, mse_se = MSEwithSE(y_test, model.predict(x_test.reshape(-1, 1)))
print(f"Lasso test-MSE: sklearn = {mse_sk:.5f}")

naming = {"n": n, "noise": noise, "type": "final_test", "exercise": "i"}
writeToFile(naming, results)


# -----------------------------------------------------------------------------------
# Save predictions for final fit
# -----------------------------------------------------------------------------------

x_plot = np.linspace(x_new.min(), x_new.max(), 200).reshape(-1, 1)

pred_OLS = model_OLS.predict(x_plot)
pred_Ridge = model_Ridge.predict(x_plot)
pred_Lasso = model_Lasso.predict(x_plot)

predictions = [
    {"x": xi, "OLS": p1, "Ridge": p2, "Lasso": p3}
    for xi, p1, p2, p3 in zip(x_plot.ravel(), pred_OLS, pred_Ridge, pred_Lasso)
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






























