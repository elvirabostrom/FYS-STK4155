import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import ast
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from utils import *
from estimator import Regressor

# Create results directory
os.makedirs("figures", exist_ok=True)

# Generate Runge data
def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)

# Helper to initialize figure
def setup_plot():
    plt.figure(figsize=(6, 4))
    plt.scatter(X_train, Y_train, color="k", s=20, label="Train data")
    plt.scatter(X_test, Y_test, facecolors="none", edgecolors="k", s=25, label="Test data")
    plt.plot(x_grid, runge(x_grid), label="Runge function", color="#e60073", linestyle="--", zorder=5)
    plt.xlabel("x", fontsize=14)
    plt.ylabel("y", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True)

"""

rng = np.random.default_rng(2026)
n = 100
x = np.sort(rng.uniform(-1.0, 1.0, n))
sigma = 0.1
y = runge(x) + rng.normal(0.0, sigma, n)

# Reshape into (n, 1) and shuffle
X = x.reshape(-1, 1)
Y = y.reshape(-1, 1)
shuffled_indices = rng.permutation(n)
X, Y = X[shuffled_indices], Y[shuffled_indices]

# Split into train/test (80/20)
X_train, X_test = X[:int(0.8 * n)], X[int(0.8 * n):]
Y_train, Y_test = Y[:int(0.8 * n)], Y[int(0.8 * n):]
 
# Compute scaling parameters
X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
Y_mean = Y_train.mean(axis=0)

# Scale data
X_train_scaled = (X_train - X_mean) / X_std
Y_train_scaled = Y_train - Y_mean
X_test_scaled = (X_test - X_mean) / X_std
Y_test_scaled = Y_test - Y_mean

# Fine grid for visualization
x_grid = np.linspace(-1.0, 1.0, 500)
X_grid_scaled = (x_grid.reshape(-1, 1) - X_mean) / X_std

# Architecture configurations: (Name, hidden tuple)
architectures = [
    ("1 Hidden Layer (50)", (50,)),
    ("2 Hidden Layers (50, 50)", (50, 50)),
]

lr_values = [0.0005, 0.001, 0.005, 0.01]
colors = ["#81C784", "#388E3C", "#0288D1", "#081d58"]


#===============================================================================================================
# Optimizer Comparison (Adam, RMSprop, Plain GD)
#===============================================================================================================

optimizers = [
    ("adam", "Adam", 16),
    ("rmsprop", "RMSprop", 16),
    ("gd", "Plain GD", None),
]

print("\nRunning optimizer comparison")

for method_key, method_name, batch_size in optimizers:
    for arch_name, hidden_dims in architectures:
        print(f"\nRunning {method_name} for: {arch_name}")
        setup_plot()

        for lr, color in zip(lr_values, colors):
            reg = Regressor(
                hidden=hidden_dims,
                activation="sigmoid",
                method=method_key,
                gamma=lr,
                batch_size=batch_size,
                epochs=2000,
                tol=1e-7,
                seed=2026,
            )
            reg.fit(X_train_scaled, Y_train_scaled.ravel())

            # Evaluate test set
            Y_pred_scaled = reg.predict(X_test_scaled).reshape(-1, 1)
            test_mse = MSE(Y_pred_scaled + Y_mean, Y_test)
            print(f"Test MSE (lr={lr}): {test_mse:.5f}")

            # Predict grid and unscale
            Y_pred_grid = reg.predict(X_grid_scaled) + Y_mean
            plt.plot(x_grid, Y_pred_grid.ravel(), label=f"lr={lr}", color=color)

        num_hidden = len(hidden_dims)
        plt.legend(fontsize=10)
        plt.savefig(f"figures/runge_{method_key}_{num_hidden}layer.pdf", bbox_inches="tight")
        plt.close()

#===============================================================================================================
# ReLU and Leaky ReLU comparison (Adam)
#===============================================================================================================

print("\nRunning ReLU and Leaky ReLU comparison")

act_configs = [
    ("relu", "ReLU", "#8C8C8C"),
    ("leakyrelu", "Leaky ReLU", "#555555"),
]

for arch_name, hidden_dims in architectures:
    num_hidden = len(hidden_dims)
    filename = f"figures/runge_actfuncs_{num_hidden}layer.pdf"
    setup_plot()

    for act_key, act_name, color in act_configs:
        reg = Regressor(
            hidden=hidden_dims,
            activation=act_key,
            method="adam",
            gamma=0.001,
            batch_size=16,
            epochs=2000,
            tol=1e-7,
            seed=2026,
        )
        reg.fit(X_train_scaled, Y_train_scaled.ravel())

        Y_pred_scaled = reg.predict(X_test_scaled).reshape(-1, 1)
        test_mse = MSE(Y_pred_scaled + Y_mean, Y_test)
        print(f"Test MSE ({arch_name}, {act_name}): {test_mse:.5f}")

        Y_pred_grid = reg.predict(X_grid_scaled) + Y_mean
        plt.plot(x_grid, Y_pred_grid.ravel(), color=color, label=act_name)

    plt.legend(fontsize=10)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
"""

# ===============================================================================================================
# Cross-Validation
# ===============================================================================================================

rng = np.random.default_rng(123)
n = 100
x = np.sort(rng.uniform(-1.0, 1.0, n))
sigma = 0.1
y = runge(x) + rng.normal(0.0, sigma, n)

# Reshape into (n, 1) and shuffle
X = x.reshape(-1, 1)
Y = y.reshape(-1, 1)
shuffled_indices = rng.permutation(n)
X, Y = X[shuffled_indices], Y[shuffled_indices]

# Split into train/test (80/20)
X_train, X_test = X[:int(0.8 * n)], X[int(0.8 * n):]
Y_train, Y_test = Y[:int(0.8 * n)], Y[int(0.8 * n):]

# Compute target mean strictly from training set for centering
Y_mean = Y_train.mean(axis=0)
Y_train_scaled = Y_train - Y_mean

# Performs 5-Fold cross-validation over architecture, lambda, and gamma
def runge_grid_search(X, y, norm_type, filename):
    cv = KFold(n_splits=5, shuffle=True, random_state=123)
    
    # Create a pipeline
    pipeline = make_pipeline(
        StandardScaler(),
        Regressor(
            activation="sigmoid", 
            method="adam", 
            epochs=2000, 
            batch_size=16, 
            tol=1e-7,
            norm=norm_type,
            seed=123,
        )
    )
    
    param_grid = {
        "regressor__hidden": [
            (50, 50)    # 2 hidden layers
        ],
        "regressor__lmbd": [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 0.01], 
        "regressor__gamma": [0.0005, 0.001, 0.003, 0.005, 0.01]       
    }
    
    search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv, 
        scoring="neg_mean_squared_error", 
        return_train_score=True, 
        n_jobs=4, 
        refit=True, 
        verbose=1
    )
    
    # Pass unscaled X and scaled y
    search.fit(X, y.ravel())
    results = search.cv_results_

    df = pd.DataFrame({
        "hidden": [str(p["regressor__hidden"]) for p in results["params"]],
        "lmbd": [p["regressor__lmbd"] for p in results["params"]],
        "gamma": [p["regressor__gamma"] for p in results["params"]],
        "cv_mse": -results["mean_test_score"],
        "cv_se": results["std_test_score"] / np.sqrt(cv.get_n_splits() - 1),
        "train_mse": -results["mean_train_score"],
        "fit_time": results["mean_fit_time"]
    })

    df.to_csv(f"results/{filename}.csv", index=False)
    
    print(f"\nBest results for {norm_type.upper()} regularization:")
    print(search.best_params_)
    print(f"Best cross-validation MSE: {-search.best_score_:.6f}\n")
    
    return df, search.best_estimator_

#print("\nStarting L2 (Ridge) search")
#runge_grid_search(X_train, Y_train_scaled, norm_type="l2", filename="gridsearch_runge_l2")
#print("\nStarting L1 (Lasso) search")
#runge_grid_search(X_train, Y_train_scaled, norm_type="l1", filename="gridsearch_runge_l1")

# ===============================================================================================================
# Final fit
# ===============================================================================================================

def load_and_train_model(csv_path, norm_type, X_train, Y_train_scaled):
    # Load the grid search results dataframe
    df = pd.read_csv(csv_path)
    
    # Find the row with the minimum cross-validation MSE
    best_row = df.loc[df["cv_mse"].idxmin()]
    
    # Extract best hyperparameters
    best_hidden = ast.literal_eval(best_row["hidden"])
    best_lmbd = float(best_row["lmbd"])
    best_gamma = float(best_row["gamma"])
    
    print(f"Loaded best parameters for {norm_type.upper()}:")
    print(f"  hidden: {best_hidden}, lmbd: {best_lmbd}, gamma: {best_gamma}")
    print(f"  CV MSE: {best_row['cv_mse']:.6f}")
    
    model = make_pipeline(
        StandardScaler(),
        Regressor(
            activation="sigmoid", 
            method="adam", 
            epochs=2000, 
            batch_size=16, 
            tol=1e-7,
            norm=norm_type,
            hidden=best_hidden,
            lmbd=best_lmbd,
            gamma=best_gamma,
            seed=123,
        )
    )
    
    # Train the network
    model.fit(X_train, Y_train_scaled.ravel())
    return model

# Load and train models from CSV files
model_l1 = load_and_train_model("results/gridsearch_runge_l1.csv", "l1", X_train, Y_train_scaled)
model_l2 = load_and_train_model("results/gridsearch_runge_l2.csv", "l2", X_train, Y_train_scaled)

# Evaluate L1 on test set
test_mse_l1 = MSE(model_l1.predict(X_test) + Y_mean, Y_test)
print(f"\nTest MSE (L1): {test_mse_l1:.5f}")

# Evaluate L2 on test set
test_mse_l2 = MSE(model_l2.predict(X_test) + Y_mean, Y_test)
print(f"Test MSE (L2): {test_mse_l2:.5f}")

# Fine grid for visualization
x_grid = np.linspace(-1.0, 1.0, 500)

# Predict on fine grid
Y_pred_grid_l1 = model_l1.predict(x_grid.reshape(-1, 1)) + Y_mean
Y_pred_grid_l2 = model_l2.predict(x_grid.reshape(-1, 1)) + Y_mean

# Plot results
plt.figure(figsize=(6, 4))
plt.scatter(X_train, Y_train, color="k", s=20, label="Train data")
plt.scatter(X_test, Y_test, facecolors="none", edgecolors="k", s=25, label="Test data")
plt.plot(x_grid, runge(x_grid), label="Runge function", color="#e60073", linestyle="--", zorder=5)
plt.xlabel("x", fontsize=10)
plt.ylabel("y", fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.grid(True)
plt.plot(x_grid, Y_pred_grid_l1.ravel(), label=r"Fit", color="#0288D1")
#plt.plot(x_grid, Y_pred_grid_l2.ravel(), label=r"L2", color="#388E3C")
plt.legend(fontsize=10)
plt.savefig("figures/runge_final_fit.pdf", bbox_inches="tight")
plt.close()

# ===============================================================================================================
# Standalone fit with gamma = 0.001
# ===============================================================================================================

df_l1 = pd.read_csv("results/gridsearch_runge_l1.csv")
df_l2 = pd.read_csv("results/gridsearch_runge_l2.csv")

# Extract best lambda for gamma = 0.001 from L1 results
df_l1_001 = df_l1[df_l1["gamma"] == 0.001]
best_lmbd_l1_001 = df_l1_001.loc[df_l1_001["cv_mse"].idxmin(), "lmbd"]
print(f"Best lambda for gamma=0.001 (L1): {best_lmbd_l1_001}")

# Extract best lambda for gamma = 0.001 from L2 results
df_l2_001 = df_l2[df_l2["gamma"] == 0.001]
best_lmbd_l2_001 = df_l2_001.loc[df_l2_001["cv_mse"].idxmin(), "lmbd"]
print(f"Best lambda for gamma=0.001 (L2): {best_lmbd_l2_001}")

model_l1_001 = make_pipeline(
    StandardScaler(),
    Regressor(
        activation="sigmoid", 
        method="adam", 
        epochs=2000, 
        batch_size=16, 
        tol=1e-7,
        norm="l1",
        seed=123,
        hidden=(50, 50),
        lmbd=best_lmbd_l1_001,
        gamma=0.001
    )
)
model_l1_001.fit(X_train, Y_train_scaled.ravel())

model_l2_001 = make_pipeline(
    StandardScaler(),
    Regressor(
        activation="sigmoid", 
        method="adam", 
        epochs=2000, 
        batch_size=16, 
        tol=1e-7,
        norm="l2",
        seed=123,
        hidden=(50, 50),
        lmbd=best_lmbd_l2_001,
        gamma=0.001
    )
)
model_l2_001.fit(X_train, Y_train_scaled.ravel())

# Evaluate on the fine grid
x_grid = np.linspace(-1.0, 1.0, 500)
Y_pred_grid_l1_001 = model_l1_001.predict(x_grid.reshape(-1, 1)) + Y_mean
Y_pred_grid_l2_001 = model_l2_001.predict(x_grid.reshape(-1, 1)) + Y_mean

# Plot
plt.figure(figsize=(6, 4))
plt.scatter(X_train, Y_train, color="k", s=20, label="Train data")
plt.scatter(X_test, Y_test, facecolors="none", edgecolors="k", s=25, label="Test data")
plt.plot(x_grid, runge(x_grid), label="Runge function", color="#e60073", linestyle="--", zorder=5)

plt.plot(
    x_grid, 
    Y_pred_grid_l1_001.ravel(), 
    label=rf"$\ell_1$ ($\gamma=0.001, \lambda={best_lmbd_l1_001}$)", 
    color="#0288D1", 
    linestyle="-"
)
plt.plot(
    x_grid, 
    Y_pred_grid_l2_001.ravel(), 
    label=rf"$\ell_2$ ($\gamma=0.001, \lambda={best_lmbd_l2_001}$)", 
    color="#388E3C", 
    linestyle="-"
)

plt.xlabel("x", fontsize=10)
plt.ylabel("y", fontsize=10)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.grid(True)
plt.legend(fontsize=8)
plt.savefig("figures/runge_fit_gamma_001_l1_l2.pdf", bbox_inches="tight")
plt.close()