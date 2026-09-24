import matplotlib.pyplot as plt
import pandas as pd 
import ast
import numpy as np
from matplotlib.ticker import MaxNLocator
from pathlib import Path

# Ensure that the figures directory exists
Path("figures").mkdir(parents=True, exist_ok=True)

# Read general .txt or .csv file and return number values in dictionary
def readResultsFile(filename):
    df = pd.read_csv(filename, sep="\t") # read file
    data = {}
    for col in df.columns:
        first_val = str(df[col].iloc[0]).strip() # labels
        if first_val.startswith("["): # if array or list
            data[col] = np.array([np.array(ast.literal_eval(v)) for v in df[col]], dtype = object)
        else: # single number values
            data[col] = df[col].to_numpy()
    return data

#
# --------------------------------------------------------------------
# CLOSED FORM
# --------------------------------------------------------------------
#

# -----------------------------------
# OLS closed form no resampling
# -----------------------------------

data = readResultsFile("results/part=a/n=100_noise=0.1_results.txt")

d = data["d"]
params = data["params"] # array of arrays, different length per row
mse = data["MSE"]
r2 = data["R2"]

# MSE
plt.figure(figsize=(3.7, 2.8))
plt.plot(d, mse, label = "MSE", color = "b")
plt.xlabel(r"Polynomial degree $(d)$")
plt.ylabel("MSE")
plt.grid()
plt.xlim(np.min(d), np.max(d))
plt.tight_layout()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig("figures/n=100_noise=0.1_exercise=a_MSE_OLS.pdf", bbox_inches='tight')
#plt.show()

# R2
plt.figure(figsize=(3.7, 2.8))
plt.plot(d, r2, label = "R2 score", color = "r")
plt.xlabel(r"Polynomial degree $(d)$")
plt.ylabel(r"R$^2$ score")
plt.grid()
plt.xlim(np.min(d), np.max(d))
plt.tight_layout()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig("figures/n=100_noise=0.1_exercise=a_R2_OLS.pdf", bbox_inches='tight')
#plt.show()

# Params
# Showing how params grow large with OLS as the model overfits at high variance
norms = [np.linalg.norm(theta) for theta in params]
plt.figure(figsize=(3.7, 2.8))
plt.plot(d, norms, color = "k")
plt.xlabel(r"Polynomial degree $(d)$")
plt.ylabel(r"$\|\boldsymbol{\theta}\|_2$")
#plt.title("Parameternorms, OLS, n = 100, noise = 0.1")
plt.xlim(np.min(d), np.max(d))
plt.grid()
plt.tight_layout()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig("figures/n=100_noise=0.1_exercise=a_param_norm_OLS.pdf", bbox_inches='tight')
#plt.show()
# Heatmap of params
max_len = max(len(theta) for theta in params)
matrix = np.full((len(d), max_len), np.nan)
for i, theta in enumerate(params):
    matrix[i, :len(theta)] = theta
plt.figure(figsize=(3.7, 2.8))
plt.imshow(matrix.T, aspect="auto", cmap="coolwarm")
plt.xlabel(r"Polynomial degree $(d)$")
plt.ylabel(r"Coefficient index $(\theta_i)$")
y_labels = np.arange(1, matrix.shape[1] + 1)
plt.yticks(ticks=np.arange(matrix.shape[1])[::2], labels=y_labels[::2])
plt.xticks(ticks=np.arange(len(d))[::2], labels=d[::2])
plt.colorbar(label=r"Coefficient value")
plt.tight_layout()
plt.savefig("figures/n=100_noise=0.1_exercise=a_param_heatmap_OLS.pdf", bbox_inches='tight')
#plt.show()

"""
# Noise and n analysis
noises = [0.05, 0.1, 0.3]
ns = [50, 100, 250, 500]
chosen_d = [1, 3, 6, 8, 10, 11, 12, 13, 14, 15]
# Structure: table[noise][d][n] = MSE
tables = {}
for noise in noises:
    table = np.full((len(chosen_d), len(ns)), np.nan)
    for col, n in enumerate(ns):
        #filename = f"results/n={n}_noise={noise}_exercise=a_results.txt"
        filename = f"results/part=a/n={n}_noise={noise}_results.txt"
        data = readResultsFile(filename)
        for row, d in enumerate(chosen_d):
            mask = data["d"] == d
            table[row, col] = data["MSE"][mask][0]  # one match per d
    tables[noise] = table
# Check
for noise, table in tables.items():
    print(f"\nNoise = {noise}")
    header = "d\\n".ljust(6) + "".join(f"{n:>10}" for n in ns)
    print(header)
    for d, row in zip(chosen_d, table):
        print(f"{d:<6}" + "".join(f"{val:>10.4g}" for val in row))
"""

# -----------------------------------
# Ridge closed form no resampling
# -----------------------------------

data = readResultsFile("results/part=b/n=100_noise=0.1_results.txt")

lamb = data["lambda"]
d = data["d"]
params = data["params"] # array of arrays, different length per row
mse = data["MSE"]
r2 = data["R2"]


# MSE and R2 as func of lambda
chosen_degree = 11
mask = d == chosen_degree
plt.figure(figsize=(3.7, 2.8))
plt.plot(lamb[mask], mse[mask], label = "MSE", color = "b")
plt.xlabel(r"Penalty $(\lambda)$")
plt.ylabel("MSE")
plt.xlim(np.min(lamb), np.max(lamb))
plt.xscale("log")
plt.grid()
plt.tight_layout()
plt.savefig("figures/n=100_noise=0.1_exercise=b_MSE_Ridge.pdf", bbox_inches='tight')
#plt.show()

plt.figure(figsize=(3.7, 2.8))
plt.plot(lamb[mask], r2[mask], label = "R2 score", color = "r")
plt.xlabel(r"Penalty $(\lambda)$")
plt.ylabel(r"R$^2$ score")
plt.xlim(np.min(lamb), np.max(lamb))
plt.xscale("log")
plt.grid()
plt.tight_layout()
plt.savefig("figures/n=100_noise=0.1_exercise=b_R2_Ridge.pdf", bbox_inches='tight')
#plt.show()

# Params
# func of d
available_lambdas = np.unique(lamb)
chosen_lambda = available_lambdas[np.argmin(np.abs(available_lambdas - 0.1))]
mask = lamb == chosen_lambda
norms_lambda = [np.linalg.norm(theta) for theta in params[mask]]
plt.figure(figsize=(3.7, 2.8))
plt.plot(d[mask], norms_lambda, color = "k")
plt.xlabel(r"Polynomial degree $(d)$")
plt.ylabel(r"$\|\boldsymbol{\theta}\|_2$")
#plt.title("Parameternorms, Ridge, n = 100, noise = 0.1")
plt.xlim(np.min(d), np.max(d))
plt.grid()
plt.tight_layout()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig("figures/n=100_noise=0.1_exercise=b_param_of_d_Ridge.pdf", bbox_inches='tight')
#plt.show()
# func of lamda
chosen_degree = 11
mask = d == chosen_degree
norms_degree = [np.linalg.norm(theta) for theta in params[mask]]
plt.figure(figsize=(3.7, 2.8))
plt.plot(lamb[mask], norms_degree, color = "k")
plt.xlabel(r"Penalty $(\lambda)$")
plt.ylabel(r"$\|\boldsymbol{\theta}\|_2$")
#plt.title("Parameternorms, Ridge, n = 100, noise = 0.1")
plt.xlim(np.min(lamb[mask]), np.max(lamb[mask]))
plt.grid()
plt.xscale("log")
plt.tight_layout()
plt.savefig("figures/n=100_noise=0.1_exercise=b_param_of_lam_Ridge.pdf", bbox_inches='tight')
#plt.show()


available_lambdas = np.unique(lamb)
markers = ["o", "s", "^"]
fig, ax = plt.subplots(figsize=(4.5, 2.8))

for lamb_val, marker in zip([available_lambdas[0], available_lambdas[5], available_lambdas[-1]], markers):
    mask = lamb == lamb_val
    ax.plot(d[mask], mse[mask], label=f"λ={lamb_val:.0e}", marker=marker, color="grey", markevery=1)

ax.set_xlabel(r"Polynomial degree $(d)$")
ax.set_ylabel("MSE")
ax.set_xlim(np.min(d), np.max(d))
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
ax.grid()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig("figures/n=100_noise=0.1_exercise=b_MSE_given_lam_Ridge.pdf", bbox_inches='tight')
#plt.show()

# --------------------------------------
# Remake Hastie figure for different n
# --------------------------------------

# n = 50
data_c = readResultsFile("results/part=c_Hastie/n=50_noise=0.1_results.txt")
d = data_c["d"]
train_MSE = data_c["Train MSE"]
test_MSE = data_c["Test MSE"]
plt.figure(figsize=(2.7, 2.3))
plt.plot(d, train_MSE, label="Training", color="b")
plt.plot(d, test_MSE, label="Testing", color="palevioletred")
plt.xlabel(r"Polynomial degree $(d)$")
plt.ylabel("MSE")
plt.xlim(np.min(d), np.max(d))
plt.title(r"$n = 50$")
plt.grid()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig("figures/n=100_noise=0.1_exercise=c_Hastie_50_OLS.pdf", bbox_inches='tight')
#plt.show()
# n = 100
data_c = readResultsFile("results/part=c_Hastie/n=100_noise=0.1_results.txt")
d = data_c["d"]
train_MSE = data_c["Train MSE"]
test_MSE = data_c["Test MSE"]
plt.figure(figsize=(2.5, 2.3))
plt.plot(d, train_MSE, label="Training", color="b")
plt.plot(d, test_MSE, label="Testing", color="palevioletred")
plt.xlabel(r"Polynomial degree $(d)$")
plt.xlim(np.min(d), np.max(d))
plt.title(r"$n = 100$")
plt.grid()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig("figures/n=100_noise=0.1_exercise=c_Hastie_OLS.pdf", bbox_inches='tight')
#plt.show()
# n = 500
data_c = readResultsFile("results/part=c_Hastie/n=500_noise=0.1_results.txt")
d = data_c["d"]
train_MSE = data_c["Train MSE"]
test_MSE = data_c["Test MSE"]
plt.figure(figsize=(3.5, 2.3))
plt.plot(d, train_MSE, label="Training", color="b")
plt.plot(d, test_MSE, label="Testing", color="palevioletred")
plt.xlabel(r"Polynomial degree $(d)$")
plt.xlim(np.min(d), np.max(d))
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
plt.grid()
plt.title(r"$n = 500$")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig("figures/n=100_noise=0.1_exercise=c_Hastie_500_OLS.pdf", bbox_inches='tight')
#plt.show()

# --------------------------------------
# Bias variance bootstrap
# --------------------------------------

# n = 50
data_c = readResultsFile("results/part=c_tradeoff/n=50_noise=0.1_results.txt")
#print(data_c)
d = data_c["degrees"][0]
MSE = data_c["MSE"][0]
bias = data_c["bias"][0]
var = data_c["variance"][0]
plt.figure(figsize=(2.5, 2.3))
plt.plot(d, MSE, label="MSE", color="b")
plt.plot(d, bias, label="Bias", color="grey", marker = "o", markevery=2)
plt.plot(d, var, label="Variance", color="grey", marker = "s", markevery=2)
plt.xlabel(r"Polynomial degree $(d)$")
plt.ylabel("Bias Variance decomposition")
plt.xlim(np.min(d), np.max(d))
plt.title(r"$n = 50$")
plt.grid()
plt.yscale("log")
#plt.ylim(0, 0.5)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig("figures/n=50_noise=0.1_exercise=c_tradeoff_OLS.pdf", bbox_inches='tight')
#plt.show()

# n = 100
data_c = readResultsFile("results/part=c_tradeoff/n=100_noise=0.1_results.txt")
d = data_c["degrees"][0]
MSE = data_c["MSE"][0]
bias = data_c["bias"][0]
var = data_c["variance"][0]
plt.figure(figsize=(2.3, 2.3))
plt.plot(d, MSE, label="MSE", color="b")
plt.plot(d, bias, label="Bias", color="grey", marker = "o", markevery=2)
plt.plot(d, var, label="Variance", color="grey", marker = "s", markevery=2)
plt.xlabel(r"Polynomial degree $(d)$")
plt.xlim(np.min(d), np.max(d))
plt.title(r"$n = 100$")
plt.grid()
plt.yscale("log")
#plt.ylim(0, 0.5)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig("figures/n=100_noise=0.1_exercise=c_tradeoff_OLS.pdf", bbox_inches='tight')
#plt.show()

# n = 500
data_c = readResultsFile("results/part=c_tradeoff/n=500_noise=0.1_results.txt")
d = data_c["degrees"][0]
MSE = data_c["MSE"][0]
bias = data_c["bias"][0]
var = data_c["variance"][0]
plt.figure(figsize=(3.5, 2.3))
plt.plot(d, MSE, label="MSE", color="b")
plt.plot(d, bias, label="Bias", color="grey", marker = "o", markevery=2)
plt.plot(d, var, label="Variance", color="grey", marker = "s", markevery=2)
plt.xlabel(r"Polynomial degree $(d)$")
plt.xlim(np.min(d), np.max(d))
plt.title(r"$n = 500$")
plt.grid()
plt.yscale("log")
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig("figures/n=500_noise=0.1_exercise=c_tradeoff_OLS.pdf", bbox_inches='tight')
#plt.show()


# -----------------------------------
# Comparison closed form no resampling
# -----------------------------------




# --------------------------------------
# Part D
# Cross-validation OLS
# --------------------------------------

# n = 50
data_d = readResultsFile("results/part=d_OLS/n=50_noise=0.1_results.txt")
# print(data_d)

k = data_d["k"]
d = data_d["d"]
MSE = data_d["MSE"]

plt.figure(figsize=(2.5, 2.3))

mask5 = k == 5
mask10 = k == 10

plt.plot(d[mask5], MSE[mask5], label="k = 5")
plt.plot(d[mask10], MSE[mask10], label="k = 10")

plt.xlabel(r"Polynomial degree $(d)$")
plt.ylabel("Mean Squared Error")
plt.xlim(np.min(d), np.max(d))
plt.title(r"$n = 50$")
plt.grid()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()

plt.savefig(
    "figures/n=50_noise=0.1_exercise=d_MSE_OLS_CV.pdf",
    bbox_inches="tight"
)
#plt.show()


# n = 100
data_d = readResultsFile("results/part=d_OLS/n=100_noise=0.1_results.txt")

k = data_d["k"]
d = data_d["d"]
MSE = data_d["MSE"]

plt.figure(figsize=(2.3, 2.3))

mask5 = k == 5
mask10 = k == 10

plt.plot(d[mask5], MSE[mask5], label="k = 5")
plt.plot(d[mask10], MSE[mask10], label="k = 10")

plt.xlabel(r"Polynomial degree $(d)$")
plt.xlim(np.min(d), np.max(d))
plt.title(r"$n = 100$")
plt.grid()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()

plt.savefig(
    "figures/n=100_noise=0.1_exercise=d_MSE_OLS_CV.pdf",
    bbox_inches="tight"
)
#plt.show()


# n = 500
data_d = readResultsFile("results/part=d_OLS/n=500_noise=0.1_results.txt")

k = data_d["k"]
d = data_d["d"]
MSE = data_d["MSE"]

plt.figure(figsize=(3.5, 2.3))

mask5 = k == 5
mask10 = k == 10

plt.plot(d[mask5], MSE[mask5], label="k = 5")
plt.plot(d[mask10], MSE[mask10], label="k = 10")

plt.xlabel(r"Polynomial degree $(d)$")
plt.xlim(np.min(d), np.max(d))
plt.title(r"$n = 500$")
plt.grid()

plt.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    frameon=False
)

plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()

plt.savefig(
    "figures/n=500_noise=0.1_exercise=d_MSE_OLS_CV.pdf",
    bbox_inches="tight"
)
#plt.show()


# -----------------------------------
# Part D
# Cross-validation Ridge
# -----------------------------------

num_points = [50, 100, 500]
noise = 0.1

for n in num_points:

    data = readResultsFile(
        f"results/part=d_Ridge/n={n}_noise={noise}_results.txt"
    )

    k = data["k"]
    d = data["d"]
    lamb = data["lambda"]
    mse = data["MSE"]

    plt.figure(figsize=(3.7, 2.8))

    for lam in np.unique(lamb):

        mask5 = (k == 5) & (lamb == lam)
        mask10 = (k == 10) & (lamb == lam)

        plt.plot(
            d[mask5], mse[mask5],
            label=fr"$\lambda={lam:.1e}$, k=5"
        )

        plt.plot(
            d[mask10], mse[mask10],
            linestyle="--",
            label=fr"$\lambda={lam:.1e}$, k=10"
        )

    plt.xlabel(r"Polynomial degree $(d)$")
    plt.ylabel("Mean Squared Error")
    plt.title(f"n = {n}")
    plt.grid()
    plt.xlim(np.min(d), np.max(d))

    # Place legend to the right of the plot
    plt.legend(
        fontsize=6,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5)
    )

    plt.tight_layout()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.savefig(
        f"figures/n={n}_noise={noise}_exercise=d_MSE_Ridge_CV.pdf",
        bbox_inches="tight"
    )

    # plt.show()


# =====================================================================
# HERE WE CAN CHANGE THE PLOT COLOURS
# =====================================================================

# Unified method styles with solid lines for all gradient descent variants
method_styles = {
    "plain": {"color": "tab:blue", "linestyle": "-", "label": "Plain GD"},
    "momentum": {"color": "tab:orange", "linestyle": "-", "label": "Momentum"},
    "adagrad": {"color": "tab:green", "linestyle": "-", "label": "AdaGrad"},
    "rmsprop": {"color": "tab:red", "linestyle": "-", "label": "RMSprop"},
    "adam": {"color": "tab:purple", "linestyle": "-", "label": "Adam"}
}

methods = list(method_styles.keys())
models = ["OLS", "Ridge"]
chosen_lambda = 0.01  # for Ridge

# ====================================================================================================
# PART E: here we have two plotting scripts - one to plot MSE and R2 score for GD vs CF (both OLS and Ridge), 
# and another one to test how many iterations it takes to reach convergence (based on the criterion we defined)
# ====================================================================================================
data_e = readResultsFile("results/part=e/n=100_noise=0.1_results.txt")

# --- OLS MSE & R2 Comparisons ---

# Extract the data we need
ols_mask = data_e["model"] == "OLS"
d_ols = data_e["d"][ols_mask]
mse_gd_ols = data_e["MSE_GD"][ols_mask]
mse_cf_ols = data_e["MSE_CF"][ols_mask]
r2_gd_ols = data_e["R2_GD"][ols_mask]
r2_cf_ols = data_e["R2_CF"][ols_mask]


plt.figure(figsize=(3.7, 2.8))
plt.plot(d_ols, mse_gd_ols, label="Gradient descent", color="tab:blue", linestyle="-", linewidth=2.0, zorder=1)
plt.plot(d_ols, mse_cf_ols, label="Closed form", color="tab:orange", linestyle="--", linewidth=1.2, zorder=2)
plt.xlabel(r"Polynomial degree $(d)$")
plt.ylabel("MSE")
plt.grid(True)
plt.xlim(np.min(d_ols), np.max(d_ols))
plt.legend()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig("figures/part=e_OLS_MSE_comparison.pdf", bbox_inches="tight")
plt.close()


plt.figure(figsize=(3.7, 2.8))
plt.plot(d_ols, r2_gd_ols, label="Gradient descent", color="tab:blue", linestyle="-", linewidth=2.0, zorder=1)
plt.plot(d_ols, r2_cf_ols, label="Closed form", color="tab:orange", linestyle="--", linewidth=1.2, zorder=2)
plt.xlabel(r"Polynomial degree $(d)$")
plt.ylabel(r"$R^2$ Score")
plt.grid(True)
plt.xlim(np.min(d_ols), np.max(d_ols))
plt.legend()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig("figures/part=e_OLS_R2_comparison.pdf", bbox_inches="tight")
plt.close()

# --- Ridge MSE & R2 Comparisons ---

# Extract the data
ridge_mask = (data_e["model"] == "Ridge") & np.isclose(
    data_e["lambda"], chosen_lambda, rtol=1e-3, atol=1e-4
)
d_ridge = data_e["d"][ridge_mask]
mse_gd_ridge = data_e["MSE_GD"][ridge_mask]
mse_cf_ridge = data_e["MSE_CF"][ridge_mask]
r2_gd_ridge = data_e["R2_GD"][ridge_mask]
r2_cf_ridge = data_e["R2_CF"][ridge_mask]


plt.figure(figsize=(3.7, 2.8))
plt.plot(d_ridge, mse_gd_ridge, label="Gradient descent", color="tab:blue", linestyle="-", linewidth=2.0, zorder=1)
plt.plot(d_ridge, mse_cf_ridge, label="Closed form", color="tab:orange", linestyle="--", linewidth=1.2, zorder=2)
plt.xlabel(r"Polynomial degree $(d)$")
plt.ylabel("MSE")
plt.grid(True)
plt.xlim(np.min(d_ridge), np.max(d_ridge))
plt.legend()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig("figures/part=e_Ridge_MSE_comparison.pdf", bbox_inches="tight")
plt.close()


plt.figure(figsize=(3.7, 2.8))
plt.plot(d_ridge, r2_gd_ridge, label="Gradient descent", color="tab:blue", linestyle="-", linewidth=2.0, zorder=1)
plt.plot(d_ridge, r2_cf_ridge, label="Closed form", color="tab:orange", linestyle="--", linewidth=1.2, zorder=2)
plt.xlabel(r"Polynomial degree $(d)$")
plt.ylabel(r"$R^2$ Score")
plt.grid(True)
plt.xlim(np.min(d_ridge), np.max(d_ridge))
plt.legend()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig("figures/part=e_Ridge_R2_comparison.pdf", bbox_inches="tight")
plt.close()

# --- Iterations to reach convergence ---
for model_name in models:
    plt.figure(figsize=(4.5, 3.2))
    
    # Extract the data
    if model_name == "OLS":
        m_mask = data_e["model"] == "OLS"
        degrees = data_e["d"][m_mask]
        iters = data_e["iters_needed"][m_mask]
    else:
        m_mask = data_e["model"] == "Ridge"
        l_mask = m_mask & np.isclose(data_e["lambda"], chosen_lambda, rtol=1e-3, atol=1e-4)
        degrees = data_e["d"][l_mask]
        iters = data_e["iters_needed"][l_mask]
        
    plt.plot(
        degrees,
        iters,
        color="tab:blue",
        linestyle="-",
        linewidth=1.5
    )

    # Plot 10,000 iteration cap line
    plt.axhline(10000, color="black", linestyle="--", alpha=0.5, linewidth=1.0, label="Iteration cap")

    plt.xlabel(r"Polynomial degree $(d)$")
    plt.ylabel(r"Iterations to reach $\|\mathbf{g}\| < 10^{-8}$")
    plt.yscale("log")
    plt.ylim(1, 20000)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(fontsize=7, loc="best", frameon=True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    
    filename = f"figures/part=e_{model_name.lower()}_iters_vs_degree.pdf"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


# ===========================================================================
# PART F: in this part we have two plotting scripts - one to plot
# the sensitivity to learning rate (for a fixed degree d = 5) and one to plot
# the iterations required to reach a convergence threshold (for all degrees) 
# with a fixed (initial) learning rate gamma = 0.01
# ===========================================================================
data_f = readResultsFile("results/part=f/n=100_type=iters_vs_degree_results.txt")

# --- Iterations to reach convergence ---

for model_name in models:
    plt.figure(figsize=(4.5, 3.2))
    
    model_mask = data_f["model"] == model_name
    
    for method, style in method_styles.items():
        method_mask = model_mask & (data_f["method"] == method)
        if not np.any(method_mask):
            continue
            
        degrees = data_f["degree"][method_mask]
        iters = data_f["iters_needed"][method_mask]
        
        plt.plot(
            degrees,
            iters,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.5
        )

    # Plot 10,000 iteration cap line
    plt.axhline(10000, color="black", linestyle="--", alpha=0.5, linewidth=1.0, label="Iteration cap")

    plt.xlabel(r"Polynomial degree $(d)$")
    plt.ylabel(r"Iterations to reach $\|\mathbf{g}\| < 10^{-8}$")
    plt.yscale("log")
    plt.ylim(1, 20000)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(fontsize=7, loc="best", frameon=True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    
    filename = f"figures/part=f_{model_name}_iters_vs_degree.pdf"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


# --- Learning rate sensitivity (final theta error as a function of (initial) learning rate) ---


sens_file = Path("results") / "part=f" / "n=100_type=lr_sensitivity_results.txt"

if sens_file.exists():
    data_sens = pd.DataFrame(readResultsFile(sens_file))

    # --- OLS LR Sensitivity ---
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    sub_sens_ols = data_sens[data_sens["lambda"] == 0.0]

    for method, style in method_styles.items():
        mask = sub_sens_ols["method"] == method
        if not np.any(mask):
            continue
        ax.plot(
            sub_sens_ols["learning_rate"][mask],
            sub_sens_ols["final_param_error"][mask],
            color=style["color"],
            label=style["label"],
            linewidth=1.8,
            linestyle=style["linestyle"],
        )

    ax.axhline(1e-8, color="black", linestyle="--", alpha=0.7, label=r"Target ($10^{-8}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-12, 1e3)
    ax.set_xlabel(r"Initial learning rate ($\gamma$)", fontsize=11)
    ax.set_ylabel(r"$\max |\boldsymbol{\theta}_{\mathrm{final}} - \boldsymbol{\theta}_{\mathrm{CF}}|$", fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9, loc="best")

    plt.tight_layout()
    plt.savefig("figures/part_f_lr_sensitivity_ols.pdf", bbox_inches="tight")
    plt.close()

    # --- Ridge LR Sensitivity ---
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    sub_sens_ridge = data_sens[data_sens["lambda"] == 0.01]

    for method, style in method_styles.items():
        mask = sub_sens_ridge["method"] == method
        if not np.any(mask):
            continue
        ax.plot(
            sub_sens_ridge["learning_rate"][mask],
            sub_sens_ridge["final_param_error"][mask],
            color=style["color"],
            label=style["label"],
            linewidth=1.8,
            linestyle=style["linestyle"],
        )

    ax.axhline(1e-8, color="black", linestyle="--", alpha=0.7, label=r"Convergence Threshold ($10^{-8}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-12, 1e3)
    ax.set_xlabel(r"Initial learning rate ($\gamma$)", fontsize=11)
    ax.set_ylabel(r"$\max |\boldsymbol{\theta}_{\mathrm{final}} - \boldsymbol{\theta}_{\mathrm{CF}}|$", fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9, loc="best")

    plt.tight_layout()
    plt.savefig("figures/part_f_lr_sensitivity_ridge.pdf", bbox_inches="tight")
    plt.close()



# ======================================================================================================
# PART H: in this part we have two plotting scripts - one to plot
# final theta error as a function of total gradient evaluations (Adam with SDG, different batch sizes M),
# another one it to compare fixed lr vs ler decaying schedule (BUT I NEED TO FIX THIS ONE I THINK)
# P.s. this is done for polynomial degree d=5
# ======================================================================================================


degree = 5
results_dir = Path("results") / "part=h"
models = ["ols", "ridge"]

for model_name in models:
    # --- File paths updated to load Adam results ---
    batch_file = results_dir / f"type=batch_sensitivity_adam_{model_name}_degree={degree}_results.txt"
    if not batch_file.exists():
        continue
        
    batch_data = readResultsFile(batch_file)
    unique_batches = np.unique(batch_data["batch_size"])

    # ==========================================
    # Batch size error vs. total gradient evaluations
    # ==========================================
    fig, ax = plt.subplots(figsize=(3.5, 3.2), dpi=300)
    for M in unique_batches:
        mask = batch_data["batch_size"] == M
        ax.plot(batch_data["total_evals"][mask], batch_data["final_param_error"][mask], label=f"M = {M}")

    ax.axhline(1e-8, color="black", linestyle="--", alpha=0.6, label=r"Target ($10^{-8}$)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Total Gradient Evaluations")
    ax.set_ylabel(r"$\max |\boldsymbol{\theta}_{\mathrm{final}} - \boldsymbol{\theta}_{\mathrm{CF}}|$")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, loc="upper right", fontsize=8)
    
    # Added bounds to frame target and evaluation budget
    ax.set_ylim(bottom=1e-9, top=10.0)
    ax.set_xlim(right=100000)

    fig.subplots_adjust(left=0.22, right=0.95, bottom=0.18, top=0.92)
    plt.savefig(f"figures/part_h_batch_sensitivity_adam_evals_{model_name}.pdf", bbox_inches="tight", pad_inches=0.2)
    plt.close()


    # ==========================================
    # Constant vs. scheduled learning rate
    # ==========================================
    sched_file = results_dir / f"type=lr_schedule_adam_{model_name}_degree={degree}_results.txt"
    if not sched_file.exists():
        continue
        
    sched_data = readResultsFile(sched_file)
    unique_schedules = np.unique(sched_data["schedule"])
    colors = {"constant": "#d62728", "scheduled_1": "#1f77b4"}
    labels = {"constant": r"Constant ($\gamma = 0.01$)", "scheduled_1": r"Scheduled ($t_0=5, t_1=50$)"}

    fig, ax = plt.subplots(figsize=(3.5, 3.2), dpi=300)
    for sched_name in unique_schedules:
        mask = sched_data["schedule"] == sched_name
        # Switched from epoch to total_evals to maintain consistent cost axis
        ax.plot(
            sched_data["total_evals"][mask], 
            sched_data["final_param_error"][mask], 
            label=labels.get(sched_name, sched_name), 
            color=colors.get(sched_name, None)
        )

    ax.axhline(1e-8, color="black", linestyle="--", alpha=0.6, label=r"Target ($10^{-8}$)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Total Gradient Evaluations")
    ax.set_ylabel(r"$\max |\boldsymbol{\theta}_{\mathrm{final}} - \boldsymbol{\theta}_{\mathrm{CF}}|$")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, loc="lower left", fontsize=8)

    # Consistent bounds for the schedule comparison plot
    ax.set_ylim(bottom=1e-9, top=10.0)
    ax.set_xlim(right=100000)

    fig.subplots_adjust(left=0.28, right=0.92, bottom=0.18, top=0.92)
    plt.savefig(f"figures/part_h_lr_schedule_adam_comparison_{model_name}.pdf", bbox_inches="tight", pad_inches=0.2)
    plt.close()