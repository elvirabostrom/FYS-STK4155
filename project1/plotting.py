import matplotlib.pyplot as plt
import pandas as pd 
import ast
import numpy as np
from matplotlib.ticker import MaxNLocator
from pathlib import Path

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
plt.show()

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
plt.show()

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
plt.show()
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
plt.show()

# Noise and n analysis
# noises = [0.05, 0.1, 0.3]
# ns = [50, 100, 250, 500]
# chosen_d = [1, 3, 6, 8, 10, 11, 12, 13, 14, 15]
# # Structure: table[noise][d][n] = MSE
# tables = {}
# for noise in noises:
#     table = np.full((len(chosen_d), len(ns)), np.nan)
#     for col, n in enumerate(ns):
#         filename = f"results/n={n}_noise={noise}_exercise=a_results.txt"
#         data = readResultsFile(filename)
#         for row, d in enumerate(chosen_d):
#             mask = data["d"] == d
#             table[row, col] = data["MSE"][mask][0]  # one match per d
#     tables[noise] = table
# # Check
# for noise, table in tables.items():
#     print(f"\nNoise = {noise}")
#     header = "d\\n".ljust(6) + "".join(f"{n:>10}" for n in ns)
#     print(header)
#     for d, row in zip(chosen_d, table):
#         print(f"{d:<6}" + "".join(f"{val:>10.4g}" for val in row))

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
plt.show()

plt.figure(figsize=(3.7, 2.8))
plt.plot(lamb[mask], r2[mask], label = "R2 score", color = "r")
plt.xlabel(r"Penalty $(\lambda)$")
plt.ylabel(r"R$^2$ score")
plt.xlim(np.min(lamb), np.max(lamb))
plt.xscale("log")
plt.grid()
plt.tight_layout()
plt.savefig("figures/n=100_noise=0.1_exercise=b_R2_Ridge.pdf", bbox_inches='tight')
plt.show()

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
plt.show()
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
plt.show()


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
plt.show()

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
plt.show()
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
plt.show()
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
plt.show()

# --------------------------------------
# Bias variance bootstrap
# --------------------------------------

# n = 50
data_c = readResultsFile("results/part=c_tradeoff/n=50_noise=0.1_results.txt")
print(data_c)
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
plt.show()

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
plt.show()

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
plt.show()


# -----------------------------------
# Comparison closed form no resampling
# -----------------------------------

# -----------------------------------
# Part D
# Cross-validation OLS 
# -----------------------------------

num_points = [50, 100, 500]

for n in num_points:

    data = readResultsFile(
        f"results/part=d_OLS/n={n}_noise=0.1_results.txt"
    )

    k = data["k"]
    d = data["d"]
    mse = data["MSE"]

    # MSE
    plt.figure(figsize=(3.7, 2.8))

    for folds in [5, 10]:
        mask = k == folds
        plt.plot(d[mask], mse[mask], label=f"k = {folds}")

    plt.xlabel(r"Polynomial degree $(d)$")
    plt.ylabel("Mean Squared Error")
    plt.title(f"n = {n}")
    plt.grid()
    plt.xlim(np.min(d), np.max(d))
    plt.legend()
    plt.tight_layout()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.savefig(
        f"figures/n={n}_noise=0.1_exercise=d_MSE_OLS_CV.pdf",
        bbox_inches="tight"
    )

    plt.show()


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

    #plt.show()

    







# --------------------------------------------------------------------
# Gradient Descent vs Closed Form Benchmarks (Part E)
# --------------------------------------------------------------------

# 1. Load Part E (Gradient Descent) Data
data_e = readResultsFile("results/part=e/n=100_noise=0.1_results.txt")

# Set the hyperparameter value for Ridge filtering
chosen_lambda = 0.1

# Filter GD data for OLS and Ridge
ols_mask = data_e["model"] == "OLS"
ridge_mask = (data_e["model"] == "Ridge") & np.isclose(
    data_e["lambda"], chosen_lambda
)

# Extract Gradient Descent arrays (fixes NameError)
d_ols = data_e["d"][ols_mask]
mse_ols = data_e["MSE"][ols_mask]
r2_ols = data_e["R2"][ols_mask]

d_ridge = data_e["d"][ridge_mask]
mse_ridge = data_e["MSE"][ridge_mask]
r2_ridge = data_e["R2"][ridge_mask]

# 2. Load Closed-Form Benchmarks (Parts A and B)
data_a = readResultsFile("results/part=a/n=100_noise=0.1_results.txt")
data_b = readResultsFile("results/part=b/n=100_noise=0.1_results.txt")

# Extract Closed-Form OLS arrays
d_cf_ols = data_a["d"]
mse_cf_ols = data_a["MSE"]
r2_cf_ols = data_a["R2"]

# Extract Closed-Form Ridge arrays
mask_b = np.isclose(data_b["lambda"], chosen_lambda)
d_cf_ridge = data_b["d"][mask_b]
mse_cf_ridge = data_b["MSE"][mask_b]
r2_cf_ridge = data_b["R2"][mask_b]


# --------------------------------------------------------------------
# PLOTTING
# --------------------------------------------------------------------

# 1. OLS MSE Comparison
plt.figure(figsize=(3.7, 2.8))
plt.plot(d_cf_ols, mse_cf_ols, label="Closed form")
plt.plot(
    d_ols, mse_ols, label="Gradient descent"
)
plt.xlabel(r"Polynomial degree $(d)$")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.xlim(np.min(d_ols), np.max(d_ols))
plt.legend()
plt.tight_layout()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig("figures/part=e_OLS_CF_vs_GD_MSE.png", bbox_inches="tight")
plt.close()

# 2. OLS R2 Comparison
plt.figure(figsize=(3.7, 2.8))
plt.plot(d_cf_ols, r2_cf_ols, label="Closed form")
plt.plot(d_ols, r2_ols, label="Gradient descent")
plt.xlabel(r"Polynomial degree $(d)$")
plt.ylabel(r"R$^2$ score")
plt.grid(True)
plt.xlim(np.min(d_ols), np.max(d_ols))
plt.legend()
plt.tight_layout()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig("figures/part=e_OLS_CF_vs_GD_R2.png", bbox_inches="tight")
plt.close()

# 3. Ridge MSE Comparison
plt.figure(figsize=(3.7, 2.8))
plt.plot(
    d_cf_ridge,
    mse_cf_ridge,
    label=f"Closed form",
)
plt.plot(
    d_ridge,
    mse_ridge,
    label=f"Gradient descent",
)
plt.xlabel(r"Polynomial degree $(d)$")
plt.ylabel("Mean Squared Error")
plt.grid(True)
plt.xlim(np.min(d_ridge), np.max(d_ridge))
plt.legend()
plt.tight_layout()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig("figures/part=e_Ridge_CF_vs_GD_MSE.png", bbox_inches="tight")
plt.close()

# 4. Ridge R2 Comparison
plt.figure(figsize=(3.7, 2.8))
plt.plot(
    d_cf_ridge,
    r2_cf_ridge,
    label=f"Closed form",
)
plt.plot(
    d_ridge,
    r2_ridge,
    label=f"Gradient descent",
)
plt.xlabel(r"Polynomial degree $(d)$")
plt.ylabel(r"R$^2$ score")
plt.grid(True)
plt.xlim(np.min(d_ridge), np.max(d_ridge))
plt.legend()
plt.tight_layout()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig("figures/part=e_Ridge_CF_vs_GD_R2.png", bbox_inches="tight")
plt.close()
