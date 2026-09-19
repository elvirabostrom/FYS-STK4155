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
print(data_d)

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





# --------------------------------------------------------------------
# PART E
# --------------------------------------------------------------------

# Load files
data_a = readResultsFile("results/part=a/n=100_noise=0.1_results.txt")
data_b = readResultsFile("results/part=b/n=100_noise=0.1_results.txt")
data_e = readResultsFile("results/part=e/n=100_noise=0.1_results.txt")

d_cf_ols = data_a["d"]
mse_cf_ols = data_a["MSE"]

n_punishers = 10
punishers = np.logspace(-6, 6, n_punishers)
chosen_lambda = punishers[3]  # lambda = 0.01

mask_b = np.isclose(data_b["lambda"], chosen_lambda)
d_cf_ridge = data_b["d"][mask_b]
mse_cf_ridge = data_b["MSE"][mask_b]

ols_mask = data_e["model"] == "OLS"
d_gd_ols = data_e["d"][ols_mask]
mse_gd_ols = data_e["MSE"][ols_mask]

ridge_mask = (data_e["model"] == "Ridge") & np.isclose(
    data_e["lambda"], chosen_lambda, rtol=1e-3, atol=1e-4
)
d_gd_ridge = data_e["d"][ridge_mask]
mse_gd_ridge = data_e["MSE"][ridge_mask]

# 1. OLS MSE comparison
plt.figure(figsize=(3.7, 2.8))
plt.plot(d_cf_ols, mse_cf_ols, label="Closed form", color="tab:blue", linewidth=2.5)
plt.plot(
    d_gd_ols, 
    mse_gd_ols, 
    label="Gradient descent", 
    color="tab:orange", 
    linestyle="--", 
    linewidth=1.5
)
plt.xlabel(r"Polynomial degree $(d)$")
plt.ylabel("MSE")
plt.grid(True)
plt.xlim(np.min(d_gd_ols), np.max(d_gd_ols))
plt.legend()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig("figures/part=e_OLS_CF_vs_GD_MSE.pdf", bbox_inches="tight")
plt.close()

# 2. Ridge MSE comparison
plt.figure(figsize=(3.7, 2.8))
plt.plot(d_cf_ridge, mse_cf_ridge, label="Closed form", color="tab:blue", linewidth=2.5)
plt.plot(
    d_gd_ridge, 
    mse_gd_ridge, 
    label="Gradient descent", 
    color="tab:orange", 
    linestyle="--", 
    linewidth=1.5
)
plt.xlabel(r"Polynomial degree $(d)$")
plt.ylabel("MSE")
plt.grid(True)
plt.xlim(np.min(d_gd_ridge), np.max(d_gd_ridge))
plt.legend()
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig("figures/part=e_Ridge_CF_vs_GD_MSE.pdf", bbox_inches="tight")
plt.close()



# =====================================================================
# PART F
# =====================================================================

method_styles = {
    "plain": {"label": "Plain GD", "marker": "o", "color": "#1f77b4"},
    "momentum": {"label": "Momentum", "marker": "s", "color": "#ff7f0e"},
    "adagrad": {"label": "AdaGrad", "marker": "^", "color": "#2ca02c"},
    "rmsprop": {"label": "RMSprop", "marker": "d", "color": "#d62728"},
    "adam": {"label": "Adam", "marker": "*", "color": "#9467bd"},
}

# Ensure figures output directory exists
Path("figures").mkdir(parents=True, exist_ok=True)


# =====================================================================
# 1. Iterations vs. Polynomial Degree (OLS)
# =====================================================================
deg_file = Path("results") / "part=f" / "n=100_type=iters_vs_degree_results.txt"

if deg_file.exists():
    data_deg = pd.DataFrame(readResultsFile(deg_file))

    # --- Plot 1: OLS ---
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    sub_data_ols = data_deg[data_deg["lambda"] == 0.0]

    for method, style in method_styles.items():
        mask = sub_data_ols["method"] == method
        ax.plot(
            sub_data_ols["degree"][mask],
            sub_data_ols["iters_needed"][mask],
            marker=style["marker"],
            color=style["color"],
            label=style["label"],
            linewidth=1.8,
            markersize=6,
        )

    ax.axhline(50000, color="black", linestyle=":", alpha=0.7, label="Iteration Cap (50,000)")
    ax.set_yscale("log")
    ax.set_xlabel("Polynomial Degree ($d$)", fontsize=11)
    ax.set_ylabel("Iterations to Reach Target Error ($10^{-4}$)", fontsize=11)
    ax.set_title(r"OLS ($\lambda = 0.0$): Convergence Speed", fontsize=12, pad=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9, loc="upper left")

    plt.tight_layout()
    plt.savefig("figures/part_f_iters_vs_degree_ols.pdf", bbox_inches="tight")
    plt.close()

    # --- Plot 2: Ridge ---
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    sub_data_ridge = data_deg[data_deg["lambda"] == 1e-3]

    for method, style in method_styles.items():
        mask = sub_data_ridge["method"] == method
        ax.plot(
            sub_data_ridge["degree"][mask],
            sub_data_ridge["iters_needed"][mask],
            marker=style["marker"],
            color=style["color"],
            label=style["label"],
            linewidth=1.8,
            markersize=6,
        )

    ax.axhline(50000, color="black", linestyle=":", alpha=0.7, label="Iteration Cap (50,000)")
    ax.set_yscale("log")
    ax.set_xlabel("Polynomial Degree ($d$)", fontsize=11)
    ax.set_ylabel("Iterations to Reach Target Error ($10^{-4}$)", fontsize=11)
    ax.set_title(r"Ridge ($\lambda = 10^{-3}$): Convergence Speed", fontsize=12, pad=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9, loc="upper left")

    plt.tight_layout()
    plt.savefig("figures/part_f_iters_vs_degree_ridge.pdf", bbox_inches="tight")
    plt.close()


# =====================================================================
# 2. Final Parameter Error vs. Initial Learning Rate
# =====================================================================
sens_file = Path("results") / "part=f" / "n=100_type=lr_sensitivity_results.txt"

if sens_file.exists():
    data_sens = pd.DataFrame(readResultsFile(sens_file))

    # --- Plot 3: OLS Sensitivity ---
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    sub_sens_ols = data_sens[data_sens["lambda"] == 0.0]

    for method, style in method_styles.items():
        mask = sub_sens_ols["method"] == method
        ax.plot(
            sub_sens_ols["learning_rate"][mask],
            sub_sens_ols["final_param_error"][mask],
            marker=style["marker"],
            color=style["color"],
            label=style["label"],
            linewidth=1.8,
            markersize=6,
        )

    ax.axhline(1e-4, color="black", linestyle="--", alpha=0.7, label=r"Convergence Threshold ($10^{-4}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1e3)
    ax.set_xlabel(r"Initial Learning Rate ($\gamma$)", fontsize=11)
    ax.set_ylabel(r"Final Parameter Error $\max |\boldsymbol{\theta}_{\mathrm{final}} - \boldsymbol{\theta}_{\mathrm{CF}}|$", fontsize=11)
    ax.set_title(r"OLS ($d=10, \lambda = 0.0$): Learning Rate Sensitivity", fontsize=12, pad=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig("figures/part_f_lr_sensitivity_ols.pdf", bbox_inches="tight")
    plt.close()

    # --- Plot 4: Ridge Sensitivity ---
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    sub_sens_ridge = data_sens[data_sens["lambda"] == 1e-3]

    for method, style in method_styles.items():
        mask = sub_sens_ridge["method"] == method
        ax.plot(
            sub_sens_ridge["learning_rate"][mask],
            sub_sens_ridge["final_param_error"][mask],
            marker=style["marker"],
            color=style["color"],
            label=style["label"],
            linewidth=1.8,
            markersize=6,
        )

    ax.axhline(1e-4, color="black", linestyle="--", alpha=0.7, label=r"Convergence Threshold ($10^{-4}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1e3)
    ax.set_xlabel(r"Initial Learning Rate ($\gamma$)", fontsize=11)
    ax.set_ylabel(r"Final Parameter Error $\max |\boldsymbol{\theta}_{\mathrm{final}} - \boldsymbol{\theta}_{\mathrm{CF}}|$", fontsize=11)
    ax.set_title(r"Ridge ($d=10, \lambda = 10^{-3}$): Learning Rate Sensitivity", fontsize=12, pad=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=9, loc="lower left")

    plt.tight_layout()
    plt.savefig("figures/part_f_lr_sensitivity_ridge.pdf", bbox_inches="tight")
    plt.close()
