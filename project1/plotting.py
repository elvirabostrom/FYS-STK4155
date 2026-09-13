import matplotlib.pyplot as plt
import pandas as pd 
import ast
import numpy as np
from matplotlib.ticker import MaxNLocator


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

data = readResultsFile("results/n=100_noise=0.1_exercise=a_results.txt")

d = data["d"]
params = data["params"] # array of arrays, different length per row
mse = data["MSE"]
r2 = data["R2"]

# MSE
plt.figure(figsize=(3.7, 2.8))
plt.plot(d, mse, label = "MSE", color = "b")
plt.xlabel(r"Polynomial degree $(d)$")
plt.ylabel("Mean Squared Error")
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
noises = [0.05, 0.1, 0.3]
ns = [50, 100, 250, 500]
chosen_d = [1, 3, 6, 8, 10, 11, 12, 13, 14, 15]
# Structure: table[noise][d][n] = MSE
tables = {}
for noise in noises:
    table = np.full((len(chosen_d), len(ns)), np.nan)
    for col, n in enumerate(ns):
        filename = f"results/n={n}_noise={noise}_exercise=a_results.txt"
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


# -----------------------------------
# Ridge closed form no resampling
# -----------------------------------

data = readResultsFile("results/n=100_noise=0.1_exercise=b_results.txt")

lamb = data["lambda"]
d = data["d"]
params = data["params"] # array of arrays, different length per row
mse = data["MSE"]
r2 = data["R2"]


# MSE with fixed lambda
for lamb in np.unique(data["lambda"]):
    mask = data["lambda"] == lamb
    plt.plot(data["d"][mask], data["MSE"][mask], label=f"λ={lamb}")

plt.xlabel("Polynomial degree")
plt.ylabel("MSE")
plt.legend()
plt.title("Ridge, n = 100, noise = 0.1")
plt.xlim(np.min(d), np.max(d))
plt.grid()
plt.show()

#R2 with fixed lambda
for lamb in np.unique(data["lambda"]):
    mask = data["lambda"] == lamb
    plt.plot(data["d"][mask], data["R2"][mask], label=f"λ={lamb}")

plt.xlabel("Polynomial degree")
plt.ylabel("R2 score")
plt.legend()
plt.title("Ridge, n = 100, noise = 0.1")
plt.xlim(np.min(d), np.max(d))
plt.grid()
plt.show()

#MSE and R2 varying with d
chosen_lambda = 0.1
mask = data["lambda"] == chosen_lambda

plt.plot(data["d"][mask], data["MSE"][mask], label = "MSE", color = "b")
plt.plot(data["d"][mask], data["R2"][mask], label = "R2 score", color = "r")
plt.xlabel("Polynomial degree")
plt.ylabel("Error")
plt.title(f"Ridge, n = 100, noise = 0.1")
plt.xlim(np.min(d), np.max(d))
plt.legend()
plt.grid()
plt.show()


#MSE and R2 varying with lambda
chosen_degree = 10
mask = data["d"] == chosen_degree

plt.plot(data["lambda"][mask], data["MSE"][mask], label = "MSE", color = "b")
plt.plot(data["lambda"][mask], data["R2"][mask], label = "R2 score", color = "r")
plt.xlabel("Punishing parameter")
plt.ylabel("Error")
plt.title(f"Ridge, n = 100, noise = 0.1")
plt.xlim(np.min(data["lambda"]), np.max(data["lambda"]))
plt.legend()
plt.grid()
plt.show()


# Params
chosen_lambda = 0.1
mask = data["lambda"] == chosen_lambda
norms = [np.linalg.norm(theta) for theta in data["params"][mask]]
plt.plot(data["d"][mask], norms, color = "k")
plt.xlabel("Polynomial degree")
plt.ylabel("Two norm of parameter set")
plt.title("Parameternorms, Ridge, n = 100, noise = 0.1")
plt.xlim(np.min(d), np.max(d))
plt.grid()
plt.show()


# -----------------------------------
# Comparison closed form no resampling
# -----------------------------------
