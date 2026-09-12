import matplotlib.pyplot as plt
import pandas as pd 
import ast
import numpy as np


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

# MSE and R2
plt.plot(d, mse, label = "MSE", color = "b")
plt.plot(d, r2, label = "R2 score", color = "r")
plt.legend()
plt.xlabel("Polynomial degree")
plt.ylabel("Error")
plt.title("OLS, n = 100, noise = 0.1")
plt.grid()
plt.xlim(np.min(d), np.max(d))
plt.show()

# Params
# showing how params grow large with OLS as the model overfits at high variance
norms = [np.linalg.norm(theta) for theta in params]
plt.plot(d, norms, color = "k")
plt.xlabel("Polynomial degree")
plt.ylabel("Two norm of parameter set")
plt.title("Parameternorms, OLS, n = 100, noise = 0.1")
plt.xlim(np.min(d), np.max(d))
plt.grid()
plt.show()

# heatmap
max_len = max(len(theta) for theta in params)
matrix = np.full((len(d), max_len), np.nan)
for i, theta in enumerate(params):
    matrix[i, :len(theta)] = theta

plt.imshow(matrix, aspect="auto", cmap="coolwarm")
plt.xlabel("Coefficient index")
plt.ylabel("Polynomial degree")
plt.colorbar(label="Value")
plt.show()


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
