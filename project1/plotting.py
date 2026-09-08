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
# OLS no resampling
# -----------------------------------

data = readResultsFile("results/n=100_noise=0.1_exercise=a_results.txt")

d = data["d"]
params = data["params"] # array of arrays, different length per row
mse = data["MSE"]
r2 = data["R2"]

# MSE and R2
plt.plot(d, mse, label = "MSE", color = "lightgreen")
plt.plot(d, r2, label = "R2 score", color = "lightsalmon")
plt.legend()
plt.xlabel("Polynomial degree")
plt.ylabel("Error")
plt.title("OLS, n = 100, noise = 0.1")
plt.grid()
plt.xlim(np.min(d), np.max(d))
plt.show()







