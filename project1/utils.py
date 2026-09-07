import numpy as np
from pathlib import Path


# Runge function
def RungeFunction(x):
	return 1 / (1 + 25 * x**2)


# Sample Runge function on random points with normally distributed noise
def MakeData(n, noise):
	rng = np.random.default_rng()
	x = rng.uniform(-1, 1, n) # n uniformly distributed values in the range [-1,1)
	y = RungeFunction(x) + noise * rng.normal(size = n)
	return x, y


# Set up design matrix - FASTER HOW?
def MakeDesignMatrix(x, d):
    p = d + 1
    X = np.zeros((len(x), p))
    X[:, 0] = 1
    for j in range(1, p):
        for i in range(len(x)):
            X[i, j] = x[i]**j
    return X


# Get mean squared error
def MSE(y_pred, y_test):
	return np.mean((y_test - y_pred)**2)


# Get R2 score
def R2Score(y_pred, y_test):
	return 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))


# Make some object into string
# Written by Claude w/ prompt "can you make a function fitting my function writeToFile that takes some value, number, array, list, and turns it into a string"
def formatValue(value):
    if isinstance(value, (list, np.ndarray)):
        arr = np.ravel(value)
        return "[" + ", ".join(f"{v:.6g}" for v in arr) + "]"
    elif isinstance(value, float):
        return f"{value:.6g}"
    else:
        return str(value)


# Write results to file
# Partly Claude w/ prompt "how to put resulting files in a folder which is in the same folder as the function file"
def writeToFile(naming, results):
	results_dir = Path("results")
	results_dir.mkdir(exist_ok = True)
	filename = "_".join(f"{key}={value}" for key, value in naming.items()) + "_results.txt"
	filepath = results_dir / filename
	keys = list(results[0].keys()) # For column titles

	with open(filepath, "w", encoding="utf-8") as file:
	    file.write("\t".join(keys) + "\n") # Column titles
	    for row in results: # Write results to file
	    	line = "\t".join(formatValue(row[key]) for key in keys)
	    	file.write(line + "\n")

