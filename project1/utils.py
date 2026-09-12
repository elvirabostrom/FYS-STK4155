import numpy as np
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample


# Runge function
def RungeFunction(x):
	return 1 / (1 + 25 * x**2)


# Sample Runge function on random points with normally distributed noise
def MakeData(n, noise, seed = None):
	rng = np.random.default_rng(seed)
	x = rng.uniform(-1, 1, n) # n uniformly distributed values in the range [-1,1)
	y = RungeFunction(x) + noise * rng.normal(size = n)
	return x, y


# Set up design matrix (excluding intercept)
def MakeDesignMatrix(x, d):
    X = np.zeros((len(x), d))
    for j in range(1, d + 1):
        for i in range(len(x)):
            X[i, j - 1] = x[i]**j
    return X


# Scale X, center y
def scaleData(X_train, X_test, y_train):
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train) # Scale
	X_test_scaled = scaler.transform(X_test) # Scale
	y_train_centered = y_train - y_train.mean() # Center
	return X_train_scaled, X_test_scaled, y_train_centered


# Solve closed form regression
def closedForm(X, y, lamb = 0.0):
    n, p = X.shape
    return np.linalg.pinv(X.T @ X + n * lamb * np.eye(p)) @ X.T @ y


# Get mean squared error
def MSE(y_pred, y_test):
	return np.mean((y_test - y_pred)**2)


# Get R2 score
def R2Score(y_pred, y_test):
	return 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))


# Complete bootstrap resampling for calculating MSE, bias, variance, for varying polynomial degree
# Only OLS
def bootStrap(x, y, degrees, iterations):
	err = np.zeros(degrees.shape)
	biasSquared = np.zeros(degrees.shape)
	var = np.zeros(degrees.shape)
	for j, d in enumerate(degrees): 
	    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=69) # Split data
	    model = make_pipeline(PolynomialFeatures(degree = d, include_bias = False), StandardScaler(), LinearRegression(fit_intercept = True))
	    y_pred = np.empty((y_test.shape[0], iterations)) # Initialize empty
	    for i in range(iterations): # Do bootstrap resampling
	        x_train_resampled, y_train_resampled = resample(x_train, y_train, random_state = i) # Resample
	        model.fit(x_train_resampled, y_train_resampled) # OLS
	        y_pred[:, i] = model.predict(x_test).ravel() # Make prediction

	    y_test = y_test.reshape(-1, 1)
	    err[j] = np.mean(np.mean((y_test - y_pred)**2, axis = 1, keepdims = True))
	    biasSquared[j] = np.mean((y_test - np.mean(y_pred, axis = 1, keepdims = True))**2)
	    var[j] = np.mean((np.mean(y_pred, axis = 1, keepdims = True) - y_pred)**2)

	return err, biasSquared, var


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

