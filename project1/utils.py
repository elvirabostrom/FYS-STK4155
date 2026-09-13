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


# Standardize X, center y
def scaleData(X_train, X_test, y_train):
	x_mean = np.mean(X_train, axis = 0)
	x_std = np.std(X_train, axis = 0)

	X_train_scaled = ( X_train - x_mean ) / x_std
	X_test_scaled = ( X_test - x_mean ) / x_std
	y_train_centered = y_train - y_train.mean()
	
	return X_train_scaled, X_test_scaled, y_train_centered


# Solve closed form regression
def ClosedForm(X, y, lamb = 0.0):
    n, p = X.shape
    return np.linalg.pinv(X.T @ X + n * lamb * np.eye(p)) @ X.T @ y


# Get mean squared error
def MSE(y_pred, y_test):
	return np.mean((y_test - y_pred)**2)


# Get R2 score
def R2Score(y_pred, y_test):
	return 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2))


<<<<<<< HEAD
# Analytic Gradients
def GradOLSAnalytic(theta, X, y):
    n = len(y)
    return (2.0 / n) * X.T @ (X @ theta - y)
def GradRidgeAnalytic(theta, X, y, lamb):
    return GradOLSAnalytic(theta, X, y) + 2.0 * lamb * theta


# Cost functions
def CostOLS(theta, X, y):
    return jnp.mean((y - X @ theta) ** 2)
def CostRidge(theta, X, y, lamb):
    return jnp.mean((y - X @ theta) ** 2) + lamb * jnp.sum(theta**2)


# Gradient Descent
def GradientDescent(
    X,
    y,
    grad_func,
    theta_init,
    eta,
    max_iter=10000,
    tol=1e-8,
    lamb=None,
):
    theta = theta_init.copy()
    for i in range(max_iter):
        if lamb is not None:
            g = grad_func(theta, X, y, lamb)
        else:
            g = grad_func(theta, X, y)

        theta_next = theta - eta * g

        # Convergence check: change in parameter vector
        if np.linalg.norm(theta_next - theta) < tol:
            return theta_next, i + 1

        theta = theta_next

    return theta, max_iter
=======
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
>>>>>>> b45cc883ffa46bfa03a2e80eb7d19a096d66b4e8


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


def writeToFile_by_folder(naming, results):
    # Extract exercise name for the directory structure
    ex_folder = f"exercise={naming.get('exercise', 'e')}"
    results_dir = Path("results") / ex_folder
    results_dir.mkdir(parents=True, exist_ok=True)

    # Omit 'exercise' key from the filename since it's now in the folder path
    file_naming = {k: v for k, v in naming.items() if k != "exercise"}
    filename = (
        "_".join(f"{key}={value}" for key, value in file_naming.items())
        + "_results.txt"
    )
    filepath = results_dir / filename

    keys = list(results[0].keys())

    with open(filepath, "w", encoding="utf-8") as file:
        file.write("\t".join(keys) + "\n")
        for row in results:
            line = "\t".join(formatValue(row[key]) for key in keys)
            file.write(line + "\n")