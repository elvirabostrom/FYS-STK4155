import numpy as np

# Ordinary Least Squares
def OLS(X, y):
	return np.linalg.pinv(X.T @ X) @ X.T @ y