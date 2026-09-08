from utils import *
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

num_points = np.array((50, 100, 250, 500))
noise = 0.1
degrees = np.arange(1, 16, 1) # MÅ 0 VÆRE MED? ISF MÅ INTERCEPT TILBAKE


# OLS, calculate and save results for given noise, varying n and polynomial degree
for n in num_points:
	x, y = MakeData(n, noise) # Sample random x and Runge function y(x) with normally distributed noise
	naming = {"n": n, "noise": noise, "exercise": "a"} # For writing to file
	results = []
	for d in degrees:
		X = MakeDesignMatrix(x, d) # Excluding intercept
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69) # Split data

		scaler = StandardScaler()
		X_train_scaled = scaler.fit_transform(X_train) # Scale
		X_test_scaled = scaler.transform(X_test) # Scale
		y_train_centered = y_train - y_train.mean() # Center

		params = closedForm(X_train_scaled, y_train_centered) # Solve OLS
		y_pred = X_test_scaled @ params + y_train.mean() # Make prediction (scale back)
		mse = MSE(y_pred, y_test) # Calculate mean squared error
		r2 = R2Score(y_pred, y_test) # Calculate r2 score

		results.append({
            "d": d,
            "params": params,
            "MSE": mse,
            "R2": r2
        })

	writeToFile(naming, results)


# Ridge, calculate and save results for given noise, varying n , punishing parameter lambda and polynomial degree
n_punishers = 10
punishers = np.logspace(-1, 10, n_punishers)

for n in num_points:
	x, y = MakeData(n, noise)
	naming = {"n": n, "noise": noise, "exercise": "b"}
	results = []
	for d in degrees:
		X = MakeDesignMatrix(x, d) # Excluding intercept
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69) # Split data

		scaler = StandardScaler()
		X_train_scaled = scaler.fit_transform(X_train) # Scale
		X_test_scaled = scaler.transform(X_test) # Scale
		y_train_centered = y_train - y_train.mean() # Center
		for lamb in punishers: 
		    params = closedForm(X_train_scaled, y_train_centered, lamb) # Solve Ridge
		    y_pred = X_test_scaled @ params + y_train.mean() # Make prediction (scale back)
		    mse = MSE(y_pred, y_test)
		    r2 = R2Score(y_pred, y_test) 


		    results.append({
                "d": d,
                "lambda": lamb,
                "params": params,
                "MSE": mse,
                "R2": r2
            })

	writeToFile(naming, results)



