from utils import *
import numpy as np
from sklearn.model_selection import train_test_split

seed = 2026

num_points = np.array((50, 100, 250, 500))
noise = 0.1
degrees = np.arange(1, 16, 1)


def plainOLS(num_points, noise, degrees):
# OLS, calculate and save results for given noise, varying n and polynomial degree
	for n in num_points:
		x, y = MakeData(n, noise, seed + n) # Sample random x and Runge function y(x) with normally distributed noise
		naming = {"n": n, "noise": noise, "exercise": "a"} # For writing to file
		results = []
		for d in degrees:
			X = MakeDesignMatrix(x, d) # Excluding intercept
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69) # Split data
			X_train_scaled, X_test_scaled, y_train_centered = scaleData(X_train, X_test, y_train) # Scale data

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


plainOLS(num_points, noise, degrees)


def plainRidge(num_points, noise, degrees, punishers):
# Ridge, calculate and save results for given noise, varying n , punishing parameter lambda and polynomial degree
	for n in num_points:
		x, y = MakeData(n, noise, seed + n)
		naming = {"n": n, "noise": noise, "exercise": "b"}
		results = []
		for d in degrees:
			X = MakeDesignMatrix(x, d) # Excluding intercept
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69) # Split data
			X_train_scaled, X_test_scaled, y_train_centered = scaleData(X_train, X_test, y_train) # Scale data
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


n_punishers = 10
punishers = np.logspace(-6, 6, n_punishers)
plainRidge(num_points, noise, degrees, punishers)


def TrainTestErr(num_points, noise, degrees):
# OLS, compute and save test and training errors, using scikit learn
	for n in num_points:
		x, y = MakeData(n, noise, seed + n)
		x = x.reshape(-1, 1)
		naming = {"n": n, "noise": noise, "exercise": "c_Hastie"}
		results = []
		for i, d in enumerate(degrees): 
			x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=69) # Split data
			model = make_pipeline(PolynomialFeatures(degree = d, include_bias = False), StandardScaler(), LinearRegression(fit_intercept = True))
			
			y_pred = model.fit(x_train, y_train).predict(x_test).ravel()
			y_pred_train = model.predict(x_train).ravel()
			MSE_test = MSE(y_test, y_pred)
			MSE_train = MSE(y_train, y_pred_train)

			results.append({
			"d": d,
			"Test MSE": MSE_test,
			"Train MSE": MSE_train
			})
		writeToFile(naming, results)

degrees = np.arange(1, 20, 1)
num_points = np.array((50, 100, 500))
TrainTestErr(num_points, noise, degrees)


def BootStrapOLS(num_points, noise, degrees, bootstrap_its):
# OLS, compute and save results for bias variance tradeoff analysis, using scikit learn
	for n in num_points:
		x, y = MakeData(n, noise, seed + n)
		x = x.reshape(-1, 1)
		naming = {"n": n, "noise": noise, "exercise": "c_tradeoff"}
		results = []
		err, biasSquared, var = bootStrap(x, y, degrees, bootstrap_its)  # Bootstrap resampling for all polynomial degrees
		results.append({
			"degrees": degrees,
			"MSE": err,
			"bias": biasSquared,
			"variance": var
		})
		writeToFile(naming, results)

degrees = np.arange(1, 16, 1)
bootstrap_its = 100
BootStrapOLS(num_points, noise, degrees, bootstrap_its)




# Part d
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge

def CrossValidationOLS(num_points, noise, degrees):
# OLS, compute and save MSE using k-fold cross-validation
    for n in num_points:
        x, y = MakeData(n, noise, seed + n)
        x = x.reshape(-1, 1)

        naming = {"n": n, "noise": noise, "exercise": "d_OLS"}
        results = []

        # Test 5-fold and 10-fold cross-validation
        for k in [5, 10]:
            kfold = KFold(n_splits=k, shuffle=True, random_state=2026)

            for d in degrees:
                # OLS model
                model = make_pipeline(
                    PolynomialFeatures(degree=d, include_bias=False),
                    StandardScaler(),
                    LinearRegression(fit_intercept=True)
                )

                # Cross-validation
                scores = -cross_val_score(
                    model,
                    x,
                    y,
                    cv=kfold,
                    scoring="neg_mean_squared_error"
                )

                mse = np.mean(scores)

                results.append({
                    "k": k,
                    "d": d,
                    "MSE": mse
                })

        writeToFile(naming, results)


def CrossValidationRidge(num_points, noise, degrees, punishers):
# Ridge, compute and save MSE using k-fold cross-validation
    for n in num_points:
        x, y = MakeData(n, noise, seed + n)
        x = x.reshape(-1, 1)

        naming = {"n": n, "noise": noise, "exercise": "d_Ridge"}
        results = []

        # Test 5-fold and 10-fold cross-validation
        for k in [5, 10]:
            kfold = KFold(n_splits=k, shuffle=True, random_state=2026)

            for d in degrees:
                for lamb in punishers:
                    # Ridge model
                    model = make_pipeline(
                        PolynomialFeatures(degree=d, include_bias=False),
                        StandardScaler(),
                        Ridge(alpha=lamb)
                    )

                    # Cross-validation
                    scores = -cross_val_score(
                        model,
                        x,
                        y,
                        cv=kfold,
                        scoring="neg_mean_squared_error"
                    )

                    mse = np.mean(scores)

                    results.append({
                        "k": k,
                        "d": d,
                        "lambda": lamb,
                        "MSE": mse
                    })

        writeToFile(naming, results)


CrossValidationOLS(num_points, noise, degrees)
CrossValidationRidge(num_points, noise, degrees, punishers)