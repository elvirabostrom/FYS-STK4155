from utils import *
from OLS import *
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#num_points = np.array((50, 100, 250, 500))
num_points = np.array((50, 100))
noise = 0.1
degrees = np.arange(0, 16, 1)

# skalerer ikke OLS pga intervallet x ligger i gjør at høye potenser faktisk blir mindre eller holder seg
# i samme størrelsesorden
# OLS vil gi samme prediksjon med og uten skalering
# SKALERER ALLIKEVEL fordi Runge funksjonen kan kreve høye polynomgrader og da blir kolonnene i designmatrise
# sterkt korrelerte. Kan få en ill-conditioned matrise etter hvert. 
# skalering krymper tallverdiene til samme statistiske skala, som gjør matriseberegninger stabile for maskinen
# DERMED kan vi være trygge på at eventuelle forskjeller i MSE kommer av algoritmene selv


# exercise a
for n in num_points:
	x, y = MakeData(n, noise) # Sample random x and Runge function y(x) with normally distributed noise
	naming = {"n": n, "noise": noise, "exercise": "a"} # For writing to file
	results = []
	for d in degrees:
		X = MakeDesignMatrix(x, d)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69) # Split data
		scaler = StandardScaler()
		X_train_scaled = scaler.fit_transform(X_train) # Scale data
		X_test_scaled = scaler.transform(X_test) # Scale data

		params = OLS(X_train_scaled, y_train) # Solve OLS - LAGRE DISSE
		y_pred = X_test_scaled @ params # Make prediction
		mse = MSE(y_pred, y_test) # Calculate mean squared error
		r2 = R2Score(y_pred, y_test) # Calculate r2 score

		results.append({
            "d": d,
            "params": params,
            "MSE": mse,
            "R2": r2
        })

	writeToFile(naming, results)




# b #
# løs Ridge med egen kode
# parametre lambda, polynomgrad, n
# beregne MSE og R2 score
# write to file
