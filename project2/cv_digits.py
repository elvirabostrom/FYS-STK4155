import numpy as np  
from sklearn import datasets
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, validation_curve
from utils import *
from estimator import *
from neural_network import NN 

"""
Cross-validating a FFNN for classifying digits

Using the training set of the MNIST dataset
"""


# load data
with np.load("mnist.npz") as data: 
	x_train, y_train = data["x_train"], data["y_train"]
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)

# scale image pixels to [0, 1]
# each picture is flattened to a row of 784 features
X_train = x_train.reshape(len(x_train), -1) / 255.0 # (60000, 784)


def grid_cv(X, y, net, cv, param_grid, filename):
    search = GridSearchCV(net, param_grid, cv=cv, scoring="accuracy", return_train_score=True, n_jobs=4, refit=False, verbose=3)
    search.fit(X, y)
    results = search.cv_results_

    df = pd.DataFrame({
        "lmbd": [p["classifier__lmbd"] for p in results["params"]],
        "gamma":  [p["classifier__gamma"] for p in results["params"]],
        "cv":     results["mean_test_score"],
        "se":     results["std_test_score"] / np.sqrt(cv.get_n_splits() - 1),
        "train":  results["mean_train_score"],
        "fit_time": results["mean_fit_time"]
        })

    df.to_csv(f"results/{filename}.csv", index = False)
    return df


def one_se(df):
    best   = df.loc[df["cv"].idxmax()]
    within = df[df["cv"] >= best["cv"] - best["se"]]
    chosen = within[within["lmbd"] == within["lmbd"].max()].sort_values("cv").iloc[-1]
    return chosen["lmbd"], chosen["gamma"]


def val_curve(X, y, net, cv, param_name, param_range, filename):
    train, val = validation_curve(net, X, y, param_name=param_name, param_range=param_range, cv=cv, scoring="accuracy", n_jobs=4, verbose=3) # train, val:shape (len(param_range), k)
    df = pd.DataFrame({
        "width": [p[0] if len(p) > 0 else 0 for p in param_range],
        "cv":    val.mean(axis=1),
        "se":    val.std(axis=1) / np.sqrt(cv.get_n_splits() - 1),
        "train": train.mean(axis=1),
    })
    df.to_csv(f"results/{filename}.csv", index=False)
    return df


# fold splitting
cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)

# search grid
penalties = [1e-6, 1e-5, 1e-4, 1e-3, 0.01]
gammas  = [1e-4, 1e-3, 3e-3, 0.01, 0.03]
param_grid = {"classifier__lmbd": penalties,
              "classifier__gamma":  gammas} # 25 combinations, 5 folds, 125 fits per model

# one layer 50 nodes
print("NETWORK")
net = make_pipeline(Classifier(hidden=(50,), method="adam", epochs=10, batch_size=130))
df = grid_cv(X_train, y_train, net, cv, param_grid, "gridsearch_MNIST")


# logistic regression
print("LOGISTIC")
net_log = make_pipeline(Classifier(hidden=(), method="adam", epochs=10, batch_size=130))
df_log = grid_cv(X_train, y_train, net_log, cv, param_grid, "gridsearch_MNIST_logistic")


# validation curve over width
print("VALIDATION CURVE")
lmbd, gamma = one_se(df)
net = make_pipeline(Classifier(lmbd=lmbd, gamma=gamma, method="adam", epochs=10, batch_size=130))
widths = [(5,), (10,), (25,), (50,), (100,)]
df_width = val_curve(X_train, y_train, net, cv, "classifier__hidden", widths, "valcurve_MNIST_width")

