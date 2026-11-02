import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  
from sklearn import datasets
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate, train_test_split, validation_curve
from utils import *
from estimator import *
from neural_network import NN 

"""
TESTING CLASSIFICATION ON THE SCIKIT DIGITS SET

Figuring out how to do the setup for the larger MNIST dataset, what results to keep and so on.
"""

# THE INPUT
# 8 X 8 = 64 pixels in each image, verdien av hver pixel er mellom 0 og 16
# 1797 stk bilder
# hvert bilde er flattened til en rad av 64 features
digits = datasets.load_digits()
inputs = digits.data / 16.0 # scaled to [0, 1], (1797, 64), SAMPLES X FEATURES


# TARGETS ER ONE-HOT ROWS
# 10 digits (0 til 9), 10 klasser
# 1 i kolonnen til riktig klasse, 0 ellers
# det er trygt å skalere før cross validation, for 16 er allerede noe kjent ved datasettet, 
# og uansett vil standardscaler() overse dette senere
labels = digits.target
targets = np.zeros((len(labels), 10))
targets[np.arange(len(labels)), labels] = 1.0 # one-hot rows, (1797, 10), SAMPLES X CLASSES


# SPLITTING
X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=2026, stratify=labels)

print(f"Training samples {len(y_train)} images, Test samples {len(y_test)} images")


# INITIALIZE NETWORK
inputSize, layerOutputSizes = X_train.shape[1], [10, y_train.shape[1]] # inkludert last layer!
activationFuncs, activationDers = [sigmoid, softmax], [sigmoid_der, None]

DigitsNet = NN(inputSize, layerOutputSizes, activationFuncs, activationDers, costFunc = crossEntropy, costDer = None, norm = None, lam = 0.0, seed = 2026)


"""
SIMPLE TRAINING
"""

# BASELINE,guess the most frequent class in the training set
# only training labels are used to pick the class
y_train_lab = np.argmax(y_train, axis=1)
y_test_lab  = np.argmax(y_test, axis=1)
majority = np.bincount(y_train_lab).argmax()
baseline_acc = np.mean(y_test_lab == majority)

print(f"Baseline guess {majority}: test accuracy {baseline_acc:.4f}")


# A TEST AGAINST JAX GRADIENTS (again)
my_grads = DigitsNet.backpropClass(X_train, y_train)
jax_grads = DigitsNet.getGradientJAX(X_train, y_train)

print("Softmax shortcut agrees with jax.grad:", [(np.allclose(dW, dW_j), np.allclose(db, db_j)) for (dW, db), (dW_j, db_j) in zip(my_grads, jax_grads)])


# TRAIN WITH SGD AND ADAM - no cross validation
history, num_epochs, trained_layers = train(DigitsNet, X_train, y_train, classification=True, method="adam", gamma=0.01, M=30, epochs=50, rho=0.99, beta1=0.9, beta2=0.999, eps=1e-8, tol=1e-12, seed=2026)

for i in (1, 2, 5, 10, 20, 50):
	if i <= num_epochs:
		print(f"epoch {i}: "
			f"train accuracy {history["trainAcc"][i - 1]:.4f} "
			f"train cost {history["trainCost"][i - 1]:.4f}")


"""
WITH WRAPPER
"""

# WRAPPER CHECK ON TRAINING DATA
est = Classifier().fit(X_train, y_train_lab)
ce_wrapper = est.history_["trainCost"][-1]
netw = NN(X_train.shape[1], [50, 10], [sigmoid, softmax], [sigmoid_der, None], crossEntropy, None, norm="l2", lam=0.0, seed=2026)
history, _, _ = train(netw, X_train, y_train, classification=True, method="none", gamma=0.03, M=10, epochs=1000, tol=None, seed=2026)
ce_own = history["trainCost"][-1]
print("WRAPPER CHECK")
print(f"wrapper: {ce_wrapper:.17g}")
print(f"egen:    {ce_own:.17g}")
print(est.score(X_train, y_train_lab))


# CROSS VALIDATE
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
net = make_pipeline(Classifier(hidden=(50,), method="adam", gamma=0.03, epochs=100, batch_size=10, lmbd=0.0, norm="l2", seed=2026))
res = cross_validate(net, X_train, y_train_lab, cv=cv, scoring=["accuracy", "neg_log_loss"], return_train_score=True)
acc, ce = res["test_accuracy"], -res["test_neg_log_loss"]
se = lambda s: s.std(ddof=1) / np.sqrt(len(s))   # standardfeil til snittet over foldene
print(f"5-fold CV accuracy {acc.mean():.4f} +/- {se(acc):.4f} (folds: {acc})")
print(f"5-fold CV cross-entropy {ce.mean():.4f} +/- {se(ce):.4f}")
print("train accuracy per fold:", res["train_accuracy"])


# GRID SEARCH CROSS-VALIDATION
import time


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

    df.to_csv(f"results/{filename}_TEST.csv", index = False)
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
    df.to_csv(f"results/{filename}_TEST.csv", index=False)
    return df


# fold splitting
cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)

# search grid
penalties = [1e-6, 1e-5, 1e-4, 1e-3, 0.01]
gammas  = [1e-4, 1e-3, 3e-3, 0.01, 0.03]
param_grid = {"classifier__lmbd": penalties,
              "classifier__gamma":  gammas} # 25 combinations, 5 folds, giving 125 fits per model

# one layer 50 nodes
print("NETWORK")
net = make_pipeline(Classifier(hidden=(50,), method="adam", epochs=100, batch_size=30))
t0 = time.perf_counter()
df = grid_cv(X_train, y_train_lab, net, cv, param_grid, "gridsearch_digits")
print(f"grid search all fits, one layer: {time.perf_counter() - t0:.1f} s")

CV_acc    = df.pivot(index="lmbd", columns="gamma", values="cv")
SE_acc    = df.pivot(index="lmbd", columns="gamma", values="se")
TRAIN_acc = df.pivot(index="lmbd", columns="gamma", values="train")
# best cell and one-standard-error rule (higher accuracy better)
best   = df.loc[df["cv"].idxmax()]
within = df[df["cv"] >= best["cv"] - best["se"]]
chosen = within[within["lmbd"] == within["lmbd"].max()].sort_values("cv").iloc[-1]
print("ONE HIDDEN LAYER, 50 NODES")
print("CV accuracy:\n", CV_acc, "\n\nSE:\n", SE_acc, "\n\ntrain accuracy:\n", TRAIN_acc)
print("\ninnenfor 1 SE av beste:\n", within)
print(f"\nvalgt (1-SE): lambda={chosen['lmbd']}, gamma={chosen['gamma']}")


# logistic regression
print("LOGISTIC")
net_log = make_pipeline(Classifier(hidden=(), method="adam", epochs=100, batch_size=30))
t0 = time.perf_counter()
df_log = grid_cv(X_train, y_train_lab, net_log, cv, param_grid, "gridsearch_digits_logistic")
print(f"grid search all fits, logistic regression: {time.perf_counter() - t0:.1f} s")

CV_acc_log    = df_log.pivot(index="lmbd", columns="gamma", values="cv")
SE_acc_log    = df_log.pivot(index="lmbd", columns="gamma", values="se")
TRAIN_acc_log = df_log.pivot(index="lmbd", columns="gamma", values="train")
# best cell and one-standard-error rule (higher accuracy better)
best_log   = df_log.loc[df_log["cv"].idxmax()]
within_log = df_log[df_log["cv"] >= best_log["cv"] - best_log["se"]]
chosen_log = within_log[within_log["lmbd"] == within_log["lmbd"].max()].sort_values("cv").iloc[-1]
print("LOGISTIC REGRESSION")
print("CV accuracy:\n", CV_acc_log, "\n\nSE:\n", SE_acc_log, "\n\ntrain accuracy:\n", TRAIN_acc_log)
print("\ninnenfor 1 SE av beste:\n", within_log)
print(f"\nvalgt (1-SE): lambda={chosen_log['lmbd']}, gamma={chosen_log['gamma']}")


# validation curve over width
lmbd, gamma = one_se(df)
net = make_pipeline(Classifier(lmbd=lmbd, gamma=gamma, method="adam", epochs=100, batch_size=30))
widths = [(5,), (10,), (25,), (50,), (100,)]
df_width = val_curve(X_train, y_train_lab, net, cv, "classifier__hidden", widths, "valcurve_digits_width")

