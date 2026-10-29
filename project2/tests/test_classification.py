import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  
from sklearn import datasets
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate, train_test_split
from utils import *
from estimator import *
from neural_network import NN 

"""
TESTING CLASSIFICATION ON THE SCIKIT DIGITS SET

Figuring out how to do the setup for the larger MNIST dataset, what results to keep and so on
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


# ACCURACY FROM GUESSING - one feed forward pass
# in theory: 1 / 10
y_guess = DigitsNet.feedForward(X_train)

print("Accuracy from guessing, before training (expected 0.1): ", accuracy(y_guess, y_train))


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
y_train_lab = np.argmax(y_train, axis=1)
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
net = make_pipeline(StandardScaler(), Classifier(hidden=(50,), method="adam", gamma=0.03, epochs=100, batch_size=10, lmbd=0.0, norm="l2", seed=2026))
res = cross_validate(net, X_train, y_train_lab, cv=cv, scoring=["accuracy", "neg_log_loss"], return_train_score=True)
acc, ce = res["test_accuracy"], -res["test_neg_log_loss"]
se = lambda s: s.std(ddof=1) / np.sqrt(len(s))   # standardfeil til snittet over foldene
print(f"5-fold CV accuracy {acc.mean():.4f} +/- {se(acc):.4f} (folds: {acc})")
print(f"5-fold CV cross-entropy {ce.mean():.4f} +/- {se(ce):.4f}")
print("train accuracy per fold:", res["train_accuracy"])


# GRID SEARCH

cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
net = make_pipeline(StandardScaler(), Classifier(method="adam", epochs=100, batch_size=10, lmbd=0.0, norm="l2", seed=2026))

hiddens = [32, 64, 128]
gammas  = [0.01, 0.03, 0.1]
param_grid = {"classifier__hidden": [(h,) for h in hiddens],
              "classifier__gamma":  gammas}

search = GridSearchCV(net, param_grid, cv=cv, scoring="accuracy", return_train_score=True, n_jobs=-1)
search.fit(X_train, y_train_lab)
results = search.cv_results_

df = pd.DataFrame({
    "hidden": [p["classifier__hidden"][0] for p in results["params"]],
    "gamma":  [p["classifier__gamma"] for p in results["params"]],
    "cv":     results["mean_test_score"],
    "se":     results["std_test_score"] / np.sqrt(cv.get_n_splits() - 1),
    "train":  results["mean_train_score"],
})
CV_acc    = df.pivot(index="hidden", columns="gamma", values="cv")
SE_acc    = df.pivot(index="hidden", columns="gamma", values="se")
TRAIN_acc = df.pivot(index="hidden", columns="gamma", values="train")

# beste celle og one-standard-error-regelen (accuracy: høyere er bedre)
best   = df.loc[df["cv"].idxmax()]
within = df[df["cv"] >= best["cv"] - best["se"]]
chosen = within[within["hidden"] == within["hidden"].min()].sort_values("cv").iloc[-1]

print("CV accuracy:\n", CV_acc, "\n\nSE:\n", SE_acc, "\n\ntrain accuracy:\n", TRAIN_acc)
print("\ninnenfor 1 SE av beste:\n", within)
print(f"\nvalgt (1-SE, minste nett): hidden={int(chosen['hidden'])}, gamma={chosen['gamma']}")







