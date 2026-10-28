import numpy as np 
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from neural_network import NN
from utils import * 


ACT = {
    "sigmoid":   (sigmoid, sigmoid_der),
    "relu":      (ReLU, ReLU_der),
    "leakyrelu": (leakyReLU, leakyReLU_der),
}


class Regressor(BaseEstimator, RegressorMixin):
    def __init__(self, hidden=(50,), activation="sigmoid", gamma=0.03, lmbd=0.0, norm = "l2", method = "none", epochs=1000, batch_size=10, tol = None, seed=2026):
        self.hidden = hidden
        self.activation = activation
        self.gamma = gamma
        self.lmbd = lmbd
        self.norm = norm
        self.method = method
        self.epochs = epochs
        self.batch_size = batch_size
        self.tol = tol
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        sizes = [*self.hidden, 1]

        names = [self.activation] * len(self.hidden) if isinstance(self.activation, str) else list(self.activation)
        activations    = [ACT[a][0] for a in names] + [linear]
        activationDers = [ACT[a][1] for a in names] + [linear_der]

        network = NN(X.shape[1], sizes, activations, activationDers, MSE, MSE_der, norm = self.norm, lam = self.lmbd, seed = self.seed)

        self.history_, _, self.layers_  = train(network, X, y, classification = False, method = self.method, gamma = self.gamma, M = self.batch_size, epochs = self.epochs, rho = 0.99, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, tol = self.tol, seed = self.seed)
        self.network_ = network

        return self


    def predict(self, X):
        return np.asarray(self.network_.feedForward(np.asarray(X, dtype=float))).ravel()


class Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden=(50,), activation="sigmoid", gamma=0.03, lmbd=0.0, norm = "l2", method = "none", epochs=1000, batch_size=10, tol = None, seed=2026):
        self.hidden = hidden
        self.activation = activation
        self.gamma = gamma
        self.lmbd = lmbd
        self.norm = norm
        self.method = method
        self.epochs = epochs
        self.batch_size = batch_size
        self.tol = tol
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype = float)
        y = np.asarray(y, dtype = int).ravel()

        self.classes_ = np.unique(y)
        K = len(self.classes_) # antall klasser
        Y = np.eye(K)[y] # one-hot, shape (len(y), K)

        sizes = [*self.hidden, K]

        names = [self.activation] * len(self.hidden) if isinstance(self.activation, str) else list(self.activation)
        activations    = [ACT[a][0] for a in names] + [softmax]
        activationDers = [ACT[a][1] for a in names] + [None]

        network = NN(X.shape[1], sizes, activations, activationDers, crossEntropy, costDer = None, norm = self.norm, lam = self.lmbd, seed = self.seed)

        self.history_, _, self.layers_ = train(network, X, Y, classification = True, method = self.method, gamma = self.gamma, M = self.batch_size, epochs = self.epochs, rho = 0.99, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, tol = self.tol, seed = self.seed)
        self.network_ = network

        return self


    def predict_proba(self, X):
        # return class probabilities
        return np.asarray(self.network_.feedForward(np.asarray(X, dtype = float)))


    def predict(self, X):
        # return integer labels, np.argmax of softmax output
        return self.classes_[np.argmax(self.predict_proba(X), axis = 1)]



























        