import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import jax
from sklearn import datasets
from utils import *
from neural_network import *


# SIMPLE TEST OF CLASS

network_input_size = 4
layer_output_sizes = [12, 10, 3]
activation_funcs = [ReLU, ReLU, softmax]

"""
Initialise object: 
a neural network with given (input size, outut sizes (number of nodes in each hidden layer),
list of activation functions corresponding to each hidden layer, the cost func, the norm if any,
the penalty if any, random seed for weights)
"""
testNN = NN(network_input_size, layer_output_sizes, activation_funcs, None)

# Print Neural network structure
testNN.getStructure()

# Make some data and perform feed forward
rng = np.random.default_rng(123)
X = rng.standard_normal((4, network_input_size))

print(testNN.feedForward(X))



# TEST ON IRIS SET (same as in the exercise)

iris = datasets.load_iris()
X = iris.data # (150, 4)
y = np.eye(3)[iris.target] # one-hot (150, 3)
N, p = X.shape
K = y.shape[1]

net = NN(p, [8, K], [sigmoid, softmax], crossEntropy) # netw setup

out = net.feedForward(X) # perform feed forward

# print network structure
net.getStructure()

# test that rows of output sum to 1 (because of softmax)
assert np.all(out > 0) and np.allclose(out.sum(axis=1), 1)

# compare our feedforward to direct calculations
(W1, b1), (W2, b2) = net.layers
a1 = 1 / (1 + np.exp(-(X @ W1 + b1)))
z2 = a1 @ W2 + b2
manual = np.exp(z2) / np.exp(z2).sum(axis=1, keepdims=True)
assert np.allclose(out, manual)

# untrained netw versus pure guess
print("Utrent accuracy:", accuracy(out, y), "(gjetting: 1/3)")
print("Utrent kostnad:", float(net.cost(X, net.layers, y)),
      "(helt flat softmax: N ln K =", N * np.log(K), ")")

accs = [accuracy(NN(p, [8, K], [sigmoid, softmax], crossEntropy, seed=s).feedForward(X), y)
        for s in range(200)]
print("Gjennomsnitt over 200 seeds:", np.mean(accs))

print("Predikerte klasser med seed=123:", np.bincount(np.argmax(out, axis=1), minlength=K))

# gradient structure matching the network structure
grads = net.getGradientJAX(X, y)
for (W, b), (dW, db) in zip(net.layers, grads):
    assert dW.shape == W.shape and db.shape == b.shape

# gradient not zero?
assert any(np.any(dW != 0) for dW, db in grads)


# test against industry standard
from sklearn.metrics import log_loss

rng = np.random.default_rng(0)
pRand = rng.dirichlet(np.ones(K), size=N)
labels = np.argmax(y, axis=1)
assert np.isclose(crossEntropy(pRand, y), N * log_loss(labels, pRand, labels=[0, 1, 2]))

out = net.feedForward(X)
assert np.isclose(crossEntropy(pRand, y), N * log_loss(labels, pRand, labels=[0, 1, 2]), rtol=1e-12)

from sklearn.metrics import mean_squared_error
rng = np.random.default_rng(1)
yTrue = rng.standard_normal((N, 3))
yPred = rng.standard_normal((N, 3))
assert np.isclose(MSE(yPred, yTrue), mean_squared_error(yTrue, yPred), rtol=1e-12)


