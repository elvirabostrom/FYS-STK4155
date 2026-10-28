import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import jax
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import *
from neural_network import *

# TEST ON IRIS SET
print("IRIS TEST")

"""
How to initialise an NN object: 
a neural network with attributes
(input size, output sizes (number of nodes in each hidden layer),
list of activation functions corresponding to each hidden layer, corresponding derivatives,
the cost func, corresponding derivative,
the norm if any (set to None as default), the penalty if any (set to 0.0 as default),
random seed for initializing the weights(set to 2026 as default))
"""

# load data, one-hot accuracy
iris = datasets.load_iris()
inputs = iris.data
targets = np.zeros((len(iris.data), 3)) # three species
for i, t in enumerate(iris.target):
    targets[i, t] = 1 # one-hot 

# structure
network_input_size = inputs.shape[1]
activation_funcs = [ReLU, sigmoid, softmax]
activation_ders = [ReLU_der, sigmoid_der, None]
layer_output_sizes = [8, 4, 3]
cost_func = crossEntropy

# make network object
irisNetwork = NN(network_input_size, layer_output_sizes, activation_funcs, activation_ders, cost_func, costDer = None, norm = "l2", lam = 0.01)

# print shape of network
irisNetwork.getStructure()

# guess
predictions = irisNetwork.feedForward(inputs)
print("Guess before training: ", accuracy(predictions, targets))

# get gradients and compare
layer_grads = irisNetwork.backpropClass(inputs, targets)
layer_grads_jax = irisNetwork.getGradientJAX(inputs, targets)

print("Own gradients:")
for (dW, db) in layer_grads:
    print(dW.shape, db.shape)

print("JAX gradients:")
for (dW, db) in layer_grads_jax:
    print(dW.shape, db.shape)

print("Same values?")
print([ (np.allclose(dW, dW_j), np.allclose(db, db_j))
        for (dW, db), (dW_j, db_j) in zip(layer_grads, layer_grads_jax) ])

# HELL YEAH I COOKED


# SGD test with momentum

# split
in_train, in_test, target_train, target_test = train_test_split(inputs, targets, test_size=0.2, random_state=123)

# scale
scaler = StandardScaler().fit(in_train)
in_train = scaler.transform(in_train)
in_test  = scaler.transform(in_test)

# re-initialize
irisNetwork = NN(network_input_size, layer_output_sizes, activation_funcs, activation_ders, cost_func, costDer = None, norm = "l2", lam = 0.01)

# pre-print
print(f"before training: train accuracy {accuracy(irisNetwork.feedForward(in_train), target_train):.3f}")

# set params
gamma = 0.1 # learning rate, how big is the step along the gradient
beta = 0.9 # momentum, how much to keep from last step
M = 32 # minibatch-size, number of samples per gradient calculation
epochs = 50 # how many times to go through the whole training set

# train
history, num_epochs, layers = train(irisNetwork, in_train, target_train, classification=True, method="none", gamma=0.1, M=30, epochs=50, rho=0.99, beta1=0.9, beta2=0.999, eps=1e-8, tol=1e-6, seed=2026)
minimizeCost, trainCost, trainAcc = history["minimizeCost"], history["trainCost"], history["trainAcc"]

for epoch in [9, 19, 29, 39, 49]:
    print("epoch: ", epoch + 1, " cost: ", minimizeCost[epoch], "train acc: ", trainAcc[epoch], "train cost: ", trainCost[epoch])


# double cook



# TEST CROSS VAL WRAPPERS


