import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import jax
from sklearn import datasets
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
activation_funcs = [ReLU, sigmoid, sigmoid, softmax]
activation_ders = [ReLU_der, sigmoid_der, sigmoid_der, None]
layer_output_sizes = [8, 10, 4, 3]
cost_func = crossEntropy

# make network object
irisNetwork = NN(network_input_size, layer_output_sizes, activation_funcs, activation_ders, cost_func, costDer = None)

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

