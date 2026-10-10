import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp               
from utils import *


class NN:
	def __init__(self, inputSize, layerOutputSizes, activationFuncs, costFunc, norm = None, lam = 0.0, seed = 123):
		# Initialize fully connected neural network
		self.inputSize = inputSize # num of features
		self.layerOutputSizes = layerOutputSizes # num of nodes in each layer
		self.activationFuncs = activationFuncs # activation functions used in each layer
		self.costFunc = costFunc # cost function (Regression og classification)
		self.norm = norm # norm (Ridge or Lasso)
		self.penalty = lam # penalty in Ridge or Lasso

		self.layers = [] # make hidden layers and output layer
		iSize = inputSize
		rng_weights = np.random.default_rng(seed)
		for i, outputSize in enumerate(layerOutputSizes):
			if activationFuncs[i] == ReLU or activationFuncs[i] == leakyReLU:
				var = 2 / iSize # He
			else:
				var = 2 / (iSize + outputSize) # Glorot/Xavier
			b = np.zeros(outputSize) + 0.01
			W = rng_weights.normal(0, np.sqrt(var), size=(iSize, outputSize))
			self.layers.append((W, b))
			iSize = outputSize

	def getStructure(self):
		# print the structure of the NN
		print("Structure: ", self.inputSize, [W.shape for W, b in self.layers])

	def feedForward(self, input, layers = None):
		# perform feed forward on the NN
		if layers is None: # this is just for the cost() to make sense with JAX
			layers = self.layers
		a = input
		for (W, b), activationFunc in zip(layers, self.activationFuncs):
			z = a @ W + b
			a = activationFunc(z)
		return a

	def cost(self, input, layers, targets):
		# compute cost after feed forward
		predict = self.feedForward(input, layers)
		return self.costFunc(predict, targets) # + self.penalty(layers) senere!

	def getGradientJAX(self, inputs, targets):
		# compute gradient from cost
		gradientFunc = jax.grad(self.cost, argnums = 1)
		layersGrad = gradientFunc(inputs, self.layers, targets)
		return layersGrad
























