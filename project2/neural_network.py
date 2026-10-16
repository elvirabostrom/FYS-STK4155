import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp               
from utils import *

"""
Class NN

Feed forward neural network (FFNN) object, a fully contracted network. 
"""


class NN:
	def __init__(self, inputSize, layerOutputSizes, activationFuncs, activationDers, costFunc, costDer, norm = None, lam = 0.0, seed = 2026):
		# Initialize NN object
		self.inputSize = inputSize # num of features
		self.layerOutputSizes = layerOutputSizes # num of nodes in each layer
		self.activationFuncs = activationFuncs # activation functions used in each layer
		self.activationDers = activationDers # derivatives of activation functions
		self.costFunc = costFunc # cost function
		self.costDer = costDer # derivative of cost function
		self.norm = norm # norm ("l1" or "l2" or None)
		self.lam = lam # penalty in Ridge or Lasso // changed self.penalty to self.lam as we already have a function called penalty

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


	def feedForwardSaver(self, input):
		# save inputs (except first layer), z and a from every layer
		layerInputs = []
		zs = []
		a = input
		for (W, b), activationFunc in zip(self.layers, self.activationFuncs):
			layerInputs.append(a)
			z = a @ W + b
			a = activationFunc(z)
			zs.append(z)
		return layerInputs, zs, a


	def penalty(self, layers):
    	# Compute L1 or L2 regularization penalty over weights (biases are excluded)
    	if self.norm is None or self.lam == 0.0:
        	return 0.0
    
    	reg = 0.0
    
    	for W, b in layers:
        	if self.norm == "l2":
            	reg += self.lam * jnp.sum(W ** 2)
        	elif self.norm == "l1":
            	reg += self.lam * jnp.sum(jnp.abs(W))
            
    	return reg


	def cost(self, input, layers, targets):
		# compute cost after feed forward
		predict = self.feedForward(input, layers)
		return self.costFunc(predict, targets) + self.penalty(layers)


	def getGradientJAX(self, inputs, targets):
		# compute gradient from cost
		gradientFunc = jax.grad(self.cost, argnums = 1)
		layersGrad = gradientFunc(inputs, self.layers, targets)
		return layersGrad


	def backpropReg(self, input, target):
		# back propagation suited for regression problem
		layerInputs, zs, predict = self.feedForwardSaver(input)
		layerGrads = [() for layer in self.layers]
		for i in reversed(range(len(self.layers))):
			layerInput, z, activationDer = layerInputs[i], zs[i], self.activationDers[i]
			
			W, b = self.layers[i]

			if i == len(self.layers) - 1:
				dC_da = self.costDer(predict, target) # last layer
			else:
				#(W, b) = self.layers[i + 1] # other layers
				#dC_da = dC_dz @ W.T
				(W_next, b_next) = self.layers[i + 1] 
        		dC_da = dC_dz @ W_next.T
			
			dC_dz = dC_da * activationDer(z)
			dC_db = np.sum(dC_dz, axis = 0)
			dC_dW = layerInput.T @ dC_dz
			if self.norm is not None and self.lam > 0.0:
    			if self.norm == "l2":
        			dC_dW += 2.0 * self.lam * W
    			elif self.norm == "l1":
        			dC_dW += self.lam * np.sign(W)
			layerGrads[i] = (dC_dW, dC_db)
		return layerGrads


	def backpropClass(self, input, target):
		# back propagation suited for classification - with softmax and cross entropy
		layerInputs, zs, predict = self.feedForwardSaver(input)
		layerGrads = [() for layer in self.layers]
		for i in reversed(range(len(self.layers))):
			layerInput, z, activationDer = layerInputs[i], zs[i], self.activationDers[i]
			
			W, b = self.layers[i]

			if i == len(self.layers) - 1:
				dC_dz = (predict - target) / input.shape[0] # last layer
			else:
				#(W, b) = self.layers[i + 1] # other layers
				#dC_da = dC_dz @ W.T
				(W_next, b_next) = self.layers[i + 1] 
        		dC_da = dC_dz @ W_next.T
				dC_dz = dC_da * activationDer(z)
			
			dC_db = np.sum(dC_dz, axis = 0)
			dC_dW = layerInput.T @ dC_dz
			if self.norm is not None and self.lam > 0.0:
    			if self.norm == "l2":
        			dC_dW += 2.0 * self.lam * W
    			elif self.norm == "l1":
        			dC_dW += self.lam * np.sign(W)
			layerGrads[i] = (dC_dW, dC_db)
		return layerGrads


	def updateWeights(self, layerSteps):
		# use computed step update (from GD method) to update W and b in the network
		self.layers = [(W - stepW, b - stepb) for (W, b), (stepW, stepb) in zip(self.layers, layerSteps)]

















