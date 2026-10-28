import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from sklearn.metrics import accuracy_score
import numpy as np

"""
Activation functions, cost functions and their derivatives, gradient descent.
"""


"""
Activation functions
"""

def ReLU(z):
    return jnp.where(z > 0, z, 0.0)

def leakyReLU(z, alpha = 0.01):
    return jnp.where(z > 0, z, alpha * z)

def sigmoid(z):
    return 1 / (1 + jnp.exp(-z))

def softmax(z):
    e_z = jnp.exp(z - jnp.max(z, axis=1, keepdims=True))
    return e_z / jnp.sum(e_z, axis=1, keepdims=True)

def linear(z):
    return z


"""
Activation function derivatives
"""

def ReLU_der(z):
    return jnp.where(z > 0, 1.0, 0.0)

def leakyReLU_der(z):
    return (z > 0).astype(float)

def sigmoid_der(z):
    s = sigmoid(z)
    return s * (1.0 - s)

def linear_der(z):
    return np.ones_like(z)


"""
Cost functions
"""

def MSE(y_pred, y):
    return jnp.mean((y_pred - y)**2)

def crossEntropy(y_pred, y):
    return -jnp.mean(jnp.sum(y*jnp.log(y_pred + 1e-12), axis=1))


"""
Cost function derivatives
"""

def MSE_der(y_pred, y):
    n = y_pred.size
    return 2.0 / n * (y_pred - y)


"""
Accuracy
"""

def accuracy(y_pred, y_one_hot):
    one_hot_predictions = np.zeros(y_pred.shape)

    for i, pred in enumerate(y_pred):
        one_hot_predictions[i, np.argmax(pred)] = 1
    return accuracy_score(one_hot_predictions, y_one_hot) # scikit-function



"""
Gradient descent methods for training
"""

def train(network, in_train, target_train, classification=True, method="none", gamma=0.01, M=None, epochs=50, rho=0.99, beta1=0.9, beta2=0.999, eps=1e-8, tol=1e-6, seed=2026):
    # Train network
    # Written with the help of Claude Opus 5.5 (October 2026)
    # w/ prompt "can you adapt this function (currently only "none" method) to also include RMSProp and Adam based on this code: *pasted code from project 1*"

    """
    classification: decides the calculation of the network gradients (classification means softmax and cross entropy)
    method: "none", "rmsprop" or "adam"
    gamma:  learningrate
    M:      minibatch size for SGD, None gives plain gradient descent
    epochs: if M is None this corresponds to the number of iterations
    rho:    RMSprop, forgetting factor
    beta1, beta2: Adam, forgetting facot for 1. and 2. moment
    eps:    fixes divide by zero
    tol:    convergence tolerance (this depends on the scale of the cost function, which is different for MSE and cross entropy!!!)
    """
    rng = np.random.default_rng(seed)
    zeros = lambda: [(np.zeros_like(W), np.zeros_like(b)) for W, b in network.layers]
    v = zeros() # Adam: 1. moment
    s = zeros() # RMSprop and Adam: 2. moment
    t = 0 # Adam: number of updates
    batchSize = len(in_train) if M is None else M # plain or stochastic

    minimizeCost, trainCost, trainAcc = [], [], []

    for epoch in range(epochs):
        order = rng.permutation(len(in_train))
        for ss in range(0, len(in_train), batchSize):
            idx = order[ss:ss + batchSize]
            if classification:
                grads = network.backpropClass(in_train[idx], target_train[idx])
            else: 
                grads = network.backpropReg(in_train[idx], target_train[idx])

            t += 1

            if method == "none":
                steps = [(gamma * dW, gamma * db) for dW, db in grads]

            elif method == "rmsprop":
                s = [(rho * sW + (1 - rho) * dW**2, rho * sb + (1 - rho) * db**2) for (sW, sb), (dW, db) in zip(s, grads)]
                steps = [(gamma * dW / (np.sqrt(sW) + eps), gamma * db / (np.sqrt(sb) + eps)) for (dW, db), (sW, sb) in zip(grads, s)]

            elif method == "adam":
                v = [(beta1 * vW + (1 - beta1) * dW, beta1 * vb + (1 - beta1) * db) for (vW, vb), (dW, db) in zip(v, grads)]
                s = [(beta2 * sW + (1 - beta2) * dW**2, beta2 * sb + (1 - beta2) * db**2) for (sW, sb), (dW, db) in zip(s, grads)]
                c1, c2 = 1 - beta1**t, 1 - beta2**t   # bias correction
                steps = [(gamma * (vW / c1) / (np.sqrt(sW / c2) + eps), gamma * (vb / c1) / (np.sqrt(sb / c2) + eps)) for (vW, vb), (sW, sb) in zip(v, s)]

            network.updateWeights(steps)

        # save
        minimizeCost.append(float(network.cost(in_train, network.layers, target_train)))   # with penalty (this is what the convergence tolerance checks because it is the function being minimized by the gradients, must be kept cus we're varying lambda)
        trainCost.append(float(network.evaluate(in_train, target_train)))               # without penalty
        if classification:
            trainAcc.append(accuracy(network.feedForward(in_train), target_train))

        # convergence check
        if tol is not None and len(minimizeCost) > 1 and abs(minimizeCost[-2] - minimizeCost[-1]) < tol:
            break

    num_epochs = len(minimizeCost)
    history = {"minimizeCost": minimizeCost, "trainCost": trainCost, "trainAcc": trainAcc}

    return history, num_epochs, network.layers







