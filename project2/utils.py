import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from sklearn.metrics import accuracy_score
import numpy as np

"""
Activation functions, cost functions and their derivatives, gradient descent
and other functions used over multiple parts of our analysis.
"""


"""
Write to file
"""

# coming - copy from last project? 


"""
Activation functions
"""

def ReLU(z):
	# copied from course notes
    return jnp.where(z > 0, z, 0.0)

def leakyReLU(z, alpha = 0.01):
    return jnp.where(z > 0, z, alpha * z)

def sigmoid(z):
	# copied from course notes
    return 1 / (1 + jnp.exp(-z))

def softmax(z):
# copied from course notes
    e_z = jnp.exp(z - jnp.max(z, axis=1, keepdims=True))
    return e_z / jnp.sum(e_z, axis=1, keepdims=True)

def linear(z):
    return z


"""
Activation function derivatives
"""

def ReLU_der(z):
    # copied from course notes
    return jnp.where(z > 0, 1.0, 0.0)

def leakyReLU_der(z):
    # NEEDS TO BE ADDED
    pass

def sigmoid_der(z):
    s = sigmoid(z)
    return s * (1.0 - s)

def linear_der(z):
    return 1


"""
Cost functions
"""

def MSE(y_pred, y):
    return jnp.mean((y_pred - y)**2)

def crossEntropy(y_pred, y):
    # copied from course notes
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
    # copied from course notes
    one_hot_predictions = np.zeros(y_pred.shape)

    for i, pred in enumerate(y_pred):
        one_hot_predictions[i, np.argmax(pred)] = 1
    return accuracy_score(one_hot_predictions, y_one_hot) # scikit-function



"""
Gradient descent methods
"""

def SGD(network, in_train, target_train, in_test, target_test, method="plain", gamma=0.01, M=30, epochs=50, beta=0.9, rho=0.99, beta1=0.9, beta2=0.999, eps=1e-8, seed=2026):
    # Written with the help of Claude Opus 5.5 (October 2026)
    # w/ prompt "can you adapt this function (currently only momentum) to also include plain GD, RMSProp and Adam based on this code: *pasted code from project 1*"

    """
    method: "plain", "momentum", "rmsprop" or "adam"
    gamma:  learningrate
    M:      minibatch size
    beta:   momentum
    rho:    RMSprop, forgetting factor
    beta1, beta2: Adam, forgetting facot for 1. and 2. moment
    eps:    fixes divide by zero
    """
    rng = np.random.default_rng(seed)
    zeros = lambda: [(np.zeros_like(W), np.zeros_like(b)) for W, b in network.layers]
    v = zeros()   # momentum: velocity, Adam: 1. moment
    s = zeros()   # RMSprop and Adam: 2. moment
    t = 0         # Adam: number of updates

    costs, trainAcc, testAcc = [], [], []

    for epoch in range(epochs):
        order = rng.permutation(len(in_train))
        for ss in range(0, len(in_train), M):
            idx = order[ss:ss + M]
            grads = network.backpropClass(in_train[idx], target_train[idx])
            t += 1

            if method == "plain":

                steps = [(gamma * dW, gamma * db) for dW, db in grads]

            elif method == "momentum":
                v = [(beta * vW + gamma * dW, beta * vb + gamma * db) for (vW, vb), (dW, db) in zip(v, grads)]

                steps = v

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
        costs.append(float(network.cost(in_train, network.layers, target_train)))
        trainAcc.append(accuracy(network.feedForward(in_train), target_train))
        testAcc.append(accuracy(network.feedForward(in_test), target_test))

    return costs, trainAcc, testAcc







