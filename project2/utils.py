import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from sklearn.metrics import accuracy_score
import numpy as np

"""
Activation functions, cost functions and their derivatives, and other functions used over 
multiple parts of our analysis, in both regression and classification.
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









