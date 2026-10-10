import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from sklearn.metrics import accuracy_score
import numpy as np

"""
Write to file
"""


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
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data: one row of scores per sample."""
    e_z = jnp.exp(z - jnp.max(z, axis=1, keepdims=True))
    return e_z / jnp.sum(e_z, axis=1, keepdims=True)

def linear(z):
    return z


"""
Cost functions
"""

def MSE(y_pred, y):
    return jnp.mean((y_pred - y)**2)


def crossEntropy(y_pred, y):
    return jnp.sum(- y * jnp.log(y_pred)) # er denne riktig?


"""
Accuracy
"""

def accuracy(y_pred, y_one_hot):
    one_hot_predictions = np.zeros(y_pred.shape)

    for i, pred in enumerate(y_pred):
        one_hot_predictions[i, np.argmax(pred)] = 1
    return accuracy_score(one_hot_predictions, y_one_hot) # scikit-function



