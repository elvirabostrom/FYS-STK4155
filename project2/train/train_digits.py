import numpy as np  
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import *
from estimator import *
from neural_network import NN 

"""
Classifying digits with a FFNN

The softmax is the final layer activation
The cross entropy is the cost function
"""

"""
---------------------------------------------------------------------------------------------
"""

# GENERAL
# download MNIST dataset, but test with small version from scikit first
# scaling, splitting
# MNIST: kommer med eget testsett, bruk dette kun til siste evaluering
# last activation is softmax, and cost is cross entropy
# accuracy score, one hot

# do: some comparison over something, yes.

# do: model selection
# gamma, lambda, structure (depth, width), activation funcs (in hidden layers)
# confusion matrix (find with scikit learn)

# compare to logistic regression (no hidden layer)

"""
---------------------------------------------------------------------------------------------
"""






