# Using feed forward neural networks for regression and classification problems

By Elvira Boström, Catherine Hughes and Vanii Hsin-Ni Yao

The full report is given as a pdf-file in the same directory as this README.

## Prerequisites
Running this project requires scikit-learn and jax, which can be installed as described on their GitHub sites. It also requires Matplotlib, NumPy, Pandas, Seaborn and Pathlib. 

## Running the regression

Fitting the Runge function.

- `regression.py`: Runs all different fits as well as cross-validation and the final evaluation.

1) optimizer comparison with different learning rates, test both 1 layer and 2 layer case.
2) activation function comparison (adam only, single lr), test both 1 layer and 2 layer case.
3) cross validation (adam only, 2 layers only).
4) final fit using adam, 2 layers, optimal hyperparameters.

## Running the classification 

Classifying the MNIST handwritten digits data set. 

- `cv_digits.py`: Cross-validates a neural network with 50 nodes in one hidden layer and does logistic regression, both over the same grid of learning rates and regularization strengths. A validation curve over the number of nodes is performed for the neural network. 
- `evaluate_digits.py`: Using results from `cv_digits.py`, the best models are determined, trained and evaluated. All plots and table-values for the classification are computed here. 

## /src
- `utils.py`: Activation functions and cost functions and their derivatives, as well as the function made for training the networks. 
- `neural_network.py`: Our neural network class. 
- `estimator.py`: Wrapper classes, for regression and classification, for using scikit_learns cross-validation.

## /tests

- `test_NN.py`: early testing of the NN class, on the iris data set, test against JAX gradients.
- `test_classification.py`: testing on the scikit learn digits set. Trying the procedure for cross validation and collecting thoughts on results to keep when classifying the whole MNIST set.

## /figures

All figures used in the final report, and some more that were not used. 
