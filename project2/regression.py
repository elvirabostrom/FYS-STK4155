import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from utils import (
    sigmoid, sigmoid_der, 
    ReLU, ReLU_der, 
    leakyReLU, leakyReLU_der, 
    linear, linear_der, 
    MSE, MSE_der
)
from neural_network import NN, train

# Generate Runge data
def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)

rng = np.random.default_rng(2026)
n = 100
x = np.sort(rng.uniform(-1.0, 1.0, n))
sigma = 0.1
y = runge(x) + rng.normal(0.0, sigma, n)

plt.plot(x, runge(x), label="Runge function")
plt.scatter(x, y, s=10, label=f"noisy data, sigma={sigma}")
plt.legend(); plt.close()

# Reshape into (n, 1)
X = x.reshape(-1, 1)
Y = y.reshape(-1, 1)

# Shuffle data before splitting into train/test
shuffled_indices = rng.permutation(n)
X = X[shuffled_indices]
Y = Y[shuffled_indices]

# Split into train/test data (80/20)
X_train, X_test = X[:int(0.8 * n)], X[int(0.8 * n):]
Y_train, Y_test = Y[:int(0.8 * n)], Y[int(0.8 * n):]

# IS IT OK THAT I AM USING THE ENTIRE DATASET TO FIND THE MEAN AND STD?
# Scale data 
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
Y_mean = Y.mean(axis=0)
X_train_scaled = (X_train - X_mean) / X_std
Y_train_scaled = Y_train - Y_mean
X_test_scaled = (X_test - X_mean) / X_std
Y_test_scaled = Y_test - Y_mean

# Create a fine grid for future plotting
x_grid = np.linspace(-1.0, 1.0, 500)
X_grid = (x_grid.reshape(-1, 1) - X_mean) / X_std


print("\nRunning optimizer comparison")

# Define network architecture configurations: (Name, layer sizes, activation funcs, activation ders)
layer_configs = [
    (
        "1 Hidden Layer (50)",
        [50, 1],
        [sigmoid, linear],
        [sigmoid_der, linear_der],
    ),
    (
        "2 Hidden Layers (50, 50)",
        [50, 50, 1],
        [sigmoid, sigmoid, linear],
        [sigmoid_der, sigmoid_der, linear_der],
    )
]

#===============================================================================================================
# Adam (1 and 2 hidden layers, different learning rates)
#===============================================================================================================

lr_values = [0.001, 0.005]
colors = ["#9ecae1", "#08519c"]

# Loop over architectures and create a separate figure for each
for arch_name, layer_sizes, act_funcs, act_ders in layer_configs:
  print(f"\nRunning Adam for: {arch_name}")

  plt.figure(figsize=(6, 4))

  # Plot data points
  plt.scatter(X_train, Y_train, color="k", s=20, label="Train data")
  plt.scatter(X_test, Y_test, facecolors="none", edgecolors="k", s=25, label="Test data")

  # Plot true analytical curve
  plt.plot(x_grid, runge(x_grid), label="Runge function", color="tab:gray")

  plt.xlabel("x")
  plt.ylabel("y")
  plt.grid(True)

  for lr, color in zip(lr_values, colors):

    # Initialize neural network with current architecture
    nn_adam = NN(
        inputSize=1,
        layerOutputSizes=layer_sizes,
        activationFuncs=act_funcs,
        activationDers=act_ders,
        costFunc=MSE,
        costDer=MSE_der,
        seed=2026,
    )

    # Train the network using Adam optimizer
    _, _, _, epochs_adam = train(
        nn_adam,
        X_train_scaled,
        Y_train_scaled,
        classification=False,
        method="adam",
        gamma=lr,
        M=16,
        epochs=2000,
        tol=1e-7,
    )

    # Evaluate on test data and scale back
    Y_pred_adam = nn_adam.feedForward(X_test_scaled) + Y_mean

    # Compute test MSE
    test_mse_adam = MSE(Y_pred_adam, Y_test)

    print(f"Test MSE (lr={lr}): {test_mse_adam:.5f} ({epochs_adam} epochs)")

    # Evaluate models on the fine grid and scale back predictions
    Y_pred_grid_adam = nn_adam.feedForward(X_grid) + Y_mean

    # Plot model prediction for this learning rate
    plt.plot(x_grid, Y_pred_grid_adam.ravel(), label=f"lr={lr}", color=color)

  # Determine number of hidden layers from layer_sizes
  num_hidden = len(layer_sizes) - 1
  
  plt.legend()
  plt.savefig(f"runge_adam_{num_hidden}layer.pdf", bbox_inches="tight")
  plt.close()

#===============================================================================================================
# RMSprop (1 and 2 hidden layers, different learning rates)
#===============================================================================================================

lr_values = [0.001, 0.005]
colors = ["#9ecae1", "#08519c"]

# Loop over architectures and create a separate figure for each
for arch_name, layer_sizes, act_funcs, act_ders in layer_configs:
  print(f"\nRunning RMSprop for: {arch_name}")

  plt.figure(figsize=(6, 4))

  # Plot data points
  plt.scatter(X_train, Y_train, color="k", s=20, label="Train data")
  plt.scatter(X_test, Y_test, facecolors="none", edgecolors="k", s=25, label="Test data")

  # Plot true analytical curve
  plt.plot(x_grid, runge(x_grid), label="Runge function", color="tab:gray")

  plt.xlabel("x")
  plt.ylabel("y")
  plt.grid(True)

  for lr, color in zip(lr_values, colors):

    # Initialize neural network with current architecture
    nn_rmsprop = NN(
        inputSize=1,
        layerOutputSizes=layer_sizes,
        activationFuncs=act_funcs,
        activationDers=act_ders,
        costFunc=MSE,
        costDer=MSE_der,
        seed=2026,
    )

    # Train the network using RMSprop optimizer
    _, _, _, epochs_rmsprop = train(
        nn_rmsprop,
        X_train_scaled,
        Y_train_scaled,
        classification=False,
        method="rmsprop",
        gamma=lr,
        M=16,
        epochs=2000,
        tol=1e-7,
    )

    # Evaluate on test data and scale back
    Y_pred_rmsprop = nn_rmsprop.feedForward(X_test_scaled) + Y_mean

    # Compute test MSE
    test_mse_rmsprop = MSE(Y_pred_rmsprop, Y_test)

    print(f"Test MSE (lr={lr}): {test_mse_rmsprop:.5f} ({epochs_rmsprop} epochs)")

    # Evaluate models on the fine grid and scale back predictions
    Y_pred_grid_rmsprop = nn_rmsprop.feedForward(X_grid) + Y_mean

    # Plot model prediction for this learning rate
    plt.plot(x_grid, Y_pred_grid_rmsprop.ravel(), label=f"lr={lr}", color=color)

  # Determine number of hidden layers from layer_sizes
  num_hidden = len(layer_sizes) - 1

  plt.legend()
  plt.savefig(f"runge_rmsprop_{num_hidden}layer.pdf", bbox_inches="tight")
  plt.close()

#===============================================================================================================
# Plain GD (1 and 2 hidden layers, different learning rates)
#===============================================================================================================

lr_values = [0.0001, 0.0005]
colors = ["#9ecae1", "#08519c"]

# Loop over architectures and create a separate figure for each
for arch_name, layer_sizes, act_funcs, act_ders in layer_configs:
  print(f"\nRunning Plain GD for: {arch_name}")

  plt.figure(figsize=(6, 4))

  # Plot data points
  plt.scatter(X_train, Y_train, color="k", s=20, label="Train data")
  plt.scatter(X_test, Y_test, facecolors="none", edgecolors="k", s=25, label="Test data")

  # Plot true analytical curve
  plt.plot(x_grid, runge(x_grid), label="Runge function", color="tab:gray")

  plt.xlabel("x")
  plt.ylabel("y")
  plt.grid(True)

  for lr, color in zip(lr_values, colors):

    # Initialize neural network with current architecture
    nn_gd = NN(
        inputSize=1,
        layerOutputSizes=layer_sizes,
        activationFuncs=act_funcs,
        activationDers=act_ders,
        costFunc=MSE,
        costDer=MSE_der,
        seed=2026,
    )

    # Train the network using Plain GD optimizer
    _, _, _, epochs_gd = train(
        nn_gd,
        X_train_scaled,
        Y_train_scaled,
        classification=False,
        method="gd",
        gamma=lr,
        M=None,
        epochs=2000,
        tol=1e-7,
    )

    # Evaluate on test data and scale back
    Y_pred_gd = nn_gd.feedForward(X_test_scaled) + Y_mean

    # Compute test MSE
    test_mse_gd = MSE(Y_pred_gd, Y_test)

    print(f"Test MSE (lr={lr}): {test_mse_gd:.5f} ({epochs_gd} epochs)")

    # Evaluate models on the fine grid and scale back predictions
    Y_pred_grid_gd = nn_gd.feedForward(X_grid) + Y_mean

    # Plot model prediction for this learning rate
    plt.plot(x_grid, Y_pred_grid_gd .ravel(), label=f"lr={lr}", color=color)
  
  # Determine number of hidden layers from layer_sizes
  num_hidden = len(layer_sizes) - 1

  plt.legend()
  plt.savefig(f"runge_gd_{num_hidden}layer.pdf", bbox_inches="tight")
  plt.close()


#===============================================================================================================
# ReLU and Leaky ReLU comparison (Adam, 1 and 2 hidden layers)
#===============================================================================================================

print("\nRunning ReLU and Leaky ReLU comparison")

# Define network architecture configurations: (Name, layer sizes, filename)
layer_configs = [
    ("1 Hidden Layer (50)", [50, 1], "runge_actfuncs_1layer.pdf"),
    ("2 Hidden Layers (50, 50)", [50, 50, 1], "runge_actfuncs_2layer.pdf"),
]

# Activation functions
act_configs = [
    ("ReLU", ReLU, ReLU_der, "#fb6a4a"),
    ("Leaky ReLU", leakyReLU, leakyReLU_der, "#de2d26"),
]

# Loop over architectures and create a separate figure for each
for arch_name, layer_sizes, filename in layer_configs:

  plt.figure(figsize=(6, 4))

  # Plot data points
  plt.scatter(X_train, Y_train, color="k", s=20, label="Train data")
  plt.scatter(X_test, Y_test, facecolors="none", edgecolors="k", s=25, label="Test data")

  # Plot true analytical curve
  plt.plot(x_grid, runge(x_grid), label="Runge function", color="tab:gray")

  plt.xlabel("x")
  plt.ylabel("y")
  plt.grid(True)

  # Determine number of hidden layers from layer_sizes
  num_hidden = len(layer_sizes) - 1

  for act_name, act_func, act_der, color in act_configs:

    # Construct activation lists: hidden layers get act_func, output layer gets linear
    activation_funcs = [act_func] * num_hidden + [linear]
    activation_ders = [act_der] * num_hidden + [linear_der]

    # Initialize neural network
    nn_model = NN(
        inputSize=1,
        layerOutputSizes=layer_sizes,
        activationFuncs=activation_funcs,
        activationDers=activation_ders,
        costFunc=MSE,
        costDer=MSE_der,
        seed=2026,
    )

    # Train using Adam with fixed learning rate 0.001
    _, _, _, epochs = train(
        nn_model,
        X_train_scaled,
        Y_train_scaled,
        classification=False,
        method="adam",
        gamma=0.001,
        M=16,
        epochs=2000,
        tol=1e-7,
    )

    # Evaluate on test data and scale back
    Y_pred = nn_model.feedForward(X_test_scaled) + Y_mean
    test_mse = MSE(Y_pred, Y_test)

    print(f"Test MSE ({arch_name}, {act_name}): {test_mse:.5f} ({epochs} epochs)")

    # Evaluate models on the fine grid and scale back predictions
    Y_pred_grid = nn_model.feedForward(X_grid) + Y_mean

    # Plot model prediction
    plt.plot(x_grid, Y_pred_grid.ravel(), color=color, label=act_name)

  plt.legend()
  plt.savefig(filename, bbox_inches="tight")
  plt.close()
