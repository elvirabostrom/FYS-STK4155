import numpy as np  
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics as skm
from src.utils import *
from src.estimator import *
from src.neural_network import NN
import time

"""
Model selection of the FFNN for classifying digits, selecting the final results and plotting/printing them

From cv_digits.py:
Cross-validation has been done over learning rates and penalties
A validation curve has been found over the width
"""

def one_se(df):
    best   = df.loc[df["cv"].idxmax()]
    within = df[df["cv"] >= best["cv"] - best["se"]]
    chosen = within[within["lmbd"] == within["lmbd"].max()].sort_values("cv").iloc[-1]
    return best, within, chosen


# load MNIST dataset
with np.load("mnist.npz") as data: 
	x_test, y_test = data["x_test"], data["y_test"]
	x_train, y_train = data["x_train"], data["y_train"]
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)


# one hot targets
targets_train = np.zeros((len(y_train), 10))
targets_test = np.zeros((len(y_test), 10))
for i, t in enumerate(y_train):
	targets_train[i, t] = 1
for i, t in enumerate(y_test):
    targets_test[i, t] = 1


# scale image pixels to [0, 1]
# each picture is flattened to a row of 784 features
X_test = x_test.reshape(len(x_test), -1) / 255.0 # (10000, 784)
X_train = x_train.reshape(len(x_train), -1) / 255.0 # (60000, 784)


# load data frame from cross validation
GRID = pd.read_csv("results/gridsearch_MNIST.csv")
GRID_LOG = pd.read_csv("results/gridsearch_MNIST_logistic.csv")

# grid used
penalties = [1e-6, 1e-5, 1e-4, 1e-3, 0.01]
gammas  = [1e-4, 1e-3, 3e-3, 0.01, 0.03]


"""
CROSS VALIDATION GRID
"""

def plot_grid(df, name, filename):
	# written by Claude Opus 5.5 (October 2026)
	# w/ prompt to make a function plotting heatmaps and returning chosen model, given the shape of the dataframe, the one_se() function and an example heatmap made in the weekly exercises (week 44)
    best, within, chosen = one_se(df)
    CV    = df.pivot(index="gamma", columns="lmbd", values="cv")
    TRAIN = df.pivot(index="gamma", columns="lmbd", values="train")

    def cell(row):
        return (CV.columns.get_loc(row["lmbd"]), CV.index.get_loc(row["gamma"]))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.0))
    sns.heatmap(CV,    cmap="GnBu", annot=True, fmt=".3f", ax=axes[0])
    sns.heatmap(TRAIN, cmap="GnBu", annot=True, fmt=".3f", ax=axes[1])
    for ax, title in zip(axes, ["CV accuracy", "Training accuracy"]):
        ax.set_xlabel(r"$\lambda$"); ax.set_ylabel(r"$\gamma$"); ax.set_title(f"{name}: {title}")

    for _, row in within.iterrows():   # black: within 1 SE of best
        axes[0].add_patch(plt.Rectangle(cell(row), 1, 1, fill=False, ec="k", lw=2))
    axes[0].add_patch(plt.Rectangle(cell(best),   1, 1, fill=False, ec="white",  lw=3))   # best
    axes[0].add_patch(plt.Rectangle(cell(chosen), 1, 1, fill=False, ec="red", lw=3))   # 1-SE choice

    plt.tight_layout()
    plt.savefig(f"figures/{filename}.pdf", bbox_inches="tight")
    plt.show()

    print(f"{name}, within 1 SE of best:\n", within[["lmbd", "gamma", "cv", "se"]].to_string(index=False))
    return within, chosen["lmbd"], chosen["gamma"]


within, lam, gamma = plot_grid(GRID, "Network (50 nodes)", "heatmap_MNIST")
within_log, lam_log, gamma_log = plot_grid(GRID_LOG, "Logistic regression", "heatmap_MNIST_logistic")


"""
VALIDATION CURVE
"""


def plot_width(df_w, ref, filename):
	# with assistance of Claude Opus 5.5 (October 2026)
	# w/ prompt to enhance already plotted results: mark standard error band and the chosen width
    best   = df_w.loc[df_w["cv"].idxmax()]
    within = df_w[df_w["cv"] >= best["cv"] - best["se"]]
    chosen = within.loc[within["width"].idxmin()]   # 1-SE fewest hidden nodes

    fig, ax = plt.subplots(figsize=(3.4, 4.2))
    ax.errorbar(df_w["width"], df_w["cv"], yerr=df_w["se"], marker="o", capsize=3, label="CV accuracy ± SE", color = "k")
    ax.plot(df_w["width"], df_w["train"], marker="s", label="Training accuracy", color = "yellowgreen")
    ax.axhline(ref["cv"], color="gray", label="Logistic regression, CV accuracy")
    ax.axhspan(ref["cv"] - ref["se"], ref["cv"] + ref["se"], color="gray", alpha=0.2, label="Logistic regression, ± 1 SE")
    ax.axhline(best["cv"] - best["se"], color="red", ls=":", label="1-SE threshold")
    ax.plot(chosen["width"], chosen["cv"], "o", ms=12, mfc="none", mec="red", mew=2, label=f"1-SE choice: {int(chosen['width'])} nodes")
    ax.grid(True)
    ax.set_xscale("log")
    ax.set_xticks(df_w["width"]); ax.set_xticklabels(df_w["width"].astype(int))
    ax.set_xlabel("Hidden nodes"); ax.set_ylabel("Accuracy")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.1), ncol=1, fontsize="medium", frameon = False)
    plt.tight_layout()
    plt.savefig(f"figures/{filename}.pdf", bbox_inches="tight")
    plt.show()
    return int(chosen["width"])

WIDTH = pd.read_csv("results/valcurve_MNIST_width.csv")
ref = GRID_LOG[(GRID_LOG["lmbd"] == lam_log) & (GRID_LOG["gamma"] == gamma_log)].iloc[0]
width = plot_width(WIDTH, ref, "valcurve_MNIST_width")

print(f"Chosen width from cross-validation: {width}")

"""
TRAIN CHOSEN MODELS
"""

chosen_width = width

# structure
network_input_size = X_train.shape[1]
layer_output_sizes = [chosen_width, 10]
activation_funcs = [sigmoid, softmax]
activation_ders = [sigmoid_der, None]
cost_func = crossEntropy

DigitsNet = NN(network_input_size, layer_output_sizes, activation_funcs, activation_ders, cost_func, costDer = None, norm = "l2", lam = lam)
DigitsLogistic = NN(network_input_size, [10], [softmax], [None], cost_func, costDer = None, norm = "l2", lam = lam_log)

# train
t0 = time.perf_counter()
history, num_epochs, layers = train(DigitsNet, X_train, targets_train, classification=True, method="adam", gamma=gamma, M=130, epochs=10, rho=0.99, beta1=0.9, beta2=0.999, eps=1e-8, tol=None, seed=2026)
time_net = time.perf_counter() - t0
#minimizeCost, trainCost, trainAcc = history["minimizeCost"], history["trainCost"], history["trainAcc"]

t0 = time.perf_counter()
history_log, num_epochs_log, layers_log = train(DigitsLogistic, X_train, targets_train, classification=True, method="adam", gamma=gamma_log, M=130, epochs=10, rho=0.99, beta1=0.9, beta2=0.999, eps=1e-8, tol=None, seed=2026)
time_log = time.perf_counter() - t0
#minimizeCost_log, trainCost_log, trainAcc_log = history_log["minimizeCost"], history_log["trainCost"], history_log["trainAcc"]

print("TRAINING DONE")

"""
TRAINING CROSS ENTROPY AND TRAINING ACCURACY
"""


# baseline
majority = np.bincount(y_train).argmax()
baseline_acc = np.mean(y_test == majority)
print(f"baseline (always guess {majority}): test accuracy {baseline_acc:.4f}")


# plot
label_log = "logistic regression"
label_net = f"one hidden layer of {chosen_width}"

# training cost
plt.figure(figsize=(3.3, 3.0))
plt.semilogy(history_log["trainCost"], color="cornflowerblue", label=label_log)
plt.semilogy(history["trainCost"], color="indianred", label=label_net)
plt.xlabel("epoch")
plt.ylabel("training cross entropy")
plt.grid(True)
plt.xlim(0,9)
plt.tight_layout()
plt.savefig("figures/train_cost_vs_epochs.pdf", bbox_inches="tight")
plt.show()

# training accuracy
plt.figure(figsize=(6.0, 3.0))
plt.plot(history_log["trainAcc"], color="cornflowerblue", label=label_log)
plt.plot(history["trainAcc"], color="indianred", label=label_net)
plt.xlabel("epoch")
plt.ylabel("training accuracy")
plt.legend(loc="upper left", bbox_to_anchor=(1.1, 1.0), ncol=1, fontsize="medium", frameon=False)
plt.grid(True)
plt.xlim(0,9)
plt.tight_layout()
plt.savefig("figures/train_acc_vs_epochs.pdf", bbox_inches="tight")
plt.show()


"""
EVALUATE MODELS
"""

# table vals
models = {
    "logistic regression": {"net": DigitsLogistic, "history": history_log, "epochs": num_epochs_log, "time": time_log},
    f"one hidden layer of {chosen_width}": {"net": DigitsNet, "history": history, "epochs": num_epochs, "time": time_net},
}

for name, m in models.items():
    m["pred"] = np.argmax(m["net"].feedForward(X_test), axis=1)
    m["n_params"] = sum(W.size + b.size for W, b in m["net"].layers)
    m["test_acc"] = np.mean(m["pred"] == y_test)
    print(f"{name}: {m['n_params']} parameters, final cost {m['history']['trainCost'][-1]:.4f}, "
    	f"test accuracy {m['test_acc']:.4f}, {m['epochs']} epochs, training time {m['time']:.1f} s")


# The following confusion matrix is by Claude, Opus 5.5 (November 2026)
# w prompt to include more digits while keeping the axis labels readable and change the row-normalizez to percentages
# in the prompt, my own code was pasted: 
"""
fig, ax = plt.subplots(figsize=(4.0, 4.0))
skm.ConfusionMatrixDisplay.from_predictions(y_test, models[f"one hidden layer of {chosen_width}"]["pred"], values_format="d", cmap="GnBu", colorbar=False, ax=ax)
plt.savefig("figures/confusion_MNIST.pdf", bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(4.0, 4.0))
skm.ConfusionMatrixDisplay.from_predictions(y_test, models[f"one hidden layer of {chosen_width}"]["pred"], normalize="true", values_format=".2f", cmap="GnBu", colorbar=False, ax=ax)
plt.savefig("figures/confusion_row_norm_MNIST.pdf", bbox_inches="tight")
plt.show()
"""
pred = models[f"one hidden layer of {chosen_width}"]["pred"]
label_size, tick_size, cell_size = 11, 9, 9.5 

def style(ax):
    ax.set_xlabel("Predicted label", fontsize=label_size)
    ax.set_ylabel("True label", fontsize=label_size)
    ax.tick_params(labelsize=tick_size)

# counts
fig, ax = plt.subplots(figsize=(4.0, 4.0))
skm.ConfusionMatrixDisplay.from_predictions(y_test, pred, values_format="d", cmap="GnBu", colorbar=False, ax=ax, text_kw={"fontsize": cell_size})
style(ax)
plt.savefig("figures/confusion_MNIST.pdf", bbox_inches="tight")
plt.show()

# row-normalized, in percent
cm = skm.confusion_matrix(y_test, pred, normalize="true") * 100
fig, ax = plt.subplots(figsize=(4.0, 4.0))
skm.ConfusionMatrixDisplay(cm).plot(values_format=".1f", cmap="GnBu", colorbar=False, ax=ax, text_kw={"fontsize": cell_size})
style(ax)
plt.savefig("figures/confusion_row_norm_MNIST.pdf", bbox_inches="tight")
plt.show()



# precision, recall, F1
report = skm.classification_report(y_test, models[f"one hidden layer of {chosen_width}"]["pred"], output_dict=True)
per_digit = pd.DataFrame(report).T.loc[[str(d) for d in range(10)], ["precision", "recall", "f1-score", "support"]]

print(per_digit.to_string(float_format="%.4f"))
per_digit.to_csv("results/per_digit_MNIST.csv")









































