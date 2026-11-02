import numpy as np  
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics as skm
from utils import *
from estimator import *
from neural_network import NN 

"""
Model selection of the FFNN for classifying digits

Cross-validation has been done over learning rates and penalties
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


# load data frame - grid from cross validation
GRID = pd.read_csv("results/gridsearch_MNIST.csv")
GRID_LOG = pd.read_csv("results/gridsearch_MNIST_logistic.csv")

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
    plt.savefig(f"results/{filename}.pdf", bbox_inches="tight")
    plt.show()

    print(f"{name}, within 1 SE of best:\n", within[["lmbd", "gamma", "cv", "se"]].to_string(index=False))
    return within, chosen["lmbd"], chosen["gamma"]


within, lam, gamma = plot_grid(GRID, "Network (50 nodes)", "heatmap_MNIST")
within_log, lam_log, gamma_log = plot_grid(GRID_LOG, "Logistic regression", "heatmap_MNIST_logistic")


def plot_width(df_w, ref, filename):
	# with assistance of Claude Opus 5.5 (October 2026)
	# w/ prompt to enhance already plotted results
    best   = df_w.loc[df_w["cv"].idxmax()]
    within = df_w[df_w["cv"] >= best["cv"] - best["se"]]
    chosen = within.loc[within["width"].idxmin()]   # 1-SE fewest hidden nodes

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
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
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize="small")
    plt.tight_layout()
    plt.savefig(f"results/{filename}.pdf", bbox_inches="tight")
    plt.show()
    return int(chosen["width"])

WIDTH = pd.read_csv("results/valcurve_MNIST_width.csv")
ref = GRID_LOG[(GRID_LOG["lmbd"] == lam_log) & (GRID_LOG["gamma"] == gamma_log)].iloc[0]
width = plot_width(WIDTH, ref, "valcurve_MNIST_width")


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

# train (both own back propagation and classifier)
history, num_epochs, layers = train(DigitsNet, X_train, targets_train, classification=True, method="adam", gamma=gamma, M=130, epochs=10, rho=0.99, beta1=0.9, beta2=0.999, eps=1e-8, tol=None, seed=2026)
minimizeCost, trainCost, trainAcc = history["minimizeCost"], history["trainCost"], history["trainAcc"]

history_log, num_epochs_log, layers_log = train(DigitsLogistic, X_train, targets_train, classification=True, method="adam", gamma=gamma_log, M=130, epochs=10, rho=0.99, beta1=0.9, beta2=0.999, eps=1e-8, tol=None, seed=2026)
minimizeCost_log, trainCost_log, trainAcc_log = history_log["minimizeCost"], history_log["trainCost"], history_log["trainAcc"]


majority = np.bincount(y_train).argmax()
baseline_acc = np.mean(y_test == majority)
print(f"baseline (always guess {majority}): test accuracy {baseline_acc:.4f}")

models = {
    "logistic regression": {"net": DigitsLogistic, "history": history_log},
    f"one hidden layer of {chosen_width}": {"net": DigitsNet, "history": history},
}
for name, m in models.items():
    m["pred"] = np.argmax(m["net"].feedForward(X_test), axis=1)
    m["n_params"] = sum(W.size + b.size for W, b in m["net"].layers)
    m["test_acc"] = np.mean(m["pred"] == y_test)
    print(f"{name}: {m['n_params']} parameters, final cost {m['history']['trainCost'][-1]:.4f}, "
          f"test accuracy {m['test_acc']:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
for name, col in zip(models, ("gray", "k")):
    ax[0].semilogy(models[name]["history"]["trainCost"], color=col, label=name)
    ax[1].plot(models[name]["history"]["trainAcc"], color=col, label=name)
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training cross entropy"); ax[0].legend(fontsize=8)
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("training accuracy"); ax[1].legend(fontsize=8)
ax[0].grid(True)
ax[1].grid(True)
plt.tight_layout()
plt.savefig(f"results/train_vs_epochs.pdf", bbox_inches="tight")
plt.show()


"""
EVALUATE MODELS
"""

n = len(y_test)
net_name = f"one hidden layer of {chosen_width}"
w = WIDTH[WIDTH["width"] == chosen_width].iloc[0]

summary = pd.DataFrame({
    "model":      ["baseline", "logistic regression", net_name],
    "parameters": [0, models["logistic regression"]["n_params"], models[net_name]["n_params"]],
    "CV":         [np.nan, ref["cv"], w["cv"]],
    "CV SE":      [np.nan, ref["se"], w["se"]],
    "test":       [baseline_acc, models["logistic regression"]["test_acc"], models[net_name]["test_acc"]],
})
summary["test"] = summary["test"].astype(float)
summary["test SE"] = np.sqrt(summary["test"] * (1 - summary["test"]) / n)

print(summary.to_string(index=False, float_format="%.4f"))
summary.to_csv("results/summary_MNIST.csv", index=False)

# confusion matrix on the test set

fig, ax = plt.subplots(figsize=(4.5, 4.5))
skm.ConfusionMatrixDisplay.from_predictions(y_test, models[f"one hidden layer of {chosen_width}"]["pred"], values_format="d", cmap="GnBu", colorbar=False, ax=ax)
plt.savefig("results/confusion_MNIST.pdf", bbox_inches="tight")
plt.show()


# precision, recall, F1

report = skm.classification_report(y_test, models[net_name]["pred"], output_dict=True)
per_digit = pd.DataFrame(report).T.loc[[str(d) for d in range(10)], ["precision", "recall", "f1-score", "support"]]

print(per_digit.to_string(float_format="%.4f"))
per_digit.to_csv("results/per_digit_MNIST.csv")














































