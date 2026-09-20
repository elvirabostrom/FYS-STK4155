import numpy as np
from pathlib import Path
from utils import writeToFile, MakeData, MakeDesignMatrix, scaleData, closedForm, GradOLSAnalytic, GradRidgeAnalytic, optimiser_step

# --- 1. Helper Functions for SGD ---

def make_batches(n, batch_size, rng):
    """Shuffle the indices and split them into minibatches (Section 4.7)."""
    idx = rng.permutation(n)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]

def step_length(t, t0, t1):
    """The learning-rate schedule of Eq. (4.40)."""
    return t0 / (t + t1)

def sgd(X, y, method="plain", n_epochs=100, batch_size=5, gamma=0.1, schedule=None,
        lam=0.0, seed=2026, theta0=None, **kw):
    """Minibatch stochastic gradient descent with any optimiser."""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    theta = np.zeros(p) if theta0 is None else np.array(theta0, dtype=float)
    state, t, history = {}, 0, [theta.copy()]
    
    eval_history = [0]
    total_evals = 0
    
    for epoch in range(n_epochs):
        for batch in make_batches(n, batch_size, rng):
            t += 1
            if lam == 0.0:
                g = GradOLSAnalytic(theta, X[batch], y[batch])
            else:
                g = GradRidgeAnalytic(theta, X[batch], y[batch], lam)
                
            g_t = gamma if schedule is None else step_length(t, *schedule)
            theta, state = optimiser_step(method, theta, g, state, t, g_t, **kw)
            total_evals += len(batch)
        history.append(theta.copy())
        eval_history.append(total_evals)
        
    return np.array(history), np.array(eval_history)


# --- 2. Data Generation & Scaling ---

degree = 5
n_samples = 100
x_raw, y_raw = MakeData(n_samples, noise=0.1, seed=2026)
X_raw = MakeDesignMatrix(x_raw, degree)

# Scale design matrix and center targets
X_train_scaled, _, y_train_centered = scaleData(X_raw, X_raw, y_raw)

# Compute closed-form solution using closedForm
lam_val = 0.0
theta_cf = closedForm(X_train_scaled, y_train_centered, lamb=lam_val)


# --- 3. Experiment Execution & Data Saving ---

# Experiment A: Mini-batch size sensitivity sweep
results_batch = []
batch_sizes = [1, 5, 20, n_samples]
n_epochs_batch = 100

for M in batch_sizes:
    hist, evals = sgd(X_train_scaled, y_train_centered, n_epochs=n_epochs_batch, batch_size=M, gamma=0.05, lam=lam_val)
    for epoch, (theta_est, total_evals) in enumerate(zip(hist, evals)):
        param_error = np.max(np.abs(theta_est - theta_cf))
        results_batch.append({
            "batch_size": M,
            "epoch": epoch,
            "total_evals": total_evals,
            "final_param_error": param_error
        })

writeToFile({"part": "h", "type": "batch_sensitivity", "degree": degree}, results_batch)

# Experiment B: Learning-rate schedule analysis
results_sched = []
n_epochs_sched = 150

for schedule_name, sched_params in [("constant", None), ("scheduled_1", (5.0, 50.0))]:
    gamma_val = 0.1 if sched_params is None else 0.0
    hist, evals = sgd(X_train_scaled, y_train_centered, n_epochs=n_epochs_sched, batch_size=5, gamma=gamma_val, schedule=sched_params, lam=lam_val)
    for epoch, (theta_est, total_evals) in enumerate(zip(hist, evals)):
        param_error = np.max(np.abs(theta_est - theta_cf))
        results_sched.append({
            "schedule": schedule_name,
            "epoch": epoch,
            "total_evals": total_evals,
            "final_param_error": param_error
        })

writeToFile({"part": "h", "type": "lr_schedule", "degree": degree}, results_sched)

print("Part H data generation complete and saved to results/")