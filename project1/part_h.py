# ---------------------------------------------------------------------------------------
# In this part, we implement stochastic gradient descent with Adam, test different 
# mini-batch sizes and compare with full batch, we also test a decaying learning schedule
# ---------------------------------------------------------------------------------------

import numpy as np
from pathlib import Path
from utils import *

# --- 1. Helper Functions for SGD ---

def make_batches(n, batch_size, rng):
    """Shuffle the indices and split them into minibatches."""
    idx = rng.permutation(n)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]

def step_length(t, t0, t1):
    """The learning-rate schedule of Eq. (4.40)."""
    return t0 / (t + t1)

def sgd_fixed_steps(X, y, method="adam", total_target_steps=10000, batch_size=5, gamma=0.01, schedule=None,
                    lam=0.0, seed=2026, theta0=None, **kw):
    """Minibatch stochastic gradient descent running for a fixed total number of steps."""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    theta = np.zeros(p) if theta0 is None else np.array(theta0, dtype=float)
    state, t = {}, 0
    
    history = [theta.copy()]
    eval_history = [0]
    total_evals = 0
    
    steps_per_epoch = int(np.ceil(n / batch_size))
    n_epochs = int(np.ceil(total_target_steps / steps_per_epoch))
    
    step_count = 0
    for epoch in range(n_epochs):
        for batch in make_batches(n, batch_size, rng):
            if step_count >= total_target_steps:
                break
            t += 1
            step_count += 1
            
            if lam == 0.0:
                g = GradOLSAnalytic(theta, X[batch], y[batch])
            else:
                g = GradRidgeAnalytic(theta, X[batch], y[batch], lam)
                
            g_t = gamma if schedule is None else step_length(t, *schedule)
            theta, state = optimiser_step(method, theta, g, state, t, g_t, **kw)
            total_evals += len(batch)
            
        history.append(theta.copy())
        eval_history.append(total_evals)
        if step_count >= total_target_steps:
            break
        
    return np.array(history), np.array(eval_history)


# --- 2. The setup ---

degree = 5
n_samples = 100
x, y = MakeData(n_samples, noise=0.1, seed=2026)
X = MakeDesignMatrix(x, degree)

X_train_scaled, _, y_train_centered = scaleData(X, X, y)

models = {
    "OLS": 0.0,
    "Ridge": 0.01
}

base_lr = 0.01 # learning rate for Adam
target_evals_budget = 100000  # we use max 10^5 total gradient evaluations (for full batch that correspoonds to 1000 iterations)


# --- 3. Generate data for plots ---

for model_name, lam_val in models.items():
    theta_cf = closedForm(X_train_scaled, y_train_centered, lamb=lam_val)

    # ----------------------------------------------------
    # Test batch size sensitivity
    # ----------------------------------------------------
    results_batch = []
    batch_sizes = [1, 5, 20, n_samples]

    for M in batch_sizes:
        total_target_steps = int(np.ceil(target_evals_budget / M))
        
        hist, evals = sgd_fixed_steps(
            X_train_scaled, y_train_centered, method="adam", 
            total_target_steps=total_target_steps, batch_size=M, 
            gamma=base_lr, lam=lam_val, schedule=None
        )
        for idx, (theta_est, total_evals) in enumerate(zip(hist, evals)):
            param_error = np.max(np.abs(theta_est - theta_cf))
            results_batch.append({
                "batch_size": M,
                "epoch": idx,
                "total_evals": total_evals,
                "final_param_error": param_error
            })

    writeToFile({"part": "h", "type": f"batch_sensitivity_adam_{model_name.lower()}", "degree": degree}, results_batch)

    # ----------------------------------------------------
    # Constant vs. scheduled learning rate
    # ----------------------------------------------------
    results_schedule = []
    schedules_to_test = {
        "constant": None,
        "scheduled_1": (5, 50)  # t0=5, t1=50 schedule configuration
    }

    
    sched_batch_size = n_samples 
    total_target_steps = int(np.ceil(target_evals_budget / sched_batch_size))

    for sched_name, sched_params in schedules_to_test.items():
        hist, evals = sgd_fixed_steps(
            X_train_scaled, y_train_centered, method="adam", 
            total_target_steps=total_target_steps, batch_size=sched_batch_size, 
            gamma=base_lr, schedule=sched_params, lam=lam_val
        )
        for idx, (theta_est, total_evals) in enumerate(zip(hist, evals)):
            param_error = np.max(np.abs(theta_est - theta_cf))
            results_schedule.append({
                "schedule": sched_name,
                "epoch": idx,
                "total_evals": total_evals,
                "final_param_error": param_error
            })

    writeToFile({"part": "h", "type": f"lr_schedule_adam_{model_name.lower()}", "degree": degree}, results_schedule)