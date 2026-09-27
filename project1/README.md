# Benchmarking linear regression methods OLS, Ridge and Lasso

## Prerequisites
This project requires scikit-learn and jax, which can be installed as described on their GitHub sites. It also requires Matplotlib, NumPy, Pandas and Pathlib. 

## Running the files

The following files compute and save to file the MSE and/or R2 score on fitting data generated from the Runge function, using the methods specified below. 

- `run_closed_form.py`: Own implementation and scikit-learn pmlementation of OLS and Ridge, either with no resampling, bootstrap or cross-validation. Compute also the MSE of Lasso regression using cross-validation. 
- `part_e.py`: Our own plain gradient descent implementation with analytical gradients and automatic differentiation, OLS and Ridge. 
- `part_f.py`: Updating the learning rate with momentum, AdaGrad, RMSProp and Adam for OLS and Ridge. 
- `part_g.py`: Lasso regression.
- `part_h.py`: Stochastic gradient descent for OLS, Ridge and Lasso. 
- `part_i.py`: ?

## Other 
- `utils.py`: Contains functions used in the running files. 
- `plotting.py`: Plots final results.
- `g_plotting.py`: Plots final results for "part g", assessing Lasso regression.
