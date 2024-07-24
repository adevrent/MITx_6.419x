import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load and prepare data
filepath = r"C:\Users\Atakan\atakan_python\MITx_6.419x\vol_model_zeynep\vol_data_close.xlsx"

df = pd.read_excel(filepath)
df = df.iloc[::-1]
df = df.set_index("date")
df["log_return"] = np.log(df["close"] / df["close"].shift(1))
df["log_return"] = df["log_return"].fillna(0)  # fill the first return value with 0
df["log_return_sq"] = df["log_return"]**2

# Visualize log returns and their squares
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(df.index, df["log_return"], label="Log Returns")
plt.xlabel('Date')
plt.ylabel('Log Return')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(df.index, df["log_return_sq"], label="Squared Log Returns")
plt.xlabel('Date')
plt.ylabel('Squared Log Return')
plt.legend()
plt.grid(True)

plt.show()

# Define the negative log-likelihood function
def negative_log_likelihood(x, var, mu=0):
    return -(-0.5 * np.log(2 * np.pi * var) - 0.5 * (x - mu)**2 / var)

# Define the objective function to minimize
def sum_negative_log_likelihood(alpha, x_arr, initial_var):
    var_arr = np.zeros_like(x_arr)
    var_arr[0] = initial_var
    for i in range(1, len(x_arr)):
        var_arr[i] = alpha * x_arr[i-1]**2 + (1 - alpha) * var_arr[i-1]
    
    nll_sum = np.sum(negative_log_likelihood(x_arr, var_arr))
    return nll_sum

# Variables to store the progress of optimization
alpha_values = []
nll_values = []

# Callback function to store alpha and NLL values
def callback(alpha):
    alpha_values.append(alpha[0])
    nll = sum_negative_log_likelihood(alpha, df["log_return"].values, initial_var)
    nll_values.append(nll)
    print(f"Alpha: {alpha[0]}, NLL: {nll}")

# Initial variance (you can choose a small number, e.g., 1e-6)
initial_var = 0.000021

# Minimize the objective function with respect to alpha
result = minimize(sum_negative_log_likelihood, x0=0.5, args=(df["log_return"].values, initial_var), bounds=[(0, 1)], callback=callback)
optimal_alpha = result.x[0]
print(f"Optimal alpha: {optimal_alpha}")

# Plot the alpha values and corresponding NLL values
plt.figure(figsize=(10, 6))
plt.plot(alpha_values, nll_values, marker='o', linestyle='-')
plt.xlabel('Alpha')
plt.ylabel('Sum of Negative Log-Likelihood')
plt.title('Optimization Progress')
plt.grid(True)
plt.show()