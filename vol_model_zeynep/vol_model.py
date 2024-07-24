import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Load and prepare data
filepath = r"C:\Users\Atakan\atakan_python\MITx_6.419x\vol_model_zeynep\vol_data_close.xlsx"  # change if necessary

df = pd.read_excel(filepath)
df = df.iloc[::-1]
df = df.set_index("date")
df["log_return"] = np.log(df["close"] / df["close"].shift(1))
df["log_return"] = df["log_return"].fillna(0)  # fill the first return value with 0
df["log_return_sq"] = df["log_return"]**2

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

x_arr = df["log_return"]
initial_var = 0.00002150

# Minimize the objective function with respect to alpha
result = minimize(sum_negative_log_likelihood, x0=0.2, args=(df["log_return"].values, initial_var), bounds=[(0, 1)])
optimal_alpha = result.x[0]
print("    optimal_alpha =", optimal_alpha)
print("    optimal_lambda =", 1-optimal_alpha)

# Plot
initial_var_arr = np.linspace(0, 0.5, 10)
alpha_arr = np.linspace(0, 1, 50)
# print("alpha_arr", alpha_arr)

fig, axs = plt.subplots(1, len(initial_var_arr), figsize=(18, 4))

sum_log_likelihood_array = []
for initial_var in initial_var_arr:
    sum_log_likelihood_array.append([-sum_negative_log_likelihood(alpha, df["log_return"], initial_var) for alpha in alpha_arr])

for i, ax in enumerate(axs):
    ax.plot((1-alpha_arr), sum_log_likelihood_array[i])
    ax.set_title(f"initial_var = {np.round(initial_var_arr[i], 2)}", fontsize=7)
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('$L$')

    # Find the alpha that maximizes the sum of log likelihood
    max_index = np.argmax(sum_log_likelihood_array[i])
    max_alpha = alpha_arr[max_index]
    ax.axvline(x=(1-max_alpha), color='red', linestyle='--')
    ax.text((1-max_alpha), ax.get_ylim()[0] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f'{1-max_alpha:.2f}', color='red', fontsize=8, ha='center')

plt.tight_layout()
plt.show()