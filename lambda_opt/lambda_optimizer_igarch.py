import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import yfinance as yf
import datetime

def optimize_lambda(ticker, years=1):
    # Dates
    end = datetime.date.today()
    timedelta = datetime.timedelta(365*years)
    start = end - timedelta

    # Download daily data
    df = pd.DataFrame(yf.download(ticker, start, end)["Close"])
    # print("df =\n", df)

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["log_return"] = df["log_return"].fillna(0)  # fill the first return value with 0
    df["log_return_sq"] = df["log_return"]**2

    print(df)

    # Define the negative log-likelihood function
    def negative_log_likelihood(x, var, mu=0, epsilon=1e-10):
        var = np.maximum(var, epsilon)  # Ensure variance is not zero
        return -(-0.5 * np.log(2 * np.pi * var) - 0.5 * (x - mu)**2 / var)

    # Define the objective function to minimize
    def sum_negative_log_likelihood(alpha, x_arr, initial_var):
        var_arr = np.zeros_like(x_arr)
        var_arr[0] = initial_var
        for i in range(1, len(x_arr)):
            var_arr[i] = alpha * x_arr[i-1]**2 + (1 - alpha) * var_arr[i-1]
        # print(var_arr)
        nll_sum = np.sum(negative_log_likelihood(x_arr, var_arr))
        return nll_sum

    x_arr = df["log_return"]
    initial_var = df["log_return"].var(ddof=1)  # unbiased variance estimator
    print("    initial_var =", initial_var)

    # Minimize the objective function with respect to alpha
    result = minimize(sum_negative_log_likelihood, x0=0.2, args=(df["log_return"].values, initial_var), bounds=[(0, 1)])
    optimal_alpha = result.x[0]
    optimal_lambda = 1-optimal_alpha
    print("    optimal_alpha =", optimal_alpha)
    print("    optimal_lambda =", optimal_lambda)

    # Plot
    initial_var_arr = np.linspace(initial_var, initial_var*5, 10)
    alpha_arr = np.linspace(0, 1, 50)
    # print("alpha_arr", alpha_arr)

    fig, axs = plt.subplots(1, len(initial_var_arr), figsize=(18, 4))

    sum_log_likelihood_array = []
    for initial_var in initial_var_arr:
        sum_log_likelihood_array.append([-sum_negative_log_likelihood(alpha, df["log_return"], initial_var) for alpha in alpha_arr])

    for i, ax in enumerate(axs):
        ax.plot((1-alpha_arr), sum_log_likelihood_array[i])
        ax.set_title(f"initial_var = {np.round(initial_var_arr[i], 4)}", fontsize=7)
        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel('$L$')

        # Find the alpha that maximizes the sum of log likelihood
        max_index = np.argmax(sum_log_likelihood_array[i])
        max_alpha = alpha_arr[max_index]
        ax.axvline(x=(1-max_alpha), color='red', linestyle='--')
        ax.text((1-max_alpha), ax.get_ylim()[0] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]), f'{1-max_alpha:.2f}', color='red', fontsize=8, ha='center')

    plt.tight_layout()
    plt.show()
    
    return optimal_lambda

# Call function
ticker="USDTRY=X"
print(optimize_lambda(ticker, 3))