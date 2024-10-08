import numpy as np
import kmeans
import common
import naive_em
import em
import pandas as pd

# X = np.loadtxt(r"C:\Users\adevr\MITx_6.419x\MITx 6.86x\week_6\netflix\toy_data.txt")
X = np.loadtxt(r"C:\Users\adevr\MITx_6.419x\MITx 6.86x\week_6\netflix\netflix_incomplete.txt")

K_arr = [1, 12]
mixtures_arr = []
posts_arr = []
costs_dict = {}

# # kmeans
# for seed in [0, 1, 2, 3, 4]:
#     costs_dict[seed] = []
#     for K in K_arr:
#         mixture, post = common.init(X, K, seed)
#         costs_dict[seed].append(kmeans.run(X, mixture, post)[-1])  # store the costs for K = 1, 2, 3, 4

# em
for seed in [0, 1, 2, 3, 4]:
    costs_dict[seed] = []
    for K in K_arr:
        mixture, post = common.init(X, K, seed)
        mixture, LL = em.run(X, mixture, post)
        costs_dict[seed].append(LL)  # store the costs for K = 1, 2, 3, 4

# # naive_em + BIC
# for seed in [0, 1, 2, 3, 4]:
#     costs_dict[seed] = []
#     for K in K_arr:
#         mixture, post = common.init(X, K, seed)
#         log_likelihood = naive_em.run(X, mixture, post)[-1]
#         costs_dict[seed].append(common.bic(X, mixture, log_likelihood))  # store the costs for K = 1, 2, 3, 4
        
df = pd.DataFrame(costs_dict)
df.index = [f"K={i}" for i in [1, 12]]
print(df)
print(df.max(axis=1))