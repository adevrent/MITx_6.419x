import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt(r"C:\Users\adevr\MITx_6.419x\MITx 6.86x\week_6\netflix\toy_data.txt")

K_arr = [1, 2, 3, 4]
mixtures_arr = []
posts_arr = []
costs_dict = {}

for seed in [0, 1, 2, 3, 4]:
    costs_dict[seed] = []
    for K in K_arr:
        mixture, post = common.init(X, K, seed)
        costs_dict[seed].append(kmeans.run(X, mixture, post)[-1])  # store the costs for K = 1, 2, 3, 4

total_cost_sum = []
for key, values in costs_dict.items():
    total_cost_sum.append(sum(values))
    
print(total_cost_sum)