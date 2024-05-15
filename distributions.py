import numpy as np
import scipy as sp
import scipy.stats as st

def pmf_binomial(n, p, k):
    return sp.special.comb(n, k) * p**k * (1-p)**(n-k)

# print("answer =", pmf_binomial(31000, 0.00203, 63))


# Gives inf error idk why
# def pmf_hypergeometric(N, K, n, x):
#     """
#     Calculates the probability that exactly x number of 1's in treatment group out of a total
#     of n number of 1's, with a grand total of N number of r.v.s, K of them belonging to the treatment group.
    
#     N (int): total number of bernoulli random variables.
#     K (int): number of bernoulli r.v. in treatment group
#     n (int): total number of 1's
#     x (int): number of 1's in treatment group
#     """
#     return (sp.special.comb(K, x) * sp.special.comb(N-K, n-x)) / (sp.special.comb(N, n))

# print("hypergeom pmf value =", pmf_hypergeometric(62000, 31000, 102, 39))
# print(sp.special.comb(62000, 102))

table = np.array([[39, 63], [30961, 30937]])
print(st.fisher_exact(table, "less"))

print("cdf:", st.norm.cdf(-3.0268))