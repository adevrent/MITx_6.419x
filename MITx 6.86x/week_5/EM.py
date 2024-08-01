import numpy as np
from scipy.stats import norm

def numerator(x, p, mu, sigma2):  # p, mu, sigma2 scalar
    return norm.pdf(x, mu, np.sqrt(sigma2)) * p

def denominator(x, P, Mu, Sigma2):  # all can be vectors
    k = len(P)
    prob_sum = 0
    for j in range(k):
        prob_sum += numerator(x, P[j], Mu[j], Sigma2[j])
        
    return prob_sum

def calc_posterior(X, P, Mu, Sigma2):
    denom = denominator(X, P, Mu, Sigma2)
    num_array = np.array([numerator(X, P[j], Mu[j], Sigma2[j]) for j in range(len(P))])
    return num_array / denom

X = np.array([0.2, -0.9, -1, 1.2, 1.8])
P = np.array([0.5, 0.5])
Mu = np.array([-3, 2])
Sigma2 = np.array([4, 4])

print(calc_posterior(X, P, Mu, Sigma2))

mysum = calc_posterior(X, P, Mu, Sigma2)[0].sum() / len(X)
print("mysum =", mysum)
mean = np.dot(calc_posterior(X, P, Mu, Sigma2)[0], X) / calc_posterior(X, P, Mu, Sigma2)[0].sum()
print("mean =", mean)
variance = np.dot(calc_posterior(X, P, Mu, Sigma2)[0], ((X - mean)**2)) / calc_posterior(X, P, Mu, Sigma2)[0].sum()
print("variance =", variance)