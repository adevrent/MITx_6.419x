"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    # Get params from GM class
    mu = mixture.mu
    var = mixture.var
    p = mixture.p
    
    # Assign shaopes
    n, d = X.shape
    K = p.shape[0]
    
    det = var**d  # (K,) determinants are equal to sigma**d for each vector
    det = det[:, np.newaxis, np.newaxis]  # Converted to shape (K, 1, 1)
    inverse_var = 1/var
    
    # Create a (K, d, d) matrix with inverse var as diagonal elements
    inverse_var_tensor = np.einsum('k,ij->kij', inverse_var, np.eye(d))  # (K, d, d)
    
    mu_tensor = mu.reshape((K, 1, d))  # different clusters 1,...,K now corresponds to DEPTH.
    X_tensor = X.reshape((1, n, d))
    
    X_mu_tensor = X_tensor - mu_tensor  # This is of shape (K, n, d)
    distance_term = -0.5 * np.einsum('kni,kij,knj->kn', X_mu_tensor, inverse_var_tensor, X_mu_tensor)  # (K, n) distance term in the PDF
    distance_term = distance_term[:, :, np.newaxis]  # converted to shape (K, n, 1)
    
    N_tensor = (2*np.pi)**(-d/2) * det**(-0.5) * np.exp(distance_term)  # (K, n, 1)  tensor where
                                                                        # rows are observations, depth is K clusters.
                                                                        # All elements are p(x(i) | cluster j)
    N_tensor = (np.squeeze(N_tensor, axis=-1)).T  # Get rid of the last dimension and transpose to convert to
                                                  # shape (n, K)
                                                  
    pre_post = N_tensor * p[np.newaxis, :]  # p vector of shape (K,) converted to (1, K)
                                        # pre_post has shape (n, K)
    post = pre_post / pre_post.sum(axis=1, keepdims=True)  # divide by denominator (sum over K clusters, collapsing the K clusters dimension)
                                                            # post still has shape (n, K)
    LL = np.sum(post * np.log(pre_post / post))  # This will summ ALL elements of the (n, K) matrix,
                                          # Summing over both observations and different clusters,
                                          # Resulting in a scalar value.
                                          
    return post, LL
                                              


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]
    
    p = post.mean(axis=0)  # (K,)
    
    mu = ((X.T @ post) / post.sum(axis=0, keepdims=True)).T  # (K, d)
    
    pre_var = X[np.newaxis, :, :] - mu[:, np.newaxis, :]  # (K, n, d)
    var_norm = (pre_var**2).sum(axis=2)  # (K, n)
    var = (var_norm.T * post).sum(axis=0) / (d * post.sum(axis=0))  # (K,)
    
    return GaussianMixture(mu=mu, var=var, p=p)

def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    LL_old = -1e6
    while True:
        post, LL = estep(X, mixture)
        mixture = mstep(X, post)
        if (LL - LL_old) / np.abs(LL) < 1e-6:
            break
        LL_old = LL
    return mixture, LL