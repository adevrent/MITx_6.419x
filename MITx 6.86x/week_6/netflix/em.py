"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a Gaussian component, handling missing values.

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    # Get params from GM class
    mu = mixture.mu  # (K, d)
    var = mixture.var  # (K,)
    p = mixture.p  # (K,)
    
    # Assign shapes
    n, d = X.shape
    K = p.shape[0]

    # Create a mask for the missing data (entries set to 0 are considered missing)
    mask = (X != 0)  # Boolean array of shape (n, d), True for non-missing entries
    
    # Initialize arrays for posterior and log-likelihood
    post = np.zeros((n, K))
    LL = 0.0

    # Loop through each Gaussian component
    for k in range(K):
        # Reshape mu for current component to (1, d)
        mu_k = mu[k].reshape(1, d)

        # Compute Mahalanobis distance considering only valid dimensions
        diff = (X - mu_k)  # Shape (n, d)
        diff[~mask] = 0  # Zero out the invalid (missing) dimensions

        # Adjust Mahalanobis distance and Gaussian density for missing data
        mahalanobis_dist = -0.5 * np.sum((diff**2) / var[k], axis=1)  # Shape (n,)
        
        # Adjust the Gaussian density, only for valid dimensions (ignore missing)
        valid_dims = mask.sum(axis=1)  # Number of valid dimensions per observation, shape (n,)
        det_adjusted = var[k]**valid_dims  # Adjust determinant for valid dimensions, shape (n,)
        normalizer = (2 * np.pi)**(-valid_dims / 2) * det_adjusted**(-0.5)  # Shape (n,)

        # Calculate the Gaussian density for each observation
        N_k = normalizer * np.exp(mahalanobis_dist)  # Shape (n,)

        # Multiply by the mixing coefficient p[k]
        pre_post_k = p[k] * N_k  # Shape (n,)

        # Add the log of this component's density to the log-likelihood
        post[:, k] = pre_post_k

    # Normalize to get the posterior (soft counts)
    sum_post = np.sum(post, axis=1, keepdims=True)
    post = post / sum_post  # Normalize to ensure the probabilities sum to 1

    # Compute log-likelihood
    LL = np.sum(np.log(sum_post))

    return post, LL



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]
    
    mask = (X != 0)  # False for missing entries.
    
    p = post.mean(axis=0)  # (K,)
    
    mu = mixture.mu  # Store old mu
    var = np.zeros(K)  # Initialize var array
    
    # Loop over K clusters
    for k in range(K):
        # Calculate the mu_mask
        mu_mask = ((post[:, k].reshape(-1, 1) * mask).sum(axis=0, keepdims=True) >= 1)  # Shape (1, d)
        
        # Proceed only if there are valid dimensions to update
        if np.any(mu_mask):
            # Calculate numerator and denominator for valid dimensions
            numerator = (post[:, k].reshape(-1, 1) * mask * X).sum(axis=0, keepdims=True)  # Shape (1, d)
            denominator = (post[:, k].reshape(-1, 1) * mask).sum(axis=0, keepdims=True)  # Shape (1, d)
            
            # Apply the mask to both numerator and denominator to ensure shapes match
            mu[k, mu_mask[0, :]] = numerator[0, mu_mask[0, :]] / denominator[0, mu_mask[0, :]]
        
        # Compute the variance for valid dimensions
        valid_dims = mask.sum(axis=1)  # Number of valid dimensions per observation, shape (n,)
        var_denom = (post[:, k] * valid_dims).sum(axis=0)
        
        # Compute difference and zero out invalid dimensions
        diff = (X - mu[k, :])
        diff[~mask] = 0
        
        # Calculate variance numerator and update
        var_num = (post[:, k] * (diff**2).sum(axis=1)).sum(axis=0)
        var[k] = var_num / var_denom
    
    # Assign a minimum variance threshold
    var[var < min_variance] = min_variance
    
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
        mixture = mstep(X, post, mixture)
        if (LL - LL_old) / np.abs(LL) < 1e-6:
            break
        LL_old = LL
    return mixture, LL


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
