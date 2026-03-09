"""
Utility functions for the voter model with stochastic resetting.
"""

import numpy as np


def laplace_from_samples(T: np.ndarray, s: float) -> float:
    """Estimate the Laplace transform of a distribution from samples.

    Computes the empirical average of exp(-s * T) over the provided samples,
    which is an unbiased estimator of the Laplace transform E[e^{-sT}].

    Args:
        T: Array of samples drawn from the distribution of interest.
        s: Laplace variable (frequency) at which to evaluate.

    Returns:
        Empirical estimate of the Laplace transform at s.
    """
    return np.mean(np.exp(-s * T))
