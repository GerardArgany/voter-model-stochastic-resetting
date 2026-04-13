"""
Analytical solution for the first passage time (FPT) distribution of the voter
model with stochastic resetting.

This module computes the Laplace transform of the first passage time distribution,
the mean FPT, and the variance of the FPT for the all-to-all voter model under
delta resetting, using an expansion in Gegenbauer polynomials.
"""

import numpy as np
from scipy.special import eval_gegenbauer


def B(n: int, m0: float) -> float:
    """Compute the Gegenbauer expansion coefficient for mode n at reset position m0.

    Args:
        n: Expansion mode index (non-negative integer).
        m0: Reset magnetisation in [-1, 1].

    Returns:
        Expansion coefficient for mode n.
    """
    C = eval_gegenbauer(n, 1.5, m0)
    return (2 * n + 3) / ((n + 1) * (n + 2)) * C


def la(n: int, N: int) -> float:
    """Compute the relaxation rate for mode n in a system of size N.

    Args:
        n: Mode index.
        N: System size (number of agents).

    Returns:
        Relaxation rate lambda_n = 2n(n+1).
    """
    return 2 * n * (n + 1)


def dist_laplace(N: int, m0: float, r: float, s: float, M: int = 1000) -> float:
    """Evaluate the Laplace transform of the FPT distribution at frequency s.

    Args:
        N: System size (number of agents).
        m0: Initial (reset) magnetisation in [-1, 1].
        r: Resetting rate (will be rescaled by 1/N internally).
        s: Laplace variable (frequency).
        M: Number of Gegenbauer modes to sum. Default is 1000.

    Returns:
        Value of the Laplace-transformed FPT distribution at s.
    """
    r = r / N

    i = np.arange(M)
    n_even = 2 * i
    n_odd = n_even + 1

    b_vals = B(n_even, m0)
    la_vals = la(n_odd, N)

    denom = s + r + la_vals

    s1 = np.sum(b_vals * la_vals / denom)
    s2 = np.sum(b_vals * (s + la_vals) / denom)

    return s1 / s2


def sol_fpt(N: int, m0: float, r: float, s_vals: np.ndarray, M: int = 1000) -> list:
    """Compute the Laplace transform of the FPT distribution at multiple frequencies.

    Args:
        N: System size (number of agents).
        m0: Initial (reset) magnetisation in [-1, 1].
        r: Resetting rate.
        s_vals: Array of Laplace-space frequencies at which to evaluate.
        M: Number of Gegenbauer modes to sum. Default is 1000.

    Returns:
        List of Laplace-transformed FPT values corresponding to each element of s_vals.
    """
    return [dist_laplace(N, m0, r, s, M) for s in s_vals]


def mean_fpt(N: int, m0: float, r: float, M: int = 1000) -> float:
    """Compute the mean first passage time to consensus.

    Uses the equivalent ratio-of-sums form:

        MFPT(r) = [sum_l B_{2l}/(r + lambda'_{2l+1})]
                  / [sum_l B_{2l} lambda'_{2l+1}/(r + lambda'_{2l+1})]

    with the internal convention r -> r/N.

    Args:
        N: System size (number of agents).
        m0: Initial (reset) magnetisation in [-1, 1].
        r: Resetting rate (will be rescaled by 1/N internally).
        M: Number of Gegenbauer modes to sum. Default is 1000.

    Returns:
        Mean first passage time to consensus (m = ±1).
    """
    r = r / N

    i = np.arange(M)
    n_even = 2 * i
    n_odd = n_even + 1

    b_vals = B(n_even, m0)
    la_vals = la(n_odd, N)

    denom = r + la_vals

    s = np.sum(b_vals / denom)
    s2 = np.sum(b_vals * la_vals / denom)

    return s / s2


def variance_fpt(N: int, m0: float, r: float, M: int = 1000) -> float:
    """Compute the standard deviation of the first passage time to consensus.

    Note: Despite the name ``variance_fpt``, this function returns the
    **standard deviation** (square root of the variance), not the variance
    itself. The name is kept for backward compatibility with the original
    scripts.

    Args:
        N: System size (number of agents).
        m0: Initial (reset) magnetisation in [-1, 1].
        r: Resetting rate (will be rescaled by 1/N internally).
        M: Number of Gegenbauer modes to sum. Default is 1000.

    Returns:
        Standard deviation of the first passage time.
    """
    r = r / N

    i = np.arange(M)
    n_even = 2 * i
    n_odd = n_even + 1

    b_vals = B(n_even, m0)
    la_vals = la(n_odd, N)

    denom = r + la_vals
    s = np.sum(b_vals / denom)
    s2 = np.sum(b_vals * la_vals / denom)
    s3 = np.sum(b_vals / (denom ** 2))

    mean = s / s2

    prefactor = 1 - m0 ** 2
    real_denom_sq = (prefactor * s2) ** 2
    real_num = 2 * prefactor * s3
    second_moment = real_num / real_denom_sq

    return np.sqrt(second_moment - mean ** 2)
