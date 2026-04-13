"""
Analytical solution for the voter model with delta (point) stochastic resetting.

This module computes the stationary and time-dependent magnetisation distribution
for the all-to-all voter model under delta resetting, using an expansion in
Gegenbauer polynomials.
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
        Relaxation rate lambda_n = 2n(n+1)/N.
    """
    return 2 * n * (n + 1) / N


def kernel(n: int, N: int, r: float, t: float) -> float:
    """Compute the time-dependent kernel for mode n.

    Args:
        n: Mode index.
        N: System size (number of agents).
        r: Resetting rate per MC step.
        t: Evaluation time. Use np.inf for the stationary state.

    Returns:
        Kernel value at time t.
    """
    lam = la(n, N)
    if np.isinf(t):
        return -r / (lam + r)
    return -lam / (lam + r) * np.exp(-(lam + r) * t) - r / (lam + r)


def pip1(N: int, r: float, m0: float, t: float = np.inf, M: int = 600) -> float:
    """Compute the probability of being in the fully ordered state m = +1.

    Args:
        N: System size (number of agents).
        r: Resetting rate per MC step.
        m0: Reset magnetisation in [-1, 1].
        t: Evaluation time. Default is np.inf (stationary state).
        M: Number of Gegenbauer modes to sum. Default is 600.

    Returns:
        Probability mass at m = +1.
    """
    A = (1 + m0) / 2
    s = np.sum([B(n, m0) * kernel(n + 1, N, r, t) for n in range(M)])
    return A * (1 + (1 - m0) * s)


def pim1(N: int, r: float, m0: float, t: float = np.inf, M: int = 600) -> float:
    """Compute the probability of being in the fully ordered state m = -1.

    Args:
        N: System size (number of agents).
        r: Resetting rate per MC step.
        m0: Reset magnetisation in [-1, 1].
        t: Evaluation time. Default is np.inf (stationary state).
        M: Number of Gegenbauer modes to sum. Default is 600.

    Returns:
        Probability mass at m = -1.
    """
    C = (1 - m0) / 2
    s = np.sum([(-1) ** n * B(n, m0) * kernel(n + 1, N, r, t) for n in range(M)])
    return C * (1 + (1 + m0) * s)


def fk(N: int, r: float, m0: float, m: float, t: float = np.inf, M: int = 600) -> float:
    """Compute the bulk probability density at magnetisation m.

    Args:
        N: System size (number of agents).
        r: Resetting rate per MC step.
        m0: Reset magnetisation in [-1, 1].
        m: Magnetisation value at which to evaluate the density, in (-1, 1).
        t: Evaluation time. Default is np.inf (stationary state).
        M: Number of Gegenbauer modes to sum. Default is 600.

    Returns:
        Probability density at m.
    """
    s = 0.0
    for n in range(M):
        Cn = eval_gegenbauer(n, 1.5, m)
        s += B(n, m0) * kernel(n + 1, N, r, t) * Cn
    return -0.5 * (1 - m0 ** 2) * s


def sol(N: int, r: float, m0: float, bins: int, t: float = np.inf) -> np.ndarray:
    """Compute the magnetisation distribution over a discretised grid.

    Evaluates the full probability distribution (bulk density plus boundary
    atoms at m = ±1) on a uniform grid of ``bins`` points in (-1, 1).

    Args:
        N: System size (number of agents).
        r: Resetting rate per MC step.
        m0: Reset magnetisation in [-1, 1].
        bins: Number of histogram bins.
        t: Evaluation time. Default is np.inf (stationary state).

    Returns:
        Array of shape (bins,) with the probability density for each bin.

    Notes:
        The extreme points m = ±1 are excluded from the bulk grid because
        the Gegenbauer series converges slowly there. Instead, the boundary
        probabilities pip1 and pim1 are added to the first and last bins.
    """
    # Avoid the extremes where the series converges slowly
    m = np.linspace(-1 + 0.1 / bins, 1 - 0.1 / bins, bins)
    out = [fk(N, r, m0, mi, t) for mi in m]

    out[0] += pim1(N, r, m0, t) / (2 / bins)
    out[-1] += pip1(N, r, m0, t) / (2 / bins)

    return np.array(out)
