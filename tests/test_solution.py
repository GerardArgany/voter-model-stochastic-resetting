"""Tests for voter_model.solution (delta resetting)."""

import numpy as np
import pytest
from voter_model import solution as sol


def test_sol_returns_array():
    """sol() should return a numpy array of the correct length."""
    result = sol.sol(N=100, r=0.1, m0=0.0, bins=50)
    assert isinstance(result, np.ndarray)
    assert len(result) == 50


def test_sol_non_negative():
    """All values returned by sol() should be non-negative."""
    result = sol.sol(N=100, r=0.1, m0=0.0, bins=50)
    assert np.all(result >= 0)


def test_pip1_between_zero_and_one():
    """pip1() should return a value in [0, 1]."""
    p = sol.pip1(N=100, r=0.1, m0=0.5)
    assert 0.0 <= p <= 1.0


def test_pim1_between_zero_and_one():
    """pim1() should return a value in [0, 1]."""
    p = sol.pim1(N=100, r=0.1, m0=0.5)
    assert 0.0 <= p <= 1.0


def test_boundary_probabilities_sum_leq_one():
    """pip1 + pim1 should not exceed 1."""
    p_plus = sol.pip1(N=100, r=0.1, m0=0.0)
    p_minus = sol.pim1(N=100, r=0.1, m0=0.0)
    assert p_plus + p_minus <= 1.0 + 1e-10


def test_la_scaling():
    """Relaxation rate should scale as n(n+1)/N."""
    assert sol.la(2, 100) == pytest.approx(6 / 100)
    assert sol.la(3, 200) == pytest.approx(12 / 200)


def test_kernel_stationary():
    """At t=inf the kernel should equal -r/(lambda+r)."""
    N, r, n = 100, 0.1, 2
    lam = sol.la(n, N)
    expected = -r / (lam + r)
    assert sol.kernel(n, N, r, np.inf) == pytest.approx(expected)


def test_sol_symmetric_for_zero_m0():
    """For m0=0 the distribution should be symmetric around m=0."""
    result = sol.sol(N=200, r=0.5, m0=0.0, bins=100)
    assert np.allclose(result, result[::-1], atol=1e-6)
