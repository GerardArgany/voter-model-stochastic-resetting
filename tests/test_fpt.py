"""Tests for voter_model.solution_fpt (first passage time)."""

import numpy as np
import pytest
from voter_model import solution_fpt as fpt


def test_mean_fpt_positive():
    """Mean FPT should be strictly positive."""
    result = fpt.mean_fpt(N=100, m0=0.0, r=0.1)
    assert result > 0


def test_variance_fpt_non_negative():
    """Standard deviation of FPT should be non-negative."""
    result = fpt.variance_fpt(N=100, m0=0.0, r=0.1)
    assert result >= 0


def test_sol_fpt_laplace_at_zero():
    """Laplace transform at s=0 should equal 1 (total probability)."""
    result = fpt.sol_fpt(N=100, m0=0.0, r=0.1, s_vals=[0.0])
    assert result[0] == pytest.approx(1.0, abs=1e-4)


def test_sol_fpt_returns_list():
    """sol_fpt() should return a list of the same length as s_vals."""
    s_vals = [0.0, 0.1, 0.5, 1.0]
    result = fpt.sol_fpt(N=100, m0=0.0, r=0.1, s_vals=s_vals)
    assert len(result) == len(s_vals)


def test_sol_fpt_decreasing_in_s():
    """Laplace transform of a positive distribution should decrease with s (approximately)."""
    s_vals = np.linspace(0.01, 2.0, 20)
    result = np.array(fpt.sol_fpt(N=100, m0=0.0, r=0.1, s_vals=s_vals))
    # Allow small numerical noise from the truncated series
    assert np.all(np.diff(result) <= 1e-5)


def test_mean_fpt_increases_with_N():
    """Mean FPT should grow with system size."""
    mfpt_small = fpt.mean_fpt(N=50, m0=0.0, r=0.05)
    mfpt_large = fpt.mean_fpt(N=200, m0=0.0, r=0.05)
    assert mfpt_large > mfpt_small


def test_la_scaling():
    """Relaxation rate should scale as n(n+1)/N."""
    assert fpt.la(2, 100) == pytest.approx(6 / 100)
