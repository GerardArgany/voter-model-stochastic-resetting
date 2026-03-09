"""
voter_model - Voter model with stochastic resetting.

This package provides analytical solutions and utilities for the voter model
with stochastic resetting on all-to-all networks.
"""

__version__ = "0.1.0"

from voter_model import solution, solution_fpt, utils

__all__ = ["solution", "solution_fpt", "utils"]
