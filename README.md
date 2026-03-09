# voter-model-stochastic-resetting

Analytical solutions and simulations for the **all-to-all voter model with stochastic resetting**.

The voter model is a paradigmatic model for opinion dynamics and consensus formation.
This repository implements the theoretical solution for the magnetisation distribution
and the first passage time (FPT) to consensus under delta (point) stochastic resetting,
and compares them against Monte Carlo simulations.

## Installation

```bash
# Clone the repository
git clone https://github.com/GerardArgany/voter-model-stochastic-resetting.git
cd voter-model-stochastic-resetting

# Install the package and its dependencies
pip install -e .

# To also install development tools (pytest, black, flake8):
pip install -e ".[dev]"
```

## Package Structure

```
voter_model/
├── __init__.py        # Package initialisation
├── solution.py        # Stationary/time-dependent distribution (delta resetting)
├── solution_fpt.py    # First passage time distribution and statistics
└── utils.py           # Utility functions (e.g. empirical Laplace transform)

notebooks/             # Jupyter notebooks with simulations and figures
scripts/               # Original standalone scripts
tests/                 # Unit tests
```

## Quick Start

```python
import numpy as np
from voter_model import solution as sol
from voter_model import solution_fpt as fpt

# Stationary magnetisation distribution
N, r, m0, bins = 1000, 0.1, 0.0, 71
p = sol.sol(N, r, m0, bins)

# Mean first passage time to consensus
mfpt = fpt.mean_fpt(N, m0, r)
print(f"Mean FPT: {mfpt:.2f} MC steps")
```

## Running Tests

```bash
pytest tests/
```

## Dependencies

- numpy, scipy, matplotlib
- networkx, numba
- jupyter, tqdm
