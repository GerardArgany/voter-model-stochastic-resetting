# Discrete-Time Voter Model: Implementation Guide

## Overview

This implementation provides **discrete-time simulations** for the voter model with stochastic resetting on complex networks. Unlike the continuous-time (Gillespie) simulations in the main module, here **r is a reset probability per step** rather than a Poisson rate.

## Key Differences: Discrete vs. Continuous Time

### Discrete-Time Model (New)
- **Reset mechanism**: At each discrete step, the system resets with probability `r` 
- **Time unit**: Steps are the natural time variable (step 1, 2, 3, ...)
- **Voter update**: One random active edge flips per step
- **Dynamics**: Synchronous, deterministic step structure
- **Use case**: Suitable for synchronous update rules, discrete time-steps

### Continuous-Time Model (Gillespie, Original)
- **Reset mechanism**: Poisson process with rate `r`
- **Time unit**: Continuous time (0, 0.5, 1.0, ...)
- **Voter update**: Variable number of events per unit time (via Gillespie sampling)
- **Dynamics**: Asynchronous, event-driven
- **Use case**: Physical systems with rate-based processes

## Quick Start

### Load the Module

```julia
using Graphs, Random, Statistics, Plots

# Load from notebook
project_root = dirname(dirname(pwd()))
include(joinpath(project_root, "src", "VoterResetting.jl"))

# All discrete-time functions are available in the VoterResetting namespace
# - simulate_pdf_discrete_complex()
# - first_passage_time_discrete_complex()
```

### Run a Simple Simulation

```julia
# Create a network
N = 200
mu = 15  # average degree
G = erdos_renyi(N, mu / (N - 1))

# Set parameters
# Note: r is now a PROBABILITY per step (not a rate)
r = 0.1           # 10% chance of reset per step
m0 = 0.0          # start from balanced magnetization
max_steps = 100   # simulate for 100 steps
nsamples = 300    # 300 independent realizations

# Create parameter struct
params = VoterResetting.ComplexParams(r, m0)

# Run discrete-time PDF simulation
times = [max_steps]  # observation time
samples = VoterResetting.simulate_pdf_discrete_complex(
    G, params, times, nsamples;
    reset=VoterResetting.delta_reset(m0)
)

# Extract magnetization samples
m_final = vec(samples[:, end])
println("Mean magnetization: $(mean(m_final))")
println("Std magnetization: $(std(m_final))")
```

### Compute First Passage Time to Consensus

```julia
# Estimate distribution of hitting times to consensus
fpt_samples = VoterResetting.first_passage_time_discrete_complex(
    G, params;
    consensus_type=:either,    # stop at any consensus
    nsamples=500,
    max_steps=1000,
    reset=VoterResetting.delta_reset(m0)
)

# Compute MFPT
mfpt = mean(fpt_samples)
println("Mean FPT: $(mfpt) steps")
```

## Core Functions

### `simulate_pdf_discrete_complex(...)`

**Purpose**: Sample probability distribution function (PDF) of magnetization at fixed times.

**Signature**:
```julia
simulate_pdf_discrete_complex(
    graph::AbstractGraph, 
    params::ComplexParams,
    times::Vector{Float64},           # observation times (in steps)
    nsamples::Int;                    # number of independent trajectories
    reset::AbstractResetProtocol      # reset protocol (e.g., delta_reset(m0))
) → samples::Matrix{Float64}          # (nsamples × ntimes) magnetizations
```

**Returns**: Matrix where `samples[i, t]` is the magnetization in trajectory `i` at observation time `times[t]`.

**Example**:
```julia
params = VoterResetting.ComplexParams(0.1, 0.0)
samples = VoterResetting.simulate_pdf_discrete_complex(
    G, params, [10, 50, 100], 200;
    reset=VoterResetting.delta_reset(0.0)
)
# samples is 200×3: 200 samples at times [10, 50, 100]
```

### `first_passage_time_discrete_complex(...)`

**Purpose**: Compute first passage times to consensus.

**Signature**:
```julia
first_passage_time_discrete_complex(
    graph::AbstractGraph,
    params::ComplexParams;
    consensus_type=:either,          # :either, :positive, :negative
    nsamples=1000,                   # number of trajectories
    max_steps=10000,                 # max steps before giving up
    reset::AbstractResetProtocol     # reset protocol
) → fpt_samples::Vector{Float64}    # hitting times for each trajectory
```

**Returns**: Vector of FPT values (in steps) for each sample.

**Example**:
```julia
fpt = VoterResetting.first_passage_time_discrete_complex(
    G, params;
    consensus_type=:either,
    nsamples=500
)
mfpt = mean(fpt)
```

## Demonstration Notebooks

Three example notebooks are provided in `notebooks/complex/`:

### 1. **discrete_time_pdf_parameter_sweep.ipynb**

Comprehensive parameter sweep over:
- **Topologies**: ER (Erdős-Rényi), RRG (Random Regular), BA (Barabási-Albert)
- **Degrees**: μ ∈ {6, 20, 30}
- **Reset probabilities**: r ∈ {0.0, 0.1, 0.5, 1.0}
- **Initial conditions**: m_reset ∈ {0.0, 0.5, 1.0}

Creates PDF histograms and summary statistics.

**Run**: Open notebook, set `quick_mode = false` for full sweep (or `true` for quick test).

### 2. **discrete_time_mfpt_complex.ipynb**

MFPT analysis:
- Compute MFPT for range of initial magnetizations m₀
- Plot MFPT curves for different reset probabilities
- Compare across topologies
- Logarithmic scale to show spanning orders of magnitude

**Key output**: How does r affect consensus time?

### 3. **discrete_time_validation.ipynb**

Validation and comparison:
- Compare discrete-time vs Gillespie (continuous-time) PDFs
- Show when they agree and where they diverge
- Benchmark computational costs
- Insights on equivalence regimes

**Key use**: Understand differences and choose appropriate model.

## Physics: Discrete-Time Stepping Details

### Network Update Rules (Complex Graph)

At each discrete time step:

```
1. With probability r:
   → RESET: all nodes return to reset state
   → Rebuild active-edge list
   
2. Else (probability 1-r):
   If there are active edges (nodes in different states):
     → Pick a random active edge (u, v)
     → Pick one endpoint at random, flip to match other
     → Update incident edges efficiently
   Else:
     → No change (system in consensus)
```

### Edge List Optimization

The active-edge list is maintained efficiently:
- **Initialization**: O(E) scan for edges between opposite states
- **After flip**: O(degree) update of incident edges only
- **On removal**: O(1) swap-and-pop deletion
- **On reset**: O(E) rebuild (only when needed)

This gives O(degree) amortized cost per step, same as Gillespie.

## Computational Performance

### Time Complexity

| Operation | Complexity |
|-----------|-----------|
| Initialize state | O(N) |
| One voter step | O(1) amortized |
| Reset event | O(N) on reset |
| PDF computation (n_samples) | O(n_samples × max_steps × avg_degree) |

### Expected Runtime (Approximate)

For N=300, μ=15, 500 samples:
- **Discrete (100 steps)**: ~2-5 seconds
- **Gillespie (t=100)**: ~1-2 seconds
- **Speedup factor**: Generally discrete is slower due to fixed per-step cost

When is discrete faster?
- Large r (reset probability > 0.5): Gillespie spends time on exponential sampling
- Synchronous system requirements: Natural fit for discrete steps
- GPU-friendly: Could be parallelized better than event-driven Gillespie

## Reset Protocols

All protocols from `VoterResetting` module are supported:

```julia
# Fixed magnetization reset (randomized node assignment)
VoterResetting.delta_reset(m0)

# Explicit state vector reset
VoterResetting.delta_reset([1, -1, 1, ..., -1])

# High-degree nodes reset to +1
VoterResetting.hub_reset(m0; highest=true)

# Each node independently reset with probability p
VoterResetting.random_node_reset(m0)

# Custom reset function
VoterResetting.custom_reset(f)
```

## Parameter Interpretation: r (Reset Probability)

### Equivalence to Gillespie Rate

**Discrete-time**: r = probability reset occurs in a step
- Range: [0, 1]
- Per-trajectory: expected resets = r × max_steps

**Gillespie continuous-time**: r = Poisson rate (resets per unit time)
- Range: [0, ∞)
- Per-trajectory: expected resets = r × T (for time interval [0,T])

**Rough equivalence** for small r: If Gillespie step time ≈ 1 unit, then:
- Discrete r_d ≈ Gillespie r_g
- But timing of resets relative to voter events differs

**Recommend**: Use discrete-time notebooks to compare at your specific r values.

## Examples and Workflow

### Workflow 1: PDF at Fixed Time

```julia
G = erdos_renyi(500, 30/(500-1))
params = VoterResetting.ComplexParams(0.1, 0.0)

# Observe at time t=100, 500, 1000 steps
samples = VoterResetting.simulate_pdf_discrete_complex(
    G, params, [100, 500, 1000], 1000;
    reset=VoterResetting.delta_reset(0.0)
)

# Create histogram
using StatsPlots
histogram(samples[:, 3], bins=50, title="PDF at t=1000")
```

### Workflow 2: MFPT Curve

```julia
m0_vals = collect(-0.9:0.1:0.9)
mfpt = zeros(length(m0_vals))

for (i, m0) in enumerate(m0_vals)
    params = VoterResetting.ComplexParams(0.1, m0)
    fpt = VoterResetting.first_passage_time_discrete_complex(
        G, params; nsamples=500
    )
    mfpt[i] = mean(fpt)
end

plot(m0_vals, mfpt, yscale=:log10, xlabel="m₀", ylabel="MFPT (steps)")
```

### Workflow 3: Parameter Sweep

See `discrete_time_pdf_parameter_sweep.ipynb` for full template.

## Troubleshooting

**Issue**: Simulation is very slow (r very small)
- **Reason**: Discrete-time always takes fixed steps; Gillespie would jump over inactive periods
- **Solution**: Use Gillespie for small r; discrete for r > 0.1

**Issue**: Very large/small magnetization changes
- **Reason**: Normal with small N or early in simulation
- **Solution**: Increase nsamples, run longer (increase max_steps)

**Issue**: PDF shows spikes at ±1
- **Reason**: Trajectories reaching consensus and staying (r=0) or resetting (r>0)
- **Solution**: Increase observation time if r>0 (allow system to equilibrate)

## Implementation Details

### File: `src/discrete_time_voter.jl`

Core module functions (included in VoterResetting):
- State initialization via `random_spin_state()`
- Active-edge tracking with `active_edge_ids_from_state()`, `update_incident_edges()`
- Stepping logic with probability checks and state flips
- Reset protocol dispatch (reuse existing infrastructure)

Key optimizations:
✓ Pre-allocated buffers
✓ No allocations in time loop
✓ Integer state representation
✓ In-place active-list updates

## Citing This Implementation

If you use the discrete-time voter model implementation, cite:
- The original VoterResetting module
- This documentation and notebook suite

## Further Development

Potential extensions:
- All-to-all discrete-time (currently only complex networks)
- Parallel batch processing for large sweeps
- GPU acceleration for massive nsamples
- Adaptive time steps (tau-leaping style)
- Hybrid continuo-discrete coupling

## References

- Gillespie algorithm: D.T. Gillespie, J. Phys. Chem. 1976
- Voter model theory: [cite your papers]
- Network models: Erdős-Rényi, Barabási-Albert, random regular graphs

---

**Questions?** See the demonstration notebooks or examine the module code in `src/discrete_time_voter.jl`.
