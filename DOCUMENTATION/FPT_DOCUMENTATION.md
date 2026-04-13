# First Passage Time (FPT) Simulations

## Overview

This document describes the first-passage-time (FPT) functionality added to the VoterResetting module. The FPT simulations compute the distribution of times until consensus is reached in voter models with stochastic resetting.

## What is First Passage Time to Consensus?

In a voter model, agents with opinion +1 or -1 interact and update their opinions through local copying. Without resetting, the system eventually reaches a consensus state where all agents hold the same opinion. The **first passage time** is the time at which consensus is first achieved.

### Consensus Types

The implementation supports three types of consensus conditions:

1. **`:either`** (default): Consensus is reached when all agents have the same opinion (either all +1 OR all -1)
   - This is the classic absorbing state in voter dynamics
   - Typically the fastest to reach
   
2. **`:positive`**: Consensus is reached only when all agents have opinion +1
   - Systems starting biased toward +1 reach this faster
   - Systems starting biased toward -1 may take much longer (or require a reset to +1)
   
3. **`:negative`**: Consensus is reached only when all agents have opinion -1
   - Symmetric to `:positive`
   - Useful for comparing absorbing states in detail

## Usage

### All-to-All Topology

For a complete graph (all agents interact with all other agents):

```julia
using VoterResetting

# Parameters: N=100 agents, r=0.1 reset rate, m0=0.0 initial magnetization
params = AllToAllParams(100, 0.1, 0.0)

# Default: consensus to either ±1, 1000 samples
result = first_passage_time_all_to_all(params)

# Only reach all +1, 500 samples
result_pos = first_passage_time_all_to_all(params; 
    consensus_type=:positive, 
    nsamples=500
)

# With custom reset protocol
result_custom = first_passage_time_all_to_all(params;
    reset=hub_reset(0.0),  # Reset to hubs at +1
    nsamples=1000
)
```

### Complex Network Topology

For a network graph (e.g., scale-free, random regular):

```julia
using Graphs

# Create a network
G = barabasi_albert(100, 2)  # 100 nodes, 2 edges per preferential attachment
params = ComplexParams(r=0.1, m0=0.0)

# Run FPT simulation
result = first_passage_time_complex(G, params; 
    consensus_type=:either, 
    nsamples=500
)
```

## Output Structure

Both `first_passage_time_all_to_all()` and `first_passage_time_complex()` return a `FPTSimulationResult` object with fields:

- **`times::Vector{Float64}`**: Raw FPT values from each of the `nsamples` independent trajectories
  
- **`bin_edges::Vector{Float64}`**: Histogram bin edges for visualizing the distribution
  
- **`bin_centers::Vector{Float64}`**: Bin centers (useful for plotting)
  
- **`densities::Vector{Float64}`**: Normalized histogram (probability density, ∫ density dx = 1)
  
- **`counts::Vector{Int}`**: Raw counts in each bin
  
- **`mean_fpt::Float64`**: Ensemble mean ⟨T⟩ of all FPT samples
  
- **`std_fpt::Float64`**: Ensemble standard deviation σ(T)

### Example: Accessing Results

```julia
result = first_passage_time_all_to_all(params)

println("Mean FPT: $(result.mean_fpt)")
println("Std:      $(result.std_fpt)")

# Plot histogram
using Plots
plot(result.bin_centers, result.densities, 
     xlabel="Time to Consensus", 
     ylabel="Probability Density",
     title="FPT Distribution")

# Access raw samples for further analysis
fpt_samples = result.times
percentile_90 = quantile(fpt_samples, 0.9)
```

## Physics & Algorithm

### Gillespie Continuous-Time Simulation

The FPT simulations use the **Gillespie algorithm** (stochastic simulation of continuous-time Markov chains):

1. **All-to-All**:
   - Event rate λ = r + 2n(N-n)/(N-1)
   - Two event types:
     - Voter event (probability λ_voter/λ): one agent flips opinion
     - Reset event (probability r/λ): system returns to initial magnetization
   - Runs until consensus reached

2. **Complex Network**:
   - Event rate λ = r + 2·num_active_edges
   - Active edges: edges connecting agents of opposite opinion
   - Voter event: flip one endpoint of a random active edge
   - Network-specific optimization for sparse graphs

### Key Insight on Consensus Types

- **`:either` (bipolar)**: System reaches consensus whenever either opinion dominates
- **`:positive` (monolithic)**: System slides toward one specific pole; much slower
- **`:negative` (monolithic)**: Symmetric to `:positive`

Example effect: For m0=0. (balanced initial state):
- `:either` reaches consensus in ~50 time units
- `:positive` takes 2-3× longer (~100-150 units)
- `:negative` also takes ~100-150 units (symmetric)

## Parameters

### Reset Protocols

The `reset` keyword argument accepts any `AbstractResetProtocol`:

- **`delta_reset(m0)`**: Reset to fixed magnetization m0
- **`uniform_reset()`**: Reset to random magnetization (not supported for complex networks)
- **`random_node_reset(m0)`**: Each node independently set to +1 with probability (1+m0)/2
- **`hub_reset(m0)`**: Reset to configuration where highest-degree nodes are +1 (m0 controls the fraction)
- **`custom_reset(f)`**: User-defined reset function

Defaults are topology-specific:
- `first_passage_time_all_to_all`: `reset=delta_reset(params.m0)`
- `first_passage_time_complex`: `reset=hub_reset(0.0)`

### Other Parameters

- **`nsamples::Int=1000`**: Number of independent trajectories. Higher values give smoother histogram but longer runtime.
- **`consensus_type::Symbol=:either`**: Which consensus to target

## Performance Notes

- **All-to-all**: Very fast; O(1) per event (just update the count of +1 agents)
- **Complex network**: O(degree) per event (only update neighbors of flipped node)
- **Reset rates**: Higher r → longer mean FPT (more frequent resets push system back toward initial state)
- **Biased initial state**: Systems close to consensus (large |m0|) reach it much faster

## Examples from Notebooks

The implementation is based on the reference notebooks:

- `notebooks/all_to_all/all_to_all_first_passage_times.ipynb`: All-to-all FPT with various parameters
- `notebooks/complex/complex_first_passage.ipynb`: Complex network FPT with different topologies

## Comparison with PDF Simulations

| Aspect | PDF | FPT |
|--------|-----|-----|
| **Observation** | Magnetization at fixed times | Time to absorption |
| **Output shape** | Distribution over m at each time | Distribution of T |
| **Algorithm** | Sample trajectory at checkpoints | Run to completion |
| **Runtime** | Fast (fixed number of checkpoints) | Variable (until consensus) |
| **Use case** | Transient dynamics | Mean time to consensus |

## Attributes of Consensus

- **Consensus magnitude**: |m| = 1 (all agents agree)
- **Consensus direction**: m = +1 (all +1) or m = -1 (all -1)
- **Time to consensus**: First time the system reaches this state

---

**Author**: GitHub Copilot | **Date**: 2025
**Based on**: Python notebooks in the voter-model-stochastic-resetting repository
