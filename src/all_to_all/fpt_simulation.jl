# =============================================================================
# all_to_all/fpt_simulation.jl
# =============================================================================
#
# First-passage-time simulation for voter model with resetting on all-to-all
# (complete graph) topologies.  Unlike pdf_simulation.jl which observes the
# magnetization at fixed times, FPT returns the distribution of hitting times
# to consensus (absorbing state where all agents agree).
#
# Physics:
#   The system evolves until |m(t)| = 1 (all +1 or all -1), or optionally
#   until a specific consensus is reached.  Using Gillespie algorithm with
#   event rate λ = r + 2n(1 - n/N) where the first term is resetting rate and
#   the second is voter dynamics.
#
# =============================================================================

"""
    first_passage_time_all_to_all(params; consensus_type=:either, nsamples=1000, reset=delta_reset(params.m0))

Estimate the distribution of first-passage times (FPT) to consensus on the 
all-to-all topology using Gillespie simulation.

# Arguments
- `params::AllToAllParams`: Parameters (N, r, m0)
- `consensus_type::Symbol`: Which consensus to target:
  - `:positive` : all agents reach +1 (all heads)
  - `:negative` : all agents reach -1 (all tails)
  - `:either`   : first to reach either (default, classic absorption)
- `nsamples::Int=1000`: Number of independent trajectories to run
- `reset::AbstractResetProtocol`: Resetting protocol (default: delta reset to m0)

# Returns
`FPTSimulationResult` containing:
- `times::Vector{Float64}`: [nsamples] raw FPT values
- `bin_edges`, `bin_centers`: Histogram structure for plotting
- `densities::Vector{Float64}`: Normalized histogram
- `counts::Vector{Int}`: Raw bin counts
- `mean_fpt::Float64`: Ensemble mean time
- `std_fpt::Float64`: Ensemble standard deviation

# Example
```julia
params = AllToAllParams(N=100, r=0.5, m0=0.0)
result = first_passage_time_all_to_all(params; consensus_type=:either, nsamples=1000)
println("Mean FPT = \$(result.mean_fpt)")
```
"""
function first_passage_time_all_to_all(
    params::AllToAllParams;
    consensus_type::Symbol=:either,
    nsamples::Int=1000,
    reset::AbstractResetProtocol=delta_reset(params.m0)
)
    # Validate consensus_type
    @assert consensus_type ∈ [:positive, :negative, :either] "consensus_type must be :positive, :negative, or :either"

    # Run nsamples independent simulations
    times = [simulate_fpt_all_to_all_trajectory(
        params.N,
        params.m0,
        params.r,
        reset,
        consensus_type
    ) for _ in 1:nsamples]

    # Create histogram with sqrt(nsamples) bins
    nbins = Int(ceil(sqrt(nsamples)))
    counts, bin_edges = compute_histogram(times, nbins)
    bin_centers = (bin_edges[1:end-1] .+ bin_edges[2:end]) ./ 2
    
    # Normalize: density * bin_width should integrate to ~1
    bin_width = bin_edges[2] - bin_edges[1]
    densities = counts ./ (sum(counts) * bin_width)

    # Summary statistics
    mean_fpt = mean(times)
    std_fpt = std(times)

    return FPTSimulationResult(
        times,
        bin_edges,
        bin_centers,
        densities,
        counts,
        mean_fpt,
        std_fpt
    )
end


"""
    simulate_fpt_all_to_all_trajectory(N, m0, r, reset_protocol, consensus_type)

Simulate a single continuous-time trajectory to consensus using Gillespie algorithm.

Runs until the system reaches the specified consensus (one sign dominates all N nodes).
Returns the continuous time of first absorption.

# Physics

The system state is tracked by n = number of agents in state +1.
The dual events are:
  1. **Voter flip** (rate 2n(N-n)/(N-1)): One agent adopts the state of a 
     neighbor on the complete graph.
  2. **Resetting** (rate r): System returns to initial magnetization m0.

At each Gillespie step, we draw time from Exp(λ) where λ is the total rate,
then decide which event based on relative probabilities.

The simulation terminates when one of these is true:
  - `:either`   : n = 0 or n = N  (classic absorbing state)
  - `:positive` : n = N            (only all-heads absorbs)
  - `:negative` : n = 0            (only all-tails absorbs)
"""
function simulate_fpt_all_to_all_trajectory(
    N::Int,
    m0::Float64,
    r::Float64,
    reset_protocol::AbstractResetProtocol,
    consensus_type::Symbol
)
    # Initial state: convert initial magnetization to count of +1 agents
    n = Int(round(N * (1 + m0) / 2))
    n_ini = n

    t = 0.0

    # Evolution loop until consensus
    while true
        # Check stopping condition based on consensus type
        if consensus_type == :positive && n == N
            return t
        elseif consensus_type == :negative && n == 0
            return t
        elseif consensus_type == :either && (n == 0 || n == N)
            return t
        end

        # Gillespie event rate λ = (reset rate) + (voter dynamics rate)
        #   Reset rate     = r
        #   Voter rate     = 2n(N-n)/(N-1)  [equal probability for n↑ and n↓]
        λ = r + 2 * n * (N - n) / (N - 1)

        # Time to next event
        τ = -log(rand()) / λ
        t += τ

        # Which event occurs?
        if rand() * λ < r
            # === RESET EVENT ===
            # Configuration returns to initial magnetization
            n = n_ini
        else
            # === VOTER EVENT ===
            # One agent flips to match a randomly chosen neighbor
            # On complete graph this is symmetric: 50/50 chance to increase or decrease n
            if rand() < 0.5
                n += 1  # One -1 agent flips to +1
            else
                n -= 1  # One +1 agent flips to -1
            end
        end
    end
end

