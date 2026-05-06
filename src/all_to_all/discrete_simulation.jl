# =============================================================================
# all_to_all/discrete_simulation.jl
# =============================================================================
#
# Discrete-time voter model simulations on the complete graph (all-to-all).
# 
# Physical picture
# ----------------
# N agents on a complete graph. At each discrete time step:
#
#   1. With probability  p_voter / (p_voter + r),  a voter event occurs:
#      A random agent picks a random neighbor and copies their state.
#      n increases or decreases by 1 with equal probability 1/2 (if possible).
#
#   2. With probability  r / (p_voter + r),  a reset event occurs:
#      The system resets to the prescribed state.
#
#   where  p_voter ~ w = 2n(N-n)/(N-1)  (normalized from continuous rate).
#
# For simplicity, we implement:
#   - First compute p = min(w / N, 1.0) as the voter step probability
#   - With probability r: reset
#   - Else with probability p: one voter step
#   - Else: nothing
#
# This is simpler than exact rate balance and works well for moderate r.

"""
    simulate_trajectory_discrete_all_to_all(
        N::Int,
        n0::Int,
        r::Real,
        nsteps::Int;
        reset::AbstractResetProtocol = delta_reset(0.0)
    ) -> n_trajectory::Vector{Int}

Simulate one discrete-time trajectory of the all-to-all voter model with reset.

# Arguments
- `N::Int`: Number of agents
- `n0::Int`: Initial count in state +1 (magnetization m0 = (2*n0 - N)/N)
- `r::Real`: Reset probability per step ∈ [0, 1]
- `nsteps::Int`: Number of discrete time steps
- `reset::AbstractResetProtocol`: Reset protocol (e.g., delta_reset(m0))

# Returns
- Vector of length `nsteps + 1` containing n at each step (including n0 at step 0)

# Example
```julia
traj = simulate_trajectory_discrete_all_to_all(1000, 100, 0.0, 500;
    reset=delta_reset(0.0))
rho = traj ./ 1000  # Convert to density
m = 2 .* rho .- 1   # Convert to magnetization
```
"""
function simulate_trajectory_discrete_all_to_all(
        N::Int,
        n0::Int,
        r::Real,
        nsteps::Int;
        reset::AbstractResetProtocol = delta_reset(0.0)
)
    0 <= r <= 1 || throw(ArgumentError("r must be in [0, 1]"))
    1 <= n0 <= N-1 || throw(ArgumentError("n0 must be in [1, N-1]"))
    
    trajectory = Vector{Int}(undef, nsteps + 1)
    trajectory[1] = n0
    n = n0
    
    params = AllToAllParams(N, r, (2 * n0 - N) / N)
    
    for step in 2:(nsteps + 1)
        # Check for reset
        if rand() < r
            n_new = apply_all_to_all_reset(reset, params, n0, n, float(step - 1))
            n = clamp(n_new, 0, N)
        else
            # Attempt voter step
            # Compute voter step probability from rate
            if n > 0 && n < N
                w = 2.0 * n * (N - n) / (N - 1)
                p_voter = w / N  # Scaled probability per step
                
                if rand() < p_voter
                    # Voter event: flip one agent
                    if rand() < 0.5
                        n = n + 1
                    else
                        n = n - 1
                    end
                end
            end
        end
        
        trajectory[step] = n
    end
    
    return trajectory
end


"""
    simulate_pdf_discrete_all_to_all(
        N::Int,
        n0::Int,
        r::Real,
        observation_steps::AbstractVector{Int},
        nsamples::Int;
        reset::AbstractResetProtocol = delta_reset(0.0)
    ) -> samples::Matrix{Float64}

Simulate an ensemble of discrete-time trajectories and sample magnetization at 
specified observation times (in steps).

# Arguments
- `N::Int`: System size
- `n0::Int`: Initial count in state +1
- `r::Real`: Reset probability per step
- `observation_steps::AbstractVector{Int}`: Which steps to observe (e.g., [100, 200, 500])
- `nsamples::Int`: Number of independent trajectories
- `reset::AbstractResetProtocol`: Reset protocol

# Returns
- Matrix of shape `(nsamples, length(observation_steps))` where entry `[i, j]` is
  the magnetization in trajectory `i` at observation time `observation_steps[j]`

# Example
```julia
params = AllToAllParams(1000, 0.1, 0.0)
samples = simulate_pdf_discrete_all_to_all(
    1000, 100, 0.1, [100, 200, 500], 300;
    reset=delta_reset(0.0)
)
# samples is 300×3: 300 samples at times [100, 200, 500] steps
m_final = vec(samples[:, end])
histogram(m_final)
```
"""
function simulate_pdf_discrete_all_to_all(
        N::Int,
        n0::Int,
        r::Real,
        observation_steps::AbstractVector{Int},
        nsamples::Int;
        reset::AbstractResetProtocol = delta_reset(0.0)
)
    max_step = maximum(observation_steps)
    observation_steps = sort(observation_steps)
    nobs = length(observation_steps)
    
    # Pre-allocate output
    samples = Matrix{Float64}(undef, nsamples, nobs)
    
    for sample_idx in 1:nsamples
        # Run one trajectory
        traj = simulate_trajectory_discrete_all_to_all(N, n0, r, max_step; reset=reset)
        
        # Extract magnetization at observation times
        for (obs_idx, step) in enumerate(observation_steps)
            n = traj[step + 1]  # +1 because trajectory[1] is step 0
            m = (2 * n - N) / N
            samples[sample_idx, obs_idx] = m
        end
    end
    
    return samples
end


"""
    simulate_fpt_discrete_all_to_all(
        N::Int,
        n0::Int,
        r::Real;
        consensus_type::Symbol = :either,
        nsamples::Int = 1000,
        max_steps::Int = 100000,
        reset::AbstractResetProtocol = delta_reset(0.0)
    ) -> fpt_samples::Vector{Float64}

Compute first-passage times to consensus for an ensemble of discrete-time trajectories.

# Arguments
- `consensus_type::Symbol`: `:either` (±1), `:positive` (+1), or `:negative` (-1)
- `max_steps::Int`: Maximum steps before giving up (to prevent infinite loops)
- Other arguments as in `simulate_trajectory_discrete_all_to_all`

# Returns
- Vector of FPT values (in steps) for each sample reaching consensus
- If a trajectory doesn't reach consensus before `max_steps`, it records `NaN`

# Example
```julia
fpt = simulate_fpt_discrete_all_to_all(
    500, 50, 0.0;
    consensus_type=:either,
    nsamples=500,
    max_steps=100000,
    reset=delta_reset(0.0)
)
mfpt = mean(filter(!isnan, fpt))
```
"""
function simulate_fpt_discrete_all_to_all(
        N::Int,
        n0::Int,
        r::Real;
        consensus_type::Symbol = :either,
        nsamples::Int = 1000,
        max_steps::Int = 100000,
        reset::AbstractResetProtocol = delta_reset(0.0)
)
    fpt_samples = Vector{Float64}(undef, nsamples)
    params = AllToAllParams(N, r, (2 * n0 - N) / N)
    
    for sample_idx in 1:nsamples
        n = n0
        reached_consensus = false
        
        for step in 1:max_steps
            # Check for reset
            if rand() < r
                n_new = apply_all_to_all_reset(reset, params, n0, n, float(step))
                n = clamp(n_new, 0, N)
            else
                # Voter step
                if n > 0 && n < N
                    w = 2.0 * n * (N - n) / (N - 1)
                    p_voter = w / N
                    
                    if rand() < p_voter
                        if rand() < 0.5
                            n = n + 1
                        else
                            n = n - 1
                        end
                    end
                end
            end
            
            # Check for consensus
            is_consensus = false
            if n == 0 && (consensus_type == :either || consensus_type == :negative)
                is_consensus = true
            elseif n == N && (consensus_type == :either || consensus_type == :positive)
                is_consensus = true
            end
            
            if is_consensus
                fpt_samples[sample_idx] = float(step)
                reached_consensus = true
                break
            end
        end
        
        if !reached_consensus
            fpt_samples[sample_idx] = NaN
        end
    end
    
    return fpt_samples
end
