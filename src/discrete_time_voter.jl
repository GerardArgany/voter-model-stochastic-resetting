# =============================================================================
# discrete_time_voter.jl
# =============================================================================
#
# Discrete-time voter model simulations with reset probability (not Gillespie rate).
#
# Physics:
# --------
# At each discrete time step:
#
#   ALL-TO-ALL:
#   1. With probability p_voter = w / (1 + w + r), a voter event:
#      A random discordant pair is chosen; one copies the other (n ±1).
#   2. With probability p_reset = r / (1 + w + r), a reset event:
#      System returns to reset state.
#   3. With probability 1 - p_voter - p_reset, nothing happens.
#      where w = 2*n*(N-n)/(N-1) is the voter dynamics rate.
#
#   COMPLEX NETWORKS:
#   1. Pick a random node.
#   2. If it has active edges (neighbors in opposite state):
#      Pick a random active neighbor, copy their state.
#   3. With probability r, reset all nodes to reset state.
#
# Optimization:
# - Pre-allocated buffers for all state and result arrays
# - Active-edge list tracking (same as Gillespie version)
# - Minimal allocations during stepping
# =============================================================================

# Helper: Extract reset target magnetization from any protocol
function reset_target_magnetization(reset::AbstractResetProtocol, fallback::Float64)
    if reset isa DeltaReset
        return reset.target_magnetization
    elseif reset isa RandomNodeReset
        return reset.target_magnetization
    elseif reset isa HubReset
        return reset.target_magnetization
    elseif reset isa StateVectorReset
        return mean(Float64.(reset.state))
    else
        return fallback
    end
end

# =============================================================================
# Complex Network Discrete-Time Simulations
# =============================================================================

"""
    simulate_pdf_discrete_complex(graph::AbstractGraph, params,
                                  times::Vector, nsamples::Int;
                                  reset::AbstractResetProtocol)

Discrete-time PDF simulation on arbitrary complex network.

At each step:
1. Pick a random active edge and flip one endpoint
2. With probability r, reset all nodes
"""
function simulate_pdf_discrete_complex(graph::AbstractGraph, params,
        times::AbstractVector{<:Real}, nsamples::Int;
        reset::AbstractResetProtocol)
    
    N = nv(graph)
    M = ne(graph)
    M > 0 || throw(ArgumentError("Graph must have at least one edge"))
    
    # Precompute graph structure
    cache = build_graph_cache(graph)
    
    # Storage for samples: (nsamples × ntimes)
    times = Float64.(times)
    samples = Matrix{Float64}(undef, nsamples, length(times))
    
    # Get max steps needed (assume 1 step ≈ 1 time unit)
    max_steps = floor(Int, maximum(times)) + 1
    
    # Precompute reset state if fixed
    reset_state_cached = nothing
    if reset isa DeltaReset
        m_reset = reset.target_magnetization
        n_reset = Int(round(N * (1.0 + m_reset) / 2.0))
        reset_state_cached = fill(Int8(-1), N)
        if n_reset > 0
            node_order = randperm(N)
            for i in 1:n_reset
                reset_state_cached[node_order[i]] = Int8(1)
            end
        end
    elseif reset isa StateVectorReset
        reset_state_cached = copy(reset.state)
    end
    
    # Run nsamples independent trajectories
    for sample_id in 1:nsamples
        # Initialize: random state with expected magnetization params.m0
        state = random_spin_state(N, params.m0)
        active_list = active_edge_ids_from_state(cache, state)
        pos_map = zeros(Int, M)
        rebuild_active_structures!(active_list, pos_map, copy(active_list))
        
        time_idx = 1
        
        for step in 1:max_steps
            # Check if we need to record at this step
            while time_idx <= length(times) && times[time_idx] <= step
                m_sample = sum(state) / N
                samples[sample_id, time_idx] = m_sample
                time_idx += 1
            end
            
            if time_idx > length(times)
                break
            end
            
            # Discrete time step
            if rand() < params.r
                # Reset event
                if !isnothing(reset_state_cached)
                    state .= reset_state_cached
                else
                    # Dynamic reset (sample new state)
                    state = random_spin_state(N, params.m0)
                end
                # Rebuild active list after reset
                active_list = active_edge_ids_from_state(cache, state)
                rebuild_active_structures!(active_list, pos_map, copy(active_list))
            elseif !isempty(active_list)
                # Voter event: pick random active edge
                edge_id = active_list[rand(1:length(active_list))]
                u = cache.edge_u[edge_id]
                v = cache.edge_v[edge_id]
                # Flip one endpoint
                flipped_node = rand(Bool) ? u : v
                state[flipped_node] = state[u == flipped_node ? v : u]
                # Update incident edges efficiently
                update_incident_edges!(active_list, pos_map, cache, state, flipped_node)
            end
        end
        
        # Record final magnetization for any remaining time points
        m_sample = sum(state) / N
        while time_idx <= length(times)
            samples[sample_id, time_idx] = m_sample
            time_idx += 1
        end
    end
    
    return samples
end

"""
    first_passage_time_discrete_complex(graph::AbstractGraph, params;
                                        consensus_type=:either, nsamples=1000,
                                        max_steps=10000, reset::AbstractResetProtocol)
                                        
Discrete-time first passage time simulation on complex network.

Runs nsamples trajectories until consensus is reached (all +1, all -1, or either).
Returns vector of FPT for each sample.
"""
function first_passage_time_discrete_complex(graph::AbstractGraph, params;
        consensus_type=:either, nsamples=1000, max_steps=10000,
        reset::AbstractResetProtocol)
    
    N = nv(graph)
    M = ne(graph)
    
    fpt_samples = Float64[]
    cache = build_graph_cache(graph)
    
    # Precompute reset state if fixed
    reset_state_cached = nothing
    if reset isa DeltaReset
        m_reset = reset.target_magnetization
        n_reset = Int(round(N * (1.0 + m_reset) / 2.0))
        reset_state_cached = fill(Int8(-1), N)
        if n_reset > 0
            node_order = randperm(N)
            for i in 1:n_reset
                reset_state_cached[node_order[i]] = Int8(1)
            end
        end
    elseif reset isa StateVectorReset
        reset_state_cached = copy(reset.state)
    end
    
    for sample_id in 1:nsamples
        state = random_spin_state(N, params.m0)
        active_list = active_edge_ids_from_state(cache, state)
        pos_map = zeros(Int, M)
        rebuild_active_structures!(active_list, pos_map, copy(active_list))
        
        for step in 1:max_steps
            # Check stopping condition
            n_plus = count(==(Int8(1)), state)
            
            if consensus_type == :positive && n_plus == N
                push!(fpt_samples, float(step))
                break
            elseif consensus_type == :negative && n_plus == 0
                push!(fpt_samples, float(step))
                break
            elseif consensus_type == :either && (n_plus == 0 || n_plus == N)
                push!(fpt_samples, float(step))
                break
            end
            
            # Discrete time step
            if rand() < params.r
                # Reset
                if !isnothing(reset_state_cached)
                    state .= reset_state_cached
                else
                    state = random_spin_state(N, params.m0)
                end
                active_list = active_edge_ids_from_state(cache, state)
                rebuild_active_structures!(active_list, pos_map, copy(active_list))
            elseif !isempty(active_list)
                # Voter flip
                edge_id = active_list[rand(1:length(active_list))]
                u = cache.edge_u[edge_id]
                v = cache.edge_v[edge_id]
                flipped_node = rand(Bool) ? u : v
                state[flipped_node] = state[u == flipped_node ? v : u]
                update_incident_edges!(active_list, pos_map, cache, state, flipped_node)
            end
        end
    end
    
    return fpt_samples
end
