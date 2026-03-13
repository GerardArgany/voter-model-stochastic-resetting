# =============================================================================
# complex/sikm_simulation.jl
# =============================================================================
#
# Monte Carlo simulation of degree-conditioned observables s_{k,m}(t) and
# i_{k,m}(t) on complex networks.
#
# State convention (matches the rest of the package):
#   s_{k,m}(t) – fraction of ALL nodes at state −1, degree k, m neighbours +1
#   i_{k,m}(t) – fraction of ALL nodes at state +1, degree k, m neighbours +1
#
# This file depends on the graph infrastructure (ComplexGraphCache, reset
# machinery, etc.) defined in complex/pdf_simulation.jl, which is included
# first in VoterResetting.jl.
#
# Public entry points:
#   simulate_degree_evolution_complex  – ensemble-averaged grid or single pair
#   simulate_sikm_pair_complex         – convenience wrapper (single pair)
# =============================================================================


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

# Build sorted observed degree classes and a degree→index map.
function degree_classes(cache::ComplexGraphCache)
    k_values  = sort(unique(cache.degrees))
    k_to_index = Dict{Int,Int}(k => idx for (idx, k) in enumerate(k_values))
    return k_values, k_to_index
end

# Run one Gillespie trajectory and return a full copy of the node-state vector
# at each requested checkpoint time.  Reuses the same dynamics as
# simulate_complex_trajectory in pdf_simulation.jl.
function simulate_complex_state_snapshots(graph::AbstractGraph,
        cache::ComplexGraphCache, params::ComplexParams,
        sorted_times::AbstractVector{<:Real};
        reset::AbstractResetProtocol)
    state       = random_spin_state(nv(graph), params.m0)
    active_list = active_edge_ids_from_state(cache, state)
    pos_map     = zeros(Int, ne(graph))
    initial_active = copy(active_list)
    rebuild_active_structures!(active_list, pos_map, initial_active)

    cached_plan = fixed_reset_plan(cache, reset, params)
    snapshots   = Vector{Vector{Int8}}(undef, length(sorted_times))
    current_time = 0.0

    for time_index in eachindex(sorted_times)
        target_time = Float64(sorted_times[time_index])

        while current_time < target_time
            voter_rate = 2.0 * length(active_list)
            total_rate = voter_rate + params.r
            total_rate > 0 || break

            dt = -log(rand()) / total_rate
            current_time + dt > target_time && break
            current_time += dt

            if rand() * total_rate < voter_rate
                edge_id      = active_list[rand(1:length(active_list))]
                u            = cache.edge_u[edge_id]
                v            = cache.edge_v[edge_id]
                flipped_node = rand() < 0.5 ? u : v
                state[flipped_node] *= Int8(-1)
                update_incident_edges!(active_list, pos_map, cache, state, flipped_node)
            else
                apply_reset!(state, active_list, pos_map, graph, cache, params,
                             reset, cached_plan, current_time)
            end
        end

        snapshots[time_index] = copy(state)
    end

    return snapshots
end

# Compute s_{k,m} and i_{k,m} fractions for a single (k, m) pair from one snapshot.
function pair_fraction_from_state(cache::ComplexGraphCache,
        state::AbstractVector{Int8}, k::Int, m::Int)
    N       = length(state)
    s_count = 0
    i_count = 0

    for node in 1:N
        cache.degrees[node] == k || continue
        m_plus = count(nb -> state[nb] == Int8(1), cache.neighbors[node])
        m_plus == m || continue
        if state[node] == Int8(-1)
            s_count += 1
        else
            i_count += 1
        end
    end

    return s_count / N, i_count / N
end

# Compute all s_{k,m} and i_{k,m} fractions from one snapshot.
function grid_fractions_from_state(cache::ComplexGraphCache,
        state::AbstractVector{Int8},
        k_values::Vector{Int}, k_to_index::Dict{Int,Int})
    s_values = [zeros(Float64, k + 1) for k in k_values]
    i_values = [zeros(Float64, k + 1) for k in k_values]
    N        = length(state)

    for node in 1:N
        k           = cache.degrees[node]
        class_index = k_to_index[k]
        m_plus      = count(nb -> state[nb] == Int8(1), cache.neighbors[node])

        if state[node] == Int8(-1)
            s_values[class_index][m_plus + 1] += 1.0 / N
        else
            i_values[class_index][m_plus + 1] += 1.0 / N
        end
    end

    return s_values, i_values
end


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

"""
    simulate_degree_evolution_complex(graph, params; reset, times, nsamples,
                                      mode, k, m)

Ensemble-averaged time evolution of degree-conditioned fractions:

- `s_{k,m}(t)`: fraction of all nodes with degree `k`, state `−1`, and exactly
  `m` neighbours in state `+1`.
- `i_{k,m}(t)`: same but node state `+1`.

**Modes**
- `mode=:all`  → `DegreeGridEvolutionResult` for every observed degree class
  and every valid `m = 0…k`.
- `mode=:pair` → `DegreePairEvolutionResult` for a single `(k, m)` pair
  (supply `k` and `m` as keyword arguments).

Set `params.r = 0.0` for a no-reset baseline.
"""
function simulate_degree_evolution_complex(graph::AbstractGraph,
        params::ComplexParams;
        reset::AbstractResetProtocol = hub_reset(params.m0),
        times,
        nsamples::Integer = 1000,
        mode::Symbol      = :all,
        k::Union{Nothing,Integer} = nothing,
        m::Union{Nothing,Integer} = nothing)
    nsamples > 0 || throw(ArgumentError("nsamples must be positive."))

    times_float = Float64.(collect(times))
    all(t -> t >= 0, times_float) || throw(ArgumentError("Observation times must be non-negative."))

    cache              = build_graph_cache(graph)
    k_values, k_to_index = degree_classes(cache)

    if mode == :pair
        k !== nothing || throw(ArgumentError("For mode=:pair, supply keyword k."))
        m !== nothing || throw(ArgumentError("For mode=:pair, supply keyword m."))

        target_k = Int(k)
        target_m = Int(m)
        target_k in k_values ||
            throw(ArgumentError("Degree k=$(target_k) is not present in the graph."))
        0 <= target_m <= target_k ||
            throw(ArgumentError("m must satisfy 0 ≤ m ≤ k."))

        s_acc = zeros(Float64, length(times_float))
        i_acc = zeros(Float64, length(times_float))

        for _ in 1:nsamples
            snaps = simulate_complex_state_snapshots(graph, cache, params,
                                                     times_float; reset = reset)
            for t_idx in eachindex(times_float)
                sv, iv = pair_fraction_from_state(cache, snaps[t_idx], target_k, target_m)
                s_acc[t_idx] += sv
                i_acc[t_idx] += iv
            end
        end

        return DegreePairEvolutionResult(times_float, target_k, target_m,
                                         s_acc ./ nsamples, i_acc ./ nsamples,
                                         Int(nsamples))

    elseif mode == :all
        s_acc = [zeros(Float64, length(times_float), kv + 1) for kv in k_values]
        i_acc = [zeros(Float64, length(times_float), kv + 1) for kv in k_values]

        for _ in 1:nsamples
            snaps = simulate_complex_state_snapshots(graph, cache, params,
                                                     times_float; reset = reset)
            for t_idx in eachindex(times_float)
                s_now, i_now = grid_fractions_from_state(cache, snaps[t_idx],
                                                          k_values, k_to_index)
                for ci in eachindex(k_values)
                    s_acc[ci][t_idx, :] .+= s_now[ci]
                    i_acc[ci][t_idx, :] .+= i_now[ci]
                end
            end
        end

        for ci in eachindex(k_values)
            s_acc[ci] ./= nsamples
            i_acc[ci] ./= nsamples
        end

        return DegreeGridEvolutionResult(times_float, k_values, s_acc, i_acc,
                                          Int(nsamples))
    end

    throw(ArgumentError("mode must be :all or :pair."))
end


"""
    simulate_sikm_pair_complex(graph, params; reset, times, nsamples, k, m)

Convenience wrapper for `simulate_degree_evolution_complex(...; mode=:pair)`.
Returns a `DegreePairEvolutionResult`.
"""
function simulate_sikm_pair_complex(graph::AbstractGraph, params::ComplexParams;
        reset::AbstractResetProtocol = hub_reset(params.m0),
        times, nsamples::Integer = 1000, k::Integer, m::Integer)
    return simulate_degree_evolution_complex(graph, params;
        reset = reset, times = times, nsamples = nsamples,
        mode = :pair, k = k, m = m)
end
