# =============================================================================
# complex/pdf_simulation.jl
# =============================================================================
#
# Voter model on an arbitrary sparse graph with stochastic resetting.
# 
# Physical picture
# ----------------
# N agents sit on the nodes of a graph.  Each node is in state +1 or −1.
# An edge is *active* if its two endpoints are in different states.
# At every infinitesimal time step dt:
#
#   - With rate  w = 2 * (number of active edges)  a voter event occurs:
#       a random active edge is picked; one of its two nodes is chosen at
#       random and copies the state of the other.  The edge becomes inactive,
#       and some neighboring edges may become active or inactive.
#
#   - With rate  r  a resetting event occurs:
#       all node states are simultaneously replaced by the reset state.
#
# Total rate:  λ = w + r
# Wait time:   dt ~ Exp(λ)
#
# The factor 2 in w comes from treating edges as undirected and counting both
# orderings of each active pair.
#
# Active-edge tracking
# --------------------
# Recomputing all active edges after each flip would cost O(M) per event.
# Instead we maintain an *active edge list* and update only the edges incident
# to the flipped node (O(degree)).  Two data structures are used together:
#
#   active_list : Vector{Int}  — contiguous list of active edge IDs,
#                                allows O(1) random selection for the voter event.
#   pos_map     : Vector{Int}  — pos_map[edge_id] = current position of that
#                                edge in active_list, or 0 if inactive.  Allows
#                                O(1) lookup and O(1) removal via swap-and-pop.
#
# Swap-and-pop removal: to remove edge_id from active_list without shifting,
#   overwrite its slot with the last element, update pos_map for that last
#   element, then pop the last element.  Cost O(1).
#
# =============================================================================


# -----------------------------------------------------------------------------
# ComplexGraphCache — precomputed graph data shared across all trajectories
# -----------------------------------------------------------------------------
#
# Building neighbor lists, edge tables, and degree orderings from the
# Graphs.jl AbstractGraph object is done once before any simulation starts,
# not inside the hot loop.  This struct stores everything the trajectory
# function needs in the most convenient integer-indexed form.
struct ComplexGraphCache
    neighbors::Vector{Vector{Int}}       # neighbors[v] = list of neighbor vertex ids
    incident_edge_ids::Vector{Vector{Int}}  # incident_edge_ids[v][i] = edge id of the i-th neighbor of v
    edge_u::Vector{Int}                  # edge_u[e] = first endpoint of edge e
    edge_v::Vector{Int}                  # edge_v[e] = second endpoint of edge e
    degrees::Vector{Int}                 # degrees[v] = degree of vertex v
    degree_order_desc::Vector{Int}       # vertex indices sorted by degree, highest first
    degree_order_asc::Vector{Int}        # vertex indices sorted by degree, lowest first
end

# FixedResetPlan — precomputed reset state for deterministic protocols
#
# For protocols where the reset target is the same every time (HubReset,
# StateVectorReset) we precompute the full reset state and its active-edge
# list once before any simulation starts.  On each reset event the arrays
# are just copied in, which is faster than recomputing the active edges.
struct FixedResetPlan
    state::Vector{Int8}       # node states (+1 or -1) after reset
    active_edge_ids::Vector{Int}  # edges active in that reset state
end


# build_graph_cache: converts a Graphs.jl AbstractGraph into a ComplexGraphCache.
#
# Edges are numbered 1 … M in the order they appear in edges(graph).
# Both endpoints of each edge are stored in edge_u / edge_v so the trajectory
# function never has to call src/dst again.
function build_graph_cache(graph::AbstractGraph)
    N = nv(graph)
    neighbors = Vector{Vector{Int}}(undef, N)
    incident_edge_ids = Vector{Vector{Int}}(undef, N)
    edge_u = Int[]
    edge_v = Int[]

    for vertex in 1:N
        neighbors[vertex] = Int[]
        incident_edge_ids[vertex] = Int[]
    end

    for edge in edges(graph)
        u = Int(src(edge))
        v = Int(dst(edge))
        push!(edge_u, u)
        push!(edge_v, v)
        edge_id = length(edge_u)  # edges are numbered in insertion order

        # Store the edge reference from both endpoints' perspectives
        push!(neighbors[u], v)
        push!(incident_edge_ids[u], edge_id)
        push!(neighbors[v], u)
        push!(incident_edge_ids[v], edge_id)
    end

    degrees = Int.(degree(graph))
    # These orderings are used by HubReset to assign +1 to the highest/lowest-k nodes
    degree_order_desc = sortperm(degrees; rev = true)
    degree_order_asc = sortperm(degrees; rev = false)

    return ComplexGraphCache(neighbors, incident_edge_ids, edge_u, edge_v, degrees, degree_order_desc, degree_order_asc)
end


# -----------------------------------------------------------------------------
# Initial and reset state generators
# -----------------------------------------------------------------------------

# random_spin_state: assign each of N nodes to +1 with probability p=(m0+1)/2
# independently.  This is the standard random initial condition for a given
# expected magnetization m0.  The resulting magnetization fluctuates around
# m0 with standard deviation ~1/√N.
function random_spin_state(N::Integer, m0::Real)
    probability = (Float64(m0) + 1.0) / 2.0
    state = Vector{Int8}(undef, N)

    for index in 1:N
        state[index] = rand() < probability ? Int8(1) : Int8(-1)
    end

    return state
end

# exact_magnetization_state: assign +1 to the first `cutoff` nodes according
# to `ordering`, and −1 to the rest.  This gives *exactly* the correct number
# of +1 nodes (no fluctuation), which is what reset protocols need.
# The `ordering` vector determines which nodes get +1:
#   - For HubReset it is the degree-sorted order.
#   - For random_exact_state it is a random permutation.
function exact_magnetization_state(ordering::AbstractVector{<:Integer}, N::Integer, m0::Real)
    cutoff = positive_count_from_magnetization(N, m0)  # exactly this many +1 nodes
    state = fill(Int8(-1), N)

    for ordered_index in 1:cutoff
        state[ordering[ordered_index]] = Int8(1)
    end

    return state
end


# -----------------------------------------------------------------------------
# Active-edge list management
# -----------------------------------------------------------------------------

# Scan all edges of the graph and return the ids of those that are active
# (endpoints in different states).  Used to initialize or rebuild the list.
function active_edge_ids_from_state(cache::ComplexGraphCache, state::AbstractVector{Int8})
    active = Int[]

    for edge_id in eachindex(cache.edge_u)
        u = cache.edge_u[edge_id]
        v = cache.edge_v[edge_id]
        if state[u] != state[v]
            push!(active, edge_id)
        end
    end

    return active
end

# rebuild_active_structures!: replace the contents of active_list and pos_map
# with the given set of active edge ids.  Used after a reset event for
# protocols that don't have a precomputed FixedResetPlan.
function rebuild_active_structures!(active_list::Vector{Int}, pos_map::Vector{Int},
        active_edge_ids::AbstractVector{<:Integer})
    empty!(active_list)
    fill!(pos_map, 0)  # 0 means "not in the active list"

    for edge_id in active_edge_ids
        push!(active_list, Int(edge_id))
        pos_map[Int(edge_id)] = length(active_list)  # record the position
    end

    return nothing
end

# add_active_edge!: insert edge_id into active_list if it is not already there.
# Uses pos_map[edge_id] == 0 as the "not present" sentinel.
function add_active_edge!(active_list::Vector{Int}, pos_map::Vector{Int}, edge_id::Int)
    if pos_map[edge_id] == 0
        push!(active_list, edge_id)
        pos_map[edge_id] = length(active_list)
    end

    return nothing
end

# remove_active_edge!: remove edge_id from active_list in O(1) using
# the swap-and-pop trick:
#   1. Find the current position p of edge_id.
#   2. Move the last element of active_list into position p.
#   3. Update pos_map for that last element.
#   4. Pop the last element (which is now a duplicate after the move).
#   5. Mark edge_id as absent (pos_map = 0).
function remove_active_edge!(active_list::Vector{Int}, pos_map::Vector{Int}, edge_id::Int)
    current_position = pos_map[edge_id]
    if current_position == 0
        return nothing  # already absent, nothing to do
    end

    last_edge_id = active_list[end]
    active_list[current_position] = last_edge_id   # overwrite the slot
    pos_map[last_edge_id] = current_position        # update the displaced element's position
    pop!(active_list)                               # shrink the list
    pos_map[edge_id] = 0                           # mark as absent

    return nothing
end

# update_incident_edges!: after a node flip, only the edges touching the flipped
# node can change their active/inactive status.  This function iterates only
# those O(degree) edges and adds/removes them from active_list as needed.
#
# This is the key optimization over rescanning all M edges after each flip.
function update_incident_edges!(active_list::Vector{Int}, pos_map::Vector{Int},
        cache::ComplexGraphCache, state::Vector{Int8}, flipped_node::Int)
    for local_index in eachindex(cache.neighbors[flipped_node])
        neighbor = cache.neighbors[flipped_node][local_index]
        edge_id = cache.incident_edge_ids[flipped_node][local_index]
        # The edge is active if and only if the two endpoints are in different states
        is_active = state[flipped_node] != state[neighbor]

        if is_active
            add_active_edge!(active_list, pos_map, edge_id)
        else
            remove_active_edge!(active_list, pos_map, edge_id)
        end
    end

    return nothing
end


# -----------------------------------------------------------------------------
# Reset plan helpers
# -----------------------------------------------------------------------------
#
# fixed_reset_plan: decide at construction time whether a given protocol has
# a time-independent reset target.  If yes, return a FixedResetPlan with
# the state and active-edge list precomputed.  If no, return nothing and
# let apply_reset! recompute the new state stochastically at each event.

# HubReset: reset state is fixed (determined by degree order) → precompute.
function fixed_reset_plan(cache::ComplexGraphCache, protocol::HubReset, params::ComplexParams)
    ordering = protocol.highest_first ? cache.degree_order_desc : cache.degree_order_asc
    state = exact_magnetization_state(ordering, length(cache.degrees), protocol.target_magnetization)
    return FixedResetPlan(state, active_edge_ids_from_state(cache, state))
end

# StateVectorReset: reset state is explicitly supplied → precompute.
function fixed_reset_plan(cache::ComplexGraphCache, protocol::StateVectorReset, params::ComplexParams)
    length(protocol.state) == length(cache.degrees) ||
        throw(ArgumentError("State-vector resets for complex simulations must match the number of vertices."))
    return FixedResetPlan(copy(protocol.state), active_edge_ids_from_state(cache, protocol.state))
end

# All other protocols are stochastic → no precomputed plan.
fixed_reset_plan(cache::ComplexGraphCache, protocol::AbstractResetProtocol, params::ComplexParams) = nothing

# random_exact_state: choose a random permutation of the nodes and assign +1
# to the first `cutoff` of them.  Used by DeltaReset and RandomNodeReset on
# complex networks where the specific node identities matter (unlike all-to-all).
function random_exact_state(N::Integer, m0::Real)
    ordering = randperm(N)  # random permutation of 1:N
    return exact_magnetization_state(ordering, N, m0)
end


# -----------------------------------------------------------------------------
# apply_dynamic_complex_reset — dispatch table for stochastic reset protocols
# -----------------------------------------------------------------------------
#
# Called only when fixed_reset_plan returned nothing (i.e. the reset target
# is not fixed).  Returns the new node-state Vector{Int8}.

# RandomNodeReset on a complex graph: random permutation so each reset
# assigns +1 to a different random set of nodes (but always the right count).
function apply_dynamic_complex_reset(protocol::RandomNodeReset,
        graph::AbstractGraph, cache::ComplexGraphCache, params::ComplexParams,
        state::Vector{Int8}, current_time::Float64)
    return random_exact_state(nv(graph), protocol.target_magnetization)
end

# FunctionalReset: call the user function; accept either a state vector or a
# magnetization scalar as the return value.
function apply_dynamic_complex_reset(protocol::FunctionalReset,
        graph::AbstractGraph, cache::ComplexGraphCache, params::ComplexParams,
        state::Vector{Int8}, current_time::Float64)
    proposal = protocol.f(graph, copy(state), params, current_time)

    if proposal isa AbstractVector{<:Integer}
        return normalize_state_vector(proposal)
    elseif proposal isa Real
        return random_exact_state(nv(graph), proposal)
    end

    throw(ArgumentError("Custom complex-network reset functions must return either a state vector or a magnetization."))
end

# DeltaReset on a complex graph: the magnetization is fixed but the specific
# nodes that are +1 are re-randomized at each reset event (unlike HubReset).
# If you want the same nodes every time, use hub_reset or a StateVectorReset.
function apply_dynamic_complex_reset(protocol::DeltaReset,
        graph::AbstractGraph, cache::ComplexGraphCache, params::ComplexParams,
        state::Vector{Int8}, current_time::Float64)
    return random_exact_state(nv(graph), protocol.target_magnetization)
end

# UniformMagnetizationReset: not yet implemented for complex networks.
function apply_dynamic_complex_reset(protocol::UniformMagnetizationReset,
        graph::AbstractGraph, cache::ComplexGraphCache, params::ComplexParams,
        state::Vector{Int8}, current_time::Float64)
    throw(ArgumentError("uniform_reset is not implemented for complex-network simulations in this first slice."))
end


# apply_reset!: top-level dispatcher called from the Gillespie loop on each
# reset event.  Mutates `state`, `active_list`, and `pos_map` in-place.
#
# Fast path (cached_plan !== nothing): the reset target is fixed.
#   Just copy the precomputed state and active-edge list.
#
# Slow path (cached_plan === nothing): the reset target is stochastic.
#   Call apply_dynamic_complex_reset, then rescan the incident edges to
#   rebuild the active-edge structures.
function apply_reset!(state::Vector{Int8}, active_list::Vector{Int}, pos_map::Vector{Int},
        graph::AbstractGraph, cache::ComplexGraphCache, params::ComplexParams,
        protocol::AbstractResetProtocol, cached_plan::Union{Nothing, FixedResetPlan}, current_time::Float64)
    if cached_plan !== nothing
        # Fast path: copy precomputed state and active edges
        copy!(state, cached_plan.state)
        rebuild_active_structures!(active_list, pos_map, cached_plan.active_edge_ids)
        return nothing
    end

    # Slow path: generate new state via the protocol, then rebuild structures
    new_state = apply_dynamic_complex_reset(protocol, graph, cache, params, state, current_time)
    length(new_state) == nv(graph) ||
        throw(ArgumentError("Dynamic reset states must match the number of vertices."))

    copy!(state, Int8.(new_state))
    # Full edge rescan needed because we don't know which edges changed
    rebuild_active_structures!(active_list, pos_map, active_edge_ids_from_state(cache, state))
    return nothing
end


# =============================================================================
# simulate_complex_trajectory
# =============================================================================
#
# Runs *one* Gillespie trajectory on the graph and records the global
# magnetization m = sum(state)/N at each checkpoint in sorted_times.
#
# Arguments:
#   graph        – the network (Graphs.jl AbstractGraph)
#   cache        – precomputed graph data (build_graph_cache, called once)
#   params       – ComplexParams (r, m0)
#   sorted_times – checkpoints in ascending order
#   reset        – resetting protocol
#
# Returns:  Vector{Float64} of length length(sorted_times).
function simulate_complex_trajectory(graph::AbstractGraph,
        cache::ComplexGraphCache, params::ComplexParams, sorted_times::AbstractVector{<:Real};
        reset::AbstractResetProtocol)
    N = nv(graph)
    # Initial condition: random assignment with expected magnetization params.m0
    state = random_spin_state(N, params.m0)

    # Build the initial active-edge list by scanning all edges once
    active_list = active_edge_ids_from_state(cache, state)
    # pos_map maps every edge id to its current position in active_list (0 = absent)
    pos_map = zeros(Int, ne(graph))
    initial_active = copy(active_list)
    rebuild_active_structures!(active_list, pos_map, initial_active)

    # If the protocol has a fixed reset target, precompute it now
    cached_plan = fixed_reset_plan(cache, reset, params)

    trajectory = zeros(Float64, length(sorted_times))
    current_time = 0.0

    for time_index in eachindex(sorted_times)
        target_time = Float64(sorted_times[time_index])

        # Advance the Gillespie process up to target_time
        while current_time < target_time
            # Total rate:  w (voter) + r (resetting)
            voter_rate = 2.0 * length(active_list)  # 2× because each active edge counts both nodes
            # Match AME-style scaling: user-facing r is O(1), while this
            # trajectory uses system-level voter rate O(N), so reset hazard is r/N.
            reset_rate = params.r / N
            total_rate = voter_rate + reset_rate

            # Nothing can happen if there are no active edges and r = 0
            total_rate > 0 || break

            # Sample the waiting time: dt ~ Exp(total_rate)
            dt = -log(rand()) / total_rate

            # Checkpoint overshoot: record current state and move on
            if current_time + dt > target_time
                break
            end

            current_time += dt

            # Decide which type of event occurred
            if rand() * total_rate < voter_rate
                # --- Voter event ---
                # Pick a uniformly random active edge
                edge_id = active_list[rand(1:length(active_list))]
                u = cache.edge_u[edge_id]
                v = cache.edge_v[edge_id]
                # Flip one of the two endpoint nodes at random
                flipped_node = rand() < 0.5 ? u : v
                state[flipped_node] *= Int8(-1)
                # Update only the O(degree) edges touching the flipped node
                update_incident_edges!(active_list, pos_map, cache, state, flipped_node)
            else
                # --- Resetting event ---
                apply_reset!(state, active_list, pos_map, graph, cache, params, reset, cached_plan, current_time)
            end
        end

        # Record the global magnetization: sum of all states divided by N
        trajectory[time_index] = sum(state) / N
    end

    return trajectory
end


# =============================================================================
# simulate_pdf_complex  — public entry point
# =============================================================================
#
# Runs `nsamples` independent trajectories of the voter model on `graph` with
# stochastic resetting and returns the empirical PDF of the global magnetization
# m(t) = sum(states)/N at each requested observation time.
#
# The graph is preprocessed into a ComplexGraphCache once outside the sampling
# loop so the O(M) preprocessing cost is not repeated per trajectory.
#
# Keyword arguments:
#   reset       – resetting protocol (default: hub_reset to m0)
#   times       – observation times (any order)
#   bins        – histogram bins (integer or edge vector)
#   nsamples    – number of independent trajectories
#   value_range – histogram range (default (-1, 1))
#
# Returns a PDFSimulationResult (see simulation_core.jl).
function simulate_pdf_complex(graph::AbstractGraph, params::ComplexParams;
        reset::AbstractResetProtocol = hub_reset(params.m0), times, bins = 50,
    nsamples::Integer = 1000,
        value_range::Tuple{<:Real, <:Real} = (-1.0, 1.0))
    times_float = Float64.(collect(times))
    all(t -> t >= 0, times_float) || throw(ArgumentError("Observation times must be non-negative."))

    # Precompute neighbor lists, edge tables, and degree orderings once.
    # All trajectories share the same cache (read-only inside the loop).
    cache = build_graph_cache(graph)

    # Run all trajectories via the generic sampling loop from simulation_core.
    # Each trajectory is independent and covers all observation times.
    sample_matrix = sample_timeseries(times_float; nsamples = nsamples) do observation_times
        simulate_complex_trajectory(graph, cache, params, observation_times; reset = reset)
    end

    edges = resolve_bin_edges(bins; value_range = value_range)
    counts, densities = summarize_pdf_samples(sample_matrix, edges)

    return PDFSimulationResult(
        times_float,
        edges,
        bin_centers_from_edges(edges),
        densities,
        counts,
        sample_matrix,
    )
end
