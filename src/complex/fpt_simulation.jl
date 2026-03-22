# =============================================================================
# complex/fpt_simulation.jl
# =============================================================================
#
# First-passage-time simulation for voter model with resetting on arbitrary
# complex networks.  Uses active-edge tracking for O(degree) per event instead
# of O(edges).
#
# Key optimization: active_list and pos_map track which edges currently
# separate nodes with different states. Only these edges can trigger voter
# flips. When a node flips, only its neighbors' edges are checked for
# activation/deactivation.
#
# Physics:
#   Stops when num_active_edges == 0 (all edges are bipartite/ordered).
#   Can also condition on specific consensus: all +1 or all -1.
#
# =============================================================================

"""
    first_passage_time_complex(graph::AbstractGraph, params::ComplexParams; 
                               consensus_type=:either, nsamples=1000, 
                               reset=hub_reset(0.0))

Estimate distribution of first-passage times (FPT) to consensus on a complex
network using Gillespie simulation with active-edge acceleration.

# Arguments
- `graph::AbstractGraph`: A Graphs.jl compatible network (e.g., SimpleGraph)
- `params::ComplexParams`: Resetting rate r and initial magnetization m0
- `consensus_type::Symbol`: `:positive`, `:negative`, or `:either` (default)
- `nsamples::Int=1000`: Number of independent runs
- `reset::AbstractResetProtocol=hub_reset(0.0)`: Resetting strategy

# Returns
`FPTSimulationResult` with histogram of FPT values and statistics.

# Example
```julia
G = barabasi_albert(100, 2)  # 100 nodes, scale-free graph
params = ComplexParams(r=0.1, m0=0.0)
result = first_passage_time_complex(G, params; consensus_type=:either, nsamples=500)
println("Mean FPT = \$(result.mean_fpt)")
```
"""
function first_passage_time_complex(
    graph::AbstractGraph,
    params::ComplexParams;
    consensus_type::Symbol=:either,
    nsamples::Int=1000,
    reset::AbstractResetProtocol=hub_reset(0.0)
)
    # Validate consensus_type
    @assert consensus_type ∈ [:positive, :negative, :either] "consensus_type must be :positive, :negative, or :either"
    
    N = nv(graph)
    M = ne(graph)

    # Build edge-indexed adjacency once (reuse for all samples).
    # This supports O(degree) active-edge updates after a single-node flip.
    edges_u = Vector{Int}(undef, M)
    edges_v = Vector{Int}(undef, M)
    incident_edge_ids = [Int[] for _ in 1:N]

    edge_id = 0
    for edge in edges(graph)
        edge_id += 1
        u, v = src(edge), dst(edge)
        edges_u[edge_id] = u
        edges_v[edge_id] = v
        push!(incident_edge_ids[u], edge_id)
        push!(incident_edge_ids[v], edge_id)
    end

    # Run nsamples independent simulations
    times = [simulate_fpt_complex_trajectory(
        N,
        M,
        edges_u,
        edges_v,
        incident_edge_ids,
        params.r,
        params.m0,
        reset,
        consensus_type
    ) for _ in 1:nsamples]

    # Create histogram
    nbins = Int(ceil(sqrt(nsamples)))
    counts, bin_edges = compute_histogram(times, nbins)
    bin_centers = (bin_edges[1:end-1] .+ bin_edges[2:end]) ./ 2
    
    # Normalize by bin width
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
    simulate_fpt_complex_trajectory(N, M, adj_list, r, m0, reset_protocol, consensus_type)

Single Gillespie trajectory for FPT on a complex network.

Uses active-edge tracking: maintains a list of edges separating different-state
nodes. When num_active_edges reaches 0, consensus is achieved.

Alternatively, can stop when all nodes have the same state (consensus verification).
"""
function simulate_fpt_complex_trajectory(
    N::Int,
    M::Int,
    edges_u::Vector{Int},
    edges_v::Vector{Int},
    incident_edge_ids::Vector{Vector{Int}},
    r::Float64,
    m0::Float64,
    reset_protocol::AbstractResetProtocol,
    consensus_type::Symbol
)
    # Initialize random spin state
    p_plus = (1 + m0) / 2
    state = Int8[rand() < p_plus ? Int8(1) : Int8(-1) for _ in 1:N]
    plus_count = count(==(Int8(1)), state)

    t = 0.0

    # Active-edge tracking structures
    active_edges = Int[]
    pos_map = zeros(Int, M)  # pos_map[eid] == 0 means inactive; otherwise index in active_edges

    for eid in 1:M
        if state[edges_u[eid]] != state[edges_v[eid]]
            push!(active_edges, eid)
            pos_map[eid] = length(active_edges)
        end
    end

    # Pre-compute reset state and reset active edges
    reset_state, reset_active_edges, reset_plus_count = compute_reset_state(
        N, m0, state, edges_u, edges_v, incident_edge_ids, reset_protocol)

    function rebuild_pos_map!(pos_map::Vector{Int}, active_edges::Vector{Int})
        fill!(pos_map, 0)
        for (idx, eid) in enumerate(active_edges)
            pos_map[eid] = idx
        end
        return nothing
    end

    function remove_active_edge!(active_edges::Vector{Int}, pos_map::Vector{Int}, eid::Int)
        pos = pos_map[eid]
        pos == 0 && return
        last_eid = active_edges[end]
        active_edges[pos] = last_eid
        pos_map[last_eid] = pos
        pop!(active_edges)
        pos_map[eid] = 0
        return nothing
    end

    function add_active_edge!(active_edges::Vector{Int}, pos_map::Vector{Int}, eid::Int)
        pos_map[eid] != 0 && return
        push!(active_edges, eid)
        pos_map[eid] = length(active_edges)
        return nothing
    end

    function update_edge_activity!(active_edges::Vector{Int}, pos_map::Vector{Int}, eid::Int)
        is_active = pos_map[eid] != 0
        should_be_active = state[edges_u[eid]] != state[edges_v[eid]]

        if is_active && !should_be_active
            remove_active_edge!(active_edges, pos_map, eid)
        elseif !is_active && should_be_active
            add_active_edge!(active_edges, pos_map, eid)
        end
        return nothing
    end

    # Evolution loop
    while true
        num_active = length(active_edges)
        
        # Check stopping conditions
        if consensus_type == :positive && plus_count == N
            return t
        elseif consensus_type == :negative && plus_count == 0
            return t
        elseif consensus_type == :either && num_active == 0
            return t
        end

        # Gillespie rate: λ = (voter rate) + (reset rate)
        #   Voter rate = 2 * num_active  (system-level, O(N))
        #   Reset rate = r / N           (AME-style user-facing r converted internally)
        λ = 2.0 * num_active + r / N

        if λ <= 0
            # At zero total rate, the process is frozen. For one-sided targets,
            # reaching the opposite absorbing state means the requested FPT is not
            # attained (infinite in the unconditional sense).
            if consensus_type == :either
                return t
            elseif consensus_type == :positive
                return plus_count == N ? t : Inf
            else # consensus_type == :negative
                return plus_count == 0 ? t : Inf
            end
        end

        # Time to next event
        τ = -log(rand()) / λ
        t += τ

        # Which event?
        if rand() * λ < 2.0 * num_active
            # === VOTER EVENT ===
            # Randomly choose one currently active edge.
            chosen_eid = active_edges[rand(1:num_active)]
            u = edges_u[chosen_eid]
            v = edges_v[chosen_eid]
            
            # Randomly choose which endpoint to flip
            node_to_flip = rand() < 0.5 ? u : v
            
            # Flip the node
            old_state = state[node_to_flip]
            state[node_to_flip] = Int8(-old_state)
            if old_state == Int8(1)
                plus_count -= 1
            else
                plus_count += 1
            end

            # Only incident edges can change active/inactive status.
            for eid in incident_edge_ids[node_to_flip]
                update_edge_activity!(active_edges, pos_map, eid)
            end
            
        else
            # === RESET EVENT ===
            if reset_protocol isa UniformMagnetizationReset
                # Uniform reset must redraw the target magnetization at every reset event.
                reset_state, reset_active_edges, reset_plus_count = compute_reset_state(
                    N,
                    m0,
                    state,
                    edges_u,
                    edges_v,
                    incident_edge_ids,
                    reset_protocol,
                )
            end
            state .= reset_state
            plus_count = reset_plus_count
            active_edges = copy(reset_active_edges)
            rebuild_pos_map!(pos_map, active_edges)
        end
    end
end


"""
    compute_reset_state(N, m0, current_state, adj_list, reset_protocol)

Compute the reset configuration based on the protocol type.
Returns (reset_state_vector, num_reset_active_edges).
"""
function compute_reset_state(
    N::Int,
    m0::Float64,
    current_state::Vector{Int8},
    edges_u::Vector{Int},
    edges_v::Vector{Int},
    incident_edge_ids::Vector{Vector{Int}},
    reset_protocol::AbstractResetProtocol
)
    reset_state = similar(current_state)
    
    if reset_protocol isa DeltaReset
        # All nodes set to match a fixed magnetization
        n_plus = Int(round(N * (1 + reset_protocol.target_magnetization) / 2))
        fill!(reset_state, Int8(-1))
        for i in 1:n_plus
            reset_state[i] = Int8(1)
        end
    elseif reset_protocol isa StateVectorReset
        # Use explicit state vector
        reset_state .= reset_protocol.state
    elseif reset_protocol isa RandomNodeReset
        # Each node independently set to +1 with probability p
        p = (1 + reset_protocol.target_magnetization) / 2
        for i in 1:N
            reset_state[i] = rand() < p ? Int8(1) : Int8(-1)
        end
    elseif reset_protocol isa UniformMagnetizationReset
        # Draw target magnetization uniformly over the discrete grid m = 2n/N - 1,
        # i.e. sample n_+ uniformly from {0, 1, ..., N} at each reset.
        n_plus = rand(0:N)
        fill!(reset_state, Int8(-1))
        if n_plus > 0
            node_order = randperm(N)
            for i in 1:n_plus
                reset_state[node_order[i]] = Int8(1)
            end
        end
    elseif reset_protocol isa HubReset
        # Highest (or lowest) degree nodes set to +1
        degrees = [length(incident_edge_ids[i]) for i in 1:N]
        node_order = sortperm(degrees; rev=reset_protocol.highest_first)
        n_plus = Int(round(N * (1 + reset_protocol.target_magnetization) / 2))
        fill!(reset_state, Int8(-1))
        for i in 1:n_plus
            reset_state[node_order[i]] = Int8(1)
        end
    else
        # Fallback: random state with target magnetization
        p = (1 + m0) / 2
        reset_state = Int8[rand() < p ? Int8(1) : Int8(-1) for _ in 1:N]
    end
    
    # Build active-edge list in reset state
    reset_active_edges = Int[]
    M = length(edges_u)
    for eid in 1:M
        if reset_state[edges_u[eid]] != reset_state[edges_v[eid]]
            push!(reset_active_edges, eid)
        end
    end

    reset_plus_count = count(==(Int8(1)), reset_state)
    
    return reset_state, reset_active_edges, reset_plus_count
end
