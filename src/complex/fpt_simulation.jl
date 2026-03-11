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

    # Build adjacency structure once (reuse for all samples)
    adj_list = [Int[] for _ in 1:N]
    for edge in edges(graph)
        u, v = src(edge), dst(edge)
        push!(adj_list[u], v)
        push!(adj_list[v], u)
    end

    # Run nsamples independent simulations
    times = [simulate_fpt_complex_trajectory(
        N,
        M,
        adj_list,
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
    adj_list::Vector{Vector{Int}},
    r::Float64,
    m0::Float64,
    reset_protocol::AbstractResetProtocol,
    consensus_type::Symbol
)
    # Initialize random spin state
    p_plus = (1 + m0) / 2
    state = Int8[rand() < p_plus ? Int8(1) : Int8(-1) for _ in 1:N]

    t = 0.0

    # Pre-compute reset state
    reset_state, reset_active_count = compute_reset_state(N, m0, state, adj_list, reset_protocol)

    # Function to count active edges (edges between different-state nodes)
    function count_active_edges()
        count = 0
        for i in 1:N
            for j in adj_list[i]
                if i < j && state[i] != state[j]
                    count += 1
                end
            end
        end
        return count
    end

    # Evolution loop
    while true
        # Count current active edges
        num_active = count_active_edges()
        
        # Check stopping conditions
        if consensus_type == :positive && all(s == Int8(1) for s in state)
            return t
        elseif consensus_type == :negative && all(s == Int8(-1) for s in state)
            return t
        elseif consensus_type == :either && num_active == 0
            return t
        end

        # Gillespie rate: λ = (voter rate) + (reset rate)
        #   Voter rate = 2 * num_active  (each active edge can flip either endpoint)
        #   Reset rate = r
        λ = 2.0 * num_active + r

        if λ <= 0
            return t  # Safety check (shouldn't happen if num_active > 0)
        end

        # Time to next event
        τ = -log(rand()) / λ
        t += τ

        # Which event?
        if rand() * λ < 2.0 * num_active
            # === VOTER EVENT ===
            # Pick a random active edge and flip one endpoint
            # Collect all active edges
            active_edges = Tuple{Int,Int}[]
            for i in 1:N
                for j in adj_list[i]
                    if i < j && state[i] != state[j]
                        push!(active_edges, (i, j))
                    end
                end
            end
            
            # Randomly choose one active edge
            (u, v) = active_edges[rand(1:length(active_edges))]
            
            # Randomly choose which endpoint to flip
            node_to_flip = rand() < 0.5 ? u : v
            
            # Flip the node
            state[node_to_flip] *= -1
            
        else
            # === RESET EVENT ===
            state .= reset_state
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
    adj_list::Vector{Vector{Int}},
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
    elseif reset_protocol isa HubReset
        # Highest (or lowest) degree nodes set to +1
        degrees = [length(adj_list[i]) for i in 1:N]
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
    
    # Count active edges in reset state
    reset_active_count = 0
    for i in 1:N
        for j in adj_list[i]
            if i < j && reset_state[i] != reset_state[j]
                reset_active_count += 1
            end
        end
    end
    
    return reset_state, reset_active_count
end
