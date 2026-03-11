# =============================================================================
# all_to_all/pdf_simulation.jl
# =============================================================================
#
# Voter model on the complete graph (all-to-all coupling) with stochastic
# resetting.  The key simplification compared to a general network is that the
# only relevant variable is n, the number of agents in state +1.  The full
# node-state vector is never needed.
#
# Physical picture
# ----------------
# N agents sit on a complete graph.  At every infinitesimal time step dt:
#
#   - With rate  w = 2 n(N-n)/(N-1)  a voter event occurs:
#       a random discordant pair is chosen and one of them copies the other.
#       n either increases or decreases by 1 with equal probability 1/2.
#
#   - With rate  r  a resetting event occurs:
#       the system is snapped to the state prescribed by the reset protocol.
#
# Total rate:  λ = w + r
# Wait time:   dt ~ Exp(λ)   (Gillespie / continuous-time Monte Carlo)
#
# The factor 2(N-1) in the denominator of w is the total number of directed
# edges in the complete graph.  See the master-equation document for derivation.
#
# =============================================================================


# voter_event_rate: total rate of all voter-flip events.
#
#   w = 2 n (N-n) / (N-1)
#
# Interpretation: there are n*(N-n) discordant pairs; each pair contributes
# a rate 2/(N-1) because we pick one of the (N-1) potential neighbors of a
# randomly chosen node.  Returns 0 for N ≤ 1 (trivial consensus).
function voter_event_rate(N::Integer, n::Integer)
    if N <= 1
        return 0.0
    end

    return 2.0 * n * (N - n) / (N - 1)
end


# =============================================================================
# apply_all_to_all_reset — dispatch table
# =============================================================================
#
# One method per resetting protocol.  Each method returns the *new value of n*
# (the number of +1 agents) after the reset event.  The simulation loop then
# uses that value directly.
#
# Arguments common to all methods:
#   protocol         – the concrete resetting-protocol struct
#   params           – AllToAllParams (N, r, m0)
#   initial_positive – n at t=0 (useful for FunctionalReset)
#   current_positive – n just before the reset event
#   current_time     – continuous time just before the reset event

# DeltaReset: reset to a fixed magnetization → fixed n every time.
function apply_all_to_all_reset(protocol::DeltaReset,
        params::AllToAllParams, initial_positive::Int, current_positive::Int, current_time::Float64)
    return positive_count_from_magnetization(params.N, protocol.target_magnetization)
end

# StateVectorReset: use the number of +1 entries in the stored state vector.
# (In the all-to-all model only the count matters, not which nodes are +1.)
function apply_all_to_all_reset(protocol::StateVectorReset,
        params::AllToAllParams, initial_positive::Int, current_positive::Int, current_time::Float64)
    length(protocol.state) == params.N ||
        throw(ArgumentError("State-vector resets for all-to-all dynamics must have length N."))
    return count(==(Int8(1)), protocol.state)
end

# UniformMagnetizationReset: draw n uniformly from {0, 1, …, N}.
# This is the "uniform resetting distribution" studied in the theory docs.
function apply_all_to_all_reset(protocol::UniformMagnetizationReset,
        params::AllToAllParams, initial_positive::Int, current_positive::Int, current_time::Float64)
    return rand(0:params.N)
end

# RandomNodeReset: each of the N agents independently becomes +1 with
# probability p = (target_magnetization + 1)/2.  The resulting n is binomial,
# not fixed — so this produces a distribution of reset states.
function apply_all_to_all_reset(protocol::RandomNodeReset,
        params::AllToAllParams, initial_positive::Int, current_positive::Int, current_time::Float64)
    probability = (protocol.target_magnetization + 1.0) / 2.0
    positive_count = 0

    for _ in 1:params.N
        positive_count += rand() < probability
    end

    return positive_count
end

# HubReset: not applicable to all-to-all (every node has the same degree).
function apply_all_to_all_reset(protocol::HubReset,
        params::AllToAllParams, initial_positive::Int, current_positive::Int, current_time::Float64)
    throw(ArgumentError("hub_reset is only meaningful for complex-network simulations."))
end

# FunctionalReset: call the user-supplied function and interpret its return
# value as either an integer count or a magnetization.
function apply_all_to_all_reset(protocol::FunctionalReset,
        params::AllToAllParams, initial_positive::Int, current_positive::Int, current_time::Float64)
    proposal = protocol.f(params, current_positive, current_time, initial_positive)

    if proposal isa Integer
        return clamp(Int(proposal), 0, params.N)
    elseif proposal isa Real
        return positive_count_from_magnetization(params.N, proposal)
    end

    throw(ArgumentError("Custom all-to-all reset functions must return either an integer count or a magnetization."))
end


# =============================================================================
# simulate_all_to_all_trajectory
# =============================================================================
#
# Runs *one* Gillespie trajectory from t=0 to the last entry in sorted_times
# and records the magnetization m = 2n/N - 1 at each checkpoint.
#
# The simulation is event-driven (continuous time).  The inner while-loop
# advances events until the next requested checkpoint time is reached, then
# exits to record m and moves on to the next checkpoint.  This means a single
# trajectory call can serve multiple observation times without restarting.
#
# Arguments:
#   params       – AllToAllParams
#   sorted_times – checkpoints in ascending order (caller must sort them)
#   reset        – resetting protocol
#
# Returns:  Vector{Float64} of length length(sorted_times).
function simulate_all_to_all_trajectory(params::AllToAllParams,
        sorted_times::AbstractVector{<:Real}; reset::AbstractResetProtocol)
    # n is the integer count of +1 agents (all state we need for all-to-all)
    initial_positive = positive_count_from_magnetization(params.N, params.m0)
    positive_count = initial_positive
    trajectory = zeros(Float64, length(sorted_times))
    current_time = 0.0

    for time_index in eachindex(sorted_times)
        target_time = Float64(sorted_times[time_index])

        # Advance the Gillespie process up to target_time
        while current_time < target_time
            total_rate = params.r + voter_event_rate(params.N, positive_count)

            # If we are in an absorbing consensus state AND r=0, nothing can happen
            total_rate > 0 || break

            # Sample the waiting time: dt ~ Exp(total_rate)
            dt = -log(rand()) / total_rate

            # Would the next event overshoot the checkpoint?  If so, stop here
            # and record the current magnetization (no event happens in [t, t+dt))
            if current_time + dt > target_time
                break
            end

            current_time += dt

            # Decide which type of event occurred
            if rand() < params.r / total_rate
                # --- Resetting event ---
                positive_count = apply_all_to_all_reset(
                    reset, params, initial_positive, positive_count, current_time)
            else
                # --- Voter event ---
                # Pick one of the two discordant nodes at random and flip it.
                # Because the complete graph is symmetric, all that matters is
                # whether n increases or decreases by 1 (equal probability).
                positive_count += rand() < 0.5 ? 1 : -1
            end
        end

        # Record the magnetization at this checkpoint
        trajectory[time_index] = magnetization_from_positive_count(positive_count, params.N)
    end

    return trajectory
end


# =============================================================================
# simulate_pdf_all_to_all  — public entry point
# =============================================================================
#
# Runs `nsamples` independent trajectories of the all-to-all voter model with
# stochastic resetting and returns the empirical PDF of the global magnetization
# m(t) at each requested observation time.
#
# Keyword arguments:
#   reset       – resetting protocol (default: delta_reset back to m0)
#   times       – vector of observation times (any order, duplicates allowed)
#   bins        – number of histogram bins or explicit edge vector
#   nsamples    – number of independent trajectories
#   value_range – range of the magnetization histogram (default (-1, 1))
#
# Returns a PDFSimulationResult (see simulation_core.jl).
function simulate_pdf_all_to_all(params::AllToAllParams; reset::AbstractResetProtocol = delta_reset(params.m0),
        times, bins = 50, nsamples::Integer = 1000,
        value_range::Tuple{<:Real, <:Real} = (-1.0, 1.0))
    times_float = Float64.(collect(times))
    all(t -> t >= 0, times_float) || throw(ArgumentError("Observation times must be non-negative."))

    # Run all trajectories using the generic sampling loop from simulation_core.
    # The do-block is the simulator closure: given observation times, it runs
    # one trajectory and returns the magnetization vector.
    sample_matrix = sample_timeseries(times_float; nsamples = nsamples) do observation_times
        simulate_all_to_all_trajectory(params, observation_times; reset = reset)
    end

    # Bin all sampled magnetizations into the histogram
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
