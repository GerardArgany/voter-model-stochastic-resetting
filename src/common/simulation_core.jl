# =============================================================================
# common/simulation_core.jl
# =============================================================================
#
# This file defines everything that is shared between the all-to-all and
# complex-network simulations:
#
#   1. Parameter structs  (AllToAllParams, ComplexParams)
#   2. Result container   (PDFSimulationResult)
#   3. Resetting-protocol types and their public constructors
#   4. Small maths helpers (magnetization ↔ positive-node-count)
#   5. Histogram / PDF estimation
#   6. The generic Monte Carlo sampling loop
#
# =============================================================================


# -----------------------------------------------------------------------------
# Section 1 – Abstract base type for resetting protocols
# -----------------------------------------------------------------------------
#
# Every concrete resetting strategy is a subtype of AbstractResetProtocol.
# This lets the simulation functions dispatch on the protocol type without
# needing if/elseif chains, which would break every time a new protocol is
# added.  Julia's multiple dispatch handles the routing automatically.

abstract type AbstractResetProtocol end


# -----------------------------------------------------------------------------
# Section 2 – Parameter structs
# -----------------------------------------------------------------------------

# AllToAllParams bundles the three numbers that fully describe any all-to-all
# voter-model-with-resetting run:
#
#   N   – number of agents (nodes in the complete graph)
#   r   – Poisson resetting rate  (events per unit time)
#   m0  – initial magnetization   m = (n_+ - n_-) / N ∈ [-1, 1]
#         where n_+ is the number of agents in state +1.
#
# The inner constructor validates the inputs so that invalid parameter
# combinations are caught immediately at construction time rather than
# leading to silent nonsense inside the simulation.
struct AllToAllParams
    N::Int
    r::Float64
    m0::Float64

    function AllToAllParams(N::Integer, r::Real, m0::Real)
        N >= 1 || throw(ArgumentError("N must be at least 1."))
        r >= 0 || throw(ArgumentError("The resetting rate r must be non-negative."))
        abs(m0) <= 1 || throw(ArgumentError("The magnetization m0 must lie in [-1, 1]."))
        new(Int(N), Float64(r), Float64(m0))
    end
end

# ComplexParams is the analogue for complex-network runs.  The network itself
# is passed separately as a Graphs.jl AbstractGraph, so only the two
# dynamical parameters live here.
struct ComplexParams
    r::Float64
    m0::Float64

    function ComplexParams(r::Real, m0::Real)
        r >= 0 || throw(ArgumentError("The resetting rate r must be non-negative."))
        abs(m0) <= 1 || throw(ArgumentError("The magnetization m0 must lie in [-1, 1]."))
        new(Float64(r), Float64(m0))
    end
end


# -----------------------------------------------------------------------------
# Section 3 – Result container
# -----------------------------------------------------------------------------
#
# Both simulate_pdf_all_to_all and simulate_pdf_complex return a
# PDFSimulationResult so that downstream plotting code only needs to know one
# data shape regardless of which topology was simulated.
#
# Fields:
#   times       – the observation times you requested, in original order
#   bin_edges   – (nbins+1) edge values of the magnetization histogram
#   bin_centers – (nbins)   midpoints of each bin, useful for plotting x-axis
#   densities   – (ntimes × nbins) matrix of normalized probability density
#                 values.  densities[i, :] is the PDF at times[i].
#                 Normalization:  sum(densities[i, :] .* bin_widths) ≈ 1
#   counts      – (ntimes × nbins) raw sample counts before normalization
#   samples     – (nsamples × ntimes) raw magnetization values from every
#                 individual trajectory, kept so you can compute any other
#                 statistic (variance, higher moments, etc.) after the fact
struct PDFSimulationResult
    times::Vector{Float64}
    bin_edges::Vector{Float64}
    bin_centers::Vector{Float64}
    densities::Matrix{Float64}
    counts::Matrix{Int}
    samples::Matrix{Float64}
end

# FPTSimulationResult contains statistics for first-passage-time distributions.
# Unlike PDFSimulationResult (which samples at fixed times), FPT collapses time
# into a single absorbing time per trajectory.
#
# Fields:
#   times        – (nsamples) raw FPT values from each independent run
#   bin_edges    – (nbins+1) histogram bin edges
#   bin_centers  – (nbins) bin midpoints, for plotting x-axis
#   densities    – (nbins) normalized histogram density ≈ counts / (sum(counts)*bin_width)
#   counts       – (nbins) raw sample counts per bin
#   mean_fpt     – ensemble mean ⟨T⟩
#   std_fpt      – ensemble standard deviation σ(T)
struct FPTSimulationResult
    times::Vector{Float64}
    bin_edges::Vector{Float64}
    bin_centers::Vector{Float64}
    densities::Vector{Float64}
    counts::Vector{Int}
    mean_fpt::Float64
    std_fpt::Float64
end

# DegreePairEvolutionResult stores ensemble-averaged time evolution for one
# selected (k, m) pair on a complex graph:
#   s_values[t] = s_{k,m}(t), fraction of all nodes that have degree k,
#                 are in state -1, and have m neighbors in state +1.
#   i_values[t] = i_{k,m}(t), same but node state +1.
struct DegreePairEvolutionResult
    times::Vector{Float64}
    k::Int
    m::Int
    s_values::Vector{Float64}
    i_values::Vector{Float64}
    nsamples::Int
end

# DegreeGridEvolutionResult stores ensemble-averaged time evolution for all
# observed degree classes and all valid neighbor counts m = 0...k.
#
# k_values defines the degree-class ordering. For each index j:
#   s_values[j] is a Matrix{Float64} of size (ntimes, k_values[j]+1), where
#   s_values[j][t, m+1] = s_{k,m}(t).
#   i_values[j] is the analogous matrix for i_{k,m}(t).
struct DegreeGridEvolutionResult
    times::Vector{Float64}
    k_values::Vector{Int}
    s_values::Vector{Matrix{Float64}}
    i_values::Vector{Matrix{Float64}}
    nsamples::Int
end

# AMESteadyStateResult: fixed-point s_{k,m}* and i_{k,m}* from the AME equations
# (d/dt = 0 solution). s_values[j][m+1] = s_{k_j, m} at steady state.
struct AMESteadyStateResult
    k_values::Vector{Int}
    s_values::Vector{Vector{Float64}}
    i_values::Vector{Vector{Float64}}
    Pk::Dict{Int,Float64}
    rho0::Float64
    r::Float64
end

# AMEEvolutionResult: time-dependent s_{k,m}(t) and i_{k,m}(t) from the AME ODE.
# s_values[j] is a (ntimes × k_j+1) Matrix where s_values[j][t, m+1] = s_{k_j, m}(t).
struct AMEEvolutionResult
    times::Vector{Float64}
    k_values::Vector{Int}
    s_values::Vector{Matrix{Float64}}
    i_values::Vector{Matrix{Float64}}
    Pk::Dict{Int,Float64}
    rho0::Float64
    r::Float64
end


# -----------------------------------------------------------------------------
# Section 4 – Resetting-protocol concrete types
# -----------------------------------------------------------------------------

# DeltaReset: the system is reset to a *fixed* magnetization target_magnetization.
#   All-to-all: n_+ is set to round(N*(target+1)/2), same every time.
#   Complex:    nodes are assigned +1/-1 randomly but with the exact right
#               count (a random permutation of the fixed-m state) unless
#               HubReset is used instead (see below).
struct DeltaReset <: AbstractResetProtocol
    target_magnetization::Float64
end

# StateVectorReset: the system is reset to a *specific* node-state vector.
#   Every reset puts every node in exactly the state specified.
#   Only meaningful when the same node arrangement matters every time
#   (e.g. the exact same hub nodes are +1 at each reset).
struct StateVectorReset <: AbstractResetProtocol
    state::Vector{Int8}   # one Int8 per node, values ∈ {-1, +1}
end

# UniformMagnetizationReset: at each reset event the target magnetization is
#   drawn uniformly from the full range of possible values.  For all-to-all
#   this means n_+ is drawn uniformly from {0, 1, …, N}.  Not yet implemented
#   for complex networks.
struct UniformMagnetizationReset <: AbstractResetProtocol end

# RandomNodeReset: the system is reset to a random node configuration whose
#   *expected* magnetization equals target_magnetization.  Each node is
#   independently set to +1 with probability p = (target+1)/2.  Unlike
#   DeltaReset the resulting magnetization varies from reset to reset.
struct RandomNodeReset <: AbstractResetProtocol
    target_magnetization::Float64
end

# HubReset: the system is reset to a fixed configuration chosen by *degree*.
#   The round(N*(target+1)/2) nodes with the highest degrees are set to +1
#   (highest_first=true, the default), or the lowest-degree nodes (false).
#   This is the protocol studied in the complex-network theory documents;
#   it is the only protocol for which the reset state is precomputed and
#   cached for speed, because the reset target never changes.
struct HubReset <: AbstractResetProtocol
    target_magnetization::Float64
    highest_first::Bool        # true → hubs are +1 at reset, false → hubs are -1
end

# FunctionalReset: escape hatch for any custom protocol not covered above.
#   The user supplies a Julia function f with the signature
#
#     all-to-all:  f(params, current_count, current_time, initial_count)
#     complex:     f(graph, current_state, params, current_time)
#
#   See the apply_*_reset methods in pdf_simulation.jl for the exact
#   signatures expected per topology.
struct FunctionalReset{F} <: AbstractResetProtocol
    f::F
end


# -- Public constructors -------------------------------------------------------
# These are the names exported in VoterResetting.jl and meant to be called
# from notebook code.  They validate inputs and return the appropriate struct.

# delta_reset(0.3)           → DeltaReset with target m = 0.3
# delta_reset([1,-1,1,...])  → StateVectorReset from an explicit node state
delta_reset(target::Real) = DeltaReset(validate_magnetization(Float64(target)))
delta_reset(target::AbstractVector{<:Integer}) = StateVectorReset(normalize_state_vector(target))

# uniform_reset()  → UniformMagnetizationReset (no parameters needed)
uniform_reset() = UniformMagnetizationReset()

# random_node_reset(0.5)  → RandomNodeReset, each node flipped independently
random_node_reset(target::Real) = RandomNodeReset(validate_magnetization(Float64(target)))

# hub_reset(0.0)             → HubReset, hubs set to +1
# hub_reset(0.0; highest=false) → HubReset, hubs set to -1
hub_reset(target::Real; highest::Bool = true) = HubReset(validate_magnetization(Float64(target)), highest)

# custom_reset(f)  → FunctionalReset wrapping any callable f
custom_reset(f::F) where {F} = FunctionalReset{F}(f)


# -- Input validation helpers --------------------------------------------------

function validate_magnetization(m::Float64)
    abs(m) <= 1 || throw(ArgumentError("Magnetization values must lie in [-1, 1]."))
    return m
end

function normalize_state_vector(state::AbstractVector{<:Integer})
    isempty(state) && throw(ArgumentError("Reset state vectors cannot be empty."))
    result = Int8.(state)
    all(value -> value == Int8(-1) || value == Int8(1), result) ||
        throw(ArgumentError("Reset state vectors must contain only -1 and 1."))
    return result
end


# -----------------------------------------------------------------------------
# Section 5 – Magnetization ↔ positive-count conversions
# -----------------------------------------------------------------------------
#
# The all-to-all simulation tracks n (integer count of +1 agents) internally
# because that avoids floating-point arithmetic inside the hot loop.  These
# two tiny functions convert between the internal representation and the
# physical observable m = (n_+ - n_-)/N = 2n/N - 1.

# n → m
magnetization_from_positive_count(n::Integer, N::Integer) = 2.0 * n / N - 1.0

# m → n  (rounds toward zero, clamps to [0, N] to handle floating-point edge cases)
function positive_count_from_magnetization(N::Integer, m::Real)
    validate_magnetization(Float64(m))
    return clamp(trunc(Int, N * (Float64(m) + 1.0) / 2.0), 0, N)
end


# No time-sorting machinery needed: user passes pre-sorted times (e.g., linspace).


# -----------------------------------------------------------------------------
# Section 7 – Histogram / PDF estimation
# -----------------------------------------------------------------------------

# resolve_bin_edges: accepts either
#   - an Integer (number of equal-width bins spanning value_range), or
#   - a sorted vector of edge values (custom bin layout).
# Returns a Vector{Float64} of (nbins+1) edge positions.
function resolve_bin_edges(bins; value_range::Tuple{<:Real, <:Real} = (-1.0, 1.0))
    left, right = Float64(value_range[1]), Float64(value_range[2])
    left < right || throw(ArgumentError("The histogram range must satisfy left < right."))

    if bins isa Integer
        bins > 0 || throw(ArgumentError("The number of histogram bins must be positive."))
        # length = nbins+1 gives exactly nbins equal-width intervals
        return collect(range(left, right; length = Int(bins) + 1))
    end

    edges = Float64.(collect(bins))
    length(edges) >= 2 || throw(ArgumentError("At least two bin edges are required."))
    issorted(edges) || throw(ArgumentError("Bin edges must be sorted."))

    return edges
end

# Midpoint of each bin — the x-axis coordinates for a bar/line plot of the PDF.
bin_centers_from_edges(edges::AbstractVector{<:Real}) =
    [(Float64(edges[i]) + Float64(edges[i + 1])) / 2.0 for i in 1:(length(edges) - 1)]

# histogram_counts: counts how many values fall in each bin.
# Convention: each bin is the half-open interval [left_edge, right_edge),
# except the last bin which is closed on the right [left, right].
# Values outside the full range are silently ignored (they represent
# consensus states ±1 when using interior bins).
function histogram_counts(values::AbstractVector{<:Real}, edges::AbstractVector{<:Real})
    nbins = length(edges) - 1
    counts = zeros(Int, nbins)

    for raw_value in values
        value = Float64(raw_value)

        # Outside the histogram range — skip (don't raise an error)
        if value < edges[1] || value > edges[end]
            continue
        end

        # Exactly at the right boundary → goes into the last bin
        if value == edges[end]
            counts[end] += 1
            continue
        end

        # searchsortedlast returns the index of the largest edge ≤ value,
        # which is the left edge of the bin that contains value.
        bin_index = searchsortedlast(edges, value)
        if 1 <= bin_index <= nbins && edges[bin_index] <= value < edges[bin_index + 1]
            counts[bin_index] += 1
        end
    end

    return counts
end

# counts_to_density: converts raw counts to a probability density.
# The density is normalized so that  ∑ density[i] * width[i] = 1,
# i.e. the area under the returned histogram equals 1.
function counts_to_density(counts::AbstractVector{<:Integer}, edges::AbstractVector{<:Real})
    widths = diff(Float64.(edges))  # vector of bin widths
    total = sum(counts)

    if total == 0
        return zeros(Float64, length(counts))
    end

    return Float64.(counts) ./ (total .* widths)
end

# magnetization_pdf: public helper to compute a PDF from a plain vector of
# magnetization samples.  Useful if you have run your own simulations and
# just want the PDF estimation without the full simulate_pdf_* pipeline.
#
# Returns a NamedTuple with fields:
#   bin_edges, bin_centers, counts, density
function magnetization_pdf(samples::AbstractVector{<:Real}; bins = 50,
        value_range::Tuple{<:Real, <:Real} = (-1.0, 1.0), normalize::Bool = true)
    edges = resolve_bin_edges(bins; value_range = value_range)
    counts = histogram_counts(samples, edges)
    densities = normalize ? counts_to_density(counts, edges) : Float64.(counts)

    return (
        bin_edges = edges,
        bin_centers = bin_centers_from_edges(edges),
        counts = counts,
        density = densities,
    )
end

# summarize_pdf_samples: converts the full (nsamples × ntimes) matrix of raw
# magnetization values produced by sample_timeseries into a (ntimes × nbins)
# counts matrix and a (ntimes × nbins) density matrix.
# Row i of the output corresponds to observation time i.
function summarize_pdf_samples(samples::AbstractMatrix{<:Real}, edges::AbstractVector{<:Real})
    nsamples, ntimes = size(samples)
    nbins = length(edges) - 1
    counts = zeros(Int, ntimes, nbins)
    densities = zeros(Float64, ntimes, nbins)

    for time_index in 1:ntimes
        # view(...) avoids allocating a copy for each slice
        counts_row = histogram_counts(view(samples, :, time_index), edges)
        counts[time_index, :] .= counts_row
        densities[time_index, :] .= counts_to_density(counts_row, edges)
    end

    return counts, densities
end


# -----------------------------------------------------------------------------
# Section 8 – Generic Monte Carlo sampling loop
# -----------------------------------------------------------------------------
#
# sample_timeseries runs `nsamples` independent trajectories and collects the
# magnetization at each checkpoint time.
#
# Arguments:
#   simulator    – a callable with signature (sorted_times) → Vector{Float64}
#                  returning one value per checkpoint.
#   sorted_times – checkpoints in ascending order.
#   nsamples     – number of independent trajectories.
#
# Returns:  (nsamples × ntimes) Matrix{Float64}.
#           Column j holds all sample values at sorted_times[j].
function sample_timeseries(simulator, sorted_times::AbstractVector{<:Real};
    nsamples::Integer)
    nsamples > 0 || throw(ArgumentError("nsamples must be positive."))

    ntimes = length(sorted_times)
    samples = Matrix{Float64}(undef, nsamples, ntimes)

    for sample_index in 1:nsamples
        samples[sample_index, :] .= simulator(sorted_times)
    end

    return samples
end


# compute_histogram: Create a histogram from 1D data (used for FPT distributions).
# Returns (counts, edges) where counts is the bin counts and edges is the bin edges.
# Automatically determines the range from min/max of data.
function compute_histogram(data::AbstractVector{<:Real}, nbins::Integer)
    isempty(data) && error("Cannot compute histogram from empty data.")
    nbins >= 1 || error("Number of bins must be at least 1.")
    
    data_min = minimum(data)
    data_max = maximum(data)
    
    # Avoid degenerate case where all data is identical
    if data_min == data_max
        data_max = data_min + 1.0
    end
    
    # Create bin edges from min to max
    edges = range(data_min, data_max, length=nbins+1) |> collect
    
    # Count samples in each bin
    counts = histogram_counts(data, edges)
    
    return counts, edges
end

