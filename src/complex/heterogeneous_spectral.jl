# =============================================================================
# complex/heterogeneous_spectral.jl
# =============================================================================
#
# Section VI-style reduced spectral workflow for uncorrelated heterogeneous
# networks. This follows the degree-class propagator approximation and builds
# a reduced Markov operator over alpha_k counts.
#
# Public entry points:
#   - section_vi_spectral_gap(graph)
#   - section_vi_mfpt(graph, m0; r=0.0)
#   - section_vi_mfpt_vs_m0(graph; m0_values, r=0.0)
#
# Notes:
#   - The user-facing reset parameter r is mapped internally to an effective
#     discrete reset probability r_eff = r / N, matching the convention used
#     in the all-to-all discrete spectral routine.
#   - State-space size grows as prod_k (N_k + 1). A max_states guard is
#     included to avoid impractical constructions.
#
# =============================================================================

using SparseArrays

const _section_vi_cache = Dict{Tuple{Vararg{Int}}, NamedTuple{(:N, :k_values, :N_k, :mu1, :mu2, :Wmax, :Q), Tuple{Int, Vector{Int}, Vector{Int}, Float64, Float64, Int, SparseMatrixCSC{Float64, Int}}}}()

function _degree_classes(graph::AbstractGraph)
    deg = degree(graph)
    N = length(deg)
    counts = Dict{Int, Int}()
    for k in deg
        counts[k] = get(counts, k, 0) + 1
    end
    k_values = sort(collect(keys(counts)))
    N_k = [counts[k] for k in k_values]

    mu1 = sum(Float64(k) * Float64(nk) for (k, nk) in zip(k_values, N_k)) / N
    mu2 = sum(Float64(k * k) * Float64(nk) for (k, nk) in zip(k_values, N_k)) / N

    return N, k_values, N_k, mu1, mu2
end

function _build_section_vi_projected_operator(N::Int, k_values::Vector{Int}, N_k::Vector{Int})
    # Section VI closure (Eq. 63) ties class occupancies to a single weighted
    # opinion coordinate w = k·n. We therefore build a projected 1D reduced
    # chain over w ∈ {0, ..., N*mu1}.
    Wmax = Int(sum(k * nk for (k, nk) in zip(k_values, N_k)))
    ntrans = max(Wmax - 1, 0)

    if ntrans == 0
        return sparse(Int[], Int[], Float64[], 0, 0), Wmax
    end

    rows = Int[]
    cols = Int[]
    vals = Float64[]
    sizehint!(rows, max(6 * ntrans, 1))
    sizehint!(cols, max(6 * ntrans, 1))
    sizehint!(vals, max(6 * ntrans, 1))

    @inbounds for w in 1:(Wmax - 1)
        i = w
        θ = Float64(w) / Float64(Wmax)
        row_out = 0.0

        for j in eachindex(k_values)
            k = k_values[j]
            nk = N_k[j]

            # From Eq. (64) with Eq. (63) projection: class-k events contribute
            # symmetric ±k jumps in w with probability proportional to N_k / N.
            p = θ * (1.0 - θ) * (Float64(nk) / Float64(N))

            if p > 0.0
                w_plus = w + k
                if w_plus <= (Wmax - 1)
                    push!(rows, i)
                    push!(cols, w_plus)
                    push!(vals, p)
                end
                row_out += p

                w_minus = w - k
                if w_minus >= 1
                    push!(rows, i)
                    push!(cols, w_minus)
                    push!(vals, p)
                end
                row_out += p
            end
        end

        stay = 1.0 - row_out
        stay >= -1e-12 || throw(ArgumentError("Invalid reduced transition row with negative stay probability=$(stay)."))
        stay = max(stay, 0.0)
        push!(rows, i)
        push!(cols, i)
        push!(vals, stay)
    end

    return sparse(rows, cols, vals, ntrans, ntrans), Wmax
end

function _get_section_vi_cache(graph::AbstractGraph; max_states::Int=200_000)
    N, k_values, N_k, mu1, mu2 = _degree_classes(graph)
    key = Tuple(vcat(k_values, -1, N_k))

    return get!(_section_vi_cache, key) do
        Wmax = Int(round(N * mu1))
        Wmax + 1 <= max_states || throw(ArgumentError("Section VI projected state space too large: $(Wmax + 1) > max_states=$(max_states)."))

        Q, Wmax_exact = _build_section_vi_projected_operator(N, k_values, N_k)

        (
            N = N,
            k_values = k_values,
            N_k = N_k,
            mu1 = mu1,
            mu2 = mu2,
            Wmax = Wmax_exact,
            Q = Q,
        )
    end
end

function _dominant_eigenvalue_power(Q::SparseMatrixCSC{Float64, Int}; maxiter::Int=5_000, tol::Float64=1e-11)
    n = size(Q, 1)
    n == 0 && return 0.0

    x = fill(1.0 / n, n)
    λ_prev = 0.0
    for _ in 1:maxiter
        y = Q * x
        λ = norm(y, Inf)
        λ == 0.0 && return 0.0
        x_new = y / λ
        if abs(λ - λ_prev) <= tol && norm(x_new - x, Inf) <= 10tol
            return λ
        end
        x = x_new
        λ_prev = λ
    end
    return λ_prev
end

function _survival_generating_vector(Q::SparseMatrixCSC{Float64, Int}, z::Float64)
    n = size(Q, 1)
    n == 0 && return Float64[]

    A = I - z * Q
    rhs = ones(Float64, n)
    return A \ rhs
end

function _w_from_m0(Wmax::Int, m0::Real)
    abs(m0) <= 1 || throw(ArgumentError("m0 must lie in [-1, 1]."))
    p_plus = (Float64(m0) + 1.0) / 2.0
    return clamp(round(Int, Wmax * p_plus), 0, Wmax)
end

"""
    section_vi_spectral_gap(graph; max_states=200_000, maxiter=5_000, tol=1e-11)

Compute the reduced Section VI spectral summary for an uncorrelated heterogeneous
network approximation.

Returns a NamedTuple with degree moments and dominant transient eigenvalue.
"""
function section_vi_spectral_gap(
    graph::AbstractGraph;
    max_states::Int=200_000,
    maxiter::Int=5_000,
    tol::Float64=1e-11,
)
    cache = _get_section_vi_cache(graph; max_states=max_states)
    Q = cache.Q

    ntrans = size(Q, 1)
    λ2 = ntrans <= 2_000 ? maximum(abs.(eigvals(Matrix(Q)))) : _dominant_eigenvalue_power(Q; maxiter=maxiter, tol=tol)
    gap = 1.0 - λ2

    return (
        N = cache.N,
        k_values = cache.k_values,
        N_k = cache.N_k,
        mu1 = cache.mu1,
        mu2 = cache.mu2,
        Wmax = cache.Wmax,
        transient_states = ntrans,
        lambda2 = λ2,
        spectral_gap = gap,
        asymptotic_gap_scale = cache.mu2 / (cache.N^2 * cache.mu1^2),
        ratio_gap_to_asymptotic = gap / (cache.mu2 / (cache.N^2 * cache.mu1^2)),
    )
end

"""
    section_vi_mfpt(graph, m0; r=0.0, max_states=200_000)

Mean first-passage time from the Section VI reduced spectral operator.

- `m0` sets the reset/initial degree-class occupancy by alpha_k = round(N_k*(1+m0)/2).
- User-facing `r` is converted to `r_eff = r/N`, then
    MFPT = S_tilde(1-r_eff) / (1 - r_eff*S_tilde(1-r_eff)).
"""
function section_vi_mfpt(
    graph::AbstractGraph,
    m0::Real;
    r::Real=0.0,
    max_states::Int=200_000,
)
        rf = Float64(r)
        rf >= 0.0 || throw(ArgumentError("r must satisfy r >= 0."))

    cache = _get_section_vi_cache(graph; max_states=max_states)
    r_eff = rf / cache.N
    0.0 <= r_eff < 1.0 || throw(ArgumentError("Effective reset probability r/N must satisfy 0 <= r/N < 1."))

    w0 = _w_from_m0(cache.Wmax, m0)

    # Consensus states are absorbing by construction.
    if w0 == 0 || w0 == cache.Wmax
        return 0.0
    end

    idx0 = w0

    z = 1.0 - r_eff
    s_vec = _survival_generating_vector(cache.Q, z)
    s = s_vec[idx0]

    return r_eff == 0.0 ? s : s / (1.0 - r_eff * s)
end

"""
    section_vi_mfpt_vs_m0(graph; m0_values=range(-0.9, 0.9, length=31), r=0.0, max_states=200_000)

Compute Section VI reduced spectral MFPT for a sweep of initial magnetizations.

User-facing `r` is converted internally to `r_eff = r/N`.
"""
function section_vi_mfpt_vs_m0(
    graph::AbstractGraph;
    m0_values=collect(range(-0.9, 0.9, length=31)),
    r::Real=0.0,
    max_states::Int=200_000,
)
    rf = Float64(r)
    rf >= 0.0 || throw(ArgumentError("r must satisfy r >= 0."))

    cache = _get_section_vi_cache(graph; max_states=max_states)
    r_eff = rf / cache.N
    0.0 <= r_eff < 1.0 || throw(ArgumentError("Effective reset probability r/N must satisfy 0 <= r/N < 1."))

    z = 1.0 - r_eff
    s_vec = _survival_generating_vector(cache.Q, z)

    m0_vec = Float64.(collect(m0_values))
    mfpt_vals = similar(m0_vec)

    @inbounds for i in eachindex(m0_vec)
        w0 = _w_from_m0(cache.Wmax, m0_vec[i])
        if w0 == 0 || w0 == cache.Wmax
            mfpt_vals[i] = 0.0
            continue
        end

        idx0 = w0

        s = s_vec[idx0]
        mfpt_vals[i] = r_eff == 0.0 ? s : s / (1.0 - r_eff * s)
    end

    return (
        m0_values = m0_vec,
        mfpt_values = mfpt_vals,
        r = rf,
        r_eff = r_eff,
        N = cache.N,
        k_values = cache.k_values,
        N_k = cache.N_k,
        mu1 = cache.mu1,
        mu2 = cache.mu2,
        Wmax = cache.Wmax,
        transient_states = size(cache.Q, 1),
    )
end
