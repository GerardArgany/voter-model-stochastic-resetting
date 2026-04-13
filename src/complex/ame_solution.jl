# =============================================================================
# complex/ame_solution.jl
# =============================================================================
#
# Numerical solution of the Approximate Master Equations (AME) for the voter
# model with stochastic resetting on complex networks.
#
# References:
#   - Gleeson (2011), "High-accuracy master equations for the voter model on
#     degree-heterogeneous networks"   (plain voter model)
#   - gleeson_approach_resetting.tex  (extension with stochastic resetting)
#
# State convention (matches the stochastic simulation):
#   s_{k,m} – fraction of ALL nodes at state −1, with degree k and m neighbours
#              at state +1.
#   i_{k,m} – fraction of ALL nodes at state +1, with degree k and m neighbours
#              at state +1.
#   (quantities are NOT P_k-weighted; P_k enters only in the β/γ rate sums)
#
# Two public entry points:
#   solve_ame_evolution     – ODE integration giving s_{k,m}(t), i_{k,m}(t)
#   solve_ame_steady_state  – NLsolve for the fixed point (d/dt = 0)
#
# Both accept either a Dict{Int,Float64} degree distribution or a Graphs.jl
# graph.  Reset protocols reuse the existing AbstractResetProtocol hierarchy:
#   delta_reset(m)          – fixed-m reset (annealed AME treats as random-node)
#   random_node_reset(m)    – same effective AME closure as delta_reset(m)
#   hub_reset(m)            – fill +1 from highest/lowest degree class first
#
# Requirements: DifferentialEquations.jl, NLsolve.jl
# =============================================================================

using DifferentialEquations
using NLsolve


# =============================================================================
# AMEIndexer – flat-vector index mapping
# =============================================================================
#
# For each degree class k (sorted ascending), two contiguous blocks of (k+1)
# entries in a single flat Float64 vector:
#
#   positions  [offset, offset+k]         → s_{k,0} … s_{k,k}
#   positions  [offset+k+1, offset+2k+1]  → i_{k,0} … i_{k,k}
#
# Only internal to this file — not exported.

struct AMEIndexer
    k_vals::Vector{Int}          # sorted unique degree classes
    k_to_j::Dict{Int,Int}        # k → index in k_vals
    offsets::Vector{Int}         # offsets[j] = first flat position for k_vals[j]
    total::Int                   # total number of variables
end

function _build_ame_indexer(k_vals::AbstractVector{<:Integer})
    sorted_k  = sort(unique(Int.(k_vals)))
    k_to_j    = Dict{Int,Int}(k => j for (j, k) in enumerate(sorted_k))
    offsets   = Vector{Int}(undef, length(sorted_k))
    counter   = 1
    for (j, k) in enumerate(sorted_k)
        offsets[j] = counter
        counter   += 2 * (k + 1)
    end
    return AMEIndexer(sorted_k, k_to_j, offsets, counter - 1)
end

@inline _idx_s(ind::AMEIndexer, k::Int, m::Int) = ind.offsets[ind.k_to_j[k]] + m
@inline _idx_i(ind::AMEIndexer, k::Int, m::Int) = ind.offsets[ind.k_to_j[k]] + (k + 1) + m


# =============================================================================
# Degree distribution helpers
# =============================================================================

function _pk_from_graph(graph::AbstractGraph)
    N  = nv(graph)
    pk = Dict{Int,Float64}()
    for k in degree(graph)
        pk[k] = get(pk, k, 0.0) + 1.0 / N
    end
    return pk
end


# =============================================================================
# Initial conditions (binomial ansatz)
# =============================================================================
#
# For a uniformly random initial state with fraction rho0 in state +1:
#   s_{k,m}(0) = (1 − rho0) · B(k, m; rho0)
#   i_{k,m}(0) = rho0       · B(k, m; rho0)
# where B(k, m; p) = C(k,m) p^m (1-p)^{k-m}.

function _ame_initial_state(ind::AMEIndexer, rho0::Float64)
    x = zeros(Float64, ind.total)
    for k in ind.k_vals
        for m in 0:k
            B = binomial(k, m) * rho0^m * (1.0 - rho0)^(k - m)
            x[_idx_s(ind, k, m)] = (1.0 - rho0) * B
            x[_idx_i(ind, k, m)] = rho0 * B
        end
    end
    return x
end


# =============================================================================
# Reset state builders (annealed AME approximation)
# =============================================================================
#
# After a reset to magnetisation m* (i.e. fraction rho* = (m*+1)/2 in +1):
#   i*_{k,m} = c_k · B(k, m; rho_eff)
#   s*_{k,m} = (1−c_k) · B(k, m; rho_eff)
# where c_k = probability that a reset node of degree k lands in state +1,
# and rho_eff = Σ_k P_k c_k is the global +1 fraction after the reset.
#
# For random / delta reset:  c_k = rho* for all k.
# For hub reset:             c_k ∈ {0, partial, 1} filled from highest/lowest k.

function _uniform_reset_state(ind::AMEIndexer, rho_reset::Float64)
    x_reset = zeros(Float64, ind.total)
    for k in ind.k_vals
        for m in 0:k
            B = binomial(k, m) * rho_reset^m * (1.0 - rho_reset)^(k - m)
            x_reset[_idx_s(ind, k, m)] = (1.0 - rho_reset) * B
            x_reset[_idx_i(ind, k, m)] = rho_reset * B
        end
    end
    return x_reset
end

function _hub_reset_state(ind::AMEIndexer, Pk::Dict{Int,Float64},
        rho_reset::Float64, highest_first::Bool)
    k_sorted   = sort(ind.k_vals; rev = highest_first)
    ck         = Dict{Int,Float64}(k => 0.0 for k in ind.k_vals)
    remaining  = rho_reset
    for k in k_sorted
        pk = get(Pk, k, 0.0)
        iszero(pk) && continue
        if remaining >= pk
            ck[k]      = 1.0
            remaining -= pk
        else
            ck[k]      = remaining / pk
            remaining  = 0.0
        end
        remaining <= 0.0 && break
    end
    # Effective global +1 fraction for the binomial neighbour distribution
    rho_eff    = sum(get(Pk, k, 0.0) * ck[k] for k in ind.k_vals)
    x_reset    = zeros(Float64, ind.total)
    for k in ind.k_vals
        c = ck[k]
        for m in 0:k
            B = binomial(k, m) * rho_eff^m * (1.0 - rho_eff)^(k - m)
            x_reset[_idx_s(ind, k, m)] = (1.0 - c) * B
            x_reset[_idx_i(ind, k, m)] = c * B
        end
    end
    return x_reset
end

# Dispatch on AbstractResetProtocol
function _build_ame_reset_state(ind::AMEIndexer, Pk::Dict{Int,Float64},
        protocol::Union{DeltaReset, RandomNodeReset})
    rho_reset = (1.0 + protocol.target_magnetization) / 2.0
    return _uniform_reset_state(ind, rho_reset)
end

function _build_ame_reset_state(ind::AMEIndexer, Pk::Dict{Int,Float64},
        protocol::HubReset)
    rho_reset = (1.0 + protocol.target_magnetization) / 2.0
    return _hub_reset_state(ind, Pk, rho_reset, protocol.highest_first)
end

function _build_ame_reset_state(::AMEIndexer, ::Dict{Int,Float64},
        ::AbstractResetProtocol)
    throw(ArgumentError("AME supports delta_reset, random_node_reset, and hub_reset only."))
end


# =============================================================================
# AME pair rates β^s, γ^s, β^i, γ^i
# =============================================================================
#
# From gleeson_approach_vm.tex:
#   β^s = [Σ_k P_k Σ_m (m(k-m)/k) s_{k,m}] / [Σ_k P_k Σ_m (k-m) s_{k,m}]
#   γ^s = [Σ_k P_k Σ_m ((k-m)²/k) i_{k,m}] / [Σ_k P_k Σ_m (k-m) i_{k,m}]
#   β^i = [Σ_k P_k Σ_m (m²/k) s_{k,m}] / [Σ_k P_k Σ_m m s_{k,m}]
#   γ^i = [Σ_k P_k Σ_m (m(k-m)/k) i_{k,m}] / [Σ_k P_k Σ_m m i_{k,m}]

function _ame_rates(x::AbstractVector{<:Real}, ind::AMEIndexer,
        Pk::Dict{Int,Float64})
    βs_n = βs_d = γs_n = γs_d = 0.0
    βi_n = βi_d = γi_n = γi_d = 0.0
    for k in ind.k_vals
        P   = get(Pk, k, 0.0)
        iszero(P) && continue
        k_f = Float64(k)
        for m in 0:k
            m_f  = Float64(m)
            km_f = k_f - m_f
            s    = x[_idx_s(ind, k, m)]
            iv   = x[_idx_i(ind, k, m)]

            βs_n += P * (m_f * km_f / k_f) * s
            βs_d += P * km_f * s

            γs_n += P * (km_f^2 / k_f) * iv
            γs_d += P * km_f * iv

            βi_n += P * (m_f^2 / k_f) * s
            βi_d += P * m_f * s

            γi_n += P * (m_f * km_f / k_f) * iv
            γi_d += P * m_f * iv
        end
    end
    eps = 1e-300
    βs = βs_d > eps ? βs_n / βs_d : 0.0
    γs = γs_d > eps ? γs_n / γs_d : 0.0
    βi = βi_d > eps ? βi_n / βi_d : 0.0
    γi = γi_d > eps ? γi_n / γi_d : 0.0
    return βs, γs, βi, γi
end


# =============================================================================
# ODE right-hand side  (also used as the steady-state residual F(x)=0)
# =============================================================================
#
# Implements the full AME with stochastic resetting:
#
#   ds_{k,m}/dt = -(m/k) s_{k,m} + ((k-m)/k) i_{k,m}
#                 - β^s(k-m) s_{k,m}   + β^s(k-m+1) s_{k,m-1}
#                 - γ^s m   s_{k,m}    + γ^s(m+1)   s_{k,m+1}
#                 + r (s*_{k,m} - s_{k,m})
#
# and analogously for i_{k,m}.  r=0 gives the plain voter model AME.
# For k=0 (isolated nodes) the voter and neighbour-flip terms are absent.

function _ame_rhs!(dx, x, p, t)
    ind, Pk, r, x_reset = p
    fill!(dx, 0.0)
    βs, γs, βi, γi = _ame_rates(x, ind, Pk)

    for k in ind.k_vals
        k_f = Float64(k)
        for m in 0:k
            is  = _idx_s(ind, k, m)
            ii  = _idx_i(ind, k, m)
            s   = x[is]
            iv  = x[ii]
            m_f = Float64(m)
            km  = k_f - m_f

            # Voter node-flip terms (vanish for isolated nodes with k=0)
            if k > 0
                volt_s =  -(m_f / k_f) * s  + (km / k_f) * iv
                volt_i =  -(km  / k_f) * iv + (m_f / k_f) * s
            else
                volt_s = 0.0
                volt_i = 0.0
            end

            # Neighbour-flip terms (boundary conditions: s_{k,-1} = s_{k,k+1} = 0)
            sm1 = m > 0 ? x[_idx_s(ind, k, m - 1)] : 0.0
            sp1 = m < k ? x[_idx_s(ind, k, m + 1)] : 0.0
            im1 = m > 0 ? x[_idx_i(ind, k, m - 1)] : 0.0
            ip1 = m < k ? x[_idx_i(ind, k, m + 1)] : 0.0

            nb_s = -βs * km * s     + βs * (km + 1) * sm1 -
                    γs * m_f * s    +  γs * (m_f + 1) * sp1
            nb_i = -βi * km * iv    + βi * (km + 1) * im1 -
                    γi * m_f * iv   +  γi * (m_f + 1) * ip1

            # Resetting term
            rst_s = r * (x_reset[is] - s)
            rst_i = r * (x_reset[ii] - iv)

            dx[is] = volt_s + nb_s + rst_s
            dx[ii] = volt_i + nb_i + rst_i
        end
    end
    return nothing
end


# =============================================================================
# Helper: extract s/i grids from flat state vector
# =============================================================================

function _extract_ame_grids(x::AbstractVector{<:Real}, ind::AMEIndexer)
    s_vals = [Vector{Float64}(undef, k + 1) for k in ind.k_vals]
    i_vals = [Vector{Float64}(undef, k + 1) for k in ind.k_vals]
    for (j, k) in enumerate(ind.k_vals)
        for m in 0:k
            s_vals[j][m + 1] = x[_idx_s(ind, k, m)]
            i_vals[j][m + 1] = x[_idx_i(ind, k, m)]
        end
    end
    return s_vals, i_vals
end


# =============================================================================
# Public API: solve_ame_evolution
# =============================================================================

"""
    solve_ame_evolution(Pk, rho0, r, times; initial_condition=:rho, reset=..., ...)
    solve_ame_evolution(graph, rho0, r, times; ...)

Integrate the AME ODEs for the voter model with stochastic resetting and return
the time evolution of `s_{k,m}(t)` and `i_{k,m}(t)`.

# Arguments
- `Pk`: `Dict{Int,Float64}` mapping degree → probability, or a Graphs.jl graph.
- `rho0`: initial value interpreted according to `initial_condition`:
    `:rho` means fraction in +1, `:m` means magnetization.
- `initial_condition`: `:rho` (default) or `:m`.
- `r`: stochastic resetting rate (r=0 = plain voter model).
- `times`: observation times (sorted internally).
- `reset`: resetting protocol — `delta_reset(m)`, `random_node_reset(m)`,
  or `hub_reset(m)`.  Default: random reset to the initial magnetisation.
- `reltol`, `abstol`: ODE solver tolerances (default 1e-8).

# Returns
An `AMEEvolutionResult`:
- `.times`: sorted observation times
- `.k_values`: sorted unique degree classes
- `.s_values[j]`: `(ntimes × k_j+1)` matrix, `s_values[j][t, m+1]` = s_{k_j,m}(t)
- `.i_values[j]`: analogous for i_{k_j,m}(t)
- `.Pk`, `.rho0`, `.r`: parameters used

Uses `Tsit5()` from DifferentialEquations.jl.
"""
function _resolve_initial_density(initial_value::Real, initial_condition::Symbol)
    value = Float64(initial_value)

    if initial_condition == :rho
        0.0 <= value <= 1.0 ||
            throw(ArgumentError("With initial_condition=:rho, initial value must be in [0, 1]."))
        return value
    elseif initial_condition == :m
        abs(value) <= 1.0 ||
            throw(ArgumentError("With initial_condition=:m, initial value must be in [-1, 1]."))
        return (value + 1.0) / 2.0
    end

    throw(ArgumentError("initial_condition must be :rho or :m."))
end

function solve_ame_evolution(Pk::Dict{Int,Float64}, rho0::Real, r::Real,
        times;
        initial_condition::Symbol = :rho,
        reset::Union{Nothing,AbstractResetProtocol} = nothing,
    reltol = 1e-8, abstol = 1e-8,
    maxiters::Int = 10_000_000)
    rho0_f = _resolve_initial_density(rho0, initial_condition)
    r_f    = Float64(r)
    r_f >= 0.0               || throw(ArgumentError("r must be non-negative."))

    reset_protocol = isnothing(reset) ? delta_reset(2.0 * rho0_f - 1.0) : reset

    times_sorted = sort(Float64.(collect(times)))
    isempty(times_sorted) && throw(ArgumentError("times must be non-empty."))
    all(t -> t >= 0.0, times_sorted) || throw(ArgumentError("All times must be non-negative."))

    ind     = _build_ame_indexer(collect(keys(Pk)))
    x0      = _ame_initial_state(ind, rho0_f)
    x_reset = r_f > 0.0 ? _build_ame_reset_state(ind, Pk, reset_protocol) :
                          zeros(Float64, ind.total)
    p       = (ind, Pk, r_f, x_reset)

    prob = ODEProblem(_ame_rhs!, x0, (0.0, times_sorted[end]), p)
    sol  = solve(prob, Tsit5(); reltol = reltol, abstol = abstol,
                 saveat = times_sorted, maxiters = maxiters)

    ntimes = length(times_sorted)
    length(sol.u) == ntimes ||
        throw(ErrorException(
            "AME evolution did not return all requested time points " *
            "($(length(sol.u)) of $(ntimes)); retcode=$(sol.retcode). " *
            "Try a smaller time horizon or increase maxiters."
        ))
    s_out  = [Matrix{Float64}(undef, ntimes, k + 1) for k in ind.k_vals]
    i_out  = [Matrix{Float64}(undef, ntimes, k + 1) for k in ind.k_vals]
    for t_idx in 1:ntimes
        x_t = sol.u[t_idx]
        for (j, k) in enumerate(ind.k_vals)
            for m in 0:k
                s_out[j][t_idx, m + 1] = x_t[_idx_s(ind, k, m)]
                i_out[j][t_idx, m + 1] = x_t[_idx_i(ind, k, m)]
            end
        end
    end

    return AMEEvolutionResult(times_sorted, ind.k_vals, s_out, i_out,
                              Pk, rho0_f, r_f)
end

function solve_ame_evolution(graph::AbstractGraph, rho0::Real, r::Real,
        times;
    initial_condition::Symbol = :rho,
    reset::Union{Nothing,AbstractResetProtocol} = nothing,
        reltol = 1e-8, abstol = 1e-8,
        maxiters::Int = 10_000_000)
    return solve_ame_evolution(_pk_from_graph(graph), rho0, r, times;
                   initial_condition = initial_condition,
                   reset = reset,
                   reltol = reltol, abstol = abstol,
                   maxiters = maxiters)
end


# =============================================================================
# Public API: solve_ame_steady_state
# =============================================================================

"""
    solve_ame_steady_state(Pk, rho0, r; initial_condition=:rho, reset=..., ...)
    solve_ame_steady_state(graph, rho0, r; ...)

Find the AME steady state (d/dt = 0) via NLsolve (trust-region method),
starting from the binomial initial guess at `rho0`.

# Arguments
- `Pk` / `graph`: degree distribution or Graphs.jl graph.
- `rho0`: initial value interpreted according to `initial_condition`:
    `:rho` means fraction in +1, `:m` means magnetization.
- `initial_condition`: `:rho` (default) or `:m`.
- `r`: resetting rate (r=0 gives the plain voter model whose steady state is
  the magnetisation-conserving fixed point closest to the initial guess).
- `reset`: resetting protocol (default: random reset to `2rho0-1`).
- `xtol`, `ftol`: NLsolve tolerances (default 1e-12).

# Returns
An `AMESteadyStateResult`:
- `.k_values`: sorted unique degree classes
- `.s_values[j]`: `(k_j+1)`-vector, `s_values[j][m+1]` = s_{k_j,m}*
- `.i_values[j]`: analogous for i_{k_j,m}*
- `.Pk`, `.rho0`, `.r`: parameters used
"""
function solve_ame_steady_state(Pk::Dict{Int,Float64}, rho0::Real, r::Real;
        initial_condition::Symbol = :rho,
        reset::Union{Nothing,AbstractResetProtocol} = nothing,
        xtol = 1e-12, ftol = 1e-12)
    rho0_f = _resolve_initial_density(rho0, initial_condition)
    r_f    = Float64(r)
    r_f >= 0.0               || throw(ArgumentError("r must be non-negative."))

    reset_protocol = isnothing(reset) ? delta_reset(2.0 * rho0_f - 1.0) : reset

    ind     = _build_ame_indexer(collect(keys(Pk)))
    x0      = _ame_initial_state(ind, rho0_f)
    x_reset = r_f > 0.0 ? _build_ame_reset_state(ind, Pk, reset_protocol) :
                          zeros(Float64, ind.total)
    p       = (ind, Pk, r_f, x_reset)

    f!(F, x) = _ame_rhs!(F, x, p, 0.0)
    sol = nlsolve(f!, x0; method = :trust_region, xtol = xtol, ftol = ftol)
    sol.f_converged ||
        @warn "AME steady-state solver did not fully converge (rho0=$(rho0_f), r=$(r_f)). " *
              "Try tightening tolerances or providing a better initial guess."

    s_vals, i_vals = _extract_ame_grids(sol.zero, ind)
    return AMESteadyStateResult(ind.k_vals, s_vals, i_vals, Pk, rho0_f, r_f)
end

function solve_ame_steady_state(graph::AbstractGraph, rho0::Real, r::Real;
    initial_condition::Symbol = :rho,
    reset::Union{Nothing,AbstractResetProtocol} = nothing,
        xtol = 1e-12, ftol = 1e-12)
    return solve_ame_steady_state(_pk_from_graph(graph), rho0, r;
                  initial_condition = initial_condition,
                  reset = reset,
                  xtol = xtol, ftol = ftol)
end
