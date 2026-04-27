# =============================================================================
# all_to_all/theory_solution.jl
# =============================================================================
#
# Analytical solutions for the all-to-all voter model with stochastic resetting.
# This file mirrors the Python API in voter_model/solution.py and
# voter_model/solution_fpt.py so Julia notebooks can use theory directly.
#
# Public functions (Python-analogous names):
#   dist_laplace, sol_fpt, mean_fpt, variance_fpt
#   pip1, pim1, fk, sol
#
# =============================================================================

try
    using Roots
catch
    import Pkg
    Pkg.add("Roots")
    using Roots
end

# Gegenbauer polynomial C_n^(3/2)(x) by three-term recurrence.
# This avoids adding a special-function dependency just for one polynomial family.
function gegenbauer_c32(n::Int, x::Float64)
    n >= 0 || throw(ArgumentError("n must be non-negative."))

    if n == 0
        return 1.0
    elseif n == 1
        return 3.0 * x
    end

    alpha = 1.5
    c_nm2 = 1.0
    c_nm1 = 2.0 * alpha * x

    for k in 2:n
        c_n = (2.0 * (k + alpha - 1.0) / k) * x * c_nm1 - ((k + 2.0 * alpha - 2.0) / k) * c_nm2
        c_nm2 = c_nm1
        c_nm1 = c_n
    end

    return c_nm1
end

"""
    B(n, m0)

Gegenbauer expansion coefficient at reset magnetization `m0`.
"""
function B(n::Int, m0::Real)
    m0f = Float64(m0)
    return (2.0 * n + 3.0) / ((n + 1.0) * (n + 2.0)) * gegenbauer_c32(n, m0f)
end

"""
    la(n, N)

Relaxation rate `lambda_n = n(n+1)/N`.
"""
la(n::Int, N::Int) = n * (n + 1) / N

"""
    kernel(n, N, r, t)

Time kernel used by the analytical magnetization distribution.
"""
function kernel(n::Int, N::Int, r::Real, t::Real)
    lam = la(n, N)
    rf = Float64(r)
    tf = Float64(t)

    if isinf(tf)
        return -rf / (lam + rf)
    end

    return -lam / (lam + rf) * exp(-(lam + rf) * tf) - rf / (lam + rf)
end

"""
    pip1(N, r, m0; t=Inf, M=600)

Analytical probability mass at `m=+1`.
"""
function pip1(N::Int, r::Real, m0::Real; t::Real=Inf, M::Int=600)
    m0f = Float64(m0)
    A = (1.0 + m0f) / 2.0

    s = 0.0
    for n in 0:(M - 1)
        s += B(n, m0f) * kernel(n + 1, N, r, t)
    end

    return A * (1.0 + (1.0 - m0f) * s)
end

"""
    pim1(N, r, m0; t=Inf, M=600)

Analytical probability mass at `m=-1`.
"""
function pim1(N::Int, r::Real, m0::Real; t::Real=Inf, M::Int=600)
    m0f = Float64(m0)
    C = (1.0 - m0f) / 2.0

    s = 0.0
    for n in 0:(M - 1)
        s += (-1.0)^n * B(n, m0f) * kernel(n + 1, N, r, t)
    end

    return C * (1.0 + (1.0 + m0f) * s)
end

"""
    fk(N, r, m0, m; t=Inf, M=600)

Bulk analytical density at magnetization `m` in `(-1, 1)`.
"""
function fk(N::Int, r::Real, m0::Real, m::Real; t::Real=Inf, M::Int=600)
    m0f = Float64(m0)
    mf = Float64(m)

    s = 0.0
    for n in 0:(M - 1)
        s += B(n, m0f) * kernel(n + 1, N, r, t) * gegenbauer_c32(n, mf)
    end

    return -0.5 * (1.0 - m0f^2) * s
end

"""
    sol(N, r, m0, bins; t=Inf, M=600)

Analytical magnetization density on a uniform grid of `bins` points in `(-1, 1)`.
Returns a vector analogous to Python `solution.sol`.
"""
function sol(N::Int, r::Real, m0::Real, bins::Int; t::Real=Inf, M::Int=600)
    bins >= 2 || throw(ArgumentError("bins must be at least 2."))

    m_grid = collect(range(-1.0 + 0.1 / bins, 1.0 - 0.1 / bins, length=bins))
    out = [fk(N, r, m0, m; t=t, M=M) for m in m_grid]

    bin_width = 2.0 / bins
    out[1] += pim1(N, r, m0; t=t, M=M) / bin_width
    out[end] += pip1(N, r, m0; t=t, M=M) / bin_width

    return out
end

"""
    dist_laplace(N, m0, r, s; M=1000)

Laplace transform of the FPT distribution at frequency `s`.
This mirrors Python behavior and uses internal `r/N` scaling.
"""
function dist_laplace(N::Int, m0::Real, r::Real, s::Real; M::Int=1000)
    m0f = Float64(m0)
    sf = Float64(s)
    r_scaled = Float64(r) / N

    s1 = 0.0
    s2 = 0.0

    for i in 0:(M - 1)
        n_even = 2 * i
        n_odd = n_even + 1

        b = B(n_even, m0f)
        lam = la(n_odd, N)
        denom = sf + r_scaled + lam

        s1 += b * lam / denom
        s2 += b * (sf + lam) / denom
    end

    return s1 / s2
end

"""
    sol_fpt(N, m0, r, s_vals; M=1000)

Vectorized Laplace-space FPT solution over frequencies `s_vals`.
"""
function sol_fpt(N::Int, m0::Real, r::Real, s_vals; M::Int=1000)
    return [dist_laplace(N, m0, r, s; M=M) for s in s_vals]
end

"""
    mean_fpt(N, m0, r; M=1000)

Analytical mean first-passage time to consensus.
Mirrors Python behavior and uses internal `r/N` scaling.

Equivalent computation form used here:

    MFPT(r) = [sum_l B_{2l}/(r + lambda'_{2l+1})]
              / [sum_l B_{2l} lambda'_{2l+1}/(r + lambda'_{2l+1})]

with internal substitution `r -> r/N`.
"""
function mean_fpt(N::Int, m0::Real, r::Real; M::Int=1000)
    m0f = Float64(m0)
    r_scaled = Float64(r) / N

    s1 = 0.0
    s2 = 0.0

    for i in 0:(M - 1)
        n_even = 2 * i
        n_odd = n_even + 1

        b = B(n_even, m0f)
        lam = la(n_odd, N)
        denom = r_scaled + lam

        s1 += b / denom
        s2 += b * lam / denom
    end

    return s1 / s2
end

"""
    variance_fpt(N, m0, r; M=1000)

Returns the standard deviation of the FPT, matching Python `variance_fpt`.
"""
function variance_fpt(N::Int, m0::Real, r::Real; M::Int=1000)
    m0f = Float64(m0)
    r_scaled = Float64(r) / N

    s1 = 0.0
    s2 = 0.0
    s3 = 0.0

    for i in 0:(M - 1)
        n_even = 2 * i
        n_odd = n_even + 1

        b = B(n_even, m0f)
        lam = la(n_odd, N)
        denom = r_scaled + lam

        s1 += b / denom
        s2 += b * lam / denom
        s3 += b / (denom^2)
    end

    mean_val = s1 / s2

    prefactor = 1.0 - m0f^2
    real_denom_sq = (prefactor * s2)^2
    real_num = 2.0 * prefactor * s3
    second_moment = real_num / real_denom_sq

    return sqrt(second_moment - mean_val^2)
end

"""
    exact_mfpt_discrete_spectral(N, n0, r; precision_bits=256, return_bigfloat=false)

Exact discrete-time MFPT to consensus for the voter model on a complete graph,
using the discrete spectral expansion (Eq. 15-18 in discrete_first_passage.pdf):

    <T(n0)> = S_tilde^(1-r)(n0) / (1 - r * S_tilde^(1-r)(n0))

with

    S_tilde^(z)(n0) = (N-1) * sum_{k=2}^N [ d_k * (v_k[1] + v_k[N-1]) / (k(k-1)) ] * 1/(1 - z*lambda_k)

and components

    lambda_k = 1 - k(k-1)/(N(N-1))
    d_k = [4(2k-1)/(k(k-1))] * [n0(N-n0)/N^2] * v_k[n0]
    v_k[j] = sum_{i=max(j,k)}^N (-1)^(i-j) * binomial(i,j) * b_i^(k)

The polynomial coefficients use Eq. 3 with the denominator typo corrected to
`l(l-1) - k(k-1)` in the product.

By default the computation runs in high precision (`BigFloat` with
`precision_bits=256`) and returns `Float64`. Set `return_bigfloat=true` to
return the high-precision value.
"""
function _b_coefficients_for_k(::Type{T}, N::Int, k::Int) where {T<:AbstractFloat}
    b_vals = zeros(T, N - k + 1)
    b_vals[1] = one(T)  # b_k^(k) = 1
    for i in (k + 1):N
        num = T((i - 1) * (N - i + 1))
        den = T(i * (i - 1) - k * (k - 1))
        b_vals[i - k + 1] = b_vals[i - k] * (num / den)
    end
    return b_vals
end

@inline function _b_at_i(b_vals, k::Int, i::Int)
    return i < k ? zero(eltype(b_vals)) : b_vals[i - k + 1]
end

function _v_component(::Type{T}, N::Int, k::Int, j::Int, b_vals) where {T<:AbstractFloat}
    start_i = max(j, k)
    s = zero(T)
    for i in start_i:N
        sign = isodd(i - j) ? -one(T) : one(T)
        s += sign * T(binomial(big(i), big(j))) * _b_at_i(b_vals, k, i)
    end
    return s
end

function _boundary_bracket(::Type{T}, N::Int, k::Int, b_vals) where {T<:AbstractFloat}
    b_nm1 = (N - 1) >= k ? _b_at_i(b_vals, k, N - 1) : zero(T)
    b_n = _b_at_i(b_vals, k, N)

    alt_sum = zero(T)
    for i in k:N
        sign = isodd(i - 1) ? -one(T) : one(T)
        alt_sum += sign * T(i) * _b_at_i(b_vals, k, i)
    end

    return b_nm1 - T(N) * b_n + alt_sum
end

function _s_tilde_pdf_sum(::Type{T}, N::Int, n0::Int, r::Real) where {T<:AbstractFloat}
    oneT = one(T)
    rT = T(r)
    zT = oneT - rT
    n0_pref = T(n0 * (N - n0)) / (T(N)^2)

    s_tilde = zero(T)
    for k in 2:N
        kk1 = T(k * (k - 1))
        lambda_k = oneT - kk1 / T(N * (N - 1))

        b_vals = _b_coefficients_for_k(T, N, k)
        vkn0 = _v_component(T, N, k, n0, b_vals)
        dk = (T(4) * T(2 * k - 1) / kk1) * n0_pref * vkn0

        bracket = _boundary_bracket(T, N, k, b_vals)
        denom = kk1 * (oneT - zT * lambda_k)
        s_tilde += T(N - 1) * dk * bracket / denom
    end
    return s_tilde
end

const _all_to_all_spectral_cache = Dict{Int, NamedTuple{(:eigvals, :coeffs_by_n0), Tuple{Vector{Float64}, Matrix{Float64}}}}()

function _build_all_to_all_spectral_cache(N::Int)
    ntrans = N - 1
    Q = zeros(Float64, ntrans, ntrans)

    for n in 1:ntrans
        a = (n * (N - n)) / (N * (N - 1))
        Q[n, n] = 1 - 2a
        if n < ntrans
            Q[n, n + 1] = a
        end
        if n > 1
            Q[n, n - 1] = a
        end
    end

    pi = [1.0 / (n * (N - n)) for n in 1:ntrans]
    d = sqrt.(pi)
    dinv = 1.0 ./ d
    S = Diagonal(d) * Q * Diagonal(dinv)
    eig = eigen(Symmetric(S))
    U = eig.vectors
    eigvals = Vector{Float64}(eig.values)

    coeffs_by_n0 = Matrix{Float64}(undef, ntrans, ntrans)
    beta = transpose(U) * d
    for n0 in 1:ntrans
        coeffs_by_n0[:, n0] = (U[n0, :] ./ d[n0]) .* beta
    end

    return (eigvals = eigvals, coeffs_by_n0 = coeffs_by_n0)
end

"""
    exact_mfpt_discrete_spectral(N, n0, r; precision_bits=256, return_bigfloat=false)

Exact MFPT for the discrete-time complete-graph voter model with resetting,
evaluated from the exact spectral decomposition of the transient no-reset kernel.

This uses the reversible similarity transform of the transient transition matrix,
so the coefficients are normalized by construction and satisfy the required
`S_tilde(z=0) = 1` identity.
"""
function exact_mfpt_discrete_spectral(
    N::Int,
    n0::Int,
    r::Real;
    precision_bits::Int=256,
    return_bigfloat::Bool=false,
    check_consistency::Bool=true,
)
    N >= 2 || throw(ArgumentError("N must be at least 2."))
    1 <= n0 <= (N - 1) || throw(ArgumentError("n0 must satisfy 1 <= n0 <= N-1."))
    precision_bits >= 64 || throw(ArgumentError("precision_bits must be at least 64."))

    rf = Float64(r)
    rf >= 0.0 || throw(ArgumentError("r must satisfy r >= 0."))
    r_eff = rf / N
    0.0 <= r_eff < 1.0 || throw(ArgumentError("Effective reset probability r/N must satisfy 0 <= r/N < 1."))

    cache = get!(_all_to_all_spectral_cache, N) do
        _build_all_to_all_spectral_cache(N)
    end

    eigvals = cache.eigvals
    coeffs = view(cache.coeffs_by_n0, :, n0)

    if check_consistency
        s0 = sum(coeffs)
        abs(s0 - 1.0) <= 1e-10 || throw(ArgumentError("Spectral coefficient normalization failed: sum(coeffs)=$(s0), expected 1."))
    end

    if return_bigfloat || precision_bits > 64
        return setprecision(precision_bits) do
            T = BigFloat
            oneT = one(T)
            zT = oneT - T(r_eff)
            rT = T(r_eff)

            s_tilde = zero(T)
            for i in eachindex(eigvals)
                s_tilde += T(coeffs[i]) / (oneT - zT * T(eigvals[i]))
            end

            mfpt_denom = oneT - rT * s_tilde
            out = s_tilde / mfpt_denom
            return return_bigfloat ? out : Float64(out)
        end
    else
        z = 1.0 - r_eff
        s_tilde = 0.0
        for i in eachindex(eigvals)
            s_tilde += coeffs[i] / (1.0 - z * eigvals[i])
        end

        mfpt_denom = 1.0 - r_eff * s_tilde
        return s_tilde / mfpt_denom
    end
end

const _optimal_return_rate_cache = Dict{Int, NamedTuple{(:choose, :b, :v, :lambda, :boundary_weight, :base_weight), Tuple{Matrix{BigFloat}, Matrix{BigFloat}, Matrix{BigFloat}, Vector{BigFloat}, Vector{BigFloat}, Vector{BigFloat}}}}()

function _build_optimal_return_rate_cache(N::Int)
    choose = zeros(BigFloat, N + 1, N + 1)
    choose[1, 1] = 1
    for i in 1:N
        choose[i + 1, 1] = 1
        choose[i + 1, i + 1] = 1
        for j in 1:(i - 1)
            choose[i + 1, j + 1] = choose[i, j] + choose[i, j + 1]
        end
    end

    b = zeros(BigFloat, N + 1, N + 1)
    for k in 2:N
        b[k, k] = 1
        for i in (k + 1):N
            num = BigFloat((i - 1) * (N - i + 1))
            den = BigFloat(i * (i - 1) - k * (k - 1))
            b[k, i] = b[k, i - 1] * (num / den)
        end
    end

    v = zeros(BigFloat, N + 1, N + 1)
    for k in 2:N
        for j in 1:(N - 1)
            start_i = max(j, k)
            s = zero(BigFloat)
            for i in start_i:N
                sign = isodd(i - j) ? -one(BigFloat) : one(BigFloat)
                s += sign * choose[i + 1, j + 1] * b[k, i]
            end
            v[k, j] = s
        end
    end

    lambda = zeros(BigFloat, N + 1)
    boundary_weight = zeros(BigFloat, N + 1)
    base_weight = zeros(BigFloat, N + 1)
    for k in 2:N
        kk1 = BigFloat(k * (k - 1))
        lambda[k] = one(BigFloat) - kk1 / BigFloat(N * (N - 1))
        boundary_weight[k] = v[k, 1] + v[k, N - 1]
        base_weight[k] = BigFloat(N - 1) * BigFloat(4) * BigFloat(2 * k - 1) * boundary_weight[k] / (kk1^2)
    end

    return (choose = choose, b = b, v = v, lambda = lambda, boundary_weight = boundary_weight, base_weight = base_weight)
end

function _optimal_return_rate_coeffs_for_n0(cache, N::Int, n0::Int)
    coeffs = zeros(BigFloat, N + 1)
    ratio = BigFloat(n0 * (N - n0)) / BigFloat(N^2)
    for k in 2:N
        coeffs[k] = cache.base_weight[k] * ratio * cache.v[k, n0]
    end
    return coeffs
end

function _optimal_return_rate_s_tilde_and_derivative(z::Real, coeffs::AbstractVector, lambda::AbstractVector, N::Int)
    zT = BigFloat(z)
    s = zero(BigFloat)
    ds = zero(BigFloat)
    for k in 2:N
        denom = one(BigFloat) - zT * lambda[k]
        ck = coeffs[k]
        s += ck / denom
        ds += ck * lambda[k] / (denom^2)
    end
    return s, ds
end

_optimal_return_rate_function(z, coeffs, lambda, N) = begin
    s, ds = _optimal_return_rate_s_tilde_and_derivative(z, coeffs, lambda, N)
    s^2 - ds
end

function _find_optimal_z(coeffs, lambda, N; zmin::Float64 = 0.5, zmax::Float64 = 1.0, ngrid::Int = 800, root_tol::Real = 1e-12)
    grid = collect(range(zmin, zmax, length = ngrid))
    values = [Float64(_optimal_return_rate_function(z, coeffs, lambda, N)) for z in grid]

    if abs(values[end]) <= root_tol
        return grid[end]
    end

    for i in (length(grid) - 1):-1:1
        f1 = values[i]
        f2 = values[i + 1]
        if !isfinite(f1) || !isfinite(f2)
            continue
        end
        if abs(f1) <= root_tol
            return grid[i]
        elseif f1 * f2 < 0
            f(z) = Float64(_optimal_return_rate_function(z, coeffs, lambda, N))
            return find_zero(f, (grid[i], grid[i + 1]), Bisection(); atol = root_tol, rtol = root_tol)
        end
    end

    return nothing
end

"""
    optimal_return_rate_curve(N; zmin=0.5, zmax=1.0, ngrid=800, root_tol=1e-12)

Compute the theoretical optimal return rate r*(m0) for the discrete voter model
on a complete graph of size N.

The returned tuple is `(m0_values, r_star, root_found)` where `m0_values` are
the magnetizations `m0 = 2n0/N - 1`, `r_star` is the minimizing return rate,
and `root_found` marks whether a nonzero interior root was detected.

If the optimality condition has no root in `(0, 1]`, the corresponding value
is set to `r* = 0`.
"""
function optimal_return_rate_curve(N::Int; zmin::Float64 = 0.5, zmax::Float64 = 1.0, ngrid::Int = 800, root_tol::Real = 1e-12)
    N >= 2 || throw(ArgumentError("N must be at least 2."))

    cache = get!(_optimal_return_rate_cache, N) do
        _build_optimal_return_rate_cache(N)
    end

    n0_values = 1:(N - 1)
    m0_values = [2.0 * n0 / N - 1.0 for n0 in n0_values]
    r_star = zeros(Float64, length(n0_values))
    root_found = falses(length(n0_values))

    for (idx, n0) in enumerate(n0_values)
        coeffs = _optimal_return_rate_coeffs_for_n0(cache, N, n0)
        z_star = _find_optimal_z(coeffs, cache.lambda, N; zmin = zmin, zmax = zmax, ngrid = ngrid, root_tol = root_tol)

        if z_star === nothing
            r_star[idx] = 0.0
            root_found[idx] = false
        else
            r_val = max(0.0, 1.0 - Float64(z_star))
            r_star[idx] = r_val
            root_found[idx] = r_val > root_tol
        end
    end

    return m0_values, r_star, root_found
end
