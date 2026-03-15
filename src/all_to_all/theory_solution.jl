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
