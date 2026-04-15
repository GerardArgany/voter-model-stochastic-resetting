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
function exact_mfpt_discrete_spectral(
    N::Int,
    n0::Int,
    r::Real;
    precision_bits::Int=256,
    return_bigfloat::Bool=false,
)
    N >= 2 || throw(ArgumentError("N must be at least 2."))
    1 <= n0 <= (N - 1) || throw(ArgumentError("n0 must satisfy 1 <= n0 <= N-1."))
    precision_bits >= 64 || throw(ArgumentError("precision_bits must be at least 64."))

    rf = Float64(r)
    0.0 <= rf < 1.0 || throw(ArgumentError("r must satisfy 0 <= r < 1 for discrete-time resetting probability."))

    return setprecision(precision_bits) do
        T = BigFloat
        oneT = one(T)
        zeroT = zero(T)

        N_T = T(N)
        n0_T = T(n0)
        r_T = T(rf)

        z = oneT - r_T
        n0_prefactor = (n0_T * (N_T - n0_T)) / (N_T^2)

        # Build b_i^(k) for i = k..N via the recursive product in Eq. 3.
        # Uses corrected denominator: l(l-1) - k(k-1).
        function b_coeffs_for_k(k::Int)
            b = zeros(T, N)
            b[k] = oneT
            prod_val = oneT
            k_T = T(k)

            for i in (k + 1):N
                l_T = T(i)
                num = (l_T - oneT) * (N_T - l_T + oneT)
                den = (l_T * (l_T - oneT)) - (k_T * (k_T - oneT))
                den == zeroT && throw(ArgumentError("Encountered zero denominator in b_i^(k) recursion."))
                prod_val *= num / den
                b[i] = prod_val
            end

            return b
        end

        # Eigenvector component v_k[j] from Eq. 11/12.
        function v_component(k::Int, j::Int, b::Vector{T}) where {T<:AbstractFloat}
            s = zero(T)
            i_start = max(j, k)

            for i in i_start:N
                sign_term = isodd(i - j) ? -one(T) : one(T)
                binom_ij = T(binomial(big(i), big(j)))
                s += sign_term * binom_ij * b[i]
            end

            return s
        end

        s_tilde = zeroT

        for k in 2:N
            k_T = T(k)
            kk1 = k_T * (k_T - oneT)
            b = b_coeffs_for_k(k)

            v_k_1 = v_component(k, 1, b)
            v_k_nminus1 = v_component(k, N - 1, b)
            v_k_n0 = v_component(k, n0, b)

            lambda_k = oneT - kk1 / (N_T * (N_T - oneT))
            d_k = (T(4) * (T(2k - 1)) / kk1) * n0_prefactor * v_k_n0

            geom_denom = oneT - z * lambda_k
            abs(geom_denom) > eps(T) || throw(ArgumentError("Singular term encountered: 1 - z*lambda_k is numerically zero."))

            s_tilde += (d_k * (v_k_1 + v_k_nminus1) / kk1) / geom_denom
        end

        s_tilde *= T(N - 1)
        mfpt_denom = oneT - r_T * s_tilde
        abs(mfpt_denom) > eps(T) || throw(ArgumentError("MFPT denominator 1 - r*S_tilde is numerically zero."))

        out = s_tilde / mfpt_denom
        return return_bigfloat ? out : Float64(out)
    end
end
