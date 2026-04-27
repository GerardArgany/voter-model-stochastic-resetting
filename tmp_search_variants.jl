cd(raw"c:\Users\gerar\Desktop\Escola\Universitat\Master\TFM\voter-model-stochastic-resetting")
include("src/VoterResetting.jl")
VR = VoterResetting

function build_b(T, N, k; num_mode::Symbol=:orig, den_mode::Symbol=:orig, start_shift::Int=0)
    b = zeros(T, N - k + 1)
    b[1] = one(T)
    for i in (k + 1):N
        num = if num_mode == :orig
            T((i - 1) * (N - i + 1))
        elseif num_mode == :minus1
            T((i - 1) * (N - i))
        elseif num_mode == :shiftk
            T(i * (N - i + 1))
        elseif num_mode == :alt
            T(i * (N - i))
        else
            error("bad num_mode")
        end
        den = if den_mode == :orig
            T(i * (i - 1) - k * (k - 1))
        elseif den_mode == :k
            T(i * (i - 1) - k)
        elseif den_mode == :shift
            T(i * (i - 1) - (k - 1) * (k - 2))
        else
            error("bad den_mode")
        end
        # optional start shift moves the effective recursion index by one
        idx = i - k + 1
        prev = idx - 1
        if start_shift != 0
            prev = idx - 1 + start_shift
        end
        if prev < 1 || prev > length(b)
            b[idx] = zero(T)
        else
            b[idx] = b[prev] * (num / den)
        end
    end
    return b
end

function v_component(T, N, k, j, b; binom_mode::Symbol=:orig, sign_mode::Symbol=:orig)
    s = zero(T)
    for i in max(j, k):N
        ii = binom_mode == :orig ? i : i - 1
        jj = binom_mode == :orig ? j : j - 1
        if ii < jj || jj < 0
            continue
        end
        comb = binomial(big(ii), big(jj))
        sgn = if sign_mode == :orig
            isodd(i - j) ? -one(T) : one(T)
        elseif sign_mode == :flip
            isodd(i - j) ? one(T) : -one(T)
        elseif sign_mode == :i1
            isodd(i - 1) ? -one(T) : one(T)
        elseif sign_mode == :j1
            isodd(j - 1) ? -one(T) : one(T)
        else
            error("bad sign_mode")
        end
        s += sgn * T(comb) * b[i - k + 1]
    end
    return s
end

function s0_variant(N::Int, n0::Int; num_mode=:orig, den_mode=:orig, binom_mode=:orig, sign_mode=:orig)
    setprecision(128) do
        T = BigFloat
        pref = T(n0 * (N - n0)) / T(N)^2
        s = zero(T)
        for k in 2:N
            kk1 = T(k * (k - 1))
            b = build_b(T, N, k; num_mode=num_mode, den_mode=den_mode)
            v = v_component(T, N, k, n0, b; binom_mode=binom_mode, sign_mode=sign_mode)
            dk = (T(4) * T(2 * k - 1) / kk1) * pref * v
            bnm1 = (N - 1) >= k ? b[N - 1 - k + 1] : zero(T)
            bn = b[N - k + 1]
            alt = zero(T)
            for i in k:N
                alt += (isodd(i - 1) ? -one(T) : one(T)) * T(i) * b[i - k + 1]
            end
            br = bnm1 - T(N) * bn + alt
            s += T(N - 1) * dk * br / kk1
        end
        return Float64(s)
    end
end

modes_num = (:orig, :minus1, :shiftk, :alt)
modes_den = (:orig, :k, :shift)
modes_bin = (:orig, :shift)
modes_sign = (:orig, :flip, :i1, :j1)

scores = NamedTuple[]
for nm in modes_num, dm in modes_den, bm in modes_bin, sm in modes_sign
    vals = [s0_variant(N, div(N,2); num_mode=nm, den_mode=dm, binom_mode=bm, sign_mode=sm) for N in (8,10,12)]
    score = sum(abs(v - 1) for v in vals)
    push!(scores, (num=nm, den=dm, bin=bm, sign=sm, vals=vals, score=score))
end
sort!(scores, by = x -> x.score)
for row in scores[1:20]
    println(row)
end