from scipy.special import eval_gegenbauer
import numpy as np


def B(n, m0):
    C = eval_gegenbauer(n, 1.5, m0)
    return ((2*n+3)/((n+1)*(n+2))*C)


def la(n, N):
    return (1/(N)*n*(n+1))


def dist_laplace(N, m0, r, s, M=1000):
    r = r/N

    i = np.arange(M)
    n_even = 2 * i
    n_odd = n_even + 1

    b_vals = B(n_even, m0)
    la_vals = la(n_odd, N)
    
    denom = s + r + la_vals

    s1 = np.sum(b_vals * la_vals / denom)
    s2 = np.sum(b_vals * (s + la_vals) / denom)

    return s1 / s2


def sol_fpt(N, m0, r, s_vals, M=1000):
    return [dist_laplace(N, m0, r, s, M) for s in s_vals]


def mean_fpt(N, m0, r, M=1000):
    #There is a simplification using an identity
    r = r/N

    i = np.arange(M)
    n_even = 2 * i
    n_odd = n_even + 1
    
    b_vals = B(n_even, m0)
    la_vals = la(n_odd, N)

    denom = r + la_vals
    
    s = np.sum(b_vals / denom)
    s2 = np.sum(b_vals * la_vals / denom)

    return s / s2

def variance_fpt(N, m0, r, M=1000):
    r = r/N

    i = np.arange(M)
    n_even = 2 * i
    n_odd = n_even + 1
    
    b_vals = B(n_even, m0)
    la_vals = la(n_odd, N)
    
    denom = r + la_vals
    s = np.sum(b_vals / denom)
    s2 = np.sum(b_vals * la_vals / denom)
    s3 = np.sum(b_vals / (denom**2))

    mean = s / s2
    
    prefactor = 1 - m0**2  
    real_denom_sq = (prefactor * s2)**2    
    real_num = 2 * prefactor * s3
    second_moment = real_num / real_denom_sq
    
    return np.sqrt(second_moment - mean**2)