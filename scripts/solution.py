from scipy.special import eval_gegenbauer
import numpy as np


def B(n, m0):
	C = eval_gegenbauer(n, 1.5, m0)
	return( (2*n+3)/((n+1)*(n+2))*C)
	

def la(n,N):
	return(1/(N)*n*(n+1))


def kernel(n, N, r, t):
    lam = la(n, N)
    if np.isinf(t):
        return -r / (lam + r)

    return -lam / (lam + r) * np.exp(-(lam + r) * t) - r / (lam + r)


def pip1(N, r, m0, t=np.inf, M = 600):
    A = (1 + m0) / 2

    s = np.sum([B(n, m0) * kernel(n+1, N, r, t) for n in range(M)])

    return A * (1 + (1 - m0) * s)    


def pim1(N, r, m0, t=np.inf, M = 600):
    C = (1 - m0) / 2

    s = np.sum([(-1)**n * B(n, m0) * kernel(n+1, N, r, t) for n in range(M)])

    return C * (1 + (1 + m0) * s)
	

def fk(N, r, m0, m, t=np.inf, M = 600):
    s = 0.0
    for n in range(M):
        Cn = eval_gegenbauer(n, 1.5, m)
        s += B(n, m0) * kernel(n+1, N, r, t) * Cn

    return -0.5 * (1 - m0**2) * s


def sol(N, r, m0, bins, t=np.inf):
    m = np.linspace(-1+0.1/bins, 1-0.1/bins, bins) #We don't input the extremes into fk(m) as they converge very slowly
    out = [fk(N, r, m0, mi, t) for mi in m]

    out[0]  += pim1(N, r, m0, t) / (2 / bins)
    out[-1] += pip1(N, r, m0, t) / (2 / bins)

    return np.array(out)








