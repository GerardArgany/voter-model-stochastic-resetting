from scipy.special import eval_gegenbauer
import numpy as np


def B(n, m0):
	C = eval_gegenbauer(n, 1.5, m0)
	return( (2*n+3)/((n+1)*(n+2))*C)
	

def la(n,N):
	return(1/(N)*n*(n+1))


def dist_laplace(N, m0, r, s, M = 600):

	s1 = np.sum( [ B(2*i, m0)*la(2*i+1, N)/(s+r+la(2*i+1, N)) for i in range(M)] )
	s2 = np.sum( [ B(2*i, m0)/(s+r+la(2*i+1, N)) for i in range(M)] )

	return( (1-m0**2)*s1/ ( 1-r*(1-m0**2)*s2 ) )


def sol_fpt(N, m0, r, s_vals, M = 600):
	return([dist_laplace(N, m0, r, s, M) for s in s_vals])


def mean_fpt(N, m0, r, M = 600):
	r = r/N
	s = np.sum( [ B(2*i, m0)/(r+la(2*i+1, N)) for i in range(M)] )
	return( (1-m0**2)*s / ( 1-r*(1-m0**2)*s ) )



