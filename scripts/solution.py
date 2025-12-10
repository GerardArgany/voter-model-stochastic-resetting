from scipy.special import eval_gegenbauer
import numpy as np

M = 300


def B(n, m0):
	C = eval_gegenbauer(n, 1.5, m0)
	return( (2*n+3)/((n+1)*(n+2))*C)

def la(n,N):
	return(1/(N)*n*(n+1))


def pi0(N, r, m0, t = np.inf):
	A = (1+m0)/2

	s = np.sum([ B(n,m0)*r/(la(n+1, N)+r) for n in range(M)])

	return( A*(1-(1-m0)*s) ) #- np.exp(-r*t)

def pi1(N, r, m0, t = np.inf):
	C = (1-m0)/2

	s = np.sum([ (-1)**n*B(n,m0)*r/(la(n+1, N)+r) for n in range(M)])

	return( C*(1-(1+m0)*s) ) #- np.exp(-r*t)
	

def fk(N, r, m0, m):
	s = 0
	for n in range(M ):
		C = eval_gegenbauer(n, 1.5, m)
		s = s + B(n,m0)*r/(la(n+1, N)+r)*C
	return(0.5*(1-m0**2)*s)



def sol(N, r,m0, bins):
	m = np.linspace(-1,1,bins)
	# print(m)
	out = [fk(N,r,m0,i) for i in m]
	out[0] += pi1(N,r,m0)/(2/bins)
	out[-1] += pi0(N,r,m0)/(2/bins)


	return(out)







