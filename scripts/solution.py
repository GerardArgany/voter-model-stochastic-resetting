from scipy.special import eval_gegenbauer
import numpy as np

N = 300


def B(n, m0):
	C = eval_gegenbauer(n, 1.5, m0)
	return( (2*n+3)/((n+1)*(n+2))*C)

def la(n):
	return(1/4*n*(n+1))


def pi0(r, m0):
	A = (1+m0)/2

	s = np.sum([ B(n,m0)*r/(la(n+1)+r) for n in range(N)])

	return( A*(1-(1-m0)*s) )

def pi1(r, m0):
	C = (1-m0)/2

	s = np.sum([ (-1)**n*B(n,m0)*r/(la(n+1)+r) for n in range(N)])

	return( C*(1-(1+m0)*s) )
	

def fk(r, m0, m):
	s = 0
	for n in range(N):
		C = eval_gegenbauer(n, 1.5, m)
		s = s + B(n,m0)*r/(la(n+1)+r)*C
	return((1-m0**2)*s)



def sol(r,m0, bins):
	m = np.linspace(-1,1,bins)
	# print(m)
	out = [fk(r,m0,i) for i in m]
	out[0] = out[0] + pi0(r,m0)*2/bins
	out[-1] = out[-1] + pi1(r,m0)*2/bins

	return(out)







