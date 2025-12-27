import numpy as np

def laplace_from_samples(T, s):
    return np.mean(np.exp(-s * T))