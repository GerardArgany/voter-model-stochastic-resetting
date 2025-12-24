import numpy as np

def laplace(t, f_t, s_vals):
    """
    Computes Laplace transform using Simpson's rule 
    """
    # Pre-allocate result array
    L_s = np.zeros(len(s_vals))
    
    for i, s in enumerate(s_vals):
        integrand = f_t * np.exp(-s * t)
        
        # Use Simpson's rule for integration
        # Note: 't' must be the x-axis points corresponding to f_t
        L_s[i] = simpson(y=integrand, x=t)
        
    return L_s