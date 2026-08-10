import torch

def lorenz96_rhs(x, F):
    """
    Computes the right-hand side of the Lorenz-96 equations for an N-dimensional state.
    Vectorized over batch dimensions.
    """
    x_plus_1 = torch.roll(x, shifts=-1, dims=-1)
    x_minus_1 = torch.roll(x, shifts=1, dims=-1)
    x_minus_2 = torch.roll(x, shifts=2, dims=-1)
    
    return (x_plus_1 - x_minus_2) * x_minus_1 - x + F

def lorenz96_rk4_step(x, dt, F=8.0):
    """
    Advances the Lorenz-96 system by one step `dt` using 4th-order Runge-Kutta.
    Vectorized over batch dimensions.
    """
    k1 = dt * lorenz96_rhs(x, F)
    k2 = dt * lorenz96_rhs(x + 0.5 * k1, F)
    k3 = dt * lorenz96_rhs(x + 0.5 * k2, F)
    k4 = dt * lorenz96_rhs(x + k3, F)
    
    return x + (k1 + 2*k2 + 2*k3 + k4) / 6.0
