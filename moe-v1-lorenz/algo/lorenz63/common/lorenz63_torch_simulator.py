import torch
import numpy as np

def lorenz63_rhs(x, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """Right-hand side of the Lorenz-63 system. Vectorized over leading dims; state in last dim (size 3).
    Matches DALearning utils.py:144 L63.forward (sig=10, rho=28, beta=8/3)."""
    x0 = x[..., 0]; x1 = x[..., 1]; x2 = x[..., 2]
    d0 = sigma * (x1 - x0)
    d1 = x0 * (rho - x2) - x1
    d2 = x0 * x1 - beta * x2
    return torch.stack([d0, d1, d2], dim=-1)

def lorenz63_rk4_step(x, dt, rho=28.0, sigma=10.0, beta=8.0 / 3.0):
    """One RK4 step. `rho` is the drift parameter (our model-error / shock knob), scalar or broadcastable
    to x[...,0]'s shape (e.g. [B,1])."""
    k1 = dt * lorenz63_rhs(x, sigma, rho, beta)
    k2 = dt * lorenz63_rhs(x + 0.5 * k1, sigma, rho, beta)
    k3 = dt * lorenz63_rhs(x + 0.5 * k2, sigma, rho, beta)
    k4 = dt * lorenz63_rhs(x + k3, sigma, rho, beta)
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def make_rho_drift_schedule(n_traj, n_steps, mode, c_max, dwell_lo, dwell_hi, obs_gap, seed):
    """Per-trajectory offset schedule [n_traj, n_steps] — OUR L63 drift generator, copied verbatim from
    `lorenz63_simulator.make_rho_drift_schedule` (used by train_simple_l63_moe.py). 'intermittent' =
    long non-periodic on/off stretches; dwell in OBS-cycles ~U[dwell_lo,dwell_hi]; random initial phase."""
    rng = np.random.default_rng(seed)
    c = np.zeros((n_traj, n_steps), dtype=np.float32)
    if mode == 'none':
        return c
    if mode == 'constant':
        c[:] = c_max
        return c
    if mode == 'pure_shock':
        pulse = max(1, int(dwell_lo * obs_gap))
        s0 = n_steps // 3
        c[:, s0:s0 + pulse] = c_max
        return c
    if mode != 'intermittent':
        raise ValueError(f'unknown drift mode {mode!r}')
    lo = max(1, int(dwell_lo * obs_gap))
    hi = max(lo + 1, int(dwell_hi * obs_gap))
    for b in range(n_traj):
        t, on = 0, bool(rng.integers(0, 2))     # random initial phase
        while t < n_steps:
            dwell = int(rng.integers(lo, hi))
            c[b, t:t + dwell] = c_max if on else 0.0
            t += dwell
            on = not on
    return c
