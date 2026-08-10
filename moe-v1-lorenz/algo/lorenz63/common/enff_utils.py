"""Flow-matching and observation-guidance utilities for analytical ENFF."""

import torch

# Default observation operator for this benchmark is arctan. Kept overridable via the
# observe_fn argument (a later linear-observation refactor can pass identity explicitly),
# but the default is arctan so a missing observe_fn can never silently become identity.
_DEFAULT_OBS_FN = torch.atan

def get_path_log_prob_and_velocities(
    zt: torch.Tensor,               # Shape: [B, J, D] -> Current positions of the particles
    z0: torch.Tensor,
    z1: torch.Tensor,
    ode_t: float,
    sigma_min: float,
):
    """Returns log_prob [B, J, J] and v_pairs [B, J, D]"""
    mu_pairs = ode_t * z1 + (1.0 - ode_t) * z0
    v_pairs = z1 - z0

    diff_path = zt.unsqueeze(2) - mu_pairs.unsqueeze(1)         # [B, J, J, D]
    sq_dist_path = torch.sum(diff_path ** 2, dim=-1)            # [B, J, J]
    log_prob = -sq_dist_path / (2 * sigma_min ** 2)             # [B, J, J]
    
    return log_prob, v_pairs


def get_marginal_vf(
    zt: torch.Tensor,               # Shape: [B, J, D]
    z0: torch.Tensor,               # Shape: [B, J, D]
    z1: torch.Tensor,               # Shape: [B, J, D]
    ode_t: float,                   # Scalar
    sigma_min: float                # Scalar
) -> torch.Tensor:
    """Returns Marginal VF u_marg [B, J, D] (The Prior Pull)"""
    log_prob, v_pairs = get_path_log_prob_and_velocities(zt, z0, z1, ode_t, sigma_min)

    base_weights = torch.softmax(log_prob, dim=2)               # [B, J, J]
    
    weighted_velocities = base_weights.unsqueeze(-1) * v_pairs.unsqueeze(1)
    u_marg = torch.sum(weighted_velocities, dim=2)              # [B, J, D]
    
    return u_marg


def get_guided_vf_local(
    zt: torch.Tensor,               # Shape: [B, J, D]
    z0: torch.Tensor,               # Shape: [B, J, D]
    z1: torch.Tensor,               # Shape: [B, J, D]
    ode_t: float,
    sigma_min: float,
    measurement: torch.Tensor,      # Shape: [B, D_y]
    obs_noise_diag: torch.Tensor,   # Shape: [B, D_y] or Scalar
    learned_scheduler_t: torch.Tensor = None,
    preconditioned: bool = False,
    observe_fn=None                 # Callable: (..., D) -> (..., D_y). Defaults to AtanObs.
) -> torch.Tensor:
    """
    Implements Local Energy Guidance from EnFF.

    observe_fn is any differentiable callable mapping state -> observation space.
    Autograd propagates gradients through it automatically — no manual Jacobian needed.
    """
    if observe_fn is None:
        observe_fn = _DEFAULT_OBS_FN

    u_unguided = get_marginal_vf(zt, z0, z1, ode_t, sigma_min)

    x1_predicted = zt + (1.0 - ode_t) * u_unguided

    with torch.enable_grad():
        x1_predicted = x1_predicted.requires_grad_(True)

        diff_obs = observe_fn(x1_predicted) - measurement.unsqueeze(1)    # [B, J, D_y]
        sq_dist_obs = diff_obs ** 2
        if not preconditioned:
            sq_dist_obs = sq_dist_obs / obs_noise_diag.unsqueeze(1)

        energy_J = 0.5 * torch.sum(sq_dist_obs, dim=-1)                   # [B, J]

        grad_energy = torch.autograd.grad(
            outputs=energy_J.sum(),
            inputs=x1_predicted,
        )[0]                                                               # [B, J, D]

    if learned_scheduler_t is not None:
        scheduler_t = learned_scheduler_t
    else:
        scheduler_t = 0.05

    grad_energy = torch.clip(grad_energy, -1000.0, 1000.0)
    u_guided = u_unguided - scheduler_t * grad_energy

    return u_guided
