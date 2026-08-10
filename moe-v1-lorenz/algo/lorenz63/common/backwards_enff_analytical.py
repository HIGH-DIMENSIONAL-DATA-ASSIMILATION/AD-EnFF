import torch
import torch.nn as nn
from common.enff_utils import get_guided_vf_local


class BackwardsENFFAnalytical(nn.Module):
    """Analytical ENFF ODE sampler with a configurable observation operator."""

    def __init__(self, Nt_ode: int, observe_fn=None):
        super().__init__()
        self.Nt_ode = Nt_ode
        self.observe_fn = observe_fn

    def forward(
        self,
        z1:             torch.Tensor,   # [B, J, D] prior ensemble
        z0:             torch.Tensor,   # [B, J, D] old posterior ensemble
        y:              torch.Tensor,   # [B, D_y]  observation
        obs_noise_diag: torch.Tensor,   # [B, D_y]  observation noise sigma^2
        external_scheduler: float = None,
        observe_fn=None,
    ) -> torch.Tensor:
        sigma_min = 0.1
        obs_fn = observe_fn if observe_fn is not None else self.observe_fn

        device = z1.device
        xt = z0.clone().to(device) + torch.randn_like(z0).to(device) * sigma_min

        y              = y.to(device)
        obs_noise_diag = obs_noise_diag.to(device)
        z1             = z1.to(device)

        t_mesh     = torch.linspace(0.0, 1.0, self.Nt_ode + 1, device=device)
        xt_history = []

        for i in range(self.Nt_ode):
            t_now  = t_mesh[i]
            dt     = t_mesh[i + 1] - t_now

            u_guided = get_guided_vf_local(
                zt=xt, z0=z0, z1=z1,
                ode_t=t_now.item(),
                sigma_min=sigma_min,
                measurement=y,
                obs_noise_diag=obs_noise_diag,
                learned_scheduler_t=external_scheduler,
                observe_fn=obs_fn,
            )

            xt = xt + dt * u_guided
            xt_history.append(xt)

        return torch.stack(xt_history, dim=1)
