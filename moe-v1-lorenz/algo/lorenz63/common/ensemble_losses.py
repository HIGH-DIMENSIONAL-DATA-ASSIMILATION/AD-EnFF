"""Training losses for posterior ensembles.

Ensembles have shape [B, T, J, D] and truth has shape [B, T, D].
"""
from __future__ import annotations
import torch

EPS = 1e-8


def _check(ens: torch.Tensor, truth: torch.Tensor):
    assert ens.dim() == 4 and truth.dim() == 3, \
        f"need ens[B,T,J,D] & truth[B,T,D], got {tuple(ens.shape)} & {tuple(truth.shape)}"
    B, T, J, D = ens.shape
    assert truth.shape == (B, T, D)
    return B, T, J, D


def _ens_mean(ens):
    return ens.mean(dim=2)


def energy_score_loss(ens, truth, beta: float = 1.0) -> torch.Tensor:
    """Energy score at each time step, averaged over batch and time."""
    B, T, J, D = _check(ens, truth)
    e = ens.reshape(B * T, J, D)
    y = truth.reshape(B * T, 1, D)
    d_xy = torch.cdist(e, y).squeeze(-1)                  # [BT,J]
    d_xx = torch.cdist(e, e)                              # [BT,J,J]
    if beta != 1.0:
        d_xy = d_xy ** beta
        d_xx = d_xx ** beta
    return (d_xy.mean(dim=1) - 0.5 * d_xx.mean(dim=(1, 2))).mean()


def trajectory_energy_score_loss(ens, truth, beta: float = 1.0,
                                 time_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Energy score after flattening each member's [T, D] trajectory."""
    B, T, J, D = _check(ens, truth)
    e = ens                                              # [B,T,J,D]
    y = truth.unsqueeze(2)                               # [B,T,1,D]
    if time_weight is not None:
        w = time_weight.reshape(1, T, 1, 1).clamp_min(0).sqrt()
        e = e * w
        y = y * w
    e = e.permute(0, 2, 1, 3).reshape(B, J, T * D)       # [B,J,T*D] member paths
    y = y.permute(0, 2, 1, 3).reshape(B, 1, T * D)       # [B,1,T*D] truth path
    d_xy = torch.cdist(e, y).squeeze(-1)                 # [B,J]
    d_xx = torch.cdist(e, e)                             # [B,J,J]
    if beta != 1.0:
        d_xy = d_xy ** beta
        d_xx = d_xx ** beta
    raw = (d_xy.mean(dim=1) - 0.5 * d_xx.mean(dim=(1, 2))).mean()
    # Keep the scale comparable across trajectory lengths and dimensions.
    return raw / float((T * D) ** 0.5)


def variogram_score_loss(ens, truth, p: float = 1.0,
                         time_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Compare consecutive state-jump magnitudes in the ensemble and truth."""
    B, T, J, D = _check(ens, truth)
    if T < 2:
        return torch.zeros((), device=ens.device, dtype=ens.dtype)
    dy = (truth[:, 1:] - truth[:, :-1]).abs() ** p                       # [B,T-1,D]
    dx = (ens[:, 1:] - ens[:, :-1]).abs() ** p                           # [B,T-1,J,D]
    diff2 = (dy - dx.mean(dim=2)) ** 2                                    # [B,T-1,D]
    if time_weight is not None:
        diff2 = diff2 * time_weight[:T - 1].reshape(1, T - 1, 1).clamp_min(0)
    return diff2.sum(dim=(1, 2)).mean()


def gaussian_nll_loss(ens, truth) -> torch.Tensor:
    """Gaussian NLL using the ensemble mean and per-dimension variance."""
    mu = _ens_mean(ens)                                  # [B,T,D]
    var = ens.var(dim=2, unbiased=True).clamp_min(EPS)   # [B,T,D]
    return (0.5 * ((truth - mu) ** 2 / var + torch.log(var))).mean()
