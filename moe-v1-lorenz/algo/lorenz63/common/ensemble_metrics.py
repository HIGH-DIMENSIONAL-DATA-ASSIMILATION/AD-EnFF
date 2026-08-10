"""Evaluation metrics for the ensemble filter.

    ens   : [B, T, J, D]  posterior ensemble  (B trajectories, T cycles, J members, D dims)
    truth : [B, T, D]     ground-truth state

Each function takes those two tensors and returns a float aggregated over B and T.
Trajectory ES and variogram live in ensemble_losses.py; latency is measured in eval_all.py.
"""
from __future__ import annotations
import torch

EPS = 1e-8


def _check(ens, truth):
    """Validate the [B,T,J,D] / [B,T,D] shapes and return (B, T, J, D)."""
    assert ens.dim() == 4, f"ens must be [B,T,J,D], got {tuple(ens.shape)}"
    assert truth.dim() == 3, f"truth must be [B,T,D], got {tuple(truth.shape)}"
    B, T, J, D = ens.shape
    assert truth.shape == (B, T, D), f"truth {tuple(truth.shape)} != ({B},{T},{D})"
    return B, T, J, D


def _tail(ens, truth, last_n):
    """Keep only the final `last_n` time steps (None/<=0 keeps all)."""
    if last_n is None or last_n <= 0:
        return ens, truth
    return ens[:, -last_n:], truth[:, -last_n:]


def ensemble_mean(ens):
    """Posterior mean over members. [B,T,J,D] -> [B,T,D]."""
    return ens.mean(dim=2)


def rmse(ens, truth, last_n=None) -> float:
    """RMSE of the ensemble mean vs truth."""
    ens, truth = _tail(ens, truth, last_n)
    err = ensemble_mean(ens) - truth
    return (err ** 2).mean(dim=-1).clamp_min(0).sqrt().mean().item()


def energy_score(ens, truth, last_n=None) -> float:
    """Multivariate CRPS: E||X - y|| - 0.5 E||X - X'||. Lower is better."""
    ens, truth = _tail(ens, truth, last_n)
    B, T, J, D = _check(ens, truth)
    e = ens.reshape(B * T, J, D)
    y = truth.reshape(B * T, 1, D)
    term1 = torch.cdist(e, y).squeeze(-1).mean(dim=1)     # E||X-y||
    term2 = torch.cdist(e, e).mean(dim=(1, 2))            # E||X-X'||
    return (term1 - 0.5 * term2).mean().item()


def coverage(ens, truth, last_n=None, level=0.95) -> float:
    """Fraction of truth values inside the central ensemble interval."""
    ens, truth = _tail(ens, truth, last_n)
    _check(ens, truth)
    tail = 0.5 * (1.0 - level)
    lower = torch.quantile(ens, tail, dim=2)
    upper = torch.quantile(ens, 1.0 - tail, dim=2)
    return ((truth >= lower) & (truth <= upper)).float().mean().item()


def _rbf_mmd2(X, Y, sigma=None):
    """Squared RBF-kernel MMD between sample sets X[n,D], Y[m,D] (median bandwidth)."""
    XX = torch.cdist(X, X) ** 2
    YY = torch.cdist(Y, Y) ** 2
    XY = torch.cdist(X, Y) ** 2
    sigma2 = torch.median(XY.flatten()).clamp_min(EPS) if sigma is None else sigma ** 2
    k = lambda d2: torch.exp(-d2 / (2.0 * sigma2 + EPS))
    return k(XX).mean() + k(YY).mean() - 2.0 * k(XY).mean()


def climatology_mmd(ens, truth, last_n=None, max_samples=2000, seed=0) -> float:
    """MMD between the filter climatology (all members pooled over b,t,j) and the truth
    climatology (all truth states pooled over b,t): does the filter match the attractor shape."""
    ens, truth = _tail(ens, truth, last_n)
    B, T, J, D = _check(ens, truth)
    X = ens.reshape(-1, D)
    Y = truth.reshape(-1, D)
    g = torch.Generator(device='cpu').manual_seed(seed)
    if X.shape[0] > max_samples:
        X = X[torch.randperm(X.shape[0], generator=g)[:max_samples]]
    if Y.shape[0] > max_samples:
        Y = Y[torch.randperm(Y.shape[0], generator=g)[:max_samples]]
    return _rbf_mmd2(X, Y).clamp_min(0).item()


@torch.no_grad()
def compute_all_metrics(ens, truth, last_n=None, level=0.95, spatial=None) -> dict:
    """Return the metrics used by the evaluator."""
    _check(ens, truth)
    return {
        'rmse':            rmse(ens, truth, last_n),
        'energy_score':    energy_score(ens, truth, last_n),
        'coverage':        coverage(ens, truth, last_n, level),
        'climatology_mmd': climatology_mmd(ens, truth, last_n),
    }


if __name__ == '__main__':
    torch.manual_seed(0)
    B, T, J, D = 4, 50, 20, 40
    truth = torch.randn(B, T, D)
    ens = truth.unsqueeze(2) + 0.5 * torch.randn(B, T, J, D)
    for k, v in compute_all_metrics(ens, truth).items():
        print(f'  {k:18s} {v:+.4f}')
