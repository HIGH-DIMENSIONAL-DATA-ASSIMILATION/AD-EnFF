import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight

class MambaBlock(nn.Module):
    """One PRE-NORM RESIDUAL selective-SSM (S6) block, operating at d_model.

    Differentiable pure-PyTorch S6 with STOCK Mamba initialization:
      * A_log = S4D-real  A = -(1..d_state)            (NOT the -0.01 slow-mode hack)
      * dt_bias = log-uniform dt in [1e-3, 1e-1]       (stock Mamba; NOT the 1e-4 hack)
    The slow-A / tiny-dt hacks made the carrier sluggish & unreactive to observation
    surprise; expressivity instead comes from DEPTH+WIDTH on stock init (the PhyxMamba
    lesson — it wins on chaotic forecasting with stock Mamba2 init + a 4-deep residual
    stack, never a hand-forced long-memory init).

    Pre-norms the d_model RESIDUAL STREAM (not the raw carrier input), so the innovation
    magnitude that in_embed carries in is preserved into the gate. Returns the block's
    output to be ADDED to the residual stream; SSM hidden h is carried out-of-place so
    autograd flows through h AND every param each cycle.
    """
    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_inner = d_model
        self.d_state = d_state
        self.norm    = RMSNorm(d_model)                      # pre-norm on the residual stream
        self.in_proj = nn.Linear(d_model, d_model)
        self.z_proj  = nn.Linear(d_model, d_model)           # gate branch (Mamba's z)
        self.dt_proj = nn.Linear(d_model, d_model)           # input-dependent step size
        self.x_proj  = nn.Linear(d_model, 2 * d_state)       # input-dependent B_t, C_t

        # dt_bias: STOCK log-uniform in [1e-3, 1e-1] (inverse-softplus parameterized)
        dt = torch.exp(torch.rand(d_model) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        # A_log: STOCK S4D-real init A = -(1..d_state) (HiPPO-style decay bank, fast..slow)
        a_init = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
        self.A_log = nn.Parameter(a_init.unsqueeze(0).repeat(d_model, 1))   # [d_inner, d_state]
        self.D     = nn.Parameter(torch.ones(d_model))

    def step(self, res, h):
        u  = self.norm(res)                                          # pre-norm the residual stream
        x  = F.silu(self.in_proj(u))                                 # [B, d_inner]
        z  = self.z_proj(u)                                          # [B, d_inner]
        dt = F.softplus(self.dt_proj(u) + self.dt_bias)            # [B, d_inner]
        BC = self.x_proj(u)
        B_t = BC[:, :self.d_state]                                   # [B, d_state]
        C_t = BC[:, self.d_state:]                                   # [B, d_state]
        A   = -torch.exp(self.A_log)                                 # [d_inner, d_state]

        if h is None:
            h = torch.zeros(res.shape[0], self.d_inner, self.d_state,
                            device=res.device, dtype=x.dtype)

        dA  = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))           # [B, d_inner, d_state]
        dBx = (dA - 1.0) / A.unsqueeze(0) * B_t.unsqueeze(1) * x.unsqueeze(-1)  # ZOH
        h_new = dA * h + dBx                                         # OUT-OF-PLACE (differentiable)

        y   = (h_new * C_t.unsqueeze(1)).sum(dim=-1) + self.D.unsqueeze(0) * x  # [B, d_inner]
        out = y * F.silu(z)                                          # gated block output
        return out, h_new


class MambaCarrier(nn.Module):
    """Pure-PyTorch DIFFERENTIABLE selective-SSM carrier — a DEEP residual S6 stack.

    Expressivity via DEPTH + WIDTH (the PhyxMamba recipe), NOT a hand-forced long-memory
    init. Per cycle:
        x       = in_embed(u)              # Linear in_dim->d_model; PRESERVES innovation magnitude
        for blk in blocks:                 # n_layers pre-norm residual S6 blocks
            y, h_i = blk.step(x, h_i)
            x      = x + y                  # residual stream
        readout = out_proj(out_norm(x))    # -> [B, out_dim]

    state = tuple of per-layer SSM hidden states (detach_state handles tuples). NO mamba_ssm
    dependency, NO in-place mutation -> autograd flows through every block's hidden AND params
    each cycle, exactly like nn.GRUCell. headdim accepted for call-site compat (unused).

    Reactivity fix vs the old single-block carrier: the old RMSNorm on the RAW input
    normalized away the *magnitude* of the innovation (h(x)-y) -> the gate could not tell a
    big surprise (shock) from a small one. in_embed (a plain Linear) now carries that
    magnitude into the model; all norms live on the d_model residual stream instead.
    """
    def __init__(self, in_dim, out_dim, d_model=64, d_state=128, headdim=64, n_layers=3):
        super().__init__()
        self.n_layers = n_layers
        self.in_embed = nn.Linear(in_dim, d_model)           # carries raw innovation magnitude in
        self.blocks   = nn.ModuleList([MambaBlock(d_model, d_state) for _ in range(n_layers)])
        self.out_norm = RMSNorm(d_model)
        self.out_proj = nn.Linear(d_model, out_dim)
        self.out_dim  = out_dim

    def step(self, u, state):
        states = list(state) if state is not None else [None] * self.n_layers
        x = self.in_embed(u)                                         # [B, d_model], magnitude preserved
        new_states = []
        for i, blk in enumerate(self.blocks):
            y, h_new = blk.step(x, states[i])
            x = x + y                                                # residual stream
            new_states.append(h_new)
        readout = self.out_proj(self.out_norm(x))
        return readout, tuple(new_states)

class GRUCarrier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.cell = nn.GRUCell(in_dim, out_dim)

    def step(self, x, h):
        h_new = self.cell(x, h)
        return h_new, h_new  # Returns (readout, state)


def detach_state(s):
    """Detach a carrier state at the TBPTT window boundary.
    Handles both a plain tensor (GRU / Mamba hidden) and a tuple of tensors."""
    if s is None:
        return None
    if isinstance(s, tuple):
        return tuple(detach_state(x) for x in s)
    return s.detach()


def make_carrier(kind, in_dim, out_dim, mamba_d_model=64, d_state=128, headdim=64,
                 n_layers=3, expand=2, d_conv=4):
    if kind == 'gru':
        return GRUCarrier(in_dim, out_dim)
    if kind == 'mamba':
        return MambaCarrier(in_dim, out_dim, mamba_d_model, d_state, headdim, n_layers=n_layers)
    raise ValueError(f"unknown carrier kind: {kind!r}")
