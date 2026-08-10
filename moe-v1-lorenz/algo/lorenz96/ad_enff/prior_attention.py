"""Particle self-attention and PMA pooling used by the attention."""

import torch
import torch.nn as nn


class SAB(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        x_norm = self.ln1(x)
        attn, _ = self.mha(x_norm, x_norm, x_norm)
        hidden = x + attn
        return hidden + self.ff(self.ln2(hidden))


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds=16):
        super().__init__()
        self.seeds = nn.Parameter(torch.empty(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.seeds)
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        if self.mha.in_proj_weight is not None:
            self.mha.in_proj_weight.register_hook(
                lambda grad: self._freeze_query_gradient(grad, dim)
            )
        self.decoder = nn.Sequential(SAB(dim, num_heads), SAB(dim, num_heads))
        self.proj = nn.Linear(num_seeds * dim, dim)

    @staticmethod
    def _freeze_query_gradient(grad, dim):
        result = grad.clone()
        result[:dim, :] = 0.0
        return result

    def forward(self, x):
        seeds = self.seeds.expand(x.size(0), -1, -1)
        pooled, _ = self.mha(seeds, x, x)
        pooled = self.decoder(pooled + seeds)
        return self.proj(pooled.reshape(x.size(0), -1))


class ParticleAttention(nn.Module):
    """Encode an ensemble cloud and pool it to one permutation-invariant summary."""

    def __init__(self, dim_x, n_fields, attn_dim=32, n_heads=4,
                 use_pma=True, pma_seeds=16):
        super().__init__()
        self.in_proj = nn.Linear(n_fields * dim_x, attn_dim)
        self.norm1 = nn.LayerNorm(attn_dim)
        self.attn = nn.MultiheadAttention(attn_dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(attn_dim)
        self.ff = nn.Sequential(
            nn.Linear(attn_dim, 2 * attn_dim),
            nn.SiLU(),
            nn.Linear(2 * attn_dim, attn_dim),
        )
        self.pma = PMA(attn_dim, n_heads, pma_seeds) if use_pma else None

    def forward(self, fields):
        hidden = self.in_proj(torch.cat(fields, dim=-1))
        normalized = self.norm1(hidden)
        attended, _ = self.attn(normalized, normalized, normalized)
        hidden = hidden + attended
        hidden = hidden + self.ff(self.norm2(hidden))
        return self.pma(hidden) if self.pma is not None else hidden.mean(dim=1)
