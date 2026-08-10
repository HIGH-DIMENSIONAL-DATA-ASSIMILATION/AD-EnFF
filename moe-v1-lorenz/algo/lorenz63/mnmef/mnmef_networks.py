"""Neural components used by the MNMEF analysis update."""

import torch
import torch.nn as nn


class Simple_MLP(nn.Module):
    def __init__(self, d_input, d_output, latent_dim=64, num_hidden_layers=2):
        super().__init__()
        layers = [nn.Linear(d_input, latent_dim), nn.ReLU()]
        for _ in range(num_hidden_layers):
            layers.extend([nn.Linear(latent_dim, latent_dim), nn.ReLU()])
        layers.append(nn.Linear(latent_dim, d_output))
        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.model(inputs)


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_KV, num_heads, freeze_WQ=False):
        super().__init__()
        del dim_KV
        self.multihead_attn = nn.MultiheadAttention(dim_Q, num_heads)
        self.ln1 = nn.LayerNorm(dim_Q)
        self.ln2 = nn.LayerNorm(dim_Q)
        self.ffn = nn.Sequential(nn.Linear(dim_Q, dim_Q), nn.ReLU(), nn.Linear(dim_Q, dim_Q))
        if freeze_WQ:
            self._freeze_query_projection()

    def _freeze_query_projection(self):
        dimension = self.multihead_attn.embed_dim
        with torch.no_grad():
            self.multihead_attn.in_proj_weight[:dimension].copy_(torch.eye(dimension))
            self.multihead_attn.in_proj_bias[:dimension].zero_()

        def freeze_weight(gradient):
            gradient[:dimension].zero_()
            return gradient

        def freeze_bias(gradient):
            gradient[:dimension].zero_()
            return gradient

        self.multihead_attn.in_proj_weight.register_hook(freeze_weight)
        self.multihead_attn.in_proj_bias.register_hook(freeze_bias)

    def forward(self, query, keys):
        query_norm = self.ln1(query).transpose(0, 1)
        keys_norm = self.ln1(keys).transpose(0, 1)
        attended, _ = self.multihead_attn(query_norm, keys_norm, keys_norm)
        hidden = query + attended.transpose(0, 1)
        return hidden + self.ffn(self.ln2(hidden))


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, num_heads)
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, inputs):
        return self.fc(self.mab(inputs, inputs))


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, freeze_WQ=False):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, num_seeds, dim))
        self.mab = MAB(dim, dim, num_heads, freeze_WQ)

    def forward(self, inputs):
        seeds = self.S.expand(inputs.shape[0], -1, -1)
        return self.mab(seeds, inputs)


class SetTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_inds, output_dim, hidden_dim,
                 num_layers=2, freeze_WQ=False):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.enc = nn.Sequential(*[
            SAB(hidden_dim, hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.pma = PMA(hidden_dim, num_heads, num_inds, freeze_WQ)
        self.dec = nn.Sequential(*[
            SAB(hidden_dim, hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(hidden_dim * num_inds, output_dim)

    def forward(self, inputs):
        hidden = self.dec(self.pma(self.enc(self.embedding(inputs))))
        return self.fc_out(hidden.reshape(hidden.shape[0], -1))
