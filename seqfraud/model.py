"""Hybrid sequence + tabular fraud model (PyTorch).

A small Transformer encoder reads the card's last K transactions (each
described relative to the current one); the current-transaction token's
output is concatenated with the Spark velocity aggregates and fed to an MLP.
"""
from __future__ import annotations

import torch
from torch import nn

from .simulate import N_MCC
from .sequences import K, N_NUM


class SeqFraudModel(nn.Module):
    def __init__(self, n_agg: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 seq_len: int = K + 1, dropout: float = 0.1):
        super().__init__()
        self.mcc_emb = nn.Embedding(N_MCC, 16)
        self.chan_emb = nn.Embedding(3, 4)
        self.step_proj = nn.Linear(N_NUM + 16 + 4, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2 * d_model,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.agg_norm = nn.BatchNorm1d(n_agg)
        self.head = nn.Sequential(
            nn.Linear(d_model + n_agg, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, num: torch.Tensor, cat: torch.Tensor, mask: torch.Tensor,
                agg: torch.Tensor) -> torch.Tensor:
        x = torch.cat([num, self.mcc_emb(cat[..., 0]), self.chan_emb(cat[..., 1])], dim=-1)
        x = self.step_proj(x) + self.pos
        h = self.encoder(x, src_key_padding_mask=~mask)
        cur = h[:, -1]                               # current-transaction token
        return self.head(torch.cat([cur, self.agg_norm(agg)], dim=-1)).squeeze(-1)
