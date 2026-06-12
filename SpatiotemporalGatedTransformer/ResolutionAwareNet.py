"""ResolutionAwareNet — multi-branch network that processes covariate bands by
their NATIVE resolution group and fuses late.

Input is the same cube the rest of the pipeline uses:  x [B, C, H, W, T]
(channels = bands in bands_list_order; H=W=window_size; T=time_steps). The model
slices channels into three groups (fine / medium / coarse) using index lists
computed from _bands.resolution_groups(), and treats each group by what its
resolution can actually support:

  * fine   (<=30 m, S2 SWIR + Landsat SRC + SRTM terrain): a real spatial patch
    through a compact CNN -> GRN gate -> 1-layer Transformer over time (the same
    front end as SimpleSGT, reused so f_fine is the proven encoder).
  * medium (250 m, MODIS NDVI/EVI/phenology + SoilGrids): a 250 m pixel barely
    varies across a 5-px patch, so take the CENTRE pixel per timestep -> MLP.
  * coarse (500 m-11 km, MODIS LAI/NPP/ET/LST + ERA5 climate): one centre value
    per band per timestep -> MLP. Context, not texture.

Late fusion: concat [f_fine, f_med, f_coarse] -> dense -> scalar (log-SOC).

ABLATIONS (the part that makes it a table, not a demo):
  branches: 'fine' | 'fine_med' | 'fine_med_coarse' | 'all_flat'
     all_flat = flatten ALL bands x H x W x T through a plain MLP (the
     resolution-NAIVE baseline). If multi-branch can't beat all_flat, that IS
     the finding.
  ablate_group: 'fine' | 'medium' | 'coarse' | None
     zero that group's input to measure its marginal R^2 contribution.

No BatchNorm anywhere (LayerNorm only) — batch stats leak across spatially
correlated samples under spatial CV.
"""
import torch
import torch.nn as nn


class _FineEncoder(nn.Module):
    """CNN -> GRN gate -> 1-layer Transformer over T, pooled to a feature vector.
    Mirrors SimpleSGT but returns the pre-head feature (mean-pooled over time)."""

    def __init__(self, in_ch, height, width, time_steps, d_model, num_heads, dropout):
        super().__init__()
        self.time_steps = time_steps
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        feat = 32 * 4 * 4
        self.grn = nn.Sequential(nn.Linear(feat, d_model), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(d_model, d_model))
        self.gate = nn.Sequential(nn.Linear(feat, d_model), nn.Sigmoid())
        self.residual_proj = nn.Linear(feat, d_model)
        self.layernorm = nn.LayerNorm(d_model)
        self.pos_embedding = nn.Parameter(torch.randn(time_steps, d_model))
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                                         dropout=dropout, dim_feedforward=2 * d_model,
                                         batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=1)
        self.out_dim = d_model

    def forward(self, x):                       # x [B, Cf, H, W, T]
        B, C, H, W, T = x.shape
        x = x.permute(0, 4, 1, 2, 3).reshape(B * T, C, H, W)
        x = self.spatial_encoder(x).reshape(B, T, -1)        # [B, T, feat]
        x = self.layernorm(self.gate(x) * self.grn(x) + self.residual_proj(x))
        x = x + self.pos_embedding
        x = self.transformer(x)                               # [B, T, d_model]
        return x.mean(dim=1)                                  # [B, d_model]


class _CentreMLP(nn.Module):
    """Centre pixel per timestep (no spatial structure) -> flatten over T -> MLP."""

    def __init__(self, n_bands, time_steps, hidden, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_bands * time_steps, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, out_dim), nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, x):                       # x [B, Cg, H, W, T]
        B, C, H, W, T = x.shape
        c = x[:, :, H // 2, W // 2, :].reshape(B, C * T)      # centre pixel, all T
        return self.net(c)


class ResolutionAwareNet(nn.Module):
    def __init__(self, input_channels, height=5, width=5, time_steps=5,
                 d_model=64, num_heads=4, dropout=0.3,
                 fine_idx=None, med_idx=None, coarse_idx=None,
                 branches='fine_med_coarse', ablate_group=None,
                 med_hidden=32, coarse_hidden=32, group_dim=32, flat_hidden=48):
        super().__init__()
        self.branches = branches
        self.ablate_group = ablate_group
        self.register_buffer('fine_idx', torch.as_tensor(fine_idx or [], dtype=torch.long))
        self.register_buffer('med_idx', torch.as_tensor(med_idx or [], dtype=torch.long))
        self.register_buffer('coarse_idx', torch.as_tensor(coarse_idx or [], dtype=torch.long))

        if branches == 'all_flat':
            # resolution-naive baseline: every band x H x W x T -> plain MLP
            flat = input_channels * height * width * time_steps
            self.flat = nn.Sequential(
                nn.Linear(flat, flat_hidden), nn.LayerNorm(flat_hidden), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(flat_hidden, flat_hidden), nn.ReLU(),
                nn.Linear(flat_hidden, 1))
            return

        use_med = branches in ('fine_med', 'fine_med_coarse') and len(self.med_idx) > 0
        use_coarse = branches == 'fine_med_coarse' and len(self.coarse_idx) > 0

        self.fine = _FineEncoder(len(self.fine_idx), height, width, time_steps,
                                 d_model, num_heads, dropout)
        fused = self.fine.out_dim
        self.med = (_CentreMLP(len(self.med_idx), time_steps, med_hidden, group_dim, dropout)
                    if use_med else None)
        if self.med is not None:
            fused += group_dim
        self.coarse = (_CentreMLP(len(self.coarse_idx), time_steps, coarse_hidden, group_dim, dropout)
                       if use_coarse else None)
        if self.coarse is not None:
            fused += group_dim
        self.head = nn.Sequential(
            nn.Linear(fused, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def _slice(self, x, idx):
        return x.index_select(1, idx)

    def forward(self, x):                       # x [B, C, H, W, T]
        if self.branches == 'all_flat':
            return self.flat(x.reshape(x.shape[0], -1)).squeeze(-1)

        xf = self._slice(x, self.fine_idx)
        if self.ablate_group == 'fine':
            xf = torch.zeros_like(xf)
        feats = [self.fine(xf)]
        if self.med is not None:
            xm = self._slice(x, self.med_idx)
            if self.ablate_group == 'medium':
                xm = torch.zeros_like(xm)
            feats.append(self.med(xm))
        if self.coarse is not None:
            xc = self._slice(x, self.coarse_idx)
            if self.ablate_group == 'coarse':
                xc = torch.zeros_like(xc)
            feats.append(self.coarse(xc))
        return self.head(torch.cat(feats, dim=1)).squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
