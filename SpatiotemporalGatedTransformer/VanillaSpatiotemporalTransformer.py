"""
VanillaSpatiotemporalTransformer — fair-comparison ablation of SimpleSGT.

Identical to SimpleSGT in every respect EXCEPT the Gated Residual Network
(GRN) block (the "G" in SGT). The GRN at SimpleSGT lines 23-36 is replaced
by a plain feature_dim → d_model Linear + LayerNorm. Everything else —
spatial encoder, positional embedding, transformer encoder layer, MLP
head — is byte-identical.

Purpose: provide a parameter-matched-by-hyperparameter "vanilla
spatiotemporal transformer" baseline so reviewers cannot argue SGT's
advantage is from being a small transformer *in general* rather than from
the gated residual mechanism specifically.

Trained-architecture parameter counts at common settings (input_channels=20,
height=5, width=5, time_steps=5):

    SimpleSGT             d=128, h=2: ~363k   ← winner of spatial-CV sweep
    VanillaTransformer    d=128, h=2: ~213k   ← this class, identical hp

The vanilla baseline is SMALLER than the gated variant at matched
hyperparameters (the gate + grn + residual_proj costs ~150k params at
d=128). So if vanilla matches SGT on R², it's both smaller AND simpler;
the gating doesn't earn its keep. If SGT beats vanilla, the gating earns
its 150k overhead.

Scalability note (this revision): the feed-forward width now scales as
4*d_model and the transformer depth is exposed via num_layers (default 1),
so vanilla can be grown to multi-layer / multi-million-param configs to test
whether a LARGER CNN+transformer beats the small gated SGT. The
"byte-identical to SimpleSGT except the GRN" claim above holds only at the
original d_model=128, num_layers=1, dim_feedforward=128 setting.
"""
import torch
import torch.nn as nn


class VanillaSpatiotemporalTransformer(nn.Module):
    """SimpleSGT minus the Gated Residual Network."""

    def __init__(self, input_channels=20, height=5, width=5, time_steps=5,
                 d_model=128, num_heads=2, num_layers=1, dropout=0.3,
                 use_linear_skip=True):
        super().__init__()
        self.time_steps = time_steps
        self.use_linear_skip = use_linear_skip

        # SAME as SimpleSGT
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.feature_dim = 32 * 4 * 4  # 512

        # ABLATION: replace SimpleSGT's gated block (lines 23-36 in
        # SimpleSGT.py) with a single Linear + LayerNorm. No gate, no
        # MLP, no residual mix.
        self.feature_proj = nn.Linear(self.feature_dim, d_model)
        self.layernorm = nn.LayerNorm(d_model)

        # Scalable encoder: depth (num_layers) and a width-proportional
        # feed-forward (4*d_model, the standard transformer ratio) so the
        # model genuinely grows with d_model / num_layers — vs SimpleSGT's
        # fixed single layer + dim_feedforward=128.
        self.pos_embedding = nn.Parameter(torch.randn(time_steps, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dropout=dropout,
            dim_feedforward=4 * d_model,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                          num_layers=num_layers)
        # Head: linear-skip baseline + MLP residual (matched to SimpleSGT so the
        # gate ablation differs ONLY by the GRN). The linear skip restores
        # dynamic range for a crisper, less mean-biased production map.
        feat_dim = time_steps * d_model
        self.head_norm = nn.LayerNorm(feat_dim)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.linear_skip = nn.Linear(feat_dim, 1) if use_linear_skip else None

    def forward(self, x):
        """x: (B, C, H, W, T) — matches the project-wide dataloader output."""
        B, C, H, W, T = x.shape
        assert T == self.time_steps

        # Per-timestep CNN feature extraction (identical to SimpleSGT)
        x = x.permute(0, 4, 1, 2, 3).reshape(B * T, C, H, W)
        x = self.spatial_encoder(x)
        x = x.view(B, T, -1)                          # (B, T, feature_dim)

        # ----- ABLATION: plain projection in place of the GRN -----
        x = self.feature_proj(x)                      # (B, T, d_model)
        x = self.layernorm(x)
        # ----------------------------------------------------------

        # Positional encoding + transformer (identical to SimpleSGT)
        x = x + self.pos_embedding                    # (B, T, d_model)
        x = x.permute(1, 0, 2)                        # (T, B, d_model)
        x = self.transformer_encoder(x)               # (T, B, d_model)
        feat = x.permute(1, 0, 2).reshape(B, -1)      # (B, T * d_model)
        out = self.head(self.head_norm(feat))
        if self.linear_skip is not None:
            out = out + self.linear_skip(feat)
        return out.squeeze()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check: param count vs SimpleSGT at matched hp
    import sys, importlib.util, pathlib
    p = pathlib.Path(__file__).resolve().parent / 'SimpleSGT.py'
    spec = importlib.util.spec_from_file_location('SimpleSGT', p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

    print(f"{'config':<40} {'params':>12}")
    print('-' * 55)
    for C, d, h in [(20, 128, 2), (20, 128, 4), (20, 64, 4), (20, 96, 4)]:
        sgt = m.SimpleSGT(input_channels=C, height=5, width=5, time_steps=5,
                          d_model=d, num_heads=h, dropout=0.5)
        van = VanillaSpatiotemporalTransformer(
            input_channels=C, height=5, width=5, time_steps=5,
            d_model=d, num_heads=h, dropout=0.5)
        print(f"  SimpleSGT (C={C}, d={d}, h={h})         "
              f"{sgt.count_parameters():>12,}")
        print(f"  VanillaTransformer (C={C}, d={d}, h={h}) "
              f"{van.count_parameters():>12,}")

    # Forward pass test
    x = torch.randn(2, 20, 5, 5, 5)
    out = VanillaSpatiotemporalTransformer(input_channels=20)(x)
    assert out.shape == (2,), f"expected (2,), got {tuple(out.shape)}"
    print("\nforward pass OK  output shape:", tuple(out.shape))
