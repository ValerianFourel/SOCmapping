"""
LightweightTransformer — pure transformer baseline with NO CNN frontend.

This is the CNN-frontend ablation companion to VanillaSpatiotemporalTransformer.
Whereas SimpleSGT / VanillaSpatiotemporalTransformer both run a 3×3 Conv2d
spatial encoder per timestep before the transformer, this class skips the
CNN entirely and feeds the raw flattened (C, H, W) per timestep straight
into a Linear input embedding.

The three-way ablation:
    SimpleSGT             = CNN spatial encoder + GRN gate + Transformer
    VanillaSpatiotemporal = CNN spatial encoder            + Transformer
    LightweightTransformer=                                 + Transformer (no CNN)

The only architectural delta from VanillaSpatiotemporalTransformer is
that the `nn.Conv2d(C, 16, 3) → Conv2d(16, 32, 3) → AvgPool(4×4)` block
is replaced by `nn.Flatten + Linear(C·H·W → d_model)`. So any R² gap
between Vanilla and Lightweight is directly attributable to the CNN's
spatial inductive bias.

Why this class exists: SimpleTransformerV2 (the current "transformer-only"
baseline in run_kfold) silently overrides d_model with C×H×W and uses a
4-layer MLP head — so its parameter count is locked at ~11.2M for 20-band
input, regardless of the `--hidden_size` flag. That makes it impossible
to compare against vanilla (~95k–215k) at matched scale. LightweightTransformer
respects d_model and gives a controllable 100k–400k param transformer-only
baseline.

Estimated parameter counts (input_channels=20, H=W=5, T=5):

    config                                       params
    -------------------------------------------- -------
    d=64,  h=4, L=1                              ~85k
    d=96,  h=4, L=1                              ~155k
    d=128, h=4, L=1                              ~240k
    d=64,  h=4, L=2                              ~150k
    d=128, h=4, L=2                              ~370k
"""
import torch
import torch.nn as nn


class LightweightTransformer(nn.Module):
    """Transformer-only spatiotemporal regressor. NO CNN frontend."""

    def __init__(self, input_channels=20, height=5, width=5, time_steps=5,
                 d_model=128, num_heads=4, num_layers=1, dropout=0.3):
        super().__init__()
        self.time_steps = time_steps
        self.flat_dim = input_channels * height * width   # e.g. 20*5*5 = 500

        # Single linear "embedding" — direct projection from the flattened
        # per-timestep input cube to d_model. This is the SOLE alternative
        # to the CNN-based spatial encoder used by SGT and Vanilla.
        self.input_proj = nn.Linear(self.flat_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Learned positional encoding over the T time steps. Same shape and
        # init scale as SimpleSGT's pos_embedding (0.02 std scales the
        # additive perturbation appropriately for LayerNorm-normalised input).
        self.pos_embedding = nn.Parameter(torch.randn(time_steps, d_model) * 0.02)

        # Transformer encoder. Pre-norm (norm_first=True) for stable
        # training without learning-rate warmup. dim_feedforward=2*d_model
        # keeps the FFN modest (typical default is 4*d, but at our scale
        # 2× keeps the parameter budget where we want it).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            norm_first=True,
            batch_first=False,    # (T, B, d_model) ordering — matches SGT/Vanilla
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                          num_layers=num_layers)
        self.head_norm = nn.LayerNorm(d_model)

        # Tiny MLP head — same shape as SGT/Vanilla so head capacity isn't
        # a confound when comparing the three architectures.
        self.head = nn.Sequential(
            nn.Linear(time_steps * d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """x: (B, C, H, W, T) — matches the project-wide dataloader output."""
        B, C, H, W, T = x.shape
        assert T == self.time_steps, \
            f"time_steps mismatch: got T={T}, model expects {self.time_steps}"

        # Move time dim to front (matches SGT/Vanilla), flatten C*H*W per
        # timestep into a single token. No spatial conv, no pooling.
        x = x.permute(0, 4, 1, 2, 3).reshape(B, T, self.flat_dim)
        # Project flat token to d_model
        x = self.input_proj(x)                 # (B, T, d_model)
        x = self.input_norm(x)

        # Add positional encoding (broadcast over batch dim)
        x = x + self.pos_embedding             # (B, T, d_model)

        # Transformer encoder expects (T, B, d_model) when batch_first=False
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)        # (T, B, d_model)
        x = x.permute(1, 0, 2)                 # (B, T, d_model)
        x = self.head_norm(x)

        # Flatten and predict
        x = x.reshape(B, -1)                   # (B, T * d_model)
        x = self.head(x)
        return x.squeeze()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print(f"{'config':<48} {'params':>12}")
    print('-' * 64)
    for C, d, h, L in [(20, 64, 4, 1),
                        (20, 96, 4, 1),
                        (20, 128, 4, 1),
                        (20, 64, 4, 2),
                        (20, 128, 4, 2),
                        (6,  64, 4, 1),
                        (6,  128, 4, 1)]:
        m = LightweightTransformer(
            input_channels=C, height=5, width=5, time_steps=5,
            d_model=d, num_heads=h, num_layers=L, dropout=0.5,
        )
        print(f"  LightweightTransformer(C={C}, d={d}, h={h}, L={L})  "
              f"{m.count_parameters():>10,}")

    # Forward-pass sanity
    x = torch.randn(2, 20, 5, 5, 5)
    out = LightweightTransformer(input_channels=20)(x)
    assert out.shape == (2,), f"expected (2,), got {tuple(out.shape)}"
    print("\nforward pass OK  output shape:", tuple(out.shape))
