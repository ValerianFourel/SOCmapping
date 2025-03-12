import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerVAE(nn.Module):
    def __init__(self, input_channels=6, input_height=33, input_width=33, input_time=4, 
                 num_heads=2, latent_dim=4, dropout_rate=0.2):
        super(TransformerVAE, self).__init__()

        # Reduce d_model size and ensure divisibility by num_heads
        self.d_model = min(64, input_channels * input_height)  # Cap at 64
        if self.d_model % num_heads != 0:
            self.d_model = (self.d_model // num_heads + 1) * num_heads

        self.input_time = input_time
        self.input_shape = (input_channels, input_height, input_width)

        # Simplified input projection
        self.input_projection = nn.Linear(input_channels * input_height * input_width, self.d_model)

        # Reduced encoder (single layer, smaller feedforward)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dropout=dropout_rate,
            dim_feedforward=16  # Reduced from 32
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # VAE components with reduced latent dim
        flat_size = self.d_model * input_time
        self.fc_mu = nn.Linear(flat_size, latent_dim)
        self.fc_var = nn.Linear(flat_size, latent_dim)

        # Transformer decoder (replacing previous linear layers)
        self.decoder_projection = nn.Linear(latent_dim, flat_size)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dropout=dropout_rate,
            dim_feedforward=16  # Reduced size
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        
        # Simplified output projection
        self.output_projection = nn.Linear(self.d_model, input_channels * input_height * input_width)

    def encode(self, x):
        batch_size, channels, height, width, time = x.size()
        
        # Reshape and project
        x = x.permute(0, 4, 1, 2, 3)  # (batch, time, channels, height, width)
        x = x.reshape(batch_size, time, -1)
        x = self.input_projection(x)

        # Transformer encoding
        x = x.transpose(0, 1)  # (time, batch, d_model)
        memory = self.transformer_encoder(x)
        x = memory.transpose(0, 1).reshape(batch_size, -1)  # (batch, time*d_model)

        # Get mu and log_var
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var, memory

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, memory):
        batch_size = z.size(0)
        
        # Project latent to transformer input
        x = self.decoder_projection(z)  # (batch, time*d_model)
        x = x.view(batch_size, self.input_time, self.d_model)
        x = x.transpose(0, 1)  # (time, batch, d_model)

        # Transformer decoding using encoder memory
        x = self.transformer_decoder(x, memory)
        
        # Project to output space
        x = x.transpose(0, 1)  # (batch, time, d_model)
        x = self.output_projection(x)  # (batch, time, channels*height*width)
        
        # Reshape to original dimensions
        x = x.view(batch_size, self.input_time, *self.input_shape)
        x = x.permute(0, 2, 3, 4, 1)  # (batch, channels, height, width, time)
        return x

    def forward(self, x):
        mu, log_var, memory = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z, memory)
        return reconstruction, mu, log_var

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Loss function remains unchanged
def vae_loss(recon_x, x, mu, log_var, kld_weight=0.01):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kld_weight * kld_loss, recon_loss, kld_loss