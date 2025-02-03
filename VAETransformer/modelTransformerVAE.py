import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerVAE(nn.Module):
    def __init__(self, input_channels=6, input_height=33, input_width=33, input_time=4, 
                 num_heads=2, latent_dim=8, dropout_rate=0.3):
        super(TransformerVAE, self).__init__()

        # Calculate d_model ensuring it is divisible by num_heads
        self.d_model = input_channels * input_height
        if self.d_model % num_heads != 0:
            self.d_model = (self.d_model // num_heads + 1) * num_heads

        self.input_time = input_time
        self.input_shape = (input_channels, input_height, input_width)

        # Encoder
        self.input_projection = nn.Linear(input_channels * input_height * input_width, self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=num_heads, 
            dropout=dropout_rate,
            dim_feedforward=32
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # VAE components
        self.fc_mu = nn.Linear(self.d_model * input_time, latent_dim)
        self.fc_var = nn.Linear(self.d_model * input_time, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.d_model * input_time)
        self.decoder_hidden = nn.Linear(self.d_model * input_time, 32)
        self.decoder_output = nn.Linear(32, input_channels * input_height * input_width * input_time)

        self.relu = nn.ReLU()

    def encode(self, x):
        batch_size, channels, height, width, time = x.size()

        # Reshape input
        x = x.permute(0, 4, 1, 2, 3)
        x = x.reshape(batch_size, time, -1)

        # Project to d_model dimension
        x = self.input_projection(x)

        # Transformer encoding
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)
        x = x.reshape(batch_size, -1)

        # Get mu and log_var
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        batch_size = z.size(0)

        # Decode from latent space
        x = self.decoder_input(z)
        x = self.relu(x)
        x = self.decoder_hidden(x)
        x = self.relu(x)
        x = self.decoder_output(x)

        # Reshape back to original dimensions
        x = x.view(batch_size, self.input_time, *self.input_shape)
        x = x.permute(0, 2, 3, 4, 1)  # Return to (batch, channels, height, width, time)

        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Loss function for VAE
def vae_loss(recon_x, x, mu, log_var, kld_weight=0.01):
    """
    VAE loss function combining reconstruction loss and KL divergence
    """
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + kld_weight * kld_loss, recon_loss, kld_loss
