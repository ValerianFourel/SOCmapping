import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerVAE(nn.Module):
    def __init__(self, input_channels=1, input_height=33, input_width=33, 
                 num_heads=2, latent_dim=4, dropout_rate=0.2):
        super(TransformerVAE, self).__init__()

        self.input_channels = input_channels  # Fixed to 1 for single band
        self.input_height = input_height
        self.input_width = input_width

        # Encoder setup
        self.d_model = min(64, input_channels * input_height)
        if self.d_model % num_heads != 0:
            self.d_model = (self.d_model // num_heads + 1) * num_heads

        self.input_projection = nn.Linear(input_channels * input_height * input_width, self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dropout=dropout_rate,
            dim_feedforward=128
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=7)

        self.fc_mu = nn.Linear(self.d_model, latent_dim)
        self.fc_var = nn.Linear(self.d_model, latent_dim)

        # Decoder with memory mechanism
        self.decoder_projection = nn.Linear(latent_dim, self.d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dropout=dropout_rate,
            dim_feedforward=128
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=7)
        
        self.output_projection = nn.Linear(self.d_model, input_channels * input_height * input_width)

    def encode(self, x):
        batch_size, channels, height, width = x.size()  # [batch_size, 1, 33, 33]
        
        x = x.reshape(batch_size, -1)  # Flatten to [batch_size, channels*height*width]
        x = self.input_projection(x)  # [batch_size, d_model]

        x = x.unsqueeze(0)  # [1, batch_size, d_model] (time=1)
        memory = self.transformer_encoder(x)  # [1, batch_size, d_model]
        x = memory.squeeze(0)  # [batch_size, d_model]

        mu = self.fc_mu(x)
        log_var = torch.clamp(F.softplus(self.fc_var(x)), min=-10, max=10)  # Stabilize log_var
        return mu, log_var, memory

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, memory):
        batch_size = z.size(0)
        
        x = self.decoder_projection(z)  # [batch_size, d_model]
        x = x.unsqueeze(0)  # [1, batch_size, d_model] (time=1)

        x = self.transformer_decoder(x, memory)  # [1, batch_size, d_model]
        x = x.squeeze(0)  # [batch_size, d_model]
        
        x = self.output_projection(x)  # [batch_size, channels*height*width]
        x = x.view(batch_size, self.input_channels, self.input_height, self.input_width)
        return x

    def forward(self, x):
        mu, log_var, memory = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z, memory)
        return reconstruction, mu, log_var

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)