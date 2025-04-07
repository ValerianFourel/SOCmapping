import torch
import torch.nn as nn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# Initialize LPIPS
# lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)

# Normalization functions
def normalize_tensor_01(tensor):
    """Normalize tensor to [0, 1] range."""
    max_val = torch.max(tensor)
    min_val = torch.min(tensor)
    return (tensor - min_val) / (max_val - min_val + 1e-6)  # Add small epsilon to avoid division by zero

def normalize_tensor_lst(tensor):
    """Normalize tensor to [-1, 1] range."""
    max_val = torch.max(tensor)
    min_val = torch.min(tensor)
    return 2 * (tensor - min_val) / (max_val - min_val + 1e-8) - 1

def normalize_tensor_lai(tensor):
    """Normalize tensor to [-1, 1] range and replace NaN values with 0."""
    # Replace NaN values with 0
    tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)

    max_val = torch.max(tensor)
    min_val = torch.min(tensor)
    return 2 * (tensor - min_val) / (max_val - min_val + 1e-8) - 1


# Loss classes
class ElevationVAELoss(nn.Module):
    def __init__(self, data_length, batch_size):
        super().__init__()
        self._loss_weight = data_length / batch_size

    def get_loss_weight(self):
        """Return the loss weight."""
        return self._loss_weight

    def forward(self, reconstructed_x, x_final, x, mu, logvar, lpips):
        # Normalize and repeat channels
        recon_repeated = normalize_tensor_01(x_final).repeat(1, 3, 1, 1)
        x_repeated = normalize_tensor_01(x).repeat(1, 3, 1, 1)
        
        # Calculate losses
        lpips_loss = lpips(recon_repeated, x_repeated)
        mse_loss = nn.MSELoss(reduction='mean')(reconstructed_x, x)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return kl_div + (mse_loss + lpips_loss * 100) * self._loss_weight

class TotalEvapotranspirationVAELoss(nn.Module):
    def __init__(self, data_length, batch_size):
        super().__init__()
        self._loss_weight = data_length / batch_size
        self.mse_loss = nn.MSELoss()

    def get_loss_weight(self):
        """Return the loss weight."""
        return self._loss_weight

    def forward(self, reconstructed_x, x, mu, logvar,lpips):
        # Normalize and repeat channels
        
        recon_repeated = normalize_tensor_01(reconstructed_x).repeat(1, 3, 1, 1)
        x_repeated = normalize_tensor_01(x).repeat(1, 3, 1, 1)
        
        # Calculate losses
        mse_loss = self.mse_loss(reconstructed_x, x)
        lpips_loss = lpips(recon_repeated, x_repeated)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return kl_div + (lpips_loss * 0.3 + mse_loss * 0.5) * self._loss_weight

class SoilEvaporationVAELoss(nn.Module):
    def __init__(self, data_length, batch_size):
        super().__init__()
        self._loss_weight = data_length / batch_size
        self.mse_loss = nn.MSELoss()

    def get_loss_weight(self):
        """Return the loss weight."""
        return self._loss_weight

    def forward(self, reconstructed_x, x, mu, logvar,lpips):
        # Normalize and repeat channels
        recon_repeated = normalize_tensor_01(reconstructed_x).repeat(1, 3, 1, 1)
        x_repeated = normalize_tensor_01(x).repeat(1, 3, 1, 1)
        
        # Calculate losses
        mse_loss = self.mse_loss(reconstructed_x, x)
        lpips_loss = lpips(recon_repeated, x_repeated)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return kl_div + (lpips_loss * 0.3 + mse_loss * 0.5) * self._loss_weight

class NPPVAELoss(nn.Module):
    def __init__(self, data_length, batch_size):
        super().__init__()
        self._loss_weight = data_length / batch_size

    def get_loss_weight(self):
        """Return the loss weight."""
        return self._loss_weight

    def forward(self, reconstructed_x, x, mu,logvar,lpips):
        # Normalize and repeat channels
        recon_repeated = normalize_tensor_01(reconstructed_x).repeat(1, 3, 1, 1)
        x_repeated = normalize_tensor_01(x).repeat(1, 3, 1, 1)
        
        # Calculate losses
        mse_loss = nn.MSELoss(reduction='mean')(reconstructed_x, x)
        lpips_loss = lpips(recon_repeated, x_repeated)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return kl_div + (lpips_loss * 0.2 * self._loss_weight + mse_loss * 0.5)

class LSTVAELoss(nn.Module):
    def __init__(self, data_length, batch_size):
        super().__init__()
        self._loss_weight = data_length / batch_size

    def get_loss_weight(self):
        """Return the loss weight."""
        return self._loss_weight

    def forward(self, x_final, x, inputs, mu, logvar, lpips):
        # Normalize and repeat channels
        recon_repeated = normalize_tensor_lst(x).repeat(1, 3, 1, 1)
        x_repeated = normalize_tensor_lst(inputs).repeat(1, 3, 1, 1)
        
        # Calculate losses
        l1_loss = nn.L1Loss(reduction='mean')(x_final, inputs)
        lpips_loss = lpips(recon_repeated, x_repeated)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return kl_div + (l1_loss + lpips_loss * 10) * self._loss_weight

class LAIVAELoss(nn.Module):
    def __init__(self, data_length, batch_size):
        super().__init__()
        self._loss_weight = data_length / batch_size
        self.mse_loss = nn.MSELoss()

    def get_loss_weight(self):
        """Return the loss weight."""
        return self._loss_weight

    def forward(self, reconstructed_x, x, mu, logvar, lpips):
        # Normalize and repeat channels
        if torch.isnan(x).any():
            print("the input : Warning: Input tensor contains NaN values. They will be replaced with 0.")
        if torch.isnan(reconstructed_x).any():
            print("the reconstructed_x : Warning: Input tensor contains NaN values. They will be replaced with 0.")

        recon_repeated = normalize_tensor_lai(reconstructed_x).repeat(1, 3, 1, 1)
        x_repeated = normalize_tensor_lai(x).repeat(1, 3, 1, 1)
        # recon_repeated = reconstructed_x.repeat(1, 3, 1, 1)
        # x_repeated = x.repeat(1, 3, 1, 1)
        
        # Calculate losses
        mse_loss = self.mse_loss(reconstructed_x, x)
        lpips_loss = lpips(recon_repeated, x_repeated)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return kl_div + (lpips_loss * 4 + mse_loss * 0.003) * self._loss_weight

import torch
import torch.nn as nn

class LAIVAELoss_v2(nn.Module):
    def __init__(self, data_length, batch_size, gamma=0.01):
        super().__init__()
        self._loss_weight = data_length / batch_size
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.gamma = gamma  # Scaling factor for L1 loss

    def get_loss_weight(self):
        """Return the loss weight."""
        return self._loss_weight

    def forward(self, reconstructed_x, x, mu, logvar):
        # Handle NaN values in inputs
        if torch.isnan(x).any():
            print("Warning: Input tensor contains NaN values. They will be replaced with 0.")
            x = torch.nan_to_num(x, nan=0.0)
        if torch.isnan(reconstructed_x).any():
            print("Warning: Reconstructed tensor contains NaN values. They will be replaced with 0.")
            reconstructed_x = torch.nan_to_num(reconstructed_x, nan=0.0)
        reconstructed_x = torch.where(torch.isnan(reconstructed_x), torch.zeros_like(reconstructed_x), reconstructed_x)
        # reconstructed_x = torch.where(torch.isnan(reconstructed_x), torch.zeros_like(reconstructed_x), reconstructed_x)

        # Calculate MSE and L1 losses
        mse_loss = self.mse_loss(reconstructed_x, x)
        l1_loss = self.l1_loss(reconstructed_x, x)

        # KL divergence term
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Composite reconstruction loss
        reconstruction_loss = self.gamma * mse_loss +  l1_loss

        # Total loss
        return kl_div + reconstruction_loss * self._loss_weight