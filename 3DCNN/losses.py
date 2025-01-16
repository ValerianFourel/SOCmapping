
import torch
import torch.nn as nn


class ChiSquareLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(ChiSquareLoss, self).__init__()
        self.eps = eps  # small constant to prevent division by zero

    def forward(self, y_pred, y_true, sigma=None):
        """
        Args:
            y_pred: predicted values
            y_true: true values
            sigma: uncertainties (optional, defaults to sqrt(y_true) if None)
        """
        if sigma is None:
            # Use Poisson statistics where variance = mean
            sigma = torch.abs(y_true) + self.eps
            #sigma = torch.sqrt(torch.abs(y_true) + self.eps)
        chi_square = torch.pow(y_pred - y_true, 2) / (2 * torch.pow(sigma, 2) + self.eps)
        return torch.mean(chi_square)
    

class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Log-Cosh Loss between predictions and ground truth.

        Args:
            y_pred (torch.Tensor): Predicted values
            y_true (torch.Tensor): Ground truth values

        Returns:
            torch.Tensor: Computed Log-Cosh Loss
        """
        def log_cosh(x):
            # Using the formula log(cosh(x)) = x + log(1 + exp(-2x)) - log(2)
            return x + torch.log1p(torch.exp(-2 * x)) - 0.6931471805599453  # log(2)

        return torch.mean(log_cosh(y_pred - y_true))
