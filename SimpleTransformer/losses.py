
import torch
import numpy as np
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


import torch
import torch.nn as nn

class HuberLoss(nn.Module):
    def __init__(self, delta: float = 50.0):
        """
        Args:
            delta (float): Threshold at which to switch between L1 and L2 loss.
                         Default is 1.0
        """
        super().__init__()
        self.delta = delta

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Huber Loss between predictions and ground truth.

        Args:
            y_pred (torch.Tensor): Predicted values
            y_true (torch.Tensor): Ground truth values

        Returns:
            torch.Tensor: Computed Huber Loss
        """
        error = y_pred - y_true
        abs_error = torch.abs(error)

        quadratic = torch.min(abs_error, torch.tensor(self.delta))
        linear = abs_error - quadratic

        loss = 0.5 * quadratic.pow(2) + self.delta * linear

        return torch.mean(loss)

def calculate_losses(all_outputs, all_targets):
    """
    Calculate L1 and L2 losses between model outputs and targets

    Args:
        all_outputs: numpy array of model predictions
        all_targets: numpy array of target values

    Returns:
        l1_loss: Mean Absolute Error (L1)
        l2_loss: Mean Squared Error (L2)
    """
    # Ensure arrays are flattened
    outputs_flat = all_outputs.flatten()
    targets_flat = all_targets.flatten()

    # L1 loss (Mean Absolute Error)
    l1_loss = np.mean(np.abs(outputs_flat - targets_flat))

    # L2 loss (Mean Squared Error)
    l2_loss = np.mean(np.square(outputs_flat - targets_flat))

    return l1_loss, l2_loss

class InverseHuberLoss(nn.Module):
    def __init__(self, delta=10.0):
        super(InverseHuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        residual = y_pred - y_true
        abs_residual = torch.abs(residual)
        
        # L1 for small errors
        l1_loss = abs_residual
        
        # L2 for large errors
        l2_loss = (residual ** 2) / (2 * self.delta) + (self.delta / 2)
        
        # Apply condition
        loss = torch.where(abs_residual <= self.delta, l1_loss, l2_loss)
        return loss.mean()
