"""
Classical reference models for BQNN benchmarking.

Provides binarized neural networks that serve as the classical (a=0)
baseline for comparison with quantum-enhanced variants.
"""

import torch
import torch.nn as nn
from typing import Optional


class StraightThroughEstimator(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for gradient flow through sign function.
    
    Forward: sign(x)
    Backward: gradient passes through unchanged (identity)
    
    This is essential for training binarized networks since sign() has
    zero gradient almost everywhere.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # Straight-through: pass gradients unchanged
        return grad_output


class ClippedSTE(torch.autograd.Function):
    """
    Clipped Straight-Through Estimator.
    
    Forward: sign(x)
    Backward: gradient clipped to [-1, 1] region (hardtanh derivative)
    
    This variant constrains gradients to the linear region, which can
    improve training stability.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return torch.sign(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors
        # Zero gradient outside [-1, 1]
        mask = (x.abs() <= 1).float()
        return grad_output * mask


def binarize_ste(x: torch.Tensor, clipped: bool = False) -> torch.Tensor:
    """
    Binarize tensor using straight-through estimator.
    
    Args:
        x: Input tensor
        clipped: Use clipped STE variant
        
    Returns:
        Binarized tensor in {-1, +1}
    """
    if clipped:
        return ClippedSTE.apply(x)
    return StraightThroughEstimator.apply(x)


class ClassicalBinarizedNet(nn.Module):
    """
    Classical binarized feed-forward model.
    
    Architecture:
        - Linear layer
        - Sign activation (with STE for gradient flow)
        - Linear output layer
    
    This serves as the a=0 "classical limit" reference for BQNN comparisons.
    The sign activation provides the binary nonlinearity that corresponds
    to the measurement-based nonlinearity in the quantum model.
    
    Args:
        n_features: Input feature dimension
        n_hidden: Hidden layer width
        n_classes: Number of output classes
        use_ste: Use straight-through estimator (recommended)
        clipped_ste: Use clipped variant of STE
    """

    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        n_classes: int,
        use_ste: bool = True,
        clipped_ste: bool = False,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.use_ste = use_ste
        self.clipped_ste = clipped_ste
        
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_classes)

    def binarize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sign activation with optional STE."""
        if self.use_ste:
            return binarize_ste(x, clipped=self.clipped_ste)
        return torch.sign(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, n_features]
            
        Returns:
            Class logits [batch_size, n_classes]
        """
        h = self.fc1(x)
        h = self.binarize(h)
        logits = self.fc2(h)
        return logits
    
    def get_model_info(self) -> dict:
        """Return model statistics."""
        return {
            "n_features": self.n_features,
            "n_hidden": self.n_hidden,
            "n_classes": self.n_classes,
            "n_parameters": sum(p.numel() for p in self.parameters()),
            "use_ste": self.use_ste,
        }


class DeepBinarizedNet(nn.Module):
    """
    Deeper binarized network with multiple hidden layers.
    
    Args:
        n_features: Input dimension
        hidden_dims: List of hidden layer widths
        n_classes: Output dimension
        use_ste: Use straight-through estimator
    """
    
    def __init__(
        self,
        n_features: int,
        hidden_dims: list,
        n_classes: int,
        use_ste: bool = True,
    ):
        super().__init__()
        self.use_ste = use_ste
        
        dims = [n_features] + hidden_dims
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1])
            for i in range(len(dims) - 1)
        ])
        self.output = nn.Linear(hidden_dims[-1], n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
            x = binarize_ste(x) if self.use_ste else torch.sign(x)
        return self.output(x)
