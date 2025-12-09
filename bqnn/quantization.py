"""
Quantization utilities for BQNN.

Converts classical features to binary representations and maps
binary values to quantum rotation angles.
"""

from typing import Optional, Tuple

import torch
import numpy as np


def binarize_features(
    x: torch.Tensor,
    threshold: float = 0.5,
    stochastic: bool = False,
) -> torch.Tensor:
    """
    Convert real-valued inputs to {0, 1} via thresholding.
    
    Args:
        x: Input tensor with values in [0, 1]
        threshold: Decision boundary for binarization
        stochastic: If True, use probabilistic binarization
        
    Returns:
        Binary tensor with values in {0, 1}
    """
    if stochastic:
        # Stochastic binarization: P(1) = x
        return (torch.rand_like(x) < x).float()
    else:
        return (x > threshold).float()


def features_to_angles(
    x_bin: torch.Tensor,
    a: float,
    base_scale: float = np.pi / 2.0,
) -> torch.Tensor:
    """
    Map binary features to rotation angles.
    
    The mapping is:
        angle = base_scale * (2 * bit - 1) * (1 + a)
    
    Where:
        - bit ∈ {0, 1} maps to {-1, +1}
        - a=0 gives fixed angles ±π/2
        - a>0 increases angular spread (more quantum variance)
    
    Physical interpretation:
        - a=0: Classical limit with fixed, deterministic encoding
        - a>0: Quantum regime with increased measurement uncertainty
        - a=1: Angles span full ±π range
        
    Args:
        x_bin: Binary features {0, 1} tensor
        a: Quantumness parameter (typically 0 to 1)
        base_scale: Base rotation angle (default π/2)
        
    Returns:
        Rotation angles tensor
    """
    bit_pm = 2 * x_bin - 1  # {0, 1} → {-1, +1}
    angles = base_scale * bit_pm * (1.0 + float(a))
    return angles


def angles_to_features(
    angles: torch.Tensor,
    a: float,
    base_scale: float = np.pi / 2.0,
) -> torch.Tensor:
    """
    Inverse mapping from angles back to binary features.
    
    Args:
        angles: Rotation angles tensor
        a: Quantumness parameter used in encoding
        base_scale: Base rotation angle
        
    Returns:
        Binary features {0, 1}
    """
    bit_pm = angles / (base_scale * (1.0 + float(a)))
    x_bin = (bit_pm + 1) / 2
    return (x_bin > 0.5).float()


def amplitude_encoding(
    x: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Alternative encoding: normalize features for amplitude encoding.
    
    This maps features to quantum amplitudes rather than rotation angles.
    Requires normalized vectors: |ψ⟩ = Σ_i x_i |i⟩ where Σ|x_i|² = 1
    
    Args:
        x: Input features [batch_size, n_features]
        normalize: Whether to L2-normalize
        
    Returns:
        Normalized amplitudes suitable for amplitude encoding
    """
    if normalize:
        norms = torch.norm(x, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)
        return x / norms
    return x


def iqp_encoding(
    x: torch.Tensor,
    order: int = 2,
) -> torch.Tensor:
    """
    IQP (Instantaneous Quantum Polynomial) style encoding.
    
    Creates angle products for data re-uploading style circuits:
        θ_ij = x_i * x_j for order=2
    
    Args:
        x: Input features [batch_size, n_features]
        order: Polynomial order (1 = linear, 2 = quadratic interactions)
        
    Returns:
        Encoded angles including interaction terms
    """
    batch_size, n_features = x.shape
    
    if order == 1:
        return x * np.pi  # Simple linear encoding
    
    elif order == 2:
        # Include linear and pairwise terms
        angles = [x * np.pi]
        
        # Pairwise products
        for i in range(n_features):
            for j in range(i + 1, n_features):
                angles.append((x[:, i:i+1] * x[:, j:j+1]) * np.pi)
        
        return torch.cat(angles, dim=1)
    
    else:
        raise ValueError(f"Order {order} not implemented")


def compute_encoding_capacity(
    n_features: int,
    a: float,
    base_scale: float = np.pi / 2.0,
) -> dict:
    """
    Compute theoretical encoding capacity metrics.
    
    Args:
        n_features: Number of input features
        a: Quantumness parameter
        base_scale: Base rotation angle
        
    Returns:
        Dictionary with capacity metrics
    """
    angle_range = 2 * base_scale * (1 + a)  # Total angular range
    
    # Theoretical bits of information per qubit
    # For rotation encoding, capacity ≈ 1 bit per qubit (binary input)
    # But effective capacity depends on measurement precision
    
    return {
        "n_features": n_features,
        "angle_range": angle_range,
        "angle_range_degrees": np.degrees(angle_range),
        "theoretical_capacity_bits": n_features,
        "quantumness": a,
    }
