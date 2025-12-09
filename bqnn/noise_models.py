"""
Noise models for BQNN experiments.

Provides various noise injection strategies to simulate real quantum hardware.
"""

from typing import Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass

import torch
import pennylane as qml

from .model import BQNNModel


class NoiseType(Enum):
    """Types of noise injection."""
    COHERENT = "coherent"      # U U† pairs (systematic error)
    DEPOLARIZING = "depolarizing"  # Random Pauli errors
    AMPLITUDE_DAMPING = "amplitude_damping"  # T1 decay
    PHASE_DAMPING = "phase_damping"  # T2 dephasing
    READOUT = "readout"        # Measurement errors


@dataclass
class NoiseConfig:
    """Configuration for noise injection."""
    noise_type: NoiseType = NoiseType.COHERENT
    n_pairs: int = 0           # For coherent noise
    angle: float = 0.1         # For coherent noise
    probability: float = 0.01  # For stochastic noise
    gamma: float = 0.01        # For damping channels
    readout_error: float = 0.0  # Readout bit-flip probability


def configure_noise(
    model: BQNNModel,
    n_pairs: int,
    angle: float = 0.1,
) -> BQNNModel:
    """
    Configure coherent U U† noise on a BQNNModel.
    
    This is the primary noise model representing systematic gate errors
    that should ideally cancel but accumulate on real hardware due to
    calibration imperfections.
    
    Args:
        model: BQNNModel instance
        n_pairs: Number of U U† gate pairs to inject
        angle: Rotation angle for noise gates
        
    Returns:
        Same model with noise configured
    """
    model.set_noise(n_pairs=n_pairs, angle=angle)
    return model


def configure_noise_full(
    model: BQNNModel,
    config: NoiseConfig,
) -> BQNNModel:
    """
    Configure noise with full configuration options.
    
    Note: Only COHERENT noise is currently implemented in the base model.
    Other noise types require device-level simulation or custom circuits.
    
    Args:
        model: BQNNModel instance
        config: NoiseConfig with noise parameters
        
    Returns:
        Configured model
    """
    if config.noise_type == NoiseType.COHERENT:
        model.set_noise(n_pairs=config.n_pairs, angle=config.angle)
    else:
        # Store config for potential future use
        model._noise_config = config
        # Only coherent noise is implemented in forward pass
        model.set_noise(n_pairs=0, angle=0.0)
        print(f"Warning: {config.noise_type} noise requires device-level simulation. "
              "Only coherent noise affects circuit execution.")
    
    return model


def estimate_noise_threshold(
    model_factory,
    train_loader,
    test_loader,
    noise_range: list = [0, 1, 2, 4, 8, 16],
    n_epochs: int = 3,
    baseline_accuracy: Optional[float] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Estimate the noise threshold where BQNN accuracy degrades below baseline.
    
    This implements a noise robustness sweep to find the critical noise level.
    
    Args:
        model_factory: Callable that returns a fresh BQNNModel
        train_loader: Training DataLoader
        test_loader: Test DataLoader
        noise_range: List of noise pair counts to test
        n_epochs: Training epochs per configuration
        baseline_accuracy: Classical baseline accuracy (estimated if None)
        device: Target device
        
    Returns:
        Dictionary with noise levels, accuracies, and estimated threshold
    """
    from .training import train_epoch
    from .inference import evaluate_accuracy
    from .classical_reference import ClassicalBinarizedNet
    
    results = {
        "noise_pairs": [],
        "accuracies": [],
        "threshold_estimate": None,
    }
    
    # Get baseline if not provided
    if baseline_accuracy is None:
        # Infer model dimensions from factory
        sample_model = model_factory()
        baseline = ClassicalBinarizedNet(
            n_features=sample_model.n_features,
            n_hidden=sample_model.n_hidden,
            n_classes=sample_model.n_classes,
        )
        for _ in range(n_epochs):
            train_epoch(baseline, train_loader, device=device)
        baseline_metrics = evaluate_accuracy(baseline, test_loader, device=device)
        baseline_accuracy = baseline_metrics["accuracy"]
    
    results["baseline_accuracy"] = baseline_accuracy
    
    # Sweep noise levels
    threshold_found = False
    for n_pairs in noise_range:
        model = model_factory()
        model.set_noise(n_pairs=n_pairs, angle=0.1)
        
        for _ in range(n_epochs):
            train_epoch(model, train_loader, device=device)
        
        metrics = evaluate_accuracy(model, test_loader, device=device)
        acc = metrics["accuracy"]
        
        results["noise_pairs"].append(n_pairs)
        results["accuracies"].append(acc)
        
        # Check if we've crossed the threshold
        if not threshold_found and acc < baseline_accuracy:
            results["threshold_estimate"] = n_pairs
            threshold_found = True
    
    return results


def create_noisy_device(
    n_qubits: int,
    noise_config: NoiseConfig,
) -> qml.Device:
    """
    Create a PennyLane device with noise model.
    
    Note: This requires pennylane >= 0.30 for mixed-state simulation.
    
    Args:
        n_qubits: Number of qubits
        noise_config: NoiseConfig with noise parameters
        
    Returns:
        PennyLane device with noise
    """
    # Default.mixed supports noise channels
    try:
        dev = qml.device("default.mixed", wires=n_qubits)
    except qml.DeviceError:
        print("Warning: default.mixed not available. Using default.qubit.")
        dev = qml.device("default.qubit", wires=n_qubits)
    
    return dev


def noise_strength_metric(model: BQNNModel) -> float:
    """
    Compute an effective noise strength metric.
    
    Combines noise pairs and angle into a single scalar for comparison.
    Metric: n_pairs * angle^2 * n_qubits
    
    Args:
        model: BQNNModel with noise configured
        
    Returns:
        Effective noise strength scalar
    """
    return model.noise_pairs * (model.noise_angle ** 2) * model.n_qubits
