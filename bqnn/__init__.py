"""
BQNN Benchmark - Tunable Quantum Neural Network Framework

This package provides a benchmarking harness for binarized quantum neural
networks (BQNNs) that interpolate between classical and quantum regimes
via a single quantumness parameter `a`.

Key Components:
    - BQNNModel: Main quantum-classical hybrid model
    - ClassicalBinarizedNet: Classical baseline with sign activation
    - Training utilities with proper optimizer persistence
    - Noise injection for hardware simulation
    - Comprehensive metrics and visualization

Example:
    >>> from bqnn import BQNNModel, get_synthetic_dataset, train_epoch
    >>> loader = get_synthetic_dataset(n_samples=1024)
    >>> model = BQNNModel(n_features=16, n_hidden=8, n_classes=2, a=0.5)
    >>> stats = train_epoch(model, loader)
"""

__version__ = "0.2.1"

from .data import (
    get_synthetic_dataset,
    get_mnist_small,
    get_train_test_loaders,
)
from .classical_reference import ClassicalBinarizedNet
from .model import BQNNModel, DeepBQNNModel
from .quantization import binarize_features, features_to_angles
from .training import (
    train_epoch,
    train_model,
    Trainer,
    TrainingConfig,
)
from .inference import (
    evaluate_accuracy,
    evaluate_full,
    get_predictions,
    get_quantum_features,
)
from .noise_models import (
    configure_noise,
    configure_noise_full,
    estimate_noise_threshold,
    NoiseConfig,
    NoiseType,
)
from .benchmark import BenchmarkRun, set_global_seed, load_run
from .sweeps import SweepConfig, SweepSpace, run_sweep

__all__ = [
    # Version
    "__version__",
    # Data
    "get_synthetic_dataset",
    "get_mnist_small",
    "get_train_test_loaders",
    # Models
    "ClassicalBinarizedNet",
    "BQNNModel",
    "DeepBQNNModel",
    # Quantization
    "binarize_features",
    "features_to_angles",
    # Training
    "train_epoch",
    "train_model",
    "Trainer",
    "TrainingConfig",
    # Inference
    "evaluate_accuracy",
    "evaluate_full",
    "get_predictions",
    "get_quantum_features",
    # Noise
    "configure_noise",
    "configure_noise_full",
    "estimate_noise_threshold",
    "NoiseConfig",
    "NoiseType",
    # Benchmarking
    "BenchmarkRun",
    "set_global_seed",
    "load_run",
    # Sweeps
    "SweepConfig",
    "SweepSpace",
    "run_sweep",
]
