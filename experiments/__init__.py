"""
BQNN Experiment Scripts

This package contains runnable experiments for benchmarking BQNNs:
    - exp_sweep_a: Sweep quantumness parameter and measure accuracy
    - train_mnist_bqnn: Compare classical vs BQNN on tiny MNIST
    - exp_noise_threshold: Study noise robustness
"""

from .exp_sweep_a import main as sweep_a_main
from .train_mnist_bqnn import main as mnist_main  
from .exp_noise_threshold import main as noise_main

__all__ = [
    "sweep_a_main",
    "mnist_main",
    "noise_main",
]
