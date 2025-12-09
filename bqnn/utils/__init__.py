"""
BQNN utilities subpackage.

Provides metrics computation and visualization tools.
"""

from .metrics import accuracy_from_counts, compute_f1_score, compute_confusion_matrix
from .plots import (
    plot_accuracy_vs_a,
    plot_heatmap,
    plot_training_curves,
    plot_gradient_norms,
    save_figure,
)

__all__ = [
    "accuracy_from_counts",
    "compute_f1_score",
    "compute_confusion_matrix",
    "plot_accuracy_vs_a",
    "plot_heatmap",
    "plot_training_curves",
    "plot_gradient_norms",
    "save_figure",
]
