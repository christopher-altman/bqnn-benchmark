"""
Plotting utilities for BQNN experiments.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt


def plot_accuracy_vs_a(
    as_list: List[float],
    acc_list: List[float],
    baseline_acc: Optional[float] = None,
    title: str = "BQNN accuracy vs quantumness",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot accuracy as a function of quantumness parameter a.
    
    Args:
        as_list: List of 'a' parameter values
        acc_list: Corresponding accuracies
        baseline_acc: Optional classical baseline accuracy for reference line
        title: Plot title
        ax: Optional axes to plot on
        
    Returns:
        matplotlib Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
    
    ax.plot(as_list, acc_list, marker="o", linewidth=2, markersize=8, 
            label="BQNN", color="#2E86AB")
    
    if baseline_acc is not None:
        ax.axhline(y=baseline_acc, color="#A23B72", linestyle="--", 
                   linewidth=2, label=f"Classical baseline ({baseline_acc:.3f})")
    
    ax.set_xlabel("Quantumness parameter $a$", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_ylim(0, 1.05)
    
    fig.tight_layout()
    return fig


def plot_heatmap(
    xs: List[Any],
    ys: List[Any],
    values: List[float],
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    cmap: str = "viridis",
    ax: Optional[plt.Axes] = None,
    annotate: bool = True,
) -> plt.Figure:
    """
    Plot a 2D heatmap of values over parameter grid.
    
    Args:
        xs: Row parameter values (y-axis in plot)
        ys: Column parameter values (x-axis in plot)  
        values: Flattened values in row-major order [len(xs) * len(ys)]
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        cmap: Colormap name
        ax: Optional axes to plot on
        annotate: Whether to show values in cells
        
    Returns:
        matplotlib Figure object
    """
    xs = list(xs)
    ys = list(ys)
    values_arr = np.array(values).reshape(len(xs), len(ys))
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    im = ax.imshow(values_arr, origin="lower", aspect="auto", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, label="Accuracy")
    
    # Set tick labels - FIXED: xs on y-axis, ys on x-axis
    ax.set_xticks(range(len(ys)))
    ax.set_xticklabels([f"{y}" for y in ys])
    ax.set_yticks(range(len(xs)))
    ax.set_yticklabels([f"{x}" for x in xs])
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Annotate cells with values
    if annotate:
        for i in range(len(xs)):
            for j in range(len(ys)):
                val = values_arr[i, j]
                # Choose text color based on background
                text_color = "white" if val < 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                       color=text_color, fontsize=9)
    
    fig.tight_layout()
    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training Progress",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot training loss and optional accuracy curves.
    
    Args:
        history: Dictionary with 'loss' and optionally 'accuracy' lists
        title: Plot title
        ax: Optional axes
        
    Returns:
        matplotlib Figure object
    """
    if ax is None:
        n_plots = 1 + ("accuracy" in history or "val_accuracy" in history)
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]
    else:
        fig = ax.figure
        axes = [ax]
    
    # Plot loss
    epochs = range(1, len(history["loss"]) + 1)
    axes[0].plot(epochs, history["loss"], marker="o", label="Train Loss", color="#2E86AB")
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], marker="s", label="Val Loss", color="#A23B72")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy if present
    if len(axes) > 1 and ("accuracy" in history or "val_accuracy" in history):
        if "accuracy" in history:
            axes[1].plot(epochs, history["accuracy"], marker="o", 
                        label="Train Acc", color="#2E86AB")
        if "val_accuracy" in history:
            axes[1].plot(epochs, history["val_accuracy"], marker="s",
                        label="Val Acc", color="#A23B72")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1.05)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_gradient_norms(
    grad_history: List[Dict[str, float]],
    title: str = "Gradient Statistics (Barren Plateau Monitor)",
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot gradient norm statistics over training for barren plateau detection.
    
    Args:
        grad_history: List of gradient stat dicts from compute_gradient_stats
        title: Plot title
        ax: Optional axes
        
    Returns:
        matplotlib Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure
    
    epochs = range(1, len(grad_history) + 1)
    means = [g["grad_mean"] for g in grad_history]
    stds = [g["grad_std"] for g in grad_history]
    maxs = [g["grad_max"] for g in grad_history]
    
    ax.semilogy(epochs, means, label="Mean", marker="o", color="#2E86AB")
    ax.semilogy(epochs, maxs, label="Max", marker="^", color="#F18F01")
    ax.fill_between(epochs, 
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.3, color="#2E86AB")
    
    # Barren plateau threshold line
    ax.axhline(y=1e-6, color="red", linestyle="--", label="Barren plateau threshold")
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Norm (log scale)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


def plot_quantum_classical_comparison(
    a_values: List[float],
    quantum_accs: List[float],
    classical_acc: float,
    noise_levels: Optional[List[float]] = None,
    title: str = "Quantum vs Classical Performance",
) -> plt.Figure:
    """
    Create comparison plot of quantum advantage region.
    
    Args:
        a_values: Quantumness parameter values
        quantum_accs: BQNN accuracies
        classical_acc: Classical baseline accuracy
        noise_levels: Optional noise levels for each point
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Fill advantage/disadvantage regions
    advantage_mask = np.array(quantum_accs) > classical_acc
    
    ax.fill_between(a_values, classical_acc, quantum_accs, 
                    where=advantage_mask, alpha=0.3, color="green",
                    label="Quantum advantage region")
    ax.fill_between(a_values, classical_acc, quantum_accs,
                    where=~advantage_mask, alpha=0.3, color="red",
                    label="Classical advantage region")
    
    ax.plot(a_values, quantum_accs, "o-", color="#2E86AB", 
            linewidth=2, markersize=8, label="BQNN")
    ax.axhline(y=classical_acc, color="#A23B72", linestyle="--",
               linewidth=2, label=f"Classical ({classical_acc:.3f})")
    
    ax.set_xlabel("Quantumness parameter $a$", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    fig.tight_layout()
    return fig



def aggregate_metrics(
    results: List[Dict[str, Any]], param_key: str, metric_key: str
) -> Dict[Any, Dict[str, float]]:
    """Aggregate a metric grouped by a parameter key.

    Returns mapping: param_value -> {mean, std, count}.
    """
    buckets: Dict[Any, List[float]] = {}
    for row in results:
        if param_key not in row or metric_key not in row:
            continue
        buckets.setdefault(row[param_key], []).append(float(row[metric_key]))

    summary: Dict[Any, Dict[str, float]] = {}
    for k, vals in buckets.items():
        arr: npt.NDArray[np.float_] = np.asarray(vals, dtype=float)
        summary[k] = {"mean": float(arr.mean()), "std": float(arr.std()), "count": int(arr.size)}
    return summary


def _prepare_series(summary: Dict[Any, Dict[str, float]]):
    items = sorted(summary.items(), key=lambda kv: kv[0])
    labels = [str(k) for k, _ in items]
    means = [v["mean"] for _, v in items]
    stds = [v.get("std", 0.0) for _, v in items]
    x = np.arange(len(items))
    return x, labels, means, stds


def plot_metric_curve(
    summary: Dict[Any, Dict[str, float]],
    param_label: str,
    metric_label: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Line plot with error bars for aggregated sweep metrics."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    if not summary:
        ax.set_title(f"{metric_label} vs {param_label} (no data)")
        return fig

    x, labels, means, stds = _prepare_series(summary)
    ax.errorbar(x, means, yerr=stds, fmt="o-", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(param_label)
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} vs {param_label}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_metric_bar(
    summary: Dict[Any, Dict[str, float]],
    param_label: str,
    metric_label: str,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Bar chart with error bars for aggregated sweep metrics."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    if not summary:
        ax.set_title(f"{metric_label} vs {param_label} (no data)")
        return fig

    x, labels, means, stds = _prepare_series(summary)
    ax.bar(x, means, yerr=stds, capsize=4, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(param_label)
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} vs {param_label}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig



def save_figure(
    fig: plt.Figure,
    path: str,
    dpi: int = 150,
    formats: List[str] = ["png", "pdf"],
) -> List[str]:
    """
    Save figure in multiple formats.
    
    Args:
        fig: matplotlib Figure
        path: Base path (without extension)
        dpi: Resolution for raster formats
        formats: List of format extensions
        
    Returns:
        List of saved file paths
    """
    saved = []
    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        fpath = base.with_suffix(f".{fmt}")
        fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
        saved.append(str(fpath))
    
    return saved
