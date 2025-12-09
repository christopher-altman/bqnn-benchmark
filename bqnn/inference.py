"""
Inference utilities for BQNN models.

Provides evaluation functions with comprehensive metrics.
"""

from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils.metrics import compute_f1_score, compute_confusion_matrix


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate model accuracy on a dataset.
    
    Args:
        model: Trained model
        loader: DataLoader for evaluation data
        device: Target device
        
    Returns:
        Dictionary with accuracy metric
    """
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        preds = torch.argmax(logits, dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    acc = correct / max(1, total)
    return {"accuracy": acc}


@torch.no_grad()
def evaluate_full(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    n_classes: int = 2,
) -> Dict[str, float]:
    """
    Comprehensive evaluation with multiple metrics.
    
    Args:
        model: Trained model
        loader: DataLoader for evaluation data
        device: Target device
        n_classes: Number of output classes
        
    Returns:
        Dictionary with accuracy, F1 scores, and confusion matrix
    """
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_logits = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())
        all_logits.append(logits.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_logits = torch.cat(all_logits)

    # Accuracy
    correct = (all_preds == all_labels).sum().item()
    total = all_labels.size(0)
    accuracy = correct / max(1, total)

    # F1 scores
    f1_metrics = compute_f1_score(all_labels, all_preds, n_classes)

    # Confusion matrix (as nested list for JSON serialization)
    cm = compute_confusion_matrix(all_labels, all_preds, n_classes)

    # Confidence statistics
    probs = torch.softmax(all_logits, dim=1)
    max_probs = probs.max(dim=1).values
    
    results = {
        "accuracy": accuracy,
        "macro_f1": f1_metrics["macro_f1"],
        "confusion_matrix": cm.tolist(),
        "mean_confidence": max_probs.mean().item(),
        "confidence_std": max_probs.std().item(),
        "n_samples": total,
    }
    
    # Add per-class F1
    for k, v in f1_metrics.items():
        if k.startswith("f1_class"):
            results[k] = v
    
    return results


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get model predictions and probabilities.
    
    Args:
        model: Trained model
        loader: DataLoader
        device: Target device
        
    Returns:
        Tuple of (predictions, probabilities, true_labels)
    """
    model.to(device)
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())
        all_labels.append(y.cpu())

    return (
        torch.cat(all_preds),
        torch.cat(all_probs),
        torch.cat(all_labels),
    )


@torch.no_grad()
def get_quantum_features(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract quantum layer outputs (expectation values) for analysis.
    
    Useful for studying the quantum feature space and expressibility.
    
    Args:
        model: BQNNModel instance
        loader: DataLoader
        device: Target device
        
    Returns:
        Tuple of (quantum_features [N, n_qubits], labels [N])
    """
    from .model import BQNNModel
    
    if not isinstance(model, BQNNModel):
        raise TypeError("Model must be a BQNNModel instance")
    
    model.to(device)
    model.eval()

    all_features = []
    all_labels = []

    for x, y in loader:
        x = x.to(device)
        
        # Run through classical preprocessing
        h = model.fc1(x)
        h_bin = torch.sign(h)
        h_bin = torch.where(h_bin == 0, torch.ones_like(h_bin), h_bin)
        h01 = ((h_bin + 1.0) / 2.0).clamp(0.0, 1.0)
        
        from .quantization import binarize_features, features_to_angles
        h_bin01 = binarize_features(h01)
        h_bin01 = h_bin01[:, :model.n_qubits]
        angle_vec = features_to_angles(h_bin01, a=model.a)
        
        # Get quantum features
        expvals = model.quantum_forward(angle_vec)
        
        all_features.append(expvals.cpu())
        all_labels.append(y)

    return torch.cat(all_features), torch.cat(all_labels)
