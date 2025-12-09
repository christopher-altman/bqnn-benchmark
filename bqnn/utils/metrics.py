"""
Metrics utilities for BQNN evaluation.
"""

from typing import Dict, List, Tuple
import torch
import numpy as np


def accuracy_from_counts(correct: int, total: int) -> float:
    """
    Compute accuracy from correct/total counts.
    
    Args:
        correct: Number of correct predictions
        total: Total number of samples
        
    Returns:
        Accuracy as float in [0, 1]
    """
    if total <= 0:
        return 0.0
    return correct / total


def compute_f1_score(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    n_classes: int = 2,
) -> Dict[str, float]:
    """
    Compute macro and per-class F1 scores.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        n_classes: Number of classes
        
    Returns:
        Dictionary with 'macro_f1' and per-class F1 scores
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    f1_scores = []
    result = {}
    
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
            
        f1_scores.append(f1)
        result[f"f1_class_{c}"] = f1
    
    result["macro_f1"] = np.mean(f1_scores)
    return result


def compute_confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    n_classes: int = 2,
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels  
        n_classes: Number of classes
        
    Returns:
        Confusion matrix as numpy array [n_classes, n_classes]
        where entry [i,j] = count of true=i, pred=j
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    
    return cm


def compute_gradient_stats(model: torch.nn.Module) -> Dict[str, float]:
    """
    Compute gradient statistics for barren plateau detection.
    
    Args:
        model: PyTorch model with gradients computed
        
    Returns:
        Dictionary with gradient norms and variance metrics
    """
    grad_norms = []
    param_names = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            param_names.append(name)
    
    if not grad_norms:
        return {"grad_mean": 0.0, "grad_std": 0.0, "grad_max": 0.0}
    
    return {
        "grad_mean": np.mean(grad_norms),
        "grad_std": np.std(grad_norms),
        "grad_max": np.max(grad_norms),
        "grad_min": np.min(grad_norms),
    }


def expressibility_metric(
    samples: torch.Tensor,
    n_bins: int = 50,
) -> float:
    """
    Estimate circuit expressibility via fidelity distribution entropy.
    
    Higher values indicate more expressible circuits (closer to Haar random).
    
    Args:
        samples: Output state samples [n_samples, n_qubits]
        n_bins: Number of histogram bins
        
    Returns:
        Expressibility estimate (entropy-based)
    """
    # Compute pairwise fidelities (simplified for expectation values)
    samples = samples.cpu().numpy()
    n = samples.shape[0]
    
    fidelities = []
    for i in range(min(n, 100)):
        for j in range(i + 1, min(n, 100)):
            # Approximate fidelity from expectation value overlap
            fid = np.abs(np.dot(samples[i], samples[j])) / (
                np.linalg.norm(samples[i]) * np.linalg.norm(samples[j]) + 1e-8
            )
            fidelities.append(fid)
    
    if not fidelities:
        return 0.0
    
    # Compute histogram entropy
    hist, _ = np.histogram(fidelities, bins=n_bins, range=(0, 1), density=True)
    hist = hist + 1e-10  # avoid log(0)
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log(hist))
    
    return entropy
