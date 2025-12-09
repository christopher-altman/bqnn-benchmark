"""
Unit tests for evaluation metrics and inference utilities.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from bqnn.inference import evaluate_accuracy
from bqnn.utils.metrics import (
    compute_f1_score,
    compute_confusion_matrix,
    accuracy_from_counts,
)


class DummyModel(nn.Module):
    """Dummy model that returns fixed predictions for testing."""
    def __init__(self, n_features=8, n_classes=2):
        super().__init__()
        self.fc = nn.Linear(n_features, n_classes)
    
    def forward(self, x):
        # Return predictable logits
        return self.fc(x)


def create_test_dataloader(x, y, batch_size=8):
    """Create a dataloader from tensors."""
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class TestEvaluateAccuracy:
    """Test evaluate_accuracy function."""
    
    def test_perfect_accuracy(self):
        """Test accuracy computation with perfect predictions."""
        # Create model that always predicts class 0
        model = DummyModel(n_features=4, n_classes=2)
        
        # Manually set weights to produce predictable output
        with torch.no_grad():
            model.fc.weight.fill_(0)
            model.fc.bias[0] = 1.0  # Class 0 logit = 1
            model.fc.bias[1] = -1.0  # Class 1 logit = -1
        
        # Create data where all labels are 0
        x = torch.randn(32, 4)
        y = torch.zeros(32, dtype=torch.long)
        loader = create_test_dataloader(x, y, batch_size=8)
        
        # Evaluate
        metrics = evaluate_accuracy(model, loader, device="cpu")
        
        # Should have perfect accuracy
        assert "accuracy" in metrics
        assert metrics["accuracy"] == 1.0
    
    def test_zero_accuracy(self):
        """Test accuracy computation with no correct predictions."""
        model = DummyModel(n_features=4, n_classes=2)
        
        # Set weights to always predict class 0
        with torch.no_grad():
            model.fc.weight.fill_(0)
            model.fc.bias[0] = 1.0
            model.fc.bias[1] = -1.0
        
        # Create data where all labels are 1
        x = torch.randn(32, 4)
        y = torch.ones(32, dtype=torch.long)
        loader = create_test_dataloader(x, y, batch_size=8)
        
        # Evaluate
        metrics = evaluate_accuracy(model, loader, device="cpu")
        
        # Should have zero accuracy
        assert metrics["accuracy"] == 0.0
    
    def test_partial_accuracy(self):
        """Test accuracy computation with partial correctness."""
        model = DummyModel(n_features=4, n_classes=2)
        
        # Set model to predict based on first feature
        with torch.no_grad():
            model.fc.weight.fill_(0)
            model.fc.weight[0, 0] = 1.0  # Class 0 depends on x[0]
            model.fc.weight[1, 0] = -1.0  # Class 1 opposite
            model.fc.bias.fill_(0)
        
        # Create data with known pattern
        x = torch.tensor([[1.0, 0, 0, 0], [-1.0, 0, 0, 0]] * 16)  # 32 samples
        y = torch.tensor([0, 1] * 16, dtype=torch.long)
        loader = create_test_dataloader(x, y, batch_size=8)
        
        # Evaluate
        metrics = evaluate_accuracy(model, loader, device="cpu")
        
        # Should have high accuracy
        assert metrics["accuracy"] > 0.9
    
    def test_multiclass_accuracy(self):
        """Test accuracy with more than 2 classes."""
        model = DummyModel(n_features=4, n_classes=3)
        
        # Create simple data
        x = torch.randn(30, 4)
        y = torch.randint(0, 3, (30,))
        loader = create_test_dataloader(x, y, batch_size=10)
        
        # Evaluate
        metrics = evaluate_accuracy(model, loader, device="cpu")
        
        # Accuracy should be between 0 and 1
        assert 0.0 <= metrics["accuracy"] <= 1.0
    
    def test_single_batch(self):
        """Test accuracy on a single batch."""
        model = DummyModel(n_features=4, n_classes=2)
        
        x = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))
        loader = create_test_dataloader(x, y, batch_size=8)
        
        metrics = evaluate_accuracy(model, loader, device="cpu")
        
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
    
    def test_empty_loader_edge_case(self):
        """Test behavior with empty dataloader."""
        model = DummyModel(n_features=4, n_classes=2)
        
        # Create empty loader
        x = torch.empty(0, 4)
        y = torch.empty(0, dtype=torch.long)
        loader = create_test_dataloader(x, y, batch_size=8)
        
        metrics = evaluate_accuracy(model, loader, device="cpu")
        
        # Should handle gracefully (may be 0 or NaN depending on implementation)
        assert "accuracy" in metrics


class TestComputeF1Score:
    """Test F1 score computation."""
    
    def test_perfect_binary_f1(self):
        """Test F1 score with perfect predictions."""
        y_true = torch.tensor([0, 0, 1, 1, 0, 1])
        y_pred = torch.tensor([0, 0, 1, 1, 0, 1])
        
        metrics = compute_f1_score(y_true, y_pred, n_classes=2)
        
        assert "macro_f1" in metrics
        assert metrics["macro_f1"] == 1.0
        assert "f1_class_0" in metrics
        assert "f1_class_1" in metrics
        assert metrics["f1_class_0"] == 1.0
        assert metrics["f1_class_1"] == 1.0
    
    def test_zero_f1(self):
        """Test F1 score with completely wrong predictions."""
        y_true = torch.tensor([0, 0, 0, 1, 1, 1])
        y_pred = torch.tensor([1, 1, 1, 0, 0, 0])
        
        metrics = compute_f1_score(y_true, y_pred, n_classes=2)
        
        # F1 should be 0 for both classes
        assert metrics["macro_f1"] == 0.0
        assert metrics["f1_class_0"] == 0.0
        assert metrics["f1_class_1"] == 0.0
    
    def test_partial_f1(self):
        """Test F1 score with partial correctness."""
        y_true = torch.tensor([0, 0, 1, 1])
        y_pred = torch.tensor([0, 1, 1, 1])  # 3/4 correct
        
        metrics = compute_f1_score(y_true, y_pred, n_classes=2)
        
        assert 0.0 < metrics["macro_f1"] < 1.0
        assert "f1_class_0" in metrics
        assert "f1_class_1" in metrics
    
    def test_multiclass_f1(self):
        """Test F1 score with 3 classes."""
        y_true = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
        
        metrics = compute_f1_score(y_true, y_pred, n_classes=3)
        
        assert metrics["macro_f1"] == 1.0
        assert "f1_class_0" in metrics
        assert "f1_class_1" in metrics
        assert "f1_class_2" in metrics
    
    def test_imbalanced_classes(self):
        """Test F1 score with imbalanced class distribution."""
        # Many class 0, few class 1
        y_true = torch.tensor([0, 0, 0, 0, 0, 1, 1])
        y_pred = torch.tensor([0, 0, 0, 0, 0, 1, 1])
        
        metrics = compute_f1_score(y_true, y_pred, n_classes=2)
        
        # Should still have perfect F1 despite imbalance
        assert metrics["macro_f1"] == 1.0
    
    def test_f1_with_missing_class(self):
        """Test F1 when a class is never predicted."""
        y_true = torch.tensor([0, 0, 1, 1])
        y_pred = torch.tensor([0, 0, 0, 0])  # Never predict class 1
        
        metrics = compute_f1_score(y_true, y_pred, n_classes=2)
        
        # Class 0 should have some F1, class 1 should have 0
        assert metrics["f1_class_0"] > 0
        assert metrics["f1_class_1"] == 0.0


class TestComputeConfusionMatrix:
    """Test confusion matrix computation."""
    
    def test_perfect_binary_confusion(self):
        """Test confusion matrix with perfect predictions."""
        y_true = torch.tensor([0, 0, 1, 1])
        y_pred = torch.tensor([0, 0, 1, 1])
        
        cm = compute_confusion_matrix(y_true, y_pred, n_classes=2)
        
        # Should be [[2, 0], [0, 2]]
        assert cm.shape == (2, 2)
        assert cm[0, 0] == 2  # True 0, pred 0
        assert cm[0, 1] == 0  # True 0, pred 1
        assert cm[1, 0] == 0  # True 1, pred 0
        assert cm[1, 1] == 2  # True 1, pred 1
    
    def test_confused_predictions(self):
        """Test confusion matrix with mixed predictions."""
        y_true = torch.tensor([0, 0, 1, 1])
        y_pred = torch.tensor([1, 0, 0, 1])
        
        cm = compute_confusion_matrix(y_true, y_pred, n_classes=2)
        
        # Should be [[1, 1], [1, 1]]
        assert cm[0, 0] == 1  # True 0, pred 0
        assert cm[0, 1] == 1  # True 0, pred 1
        assert cm[1, 0] == 1  # True 1, pred 0
        assert cm[1, 1] == 1  # True 1, pred 1
    
    def test_multiclass_confusion(self):
        """Test confusion matrix with 3 classes."""
        y_true = torch.tensor([0, 1, 2, 0, 1, 2])
        y_pred = torch.tensor([0, 1, 2, 0, 1, 2])
        
        cm = compute_confusion_matrix(y_true, y_pred, n_classes=3)
        
        assert cm.shape == (3, 3)
        # Diagonal should all be 2
        assert cm[0, 0] == 2
        assert cm[1, 1] == 2
        assert cm[2, 2] == 2
        # Off-diagonal should be 0
        assert cm.sum() - cm.trace() == 0
    
    def test_confusion_matrix_sum(self):
        """Test that confusion matrix sums to total samples."""
        y_true = torch.tensor([0, 1, 0, 1, 0, 1, 1, 0])
        y_pred = torch.tensor([0, 1, 1, 0, 0, 1, 0, 1])
        
        cm = compute_confusion_matrix(y_true, y_pred, n_classes=2)
        
        assert cm.sum() == 8  # Total number of samples
    
    def test_confusion_matrix_dtype(self):
        """Test that confusion matrix has correct dtype."""
        y_true = torch.tensor([0, 1, 0, 1])
        y_pred = torch.tensor([0, 1, 1, 0])
        
        cm = compute_confusion_matrix(y_true, y_pred, n_classes=2)
        
        # Should be integer type
        import numpy as np
        assert cm.dtype == np.int64


class TestAccuracyFromCounts:
    """Test accuracy_from_counts utility function."""
    
    def test_perfect_accuracy_counts(self):
        """Test accuracy with all correct predictions."""
        acc = accuracy_from_counts(correct=100, total=100)
        assert acc == 1.0
    
    def test_zero_accuracy_counts(self):
        """Test accuracy with no correct predictions."""
        acc = accuracy_from_counts(correct=0, total=100)
        assert acc == 0.0
    
    def test_partial_accuracy_counts(self):
        """Test accuracy with partial correctness."""
        acc = accuracy_from_counts(correct=75, total=100)
        assert acc == 0.75
    
    def test_zero_total_edge_case(self):
        """Test behavior when total is zero."""
        acc = accuracy_from_counts(correct=0, total=0)
        assert acc == 0.0  # Should handle gracefully
    
    def test_negative_total_edge_case(self):
        """Test behavior with negative total."""
        acc = accuracy_from_counts(correct=10, total=-5)
        # Should return 0 for invalid input
        assert acc == 0.0


class TestClassificationMetricsIntegration:
    """Integration tests for classification metrics."""
    
    def test_metrics_consistency(self):
        """Test that accuracy, F1, and confusion matrix are consistent."""
        y_true = torch.tensor([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = torch.tensor([0, 0, 1, 1, 0, 1, 1, 0])
        
        # Compute all metrics
        cm = compute_confusion_matrix(y_true, y_pred, n_classes=2)
        f1_metrics = compute_f1_score(y_true, y_pred, n_classes=2)
        
        # Accuracy from confusion matrix
        acc_from_cm = cm.diagonal().sum() / cm.sum()
        
        # Accuracy from counts
        correct = (y_true == y_pred).sum().item()
        total = len(y_true)
        acc_from_counts_val = accuracy_from_counts(correct, total)
        
        # All should agree
        assert abs(acc_from_cm - 1.0) < 1e-6
        assert acc_from_counts_val == 1.0
        assert f1_metrics["macro_f1"] == 1.0
    
    def test_metrics_with_errors(self):
        """Test metrics consistency with prediction errors."""
        y_true = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = torch.tensor([0, 0, 1, 1, 1, 1, 0, 0])  # 50% accuracy
        
        cm = compute_confusion_matrix(y_true, y_pred, n_classes=2)
        f1_metrics = compute_f1_score(y_true, y_pred, n_classes=2)
        
        # Accuracy should be 0.5
        acc_from_cm = cm.diagonal().sum() / cm.sum()
        assert abs(acc_from_cm - 0.5) < 1e-6
        
        # F1 should also reflect this
        assert 0.0 <= f1_metrics["macro_f1"] <= 1.0
