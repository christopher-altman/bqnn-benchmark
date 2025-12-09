"""
Unit tests for Straight-Through Estimator (STE) variants.
"""

import pytest
import torch
from bqnn.classical_reference import (
    StraightThroughEstimator,
    ClippedSTE,
    binarize_ste,
)


class TestStraightThroughEstimator:
    """Test STE forward and backward passes."""
    
    def test_forward_sign_function(self):
        """Test that forward pass correctly applies sign function."""
        x = torch.tensor([-2.5, -0.5, 0.0, 0.5, 2.5])
        expected = torch.tensor([-1.0, -1.0, 0.0, 1.0, 1.0])
        
        output = StraightThroughEstimator.apply(x)
        
        assert torch.allclose(output, expected), \
            f"Expected {expected}, got {output}"
    
    def test_backward_identity_gradient(self):
        """Test that backward pass propagates gradients unchanged."""
        x = torch.tensor([-2.0, 0.5, 1.5], requires_grad=True)
        
        # Apply STE
        y = StraightThroughEstimator.apply(x)
        
        # Create a simple loss
        loss = (y ** 2).sum()
        loss.backward()
        
        # Gradient should pass through unchanged
        # dy/dx via STE = 1 (identity), so d(y^2)/dx = 2*y * 1
        # But y = sign(x), so we expect grad = 2*sign(x)
        expected_grad = 2 * torch.sign(x)
        
        assert torch.allclose(x.grad, expected_grad), \
            f"Expected grad {expected_grad}, got {x.grad}"
    
    def test_gradient_flow_through_network(self):
        """Test that STE allows gradient flow in a simple network."""
        x = torch.randn(10, 5, requires_grad=True)
        
        # Simple layer with STE
        weight = torch.randn(5, 3, requires_grad=True)
        h = x @ weight
        h_bin = StraightThroughEstimator.apply(h)
        loss = h_bin.sum()
        
        loss.backward()
        
        # Both x and weight should have gradients
        assert x.grad is not None, "Input should have gradients"
        assert weight.grad is not None, "Weight should have gradients"
        assert torch.isfinite(x.grad).all(), "Input gradients should be finite"
        assert torch.isfinite(weight.grad).all(), "Weight gradients should be finite"


class TestClippedSTE:
    """Test Clipped STE forward and backward passes."""
    
    def test_forward_sign_function(self):
        """Test that forward pass correctly applies sign function."""
        x = torch.tensor([-2.5, -0.5, 0.0, 0.5, 2.5])
        expected = torch.tensor([-1.0, -1.0, 0.0, 1.0, 1.0])
        
        output = ClippedSTE.apply(x)
        
        assert torch.allclose(output, expected), \
            f"Expected {expected}, got {output}"
    
    def test_backward_clipped_gradient(self):
        """Test that backward pass clips gradients outside [-1, 1]."""
        # Values outside [-1, 1] should have zero gradient
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
        
        y = ClippedSTE.apply(x)
        loss = y.sum()
        loss.backward()
        
        # Gradient mask: 1 for |x| <= 1, 0 otherwise
        expected_mask = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0])
        expected_grad = expected_mask  # dloss/dy = 1, so grad = mask * 1
        
        assert torch.allclose(x.grad, expected_grad), \
            f"Expected grad {expected_grad}, got {x.grad}"
    
    def test_gradient_clipping_behavior(self):
        """Test that gradients are clipped for large values."""
        x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], requires_grad=True)
        
        y = ClippedSTE.apply(x)
        # Create loss with non-uniform gradients
        loss = (y * torch.tensor([2.0, 3.0, 1.0, 3.0, 2.0])).sum()
        loss.backward()
        
        # Gradients should be clipped for |x| > 1
        assert x.grad[0] == 0.0, "Gradient should be 0 for x=-3"
        assert x.grad[1] == 3.0, "Gradient should pass through for x=-1"
        assert x.grad[2] == 1.0, "Gradient should pass through for x=0"
        assert x.grad[3] == 3.0, "Gradient should pass through for x=1"
        assert x.grad[4] == 0.0, "Gradient should be 0 for x=3"
    
    def test_saves_input_for_backward(self):
        """Test that ClippedSTE correctly saves input for backward pass."""
        x = torch.tensor([0.5, 1.5], requires_grad=True)
        y = ClippedSTE.apply(x)
        loss = y.sum()
        loss.backward()
        
        # Should have gradient for first element, not second
        assert x.grad[0] == 1.0, "Within [-1,1] should have gradient"
        assert x.grad[1] == 0.0, "Outside [-1,1] should have zero gradient"


class TestBinarizeSTE:
    """Test binarize_ste convenience function."""
    
    def test_default_ste(self):
        """Test that default behavior uses StraightThroughEstimator."""
        x = torch.tensor([-1.5, 0.5], requires_grad=True)
        y = binarize_ste(x, clipped=False)
        
        loss = y.sum()
        loss.backward()
        
        # Should use identity gradient
        expected_grad = torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad), \
            "Default should use StraightThroughEstimator"
    
    def test_clipped_ste(self):
        """Test that clipped=True uses ClippedSTE."""
        x = torch.tensor([-2.0, 0.5], requires_grad=True)
        y = binarize_ste(x, clipped=True)
        
        loss = y.sum()
        loss.backward()
        
        # First element outside [-1,1] should have zero gradient
        assert x.grad[0] == 0.0, "Outside [-1,1] should have zero gradient"
        assert x.grad[1] == 1.0, "Inside [-1,1] should have gradient"
    
    def test_output_is_binary(self):
        """Test that output is binarized to {-1, 0, +1}."""
        x = torch.tensor([-2.5, -0.1, 0.0, 0.1, 2.5])
        y = binarize_ste(x)
        
        # All values should be in {-1, 0, +1}
        assert torch.all((y == -1) | (y == 0) | (y == 1)), \
            "Output should be binary"
    
    def test_gradient_magnitude_preservation(self):
        """Test that STE preserves gradient magnitude."""
        x = torch.randn(100, requires_grad=True)
        y = binarize_ste(x, clipped=False)
        
        # Create a loss with known gradient
        grad_out = torch.randn(100)
        y.backward(grad_out)
        
        # Gradients should match (STE is identity in backward)
        assert torch.allclose(x.grad, grad_out), \
            "STE should preserve gradient magnitude"
    
    def test_comparison_ste_vs_clipped(self):
        """Compare gradient behavior of STE vs ClippedSTE."""
        x_vals = torch.tensor([-2.0, -0.5, 0.5, 2.0], requires_grad=True)
        
        # Standard STE
        x1 = x_vals.clone().detach().requires_grad_(True)
        y1 = binarize_ste(x1, clipped=False)
        y1.sum().backward()
        grad_ste = x1.grad.clone()
        
        # Clipped STE
        x2 = x_vals.clone().detach().requires_grad_(True)
        y2 = binarize_ste(x2, clipped=True)
        y2.sum().backward()
        grad_clipped = x2.grad.clone()
        
        # STE should have uniform gradients
        assert torch.allclose(grad_ste, torch.ones_like(grad_ste)), \
            "Standard STE should have uniform gradients"
        
        # Clipped STE should zero out large values
        expected_clipped = torch.tensor([0.0, 1.0, 1.0, 0.0])
        assert torch.allclose(grad_clipped, expected_clipped), \
            f"Clipped STE should zero out large values, got {grad_clipped}"
