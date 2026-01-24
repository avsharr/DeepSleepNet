"""
Tests for training stage: loss computation, optimizer, class weights.
"""
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pytest

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models import DeepSleepNet
from preprocessing import compute_class_weights


class TestLossComputation:
    """Test loss computation with class weights."""
    
    def test_crossentropy_loss(self):
        """Test that CrossEntropyLoss works with model output."""
        model = DeepSleepNet(n_classes=5)
        criterion = nn.CrossEntropyLoss()
        
        # Model expects (batch, seq, 1, 3000)
        dummy_input = torch.randn(2, 25, 1, 3000)
        dummy_labels = torch.randint(0, 5, (2 * 25,))
        
        output = model(dummy_input)
        loss = criterion(output, dummy_labels)
        
        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"
    
    def test_weighted_loss(self):
        """Test that weighted CrossEntropyLoss works correctly."""
        model = DeepSleepNet(n_classes=5)
        
        # Create imbalanced labels
        labels = [0] * 100 + [1] * 20 + [2] * 200 + [3] * 50 + [4] * 150
        weights = compute_class_weights(labels, num_classes=5)
        class_weights = torch.tensor(weights, dtype=torch.float32)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Model expects (batch, seq, 1, 3000)
        dummy_input = torch.randn(2, 25, 1, 3000)
        dummy_labels = torch.randint(0, 5, (2 * 25,))
        
        output = model(dummy_input)
        loss = criterion(output, dummy_labels)
        
        assert loss.item() > 0, "Weighted loss should be positive"
        assert not torch.isnan(loss), "Weighted loss should not be NaN"
    
    def test_loss_gradient(self):
        """Test that loss produces gradients."""
        model = DeepSleepNet(n_classes=5)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Model expects (batch, seq, 1, 3000)
        dummy_input = torch.randn(1, 25, 1, 3000)
        dummy_labels = torch.randint(0, 5, (25,))
        
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = criterion(output, dummy_labels)
        loss.backward()
        
        # Check that gradients exist
        has_gradients = any(p.grad is not None for p in model.parameters())
        assert has_gradients, "Model should have gradients after backward pass"


class TestOptimizer:
    """Test optimizer configuration."""
    
    def test_optimizer_step(self):
        """Test that optimizer step updates parameters."""
        model = DeepSleepNet(n_classes=5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]
        
        # Forward and backward pass
        # Model expects (batch, seq, 1, 3000)
        dummy_input = torch.randn(1, 25, 1, 3000)
        dummy_labels = torch.randint(0, 5, (25,))
        
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = criterion(output, dummy_labels)
        loss.backward()
        optimizer.step()
        
        # Check that parameters changed
        params_changed = any(
            not torch.allclose(p1, p2)
            for p1, p2 in zip(initial_params, model.parameters())
        )
        assert params_changed, "Optimizer step should update parameters"
    
    def test_gradient_clipping(self):
        """Test that gradient clipping works."""
        model = DeepSleepNet(n_classes=5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Model expects (batch, seq, 1, 3000)
        dummy_input = torch.randn(1, 25, 1, 3000)
        dummy_labels = torch.randint(0, 5, (25,))
        
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = criterion(output, dummy_labels)
        loss.backward()
        
        # Clip gradients
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        
        # Check that gradients are clipped
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        assert total_norm <= max_norm + 1e-6, f"Gradient norm should be <= {max_norm}, got {total_norm:.4f}"


class TestClassWeights:
    """Test class weight computation for training."""
    
    def test_class_weights_shape(self):
        """Test that class weights have correct shape."""
        labels = [0] * 100 + [1] * 20 + [2] * 200 + [3] * 50 + [4] * 150
        weights = compute_class_weights(labels, num_classes=5)
        
        assert weights.shape == (5,), "Class weights should have shape (5,)"
    
    def test_class_weights_usage(self):
        """Test that class weights can be used in loss."""
        labels = [0] * 100 + [1] * 20 + [2] * 200 + [3] * 50 + [4] * 150
        weights = compute_class_weights(labels, num_classes=5)
        class_weights = torch.tensor(weights, dtype=torch.float32)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        model = DeepSleepNet(n_classes=5)
        
        # Model expects (batch, seq, 1, 3000)
        dummy_input = torch.randn(1, 25, 1, 3000)
        dummy_labels = torch.randint(0, 5, (25,))
        
        output = model(dummy_input)
        loss = criterion(output, dummy_labels)
        
        assert loss.item() > 0, "Loss with class weights should be positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
