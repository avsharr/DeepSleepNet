"""
Tests for preprocessing stage: EDF to NPZ conversion, class weights computation.
"""
import sys
import os
import numpy as np
import pytest

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from preprocessing import compute_class_weights, CLASS_NAMES, NUM_CLASSES


class TestClassWeights:
    """Test class weight computation."""
    
    def test_compute_class_weights_balanced(self):
        """Test that balanced classes produce equal weights."""
        # Create balanced labels (equal counts for each class)
        labels = [0] * 100 + [1] * 100 + [2] * 100 + [3] * 100 + [4] * 100
        weights = compute_class_weights(labels, num_classes=5)
        
        # All weights should be equal (1.0)
        assert np.allclose(weights, 1.0, atol=0.01), "Balanced classes should have equal weights"
    
    def test_compute_class_weights_imbalanced(self):
        """Test that imbalanced classes produce appropriate weights."""
        # Create imbalanced labels (N1 is rare)
        labels = [0] * 500 + [1] * 50 + [2] * 500 + [3] * 200 + [4] * 300
        weights = compute_class_weights(labels, num_classes=5)
        
        # N1 (index 1) should have highest weight (rarest)
        assert weights[1] > weights[0], "Rare class (N1) should have higher weight"
        assert weights[1] > weights[2], "Rare class (N1) should have higher weight"
        
        # Most common classes should have lower weights
        assert weights[0] < weights[1], "Common class should have lower weight"
        assert weights[2] < weights[1], "Common class should have lower weight"
    
    def test_compute_class_weights_shape(self):
        """Test that weights have correct shape."""
        labels = [0, 1, 2, 3, 4] * 10
        weights = compute_class_weights(labels, num_classes=5)
        
        assert weights.shape == (5,), "Weights should have shape (5,)"
        assert len(weights) == NUM_CLASSES, "Weights length should match NUM_CLASSES"
    
    def test_compute_class_weights_all_positive(self):
        """Test that all weights are positive."""
        labels = [0] * 100 + [1] * 20 + [2] * 200 + [3] * 50 + [4] * 150
        weights = compute_class_weights(labels, num_classes=5)
        
        assert np.all(weights > 0), "All weights should be positive"
    
    def test_compute_class_weights_empty_class(self):
        """Test handling of empty class (should default to 1.0)."""
        # Class 3 is missing
        labels = [0] * 100 + [1] * 50 + [2] * 200 + [4] * 150
        weights = compute_class_weights(labels, num_classes=5)
        
        # Empty class should have weight 1.0
        assert weights[3] == 1.0, "Empty class should have default weight 1.0"
        assert weights[3] > 0, "Empty class weight should be positive"


class TestConstants:
    """Test preprocessing constants."""
    
    def test_class_names(self):
        """Test that CLASS_NAMES matches expected sleep stages."""
        expected = ['Wake', 'N1', 'N2', 'N3', 'REM']
        assert CLASS_NAMES == expected, "CLASS_NAMES should match AASM standard"
    
    def test_num_classes(self):
        """Test that NUM_CLASSES is correct."""
        assert NUM_CLASSES == 5, "Should have 5 sleep stage classes"
        assert len(CLASS_NAMES) == NUM_CLASSES, "CLASS_NAMES length should match NUM_CLASSES"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
