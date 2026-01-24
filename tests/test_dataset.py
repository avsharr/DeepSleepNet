"""
Tests for dataset loading: SequentialSleepDataset, normalization, data splits.
"""
import sys
import os
import numpy as np
import torch
import pytest

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from datasets import SequentialSleepDataset


class TestDatasetStructure:
    """Test dataset structure and basic properties."""
    
    def test_dataset_initialization(self):
        """Test that dataset can be initialized."""
        data_path = os.path.join(ROOT, 'data', 'preprocessed')
        
        # Skip if no preprocessed data exists
        if not os.path.exists(data_path) or not os.listdir(data_path):
            pytest.skip("No preprocessed data available")
        
        dataset = SequentialSleepDataset(data_path, mode='train', normalize=False)
        assert len(dataset) > 0, "Dataset should have at least one sequence"
    
    def test_dataset_modes(self):
        """Test that train/val/test modes work correctly."""
        data_path = os.path.join(ROOT, 'data', 'preprocessed')
        
        if not os.path.exists(data_path) or not os.listdir(data_path):
            pytest.skip("No preprocessed data available")
        
        train_ds = SequentialSleepDataset(data_path, mode='train', normalize=False)
        val_ds = SequentialSleepDataset(data_path, mode='val', normalize=False)
        test_ds = SequentialSleepDataset(data_path, mode='test', normalize=False)
        
        # All modes should have data
        assert len(train_ds) > 0, "Train dataset should have data"
        assert len(val_ds) >= 0, "Val dataset should exist (may be empty)"
        assert len(test_ds) >= 0, "Test dataset should exist (may be empty)"
    
    def test_dataset_item_shape(self):
        """Test that dataset items have correct shape."""
        data_path = os.path.join(ROOT, 'data', 'preprocessed')
        
        if not os.path.exists(data_path) or not os.listdir(data_path):
            pytest.skip("No preprocessed data available")
        
        dataset = SequentialSleepDataset(data_path, mode='train', normalize=False)
        signals, labels = dataset[0]
        
        # Signals: (seq_length, n_channels, signal_length) = (25, 1, 3000)
        assert signals.shape[0] == 25, "Sequence length should be 25"
        assert len(signals.shape) == 3, "Signals should have 3 dimensions (seq, channels, time)"
        assert signals.shape[1] == 1, "Number of channels should be 1"
        assert signals.shape[2] == 3000, "Signal length should be 3000 (30s * 100Hz)"
        
        # Labels: (seq_length,)
        assert labels.shape == (25,), "Labels should have shape (seq_length,)"


class TestNormalization:
    """Test data normalization."""
    
    def test_normalization_stats(self):
        """Test that normalization computes correct statistics."""
        data_path = os.path.join(ROOT, 'data', 'preprocessed')
        
        if not os.path.exists(data_path) or not os.listdir(data_path):
            pytest.skip("No preprocessed data available")
        
        dataset = SequentialSleepDataset(data_path, mode='train', normalize=True)
        
        # Check that normalization stats exist
        assert hasattr(dataset, 'global_mean'), "Dataset should have global_mean"
        assert hasattr(dataset, 'global_std'), "Dataset should have global_std"
        assert dataset.global_std > 0, "Standard deviation should be positive"
    
    def test_normalized_data_range(self):
        """Test that normalized data has reasonable range."""
        data_path = os.path.join(ROOT, 'data', 'preprocessed')
        
        if not os.path.exists(data_path) or not os.listdir(data_path):
            pytest.skip("No preprocessed data available")
        
        dataset = SequentialSleepDataset(data_path, mode='train', normalize=True)
        signals, _ = dataset[0]
        
        # Normalized data should be roughly centered around 0
        mean = signals.numpy().mean()
        std = signals.numpy().std()
        
        # Mean should be close to 0 (within 0.2 for more tolerance)
        assert abs(mean) < 0.2, f"Normalized mean should be ~0, got {mean:.4f}"
        # Std should be close to 1 (within 0.2 for more tolerance due to sequence-level normalization)
        assert abs(std - 1.0) < 0.2, f"Normalized std should be ~1, got {std:.4f}"


class TestDataSplits:
    """Test train/val/test data splits."""
    
    def test_split_consistency(self):
        """Test that splits are consistent across runs."""
        data_path = os.path.join(ROOT, 'data', 'preprocessed')
        
        if not os.path.exists(data_path) or not os.listdir(data_path):
            pytest.skip("No preprocessed data available")
        
        # Create two datasets with same mode
        ds1 = SequentialSleepDataset(data_path, mode='train', normalize=False)
        ds2 = SequentialSleepDataset(data_path, mode='train', normalize=False)
        
        # Should have same length (deterministic split)
        assert len(ds1) == len(ds2), "Splits should be deterministic"
    
    def test_split_proportions(self):
        """Test that splits have approximately correct proportions."""
        data_path = os.path.join(ROOT, 'data', 'preprocessed')
        
        if not os.path.exists(data_path) or not os.listdir(data_path):
            pytest.skip("No preprocessed data available")
        
        train_ds = SequentialSleepDataset(data_path, mode='train', normalize=False)
        val_ds = SequentialSleepDataset(data_path, mode='val', normalize=False)
        test_ds = SequentialSleepDataset(data_path, mode='test', normalize=False)
        
        total = len(train_ds) + len(val_ds) + len(test_ds)
        
        if total > 0:
            train_ratio = len(train_ds) / total
            val_ratio = len(val_ds) / total
            test_ratio = len(test_ds) / total
            
            # Train should be ~70%, val ~15%, test ~15%
            assert 0.65 < train_ratio < 0.75, f"Train ratio should be ~0.70, got {train_ratio:.2f}"
            assert 0.10 < val_ratio < 0.20, f"Val ratio should be ~0.15, got {val_ratio:.2f}"
            assert 0.10 < test_ratio < 0.20, f"Test ratio should be ~0.15, got {test_ratio:.2f}"


class TestLabels:
    """Test label encoding and range."""
    
    def test_labels_range(self):
        """Test that labels are in valid range [0, 4]."""
        data_path = os.path.join(ROOT, 'data', 'preprocessed')
        
        if not os.path.exists(data_path) or not os.listdir(data_path):
            pytest.skip("No preprocessed data available")
        
        dataset = SequentialSleepDataset(data_path, mode='train', normalize=False)
        _, labels = dataset[0]
        
        # Labels should be in range [0, 4] (5 classes)
        assert labels.min() >= 0, "Labels should be >= 0"
        assert labels.max() <= 4, "Labels should be <= 4"
        assert labels.dtype == torch.long, "Labels should be long integers"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
